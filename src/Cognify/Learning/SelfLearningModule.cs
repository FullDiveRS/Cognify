using Cognify.Core;

namespace Cognify.Learning;

/// <summary>
/// The self-learning module — a HYBRID of symbolic and neural learning:
///
/// <list type="bullet">
/// <item><b>Neural</b>: a <see cref="MicroNet"/> (real backpropagation) learns to predict
/// reward and valence from stimulus features, generalizing across similar inputs.</item>
/// <item><b>Neural, style-conditioned</b>: a second network — the
/// <see cref="NeuralStyleEngine"/> — learns how rewarding each response STYLE is in
/// the current context and mood, so the mind gets close to "an AI model" feel:
/// every decision is a learned function, not a fixed rule.</item>
/// <item><b>Reinforcement</b>: per-style reward averages (Q-table) are the symbolic
/// half; neural and symbolic scores are blended adaptively — the better the
/// network's predictions get, the more the mind trusts the neural side
/// (see <see cref="NeuralWeight"/>).</item>
/// <item><b>Self-supervised</b>: every cycle trains valence prediction from the
/// emotion module's appraisal — the mind continuously learns without feedback.</item>
/// <item><b>Vocabulary growth</b>: a <see cref="Lexicon"/> picks up new words from the
/// user every cycle; the composer weaves them back into conversation, so the
/// mind literally adds words as it talks.</item>
/// </list>
/// Call <see cref="Mind.Reinforce(double)"/> to close the learning loop.
/// </summary>
public sealed class SelfLearningModule : CognitiveModuleBase
{
    private readonly LearningOptions _options;
    private readonly Queue<Experience> _experiences;
    private readonly Dictionary<int, (double Q, int Count)> _qTable = new();

    private double[]? _lastFeatures;
    private int _lastAction;
    private double _lastValenceTarget;
    private double _lastValence;
    private double _lastArousal = 0.5;
    private int _lastObservedCycle = -1;

    internal SelfLearningModule(LearningOptions options)
    {
        _options = options;
        Net = new MicroNet(options.FeatureDim, options.HiddenUnits, OutputCount, options.Seed);
        StyleNet = new NeuralStyleEngine(options.FeatureDim, StyleCount,
            Math.Max(8, options.HiddenUnits / 2), options.Seed);
        _experiences = new Queue<Experience>(options.ExperienceCapacity);
        Stats = new LearningStats { ExplorationRate = options.BaseExploration };
    }

    /// <summary>
    /// How many style slots the style-conditioned network discriminates.
    /// ResponseComposer defines 7 styles (ids 0..6); two spare slots keep the
    /// input layout stable as styles evolve across versions.
    /// </summary>
    public const int StyleCount = 9;

    /// <summary>Number of network outputs: [0]=reward, [1]=valence, [2]=novelty.</summary>
    public const int OutputCount = 3;

    /// <summary>Default style id used for credit assignment when thinking is disabled.</summary>
    public const int StyleDefault = 0;

    /// <summary>The underlying micro neural network.</summary>
    public MicroNet Net { get; internal set; }

    /// <summary>
    /// The style-conditioned neural network — the neural half of the hybrid
    /// engine that scores (context, style, mood) triples.
    /// </summary>
    public NeuralStyleEngine StyleNet { get; internal set; }

    /// <summary>
    /// How much of the final style score comes from the neural side (0..1).
    /// Starts symbolic-leaning; drifts toward the neural side as the style
    /// net's prediction error shrinks, and back when it degrades — a real
    /// hybrid: learned behavior is trusted only as fast as it earns it.
    /// </summary>
    public double NeuralWeight { get { lock (_gate) return _neuralWeight; } }
    private double _neuralWeight = 0.45;

    private double _stylePredictionError = 0.25; // EMA of |predicted - observed|

    /// <summary>Last smoothed |prediction error| of the style network.</summary>
    public double StylePredictionError { get { lock (_gate) return _stylePredictionError; } }

    /// <summary>The mind's growing word vocabulary (picks up words from the user).</summary>
    public Lexicon Lexicon { get; } = new();

    /// <summary>Running learning statistics.</summary>
    public LearningStats Stats { get; }

    /// <summary>Stored experiences (newest last).</summary>
    public IReadOnlyList<Experience> Experiences { get { lock (_gate) return _experiences.ToList(); } }

    private readonly object _gate = new();

    /// <summary>Read-only view of the Q-table (style id -> (average reward, count)).</summary>
    public IReadOnlyDictionary<int, (double Q, int Count)> QTable { get { lock (_gate) return _qTable; } }

    /// <inheritdoc />
    public override string Name => "learning";

    /// <inheritdoc />
    public override int Order => 60;

    // ------------------------------------------------------------------
    // Per-cycle behavior
    // ------------------------------------------------------------------

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        // Encode features once per cycle for thinking + learning.
        var features = FeatureEncoder.Encode(context.EffectiveContent, _options.FeatureDim);
        context.Features = features;

        // Grow the word vocabulary: listen for words the mind has never
        // heard before. The composer may weave them back into a reply —
        // every new word is one more thing the mind can say.
        var picked = Lexicon.Observe(context.EffectiveContent, context.Cycle);
        if (picked.Count > 0)
            context.Bag["lexicon.picked"] = picked;

        // If no upstream module (thinking) observed an action this cycle,
        // default the credit assignment to the informative style.
        if (_lastObservedCycle != context.Cycle)
            ObserveAction(features, StyleDefault);

        // Self-supervised: train valence prediction from current appraisal.
        if (context.Emotions is { } emotions)
        {
            _lastValenceTarget = (emotions.Valence + 1) / 2.0; // map -1..1 -> 0..1
            Train(features, targetReward: null, targetValence: _lastValenceTarget);
        }

        // Feedback stimuli carry reward metadata -> close the loop automatically.
        if (context.Stimulus.Kind == StimulusKind.Feedback &&
            context.Stimulus.Metadata.TryGetValue("reward", out var raw) &&
            double.TryParse(raw, System.Globalization.CultureInfo.InvariantCulture, out var reward))
        {
            Reinforce(reward);
        }
    }

    /// <summary>
    /// Scores a candidate response style with the HYBRID engine: the
    /// symbolic side (Q-table evidence + the reward net's context
    /// prediction) blended with the style-conditioned neural prediction.
    /// Higher is better. Neutral mood is assumed.
    /// </summary>
    public double Score(double[] features, int styleId) =>
        Score(features, styleId, valence: 0, arousal: 0.5);

    /// <summary>
    /// Hybrid scoring with the mood taken into account — this is the call
    /// deliberation uses. symbolic = 0.4·Q(style) + 0.3·prior + 0.3·rewardNet(context);
    /// neural = styleNet(context, style, mood); the two are mixed by
    /// <see cref="NeuralWeight"/>. The prior carries personality/emotion fit.
    /// </summary>
    public double Score(double[] features, int styleId, double valence, double arousal, double prior = 0.5)
    {
        double q = _qTable.TryGetValue(styleId, out var entry) ? entry.Q : 0.5; // optimistic prior
        double contextReward = Net.Predict(features)[0];
        double symbolic = 0.4 * q + 0.3 * prior + 0.3 * contextReward;
        double neural = StyleNet.Predict(features, styleId, valence, arousal);
        return (1 - _neuralWeight) * symbolic + _neuralWeight * neural;
    }

    /// <summary>Records which style was chosen for this cycle (credit assignment).</summary>
    public void ObserveAction(double[] features, int styleId) =>
        ObserveAction(features, styleId, valence: 0, arousal: 0.5);

    /// <summary>Records the chosen style plus the mood it was chosen in (credit assignment).</summary>
    public void ObserveAction(double[] features, int styleId, double valence, double arousal)
    {
        _lastFeatures = features;
        _lastAction = styleId;
        _lastValence = valence;
        _lastArousal = arousal;
        _lastObservedCycle = Host?.CycleCount ?? 0;
    }

    /// <summary>
    /// Reinforces the last action with a reward (-1..1): trains the network
    /// (gradient descent), updates the Q-table and stores the experience.
    /// </summary>
    public void Reinforce(double reward)
    {
        reward = Math.Clamp(reward, -1, 1);
        double[] features;
        int action;
        lock (_gate)
        {
            if (_lastFeatures is null) return;
            features = _lastFeatures;
            action = _lastAction;
        }

        Train(features, targetReward: (reward + 1) / 2.0, targetValence: null);

        // Train the style-conditioned half of the hybrid engine on the same
        // outcome, and let its prediction accuracy steer how much the mind
        // trusts the neural side next time.
        double targetReward01 = (reward + 1) / 2.0;
        double predicted;
        lock (_gate)
        {
            predicted = StyleNet.Predict(features, action, _lastValence, _lastArousal);
        }
        StyleNet.Learn(features, action, _lastValence, _lastArousal, targetReward01, _options.LearningRate);
        AdaptNeuralWeight(predicted, targetReward01);

        lock (_gate)
        {
            var (q, count) = _qTable.TryGetValue(action, out var e) ? e : (0.5, 0);
            double alpha = 1.0 / (1.0 + count * 0.5); // decaying learning rate per style
            _qTable[action] = (q + alpha * (reward - q), count + 1);

            _experiences.Enqueue(new Experience((double[])features.Clone(), action, reward, DateTimeOffset.UtcNow));
            while (_experiences.Count > _options.ExperienceCapacity) _experiences.Dequeue();

            Stats.TotalExperiences++;
            Stats.AverageReward += 0.1 * (reward - Stats.AverageReward);
        }

        Host?.RaiseLearned(new LearnedEventArgs(
            reward, Stats.AverageReward, Stats.TotalExperiences,
            $"style #{action} reinforced; net loss {Stats.LastLoss:0.000}"));
    }

    /// <summary>
    /// Moves the symbolic/neural blend: accurate predictions push weight
    /// toward the neural side (max 0.8), poor ones back toward symbolic
    /// (min 0.25). Deliberate, small steps keep behavior stable.
    /// </summary>
    private void AdaptNeuralWeight(double predicted, double actual)
    {
        var err = Math.Abs(predicted - actual);
        lock (_gate)
        {
            _stylePredictionError += 0.1 * (err - _stylePredictionError);
            double target = _stylePredictionError < 0.15 ? 0.75
                          : _stylePredictionError > 0.30 ? 0.30
                          : 0.45;
            _neuralWeight += 0.05 * (target - _neuralWeight);
            _neuralWeight = Math.Clamp(_neuralWeight, 0.25, 0.80);
        }
    }

    /// <summary>Runs one gradient-descent step. Either target may be null to skip that head.</summary>
    private void Train(double[] features, double? targetReward, double? targetValence)
    {
        var prediction = Net.Predict(features);
        var target = (double[])prediction.Clone();
        if (targetReward.HasValue) target[0] = targetReward.Value;
        if (targetValence.HasValue) target[1] = targetValence.Value;

        double loss = Net.Train(features, target, _options.LearningRate);
        lock (_gate)
        {
            Stats.TrainingSteps++;
            Stats.LastLoss = loss;
        }
    }

    /// <summary>
    /// Replay: retrains the network on stored experiences (experience replay).
    /// Useful after loading a mind to re-solidify learned behavior.
    /// </summary>
    public double Replay(int epochs = 3)
    {
        double totalLoss = 0;
        int steps = 0;
        var buffer = Experiences;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var experience in buffer)
            {
                var prediction = Net.Predict(experience.Features);
                var target = (double[])prediction.Clone();
                target[0] = (experience.Reward + 1) / 2.0;
                totalLoss += Net.Train(experience.Features, target, _options.LearningRate);
                steps++;
            }
        }
        if (steps > 0) lock (_gate) Stats.LastLoss = totalLoss / steps;
        return steps > 0 ? totalLoss / steps : 0;
    }

    /// <summary>Restores network weights, the style net, Q-table and lexicon from persistence.</summary>
    internal void RestoreState(string netJson,
        IReadOnlyDictionary<int, (double Q, int Count)> qTable, LearningStats stats,
        string? styleNetJson = null, double? neuralWeight = null,
        IEnumerable<string[]>? lexiconEntries = null, IEnumerable<string>? lexiconSpoken = null)
    {
        Net = MicroNet.Deserialize(netJson);
        if (!string.IsNullOrWhiteSpace(styleNetJson))
        {
            try { StyleNet = NeuralStyleEngine.Deserialize(styleNetJson); }
            catch (Cognify.Core.PersistenceException)
            {
                // A style net from an incompatible layout is worse than a
                // fresh one — keep the freshly initialized network.
            }
        }
        if (neuralWeight.HasValue) _neuralWeight = Math.Clamp(neuralWeight.Value, 0.25, 0.80);
        lock (_gate)
        {
            _qTable.Clear();
            foreach (var (key, value) in qTable) _qTable[key] = value;
        }
        Stats.TotalExperiences = stats.TotalExperiences;
        Stats.TrainingSteps = stats.TrainingSteps;
        Stats.AverageReward = stats.AverageReward;
        Stats.LastLoss = stats.LastLoss;
        Stats.ExplorationRate = stats.ExplorationRate;
        if (lexiconEntries is not null) Lexicon.Restore(lexiconEntries, lexiconSpoken);
    }

    /// <summary>Serializes the Q-table for persistence.</summary>
    public IReadOnlyDictionary<int, (double Q, int Count)> SnapshotQTable()
    {
        lock (_gate) return new Dictionary<int, (double, int)>(_qTable);
    }
}
