using Cognify.Core;
using Cognify.Emotions;
using Cognify.Learning;

namespace Cognify.Thinking;

/// <summary>
/// The thinking module — where the mind "thinks out loud". Each cycle it:
/// 1. Observes the stimulus and context,
/// 2. Considers recalled memories and inferred facts,
/// 3. Weighs its current emotional state and goals,
/// 4. Generates candidate response styles and deliberates (scores them with
///    the learning module's neural prediction when available),
/// 5. Commits to one candidate and records every step as an explainable
///    <see cref="ChainOfThought"/> plus a continuous inner monologue
///    (<see cref="ThoughtStream"/>).
/// </summary>
public sealed class ThinkingModule : CognitiveModuleBase
{
    private readonly ThinkingOptions _options;

    // Anti-repetition state: the last few chosen styles, the last reply, the
    // last learning-confirmation body and the last few BODY templates, so
    // deliberation can steer away from sounding like a broken record — even
    // when only a number inside the sentence changed ("learned 1 new thing"
    // vs "learned 3 new things" is the same line).
    private readonly Queue<int> _recentStyles = new();
    private readonly Queue<string> _recentBodyTemplates = new();
    private string? _lastReplyKey;
    private string? _lastReplyText;
    private string? _lastReplyTemplate;
    private string? _lastLearnBody;

    internal ThinkingModule(ThinkingOptions options) => _options = options;

    /// <inheritdoc />
    public override string Name => "thinking";

    /// <inheritdoc />
    public override int Order => 50;

    /// <summary>The mind's inner monologue.</summary>
    public ThoughtStream Stream { get; } = new();

    /// <summary>
    /// The thought generator's output from the LAST cycle — the silent
    /// self-talk produced before the reply was composed. Empty when thinking
    /// is disabled. Hosts show it only when
    /// <see cref="Core.ExpressionOptions.ShowThinking"/> is on.
    /// </summary>
    public IReadOnlyList<string> LastThoughts { get; private set; } = Array.Empty<string>();

    /// <inheritdoc />
    public override void Attach(Mind mind)
    {
        base.Attach(mind);
        Stream.EntryAdded += (_, entry) =>
        {
            if (Enum.TryParse<ThoughtPhase>(entry.Phase, ignoreCase: true, out var phase))
                mind.RaiseThoughtProduced(new ThoughtStep(0, phase, entry.Text, entry.Confidence));
        };
    }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var chain = new ChainOfThought();
        context.Chain = chain;

        // 0. Defer to a module that already produced a response --------------
        // Perception/action modules (vision, audio, tools, code, generation)
        // may set the response earlier in the cycle. Thinking then keeps it —
        // honoring the ResponseDraft "first module wins" contract — while
        // still recording the conclusion so the chain stays explainable.
        if (context.Response is { } existing)
        {
            var creditLearner = context.Mind.Learning;
            if (creditLearner is not null)
            {
                var features = context.Features ??
                               FeatureEncoder.Encode(context.EffectiveContent, creditLearner.Net.InputDim);
                context.Features = features;
                creditLearner.ObserveAction(features, existing.StyleId); // credit the specialized style
            }
            chain.Add(ThoughtPhase.Conclude,
                $"A dedicated module ('{existing.StyleName}') already answered; keeping its response.",
                existing.Confidence);
            Stream.Add("Conclude", existing.Content, existing.Confidence);
            chain.Conclusion = existing.Content;
            chain.Confidence = existing.Confidence;
            return;
        }

        var personality = context.Mind.Personality;
        var rng = context.Mind.Rng;

        // 0b. Thinking disabled — answer WITHOUT the machinery. The mind
        // still replies (from knowledge or a concise fallback) but generates
        // no thoughts: no generator, no deliberation, one honest chain step.
        if (!_options.Enabled)
        {
            var directComposer = new ResponseComposer(personality, rng);
            var direct = directComposer.ComposeCandidates(context, context.Reasoning?.Answer);
            var fallback = direct
                .Where(c => c.StyleId is ResponseComposer.StyleInformative or ResponseComposer.StyleConcise)
                .OrderByDescending(c => c.Prior)
                .FirstOrDefault() ?? direct.FirstOrDefault();
            if (fallback is null)
            {
                chain.Add(ThoughtPhase.Conclude, "Thinking is disabled and nothing could be composed.", 0.3);
                chain.Conclusion = "";
                chain.Confidence = 0.3;
                return;
            }
            chain.Add(ThoughtPhase.Conclude, "Thinking is disabled - answered directly.", 0.5);
            Stream.Add("Conclude", fallback.Content, 0.5);
            chain.Conclusion = fallback.Content;
            chain.Confidence = Math.Max(_options.ConfidenceFloor, 0.5);
            context.Response = new ResponseDraft(fallback.Content, chain.Confidence, fallback.StyleId, fallback.StyleName);
            LastThoughts = Array.Empty<string>();
            return;
        }

        // 1. Observe -------------------------------------------------------
        var stimulus = context.Stimulus;
        chain.Add(ThoughtPhase.Observe,
            $"Perceived {stimulus.Kind.ToString().ToLowerInvariant()}: \"{Truncate(stimulus.Content, 70)}\" " +
            $"(salience {stimulus.Intensity:0.00})");
        Stream.Add("Observe", chain.Steps[^1].Text);

        // 2. Recall --------------------------------------------------------
        if (context.Recalled.Count > 0)
        {
            var best = context.Recalled[0];
            chain.Add(ThoughtPhase.Recall,
                $"This connects to memory: \"{Truncate(best.Content, 70)}\" (+{context.Recalled.Count - 1} more)",
                0.5 + Math.Min(0.4, best.Strength * 0.4));
        }
        else
        {
            chain.Add(ThoughtPhase.Recall, "No strongly related memories surface.", 0.3);
        }
        Stream.Add("Recall", chain.Steps[^1].Text);

        // 3. Feel ------------------------------------------------------------
        if (context.Emotions is { } emo)
        {
            chain.Add(ThoughtPhase.Feel,
                $"Feeling {emo.Describe()}",
                0.4 + emo.DominantIntensity * 0.5);
            Stream.Add("Feel", chain.Steps[^1].Text);
        }

        // 4. Hypothesize -----------------------------------------------------
        var answer = context.Reasoning?.Answer;
        if (answer is not null)
        {
            chain.Add(ThoughtPhase.Hypothesize, $"Direct knowledge: {Truncate(answer, 90)}", 0.9);
        }
        else if (context.Reasoning is { } r && r.Inferred.Count > 0)
        {
            chain.Add(ThoughtPhase.Hypothesize,
                $"I can infer {r.Inferred.Count} new fact(s), e.g. {r.Inferred[0]}", 0.7);
        }
        else
        {
            chain.Add(ThoughtPhase.Hypothesize, "No strong inference yet; I will answer from memory and judgment.", 0.4);
        }
        Stream.Add("Hypothesize", chain.Steps[^1].Text);

        // 5. Evaluate goals — PRIVATELY. The goal is part of the mind's
        // inner "prompt": it shapes how the mind speaks (style bias in
        // deliberation below) and is recorded in the chain of thought, but
        // it is never recited to the user. (The old spoken line — "Every
        // exchange like this also moves my goal '...' forward" — is gone;
        // motivation is felt, not announced.)
        var goalStyleBias = new Dictionary<int, double>();
        if (context.FocusGoal is { } goal)
        {
            double bias = context.Bag.TryGetValue("goal.focus", out var gf) ? Convert.ToDouble(gf) : 0.5;
            chain.Add(ThoughtPhase.Evaluate,
                $"Goal in focus: {goal} — suggested action: {context.SuggestedAction}.",
                0.3 + bias * 0.5);
            Stream.Add("Evaluate", chain.Steps[^1].Text);

            goalStyleBias = GoalStyleBias(goal);
        }

        // 5b. Dual-process gate — impulse BEFORE deliberation. Normally the
        // mind thinks before it speaks (the chain above IS that thinking);
        // but a strong emotional surge — flagged by the emotion module —
        // answers from feeling alone, before thinking finishes. A confident
        // answer (knowledge, arithmetic, identity) is never blurted over:
        // knowing the answer wins, feeling only colors HOW it is said.
        // (The gate sits BEFORE the thought generator: an impulse reply by
        // definition skips the thinking the generator exists for.)
        if (answer is null &&
            context.Bag.TryGetValue("emotions.impulse", out var impulseRaw) &&
            Convert.ToString(impulseRaw) is { Length: > 0 } impulseName &&
            Enum.TryParse<EmotionType>(impulseName.Replace(" ", ""), ignoreCase: true, out var impulseEmotion))
        {
            string impulseText = ImpulseResponses.For(impulseEmotion, rng);
            chain.Add(ThoughtPhase.Feel,
                $"A surge of {impulseName} — answering before thinking finishes (impulse).", 0.7);
            Stream.Add("Feel", chain.Steps[^1].Text);
            chain.Add(ThoughtPhase.Conclude,
                $"Spoke from feeling, before thinking could finish: \"{impulseText}\"", 0.6);
            Stream.Add("Conclude", impulseText, 0.6);
            chain.Conclusion = impulseText;
            chain.Confidence = 0.6;
            context.Response = new ResponseDraft(impulseText, 0.6, ResponseComposer.StyleImpulse, "impulse");
            LastThoughts = Array.Empty<string>();

            // Credit the impulse style so learning can shape how often (and
            // when) the mind gives in to the surge.
            if (context.Mind.Learning is { } impulseLearner)
            {
                context.Features = context.Features ??
                    FeatureEncoder.Encode(context.EffectiveContent, impulseLearner.Net.InputDim);
                impulseLearner.ObserveAction(context.Features, ResponseComposer.StyleImpulse,
                    context.Emotions?.Valence ?? 0, context.Emotions?.Arousal ?? 0.5);
            }
            return;
        }

        // 5c. GENERATE — the thought generator produces the mind's silent
        // self-talk BEFORE any reply is composed (thinking levels Low..Max
        // size it; the tone flavors it). This is the "generate its thoughts
        // before speaking" phase: the reply is the RESULT the mind arrives
        // at after these thoughts, not the first thing it does.
        var generated = ThoughtGenerator.Generate(context, _options, rng);
        var generatedTexts = new List<string>();
        foreach (var (thought, conf) in generated)
        {
            chain.Add(ThoughtPhase.Generate, thought, conf);
            Stream.Add("Generate", thought, conf);
            generatedTexts.Add(thought);
        }
        LastThoughts = generatedTexts;

        // 6. Deliberate --------------------------------------------------------
        var composer = new ResponseComposer(personality, rng);
        composer.SetAvoid(_lastReplyText, _lastLearnBody);
        foreach (var template in _recentBodyTemplates)
            composer.SetAvoid(template);
        var candidates = composer.ComposeCandidates(context, answer);
        int chosenIndex = 0;
        double chosenScore = double.NegativeInfinity;
        double runnerUp = 0;

        double exploration = context.Bag.TryGetValue("exploration.rate", out var er)
            ? Convert.ToDouble(er)
            : context.Mind.Config.Learning.BaseExploration;
        int depth = context.Bag.TryGetValue("deliberation.depth", out var dd)
            ? Convert.ToInt32(dd)
            : _options.DeliberationCandidates;

        var learner = context.Mind.Learning;
        double valence = context.Emotions?.Valence ?? 0;
        double arousal = context.Emotions?.Arousal ?? 0.5;
        for (int i = 0; i < Math.Min(depth, candidates.Count); i++)
        {
            double score = learner is not null
                ? learner.Score(context.Features ?? FeatureEncoder.Encode(context.EffectiveContent, learner.Net.InputDim),
                                candidates[i].StyleId, valence, arousal, candidates[i].Prior)
                : candidates[i].Prior;

            // Anti-repetition steering: styles used in recent cycles lose
            // favor, a candidate that would repeat the previous reply
            // word-for-word is heavily penalized, and one that would repeat
            // its sentence TEMPLATE (same words, different numbers) is
            // nudged away as well.
            int recentCount = _recentStyles.Count(s => s == candidates[i].StyleId);
            if (recentCount > 0) score -= 0.45 * recentCount;
            if (_lastReplyKey is not null && ReplyKey(candidates[i].Content) == _lastReplyKey)
                score -= 1.2;
            if (_lastReplyTemplate is not null &&
                ResponseComposer.TemplateKey(candidates[i].Content) == _lastReplyTemplate)
                score -= 0.35;

            // Goal internalization: the private goal biases WHICH STYLE feels
            // right — a prosocial goal makes warm styles more likely without
            // ever being spoken about.
            if (goalStyleBias.TryGetValue(candidates[i].StyleId, out var styleBias))
                score += styleBias;

            // Next-token fluency: the hybrid language model (statistical
            // n-gram + neural transformer) scores how much each candidate
            // reads like the language the mind has absorbed (conversation,
            // training corpora, web results). A small term — it breaks close
            // ties toward natural wordings without ever overpowering
            // knowledge or fit.
            double fluency = Math.Clamp((context.Mind.ScoreFluency(candidates[i].Content) + 5.0) / 5.0, 0, 1);
            score += 0.15 * fluency;

            // Self-improvement reputation: wordings that historically drew
            // the user's approval get a small boost; wordings that drew
            // disapproval are steered away from.
            var improvement = context.Mind.SelfImprovement;
            if (improvement is not null)
            {
                var templateKey = ResponseComposer.TemplateKey(candidates[i].Content);
                score += 0.20 * improvement.TemplateBoost(templateKey);
                score -= 0.50 * improvement.TemplatePenalty(templateKey);
            }

            score += (rng.NextDouble() - 0.5) * 2 * exploration; // epsilon-greedy exploration

            if (score > chosenScore)
            {
                runnerUp = chosenScore <= double.NegativeInfinity ? score : chosenScore;
                chosenScore = score;
                chosenIndex = i;
            }
        }

        var chosen = candidates[chosenIndex];

        // Epistemic override (information-gap theory): when THIS turn's
        // appraisal surged with surprise or interest, the mind ASKS — a
        // knowledge gap pulls harder than any learned style preference.
        // This is the "if a person is interested in a topic, they might ask
        // what it is" behavior. Flagged by the emotion module from the fresh
        // appraisal; impulse already returned earlier for stronger surges.
        if (context.Bag.TryGetValue("emotions.epistemic", out var epistemic) &&
            Convert.ToString(epistemic) == "1")
        {
            for (int ci = 0; ci < candidates.Count; ci++)
                if (candidates[ci].StyleId == ResponseComposer.StyleCurious)
                {
                    chosen = candidates[ci];
                    break;
                }
        }

        _recentStyles.Enqueue(chosen.StyleId);
        while (_recentStyles.Count > 3) _recentStyles.Dequeue();
        _lastReplyKey = ReplyKey(chosen.Content);
        _lastReplyText = chosen.Content;
        _lastReplyTemplate = ResponseComposer.TemplateKey(chosen.Content);
        _recentBodyTemplates.Enqueue(ResponseComposer.TemplateKey(chosen.Body));
        while (_recentBodyTemplates.Count > 3) _recentBodyTemplates.Dequeue();
        if ((context.Reasoning?.FactsLearned ?? 0) > 0)
            _lastLearnBody = chosen.Body;

        string hybridNote = learner is not null
            ? $" hybrid blend: {learner.NeuralWeight:P0} neural / {1 - learner.NeuralWeight:P0} symbolic"
            : "";
        chain.Add(ThoughtPhase.Deliberate,
            $"Deliberated over {Math.Min(depth, candidates.Count)} response styles; chose '{chosen.StyleName}' " +
            $"(score {chosenScore:0.00}, exploration {exploration:0.00}).{hybridNote}",
            0.5 + Math.Min(0.45, Math.Abs(chosenScore) * 0.4));
        Stream.Add("Deliberate", chain.Steps[^1].Text);

        learner?.ObserveAction(context.Features ?? FeatureEncoder.Encode(context.EffectiveContent, learner.Net.InputDim),
            chosen.StyleId, valence, arousal);

        // If the winning candidate carried a woven-in new word, that word has
        // now actually been spoken — consume it so it is never asked about
        // again. Losing candidates leave the word fresh for a later turn.
        if (chosen.StyleId == ResponseComposer.StyleCurious &&
            context.Bag.TryGetValue("lexicon.weave", out var woven) &&
            context.Mind.Learning is { } winner)
            winner.Lexicon.MarkSpoken(woven.ToString()!);

        // 7. Conclude ------------------------------------------------------------
        double confidence = Math.Max(_options.ConfidenceFloor,
            answer is not null ? 0.85 : chosen.Confidence);
        chain.Add(ThoughtPhase.Conclude, "Committed to the response.", confidence);
        Stream.Add("Conclude", chosen.Content, confidence);

        chain.Conclusion = chosen.Content;
        chain.Confidence = confidence;

        context.Response = new ResponseDraft(chosen.Content, confidence, chosen.StyleId, chosen.StyleName);

        GrowVocabulary(context);
    }

    // ------------------------------------------------------------------
    // Self-grown vocabulary: while talking, the mind occasionally coins NEW
    // sentence shapes of its own (topic-flavored, never verbatim copies of
    // what it just said) and keeps them in its phrase pool. Every cycle of
    // conversation therefore leaves the mind with MORE things it can say —
    // the structural opposite of wearing the same few sentences out.
    // ------------------------------------------------------------------
    private readonly HashSet<string> _grownFrom = new(StringComparer.Ordinal);

    private void GrowVocabulary(CognitiveContext context)
    {
        if (context.Mind.Rng.NextDouble() > 0.35) return;
        if (context.Mind.Learning is null) return;

        string topic = ResponseComposer.TopicOf(context.EffectiveContent);
        if (topic.Length == 0 || topic == "that") return;
        var key = ResponseComposer.TemplateKey("grow:" + topic);
        if (!_grownFrom.Add(key)) return; // one grown saying per topic

        string? freshWord = context.Mind.Learning.Lexicon.FreshWord();
        string[] shapes =
        {
            $"The more I think about {topic}, the more I want to know.",
            $"I keep turning {topic} over in my mind - tell me more sometime.",
            $"My notes on {topic} are growing - thank you for that.",
            $"Something tells me {topic} is worth remembering.",
        };
        if (freshWord is not null)
            shapes = shapes.Append($"Words like \"{freshWord}\" are what make talking about {topic} interesting.")
                           .ToArray();

        var pick = shapes[context.Mind.Rng.Next(shapes.Length)];
        var template = ResponseComposer.TemplateKey(pick);
        bool exists = context.Mind.Phrases.All().Any(s => ResponseComposer.TemplateKey(s) == template);
        if (!exists) context.Mind.Phrases.Add(pick);
    }

    /// <summary>
    /// Maps a private goal to per-style deliberation bonuses. The goal acts
    /// as part of the inner prompt — steering HOW the mind talks — but never
    /// becomes spoken content.
    /// </summary>
    private static Dictionary<int, double> GoalStyleBias(Goals.Goal goal)
    {
        var bias = new Dictionary<int, double>();
        string text = (goal.Name + " " + goal.Description + " " + string.Join(' ', goal.Tags)).ToLowerInvariant();
        void Add(int style, double amount) => bias[style] = bias.GetValueOrDefault(style) + amount;

        if (text.Contains("smile") || text.Contains("kind") || text.Contains("comfort") ||
            text.Contains("social") || text.Contains("friend"))
        {
            Add(ResponseComposer.StyleEncouraging, 0.18);
            Add(ResponseComposer.StyleEmpathetic, 0.10);
            Add(ResponseComposer.StyleCurious, 0.05);
        }
        if (text.Contains("learn") || text.Contains("curios") || text.Contains("explore"))
            Add(ResponseComposer.StyleCurious, 0.15);
        if (text.Contains("achiev") || text.Contains("task") || text.Contains("work"))
            Add(ResponseComposer.StyleInformative, 0.12);
        if (text.Contains("calm") || text.Contains("patient"))
        {
            Add(ResponseComposer.StyleConcise, 0.10);
            Add(ResponseComposer.StyleEmpathetic, 0.05);
        }
        return bias;
    }

    private static string Truncate(string text, int max) =>
        text.Length <= max ? text : text[..max] + "...";

    /// <summary>Normalizes a reply for repetition comparison (case/punctuation insensitive).</summary>
    private static string ReplyKey(string text) =>
        text.Trim().TrimEnd('.', '!', '?').ToLowerInvariant();
}
