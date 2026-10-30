using Cognify.Core;

namespace Cognify.Emotions;

/// <summary>
/// The emotion module. Every cognitive cycle it appraises the stimulus
/// (what does this mean for me?), blends the fresh appraisal into the
/// standing emotional state, updates mood, and exposes the state to
/// downstream modules (thinking, learning).
/// </summary>
public sealed class EmotionModule : CognitiveModuleBase
{
    private readonly EmotionOptions _options;
    private readonly MoodEngine _engine = new();
    private EmotionalState _current = new();
    private EmotionType? _lastReported;

    internal EmotionModule(EmotionOptions options) => _options = options;

    /// <inheritdoc />
    public override string Name => "emotions";

    /// <inheritdoc />
    public override int Order => 20;

    /// <summary>The mind's current emotional state.</summary>
    public EmotionalState Current => _current;

    /// <summary>Emotions above this intensity raise <see cref="Mind.EmotionChanged"/>.</summary>
    public double ReportThreshold { get; set; } = 0.30;

    /// <summary>Directly makes the mind feel an emotion (e.g. from your application).</summary>
    public void Feel(EmotionType emotion, double amount)
    {
        _current.Feel(emotion, amount * _options.IntensityScale);
        Report();
    }

    /// <summary>Clears all emotions back to neutral.</summary>
    public void Calm() => _current = new EmotionalState();

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var reward = ExtractReward(context.Stimulus);
        var appraisal = _engine.Appraise(new AppraisalInput
        {
            Text = context.EffectiveContent,
            Intensity = context.Stimulus.Intensity,
            Unexpected = context.Stimulus.Kind == StimulusKind.Event &&
                         context.Mind.Rng.NextDouble() < 0.2,
            Reward = reward
        }, context.Mind.Personality);

        _current.Blend(appraisal, weight: 0.9);

        // Expectation violation (appraisal research: surprise IS violated
        // expectation): a statement that negates something the mind knows
        // startles it, even when nothing in the wording is emotional.
        bool contradictsKnown = context.Mind.Reasoning is { } reasoner &&
                                reasoner.KnownContradiction(context.EffectiveContent);
        if (contradictsKnown)
        {
            _current.Feel(EmotionType.Surprise, 0.45);
            appraisal.Feel(EmotionType.Surprise, 0.45);
        }

        _current.UpdateMood(_options.MoodInertia);
        context.Emotions = _current.Clone();

        // Dual-process gate (Zajonc's affect primacy): usually the mind
        // thinks BEFORE it speaks — but a strong enough surge answers from
        // feeling alone. Blame and dismissal are reflexive (appraisal
        // research: other-accountability triggers anger instantly); other
        // emotions must cross ImpulseThreshold. The thinking module reads
        // this flag and, when it has nothing confident to say, answers
        // impulsively instead of deliberating.
        var surge = appraisal.Dominant;
        var surgeIntensity = appraisal.DominantIntensity;
        var triggers = MoodEngine.DetectTriggers(context.EffectiveContent);
        bool reflexive = triggers.SelfInsult || triggers.Dismissal || contradictsKnown;
        if (ImpulseResponses.IsImpulseProne(surge) &&
            (reflexive || surgeIntensity >= _options.ImpulseThreshold))
        {
            context.Bag["emotions.impulse"] = EmotionalState.Humanize(surge);
        }
        else
        {
            context.Bag.Remove("emotions.impulse");
        }

        // Epistemic surge (information-gap theory): a FRESH surprise or a
        // spike of interest opens a knowledge gap — the mind will want to
        // ASK this turn. Flagged from the fresh appraisal (not the standing
        // blend) so a single spike does not latch the question mode forever.
        if (appraisal.Intensities[EmotionType.Surprise] > 0.25 ||
            appraisal.Intensities[EmotionType.Anticipation] > 0.25)
        {
            context.Bag["emotions.epistemic"] = "1";
        }
        else
        {
            context.Bag.Remove("emotions.epistemic");
        }

        Report();
    }

    /// <inheritdoc />
    public override void Tick()
    {
        _current.Decay(_options.DecayRate);
        _current.UpdateMood(_options.MoodInertia * 0.5);
        Report();
    }

    private void Report()
    {
        var dominant = _current.Dominant;
        var intensity = _current.DominantIntensity;
        if (_lastReported != dominant && intensity > ReportThreshold)
        {
            _lastReported = dominant;
            Host?.RaiseEmotionChanged(dominant, intensity, _current.Valence);
        }
    }

    private static double? ExtractReward(Stimulus stimulus)
    {
        if (stimulus.Kind != StimulusKind.Feedback) return null;
        return stimulus.Metadata.TryGetValue("reward", out var raw) &&
               double.TryParse(raw, System.Globalization.CultureInfo.InvariantCulture, out var reward)
            ? reward
            : null;
    }
}
