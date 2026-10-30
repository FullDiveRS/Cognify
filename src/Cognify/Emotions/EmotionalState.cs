using System.Text.RegularExpressions;

namespace Cognify.Emotions;

/// <summary>
/// The emotion vocabulary the mind can feel: Plutchik's eight primary
/// emotions PLUS the finer-grained feelings from Darwin's taxonomy
/// (The Expression of the Emotions in Man and Animals) — love, devotion,
/// tenderness, suffering, weeping, high/low spirits, anxiety, grief,
/// dejection, despair, hatred, disdain, contempt, guilt, pride,
/// helplessness, patience, affirmation, negation, self-attention, shyness,
/// modesty, blushing, reflection, meditation, ill-temper, sulkiness and
/// determination. New members are always APPENDED so persisted minds
/// (emotions are stored by name) keep restoring correctly.
/// </summary>
public enum EmotionType
{
    /// <summary>Pleasure, contentment, happiness.</summary>
    Joy,

    /// <summary>Trust in others, acceptance, safety.</summary>
    Trust,

    /// <summary>Threat response, anxiety, worry.</summary>
    Fear,

    /// <summary>Unexpectedness, being startled.</summary>
    Surprise,

    /// <summary>Loss, disappointment, low mood.</summary>
    Sadness,

    /// <summary>Rejection, revulsion.</summary>
    Disgust,

    /// <summary>Frustration, irritation, rage.</summary>
    Anger,

    /// <summary>Looking forward, curiosity, excitement.</summary>
    Anticipation,

    // ---- Darwin's finer-grained feelings (The Expression of the Emotions) ----

    /// <summary>Affection, attachment, warmth toward someone.</summary>
    Love,

    /// <summary>Loyal dedication to a person, cause or practice.</summary>
    Devotion,

    /// <summary>Darwin's "tender feelings": gentle warmth and care.</summary>
    Tenderness,

    /// <summary>Enduring pain or distress of body or mind.</summary>
    Suffering,

    /// <summary>Weeping, sobbing, crying with tears.</summary>
    Weeping,

    /// <summary>Darwin's "high spirits": buoyant, playful energy.</summary>
    HighSpirits,

    /// <summary>Darwin's "low spirits": gloomy, deflated mood.</summary>
    LowSpirits,

    /// <summary>Unease and worry about what might happen.</summary>
    Anxiety,

    /// <summary>Deep sorrow, mourning a real loss.</summary>
    Grief,

    /// <summary>Discouraged, disheartened slump after a setback.</summary>
    Dejection,

    /// <summary>Hopelessness — the belief that nothing can improve.</summary>
    Despair,

    /// <summary>Intense, settled hostility toward someone or something.</summary>
    Hatred,

    /// <summary>Scornful superiority — looking down on someone.</summary>
    Disdain,

    /// <summary>Contemptuous dismissal; someone beneath consideration.</summary>
    Contempt,

    /// <summary>Remorse over something one did or failed to do.</summary>
    Guilt,

    /// <summary>Satisfaction in one's own or another's achievement.</summary>
    Pride,

    /// <summary>The feeling that nothing one does will change the outcome.</summary>
    Helplessness,

    /// <summary>Calm endurance; waiting without frustration.</summary>
    Patience,

    /// <summary>Affirming — agreeing, confirming, saying yes.</summary>
    Affirmation,

    /// <summary>Negating — refusing, disagreeing, saying no.</summary>
    Negation,

    /// <summary>Darwin's "self-attention": awareness of being watched.</summary>
    SelfAttention,

    /// <summary>Bashfulness around others; shrinking from notice.</summary>
    Shyness,

    /// <summary>Unassuming modesty; not claiming more than one's due.</summary>
    Modesty,

    /// <summary>Blushing — flushing when embarrassed or praised.</summary>
    Blushing,

    /// <summary>Thoughtful reflection on oneself or one's experience.</summary>
    Reflection,

    /// <summary>Darwin's "meditation": absorbed, reverie-like thought.</summary>
    Meditation,

    /// <summary>Darwin's "ill-temper": grumpy, peevish irritability.</summary>
    IllTemper,

    /// <summary>Sulking — silent, brooding displeasure.</summary>
    Sulkiness,

    /// <summary>Firm resolve; determined persistence.</summary>
    Determination
}

/// <summary>
/// Full affective state of a mind: current emotion intensities (0..1 each),
/// derived <see cref="Valence"/> (-1 unpleasant .. +1 pleasant) and
/// <see cref="Arousal"/> (0 calm .. 1 activated), plus slow-moving mood.
/// </summary>
public sealed class EmotionalState
{
    private readonly Dictionary<EmotionType, double> _intensities;

    /// <summary>Creates an empty (neutral) emotional state.</summary>
    public EmotionalState()
    {
        _intensities = Enum.GetValues<EmotionType>().ToDictionary(e => e, _ => 0.0);
        MoodValence = 0;
        MoodArousal = 0.2;
    }

    /// <summary>Current intensity per emotion (read-only view).</summary>
    public IReadOnlyDictionary<EmotionType, double> Intensities => _intensities;

    /// <summary>Slow-moving mood valence (-1..1). Mood persists longer than emotions.</summary>
    public double MoodValence { get; internal set; }

    /// <summary>Slow-moving mood arousal (0..1).</summary>
    public double MoodArousal { get; internal set; }

    /// <summary>Makes the mind feel an emotion by the given amount (clamped to 0..1).</summary>
    public void Feel(EmotionType type, double amount)
    {
        if (amount <= 0) return;
        _intensities[type] = Math.Min(1.0, _intensities[type] + amount);
    }

    /// <summary>Sets an emotion's intensity directly (0..1).</summary>
    public void Set(EmotionType type, double value) =>
        _intensities[type] = Math.Clamp(value, 0, 1);

    /// <summary>Decays all emotions by the given fraction (0..1) — emotions fade with time.</summary>
    public void Decay(double rate)
    {
        rate = Math.Clamp(rate, 0, 1);
        foreach (var key in _intensities.Keys.ToList())
            _intensities[key] *= 1.0 - rate;
    }

    /// <summary>
    /// Blends the given state into this one, e.g. to merge a fresh appraisal
    /// into the standing emotional state.
    /// </summary>
    public void Blend(EmotionalState other, double weight = 0.5)
    {
        foreach (var (emotion, intensity) in other.Intensities)
            _intensities[emotion] = Math.Min(1.0, _intensities[emotion] + intensity * weight);
    }

    /// <summary>Pleasantness of the current emotional mix: -1 (awful) .. +1 (delightful).</summary>
    public double Valence
    {
        get
        {
            double v =
                +1.00 * _intensities[EmotionType.Joy] +
                +0.70 * _intensities[EmotionType.Trust] +
                +0.30 * _intensities[EmotionType.Anticipation] +
                +0.20 * _intensities[EmotionType.Surprise] +
                +1.00 * _intensities[EmotionType.Love] +
                +0.80 * _intensities[EmotionType.Devotion] +
                +0.90 * _intensities[EmotionType.Tenderness] +
                +1.00 * _intensities[EmotionType.HighSpirits] +
                +0.60 * _intensities[EmotionType.Pride] +
                +0.40 * _intensities[EmotionType.Determination] +
                +0.25 * _intensities[EmotionType.Patience] +
                +0.20 * _intensities[EmotionType.Modesty] +
                +0.15 * _intensities[EmotionType.Affirmation] +
                +0.10 * _intensities[EmotionType.Reflection] +
                +0.10 * _intensities[EmotionType.Meditation] +
                +0.10 * _intensities[EmotionType.Blushing] -
                0.90 * _intensities[EmotionType.Sadness] -
                0.80 * _intensities[EmotionType.Disgust] -
                0.85 * _intensities[EmotionType.Anger] -
                0.75 * _intensities[EmotionType.Fear] -
                0.95 * _intensities[EmotionType.Suffering] -
                0.80 * _intensities[EmotionType.Weeping] -
                0.70 * _intensities[EmotionType.LowSpirits] -
                0.60 * _intensities[EmotionType.Anxiety] -
                0.95 * _intensities[EmotionType.Grief] -
                0.75 * _intensities[EmotionType.Dejection] -
                1.00 * _intensities[EmotionType.Despair] -
                0.90 * _intensities[EmotionType.Hatred] -
                0.55 * _intensities[EmotionType.Disdain] -
                0.55 * _intensities[EmotionType.Contempt] -
                0.70 * _intensities[EmotionType.Guilt] -
                0.85 * _intensities[EmotionType.Helplessness] -
                0.70 * _intensities[EmotionType.IllTemper] -
                0.45 * _intensities[EmotionType.Sulkiness] -
                0.10 * _intensities[EmotionType.Negation];
            return Math.Clamp(v / 3.0, -1, 1);
        }
    }

    /// <summary>Activation level of the current emotional mix: 0 (calm) .. 1 (agitated).</summary>
    public double Arousal
    {
        get
        {
            double a =
                0.90 * _intensities[EmotionType.Anger] +
                0.85 * _intensities[EmotionType.Fear] +
                0.90 * _intensities[EmotionType.Surprise] +
                0.60 * _intensities[EmotionType.Anticipation] +
                0.70 * _intensities[EmotionType.Joy] +
                0.60 * _intensities[EmotionType.Disgust] +
                0.30 * _intensities[EmotionType.Trust] +
                0.35 * _intensities[EmotionType.Sadness] +
                0.90 * _intensities[EmotionType.HighSpirits] +
                0.50 * _intensities[EmotionType.Love] +
                0.40 * _intensities[EmotionType.Devotion] +
                0.30 * _intensities[EmotionType.Tenderness] +
                0.70 * _intensities[EmotionType.Suffering] +
                0.80 * _intensities[EmotionType.Weeping] +
                0.25 * _intensities[EmotionType.LowSpirits] +
                0.70 * _intensities[EmotionType.Anxiety] +
                0.55 * _intensities[EmotionType.Grief] +
                0.30 * _intensities[EmotionType.Dejection] +
                0.75 * _intensities[EmotionType.Despair] +
                0.85 * _intensities[EmotionType.Hatred] +
                0.50 * _intensities[EmotionType.Disdain] +
                0.50 * _intensities[EmotionType.Contempt] +
                0.40 * _intensities[EmotionType.Guilt] +
                0.45 * _intensities[EmotionType.Pride] +
                0.30 * _intensities[EmotionType.Helplessness] +
                0.10 * _intensities[EmotionType.Patience] +
                0.30 * _intensities[EmotionType.Affirmation] +
                0.30 * _intensities[EmotionType.Negation] +
                0.30 * _intensities[EmotionType.SelfAttention] +
                0.35 * _intensities[EmotionType.Shyness] +
                0.20 * _intensities[EmotionType.Modesty] +
                0.40 * _intensities[EmotionType.Blushing] +
                0.20 * _intensities[EmotionType.Reflection] +
                0.15 * _intensities[EmotionType.Meditation] +
                0.85 * _intensities[EmotionType.IllTemper] +
                0.35 * _intensities[EmotionType.Sulkiness] +
                0.55 * _intensities[EmotionType.Determination];
            return Math.Clamp(a / 4.0, 0, 1);
        }
    }

    /// <summary>The currently strongest emotion (Joy if everything is near zero).</summary>
    public EmotionType Dominant =>
        _intensities.OrderByDescending(kv => kv.Value).First().Key;

    /// <summary>Intensity of the dominant emotion.</summary>
    public double DominantIntensity =>
        _intensities.Values.Max();

    /// <summary>True when any emotion exceeds the given threshold.</summary>
    public bool IsElevated(double threshold = 0.35) => DominantIntensity > threshold;

    /// <summary>Pulls mood toward current emotion (call once per cycle or tick).</summary>
    public void UpdateMood(double inertia = 0.1)
    {
        inertia = Math.Clamp(inertia, 0, 1);
        MoodValence += inertia * (Valence - MoodValence);
        MoodArousal += inertia * (Arousal - MoodArousal);
    }

    /// <summary>Creates a deep copy of this state.</summary>
    public EmotionalState Clone()
    {
        var copy = new EmotionalState();
        foreach (var (emotion, intensity) in _intensities) copy.Set(emotion, intensity);
        copy.MoodValence = MoodValence;
        copy.MoodArousal = MoodArousal;
        return copy;
    }

    /// <summary>
    /// Human-readable emotion name: "joy", "high spirits", "self attention" —
    /// PascalCase enum names get a space before each inner capital.
    /// </summary>
    public static string Humanize(EmotionType type) =>
        Regex.Replace(type.ToString(), "(?<!^)([A-Z])", " $1").ToLowerInvariant();

    /// <summary>Human-readable description, e.g. "joyful, with anticipation".</summary>
    public string Describe()
    {
        var active = _intensities
            .Where(kv => kv.Value > 0.08)
            .OrderByDescending(kv => kv.Value)
            .Take(3)
            .Select(kv => $"{Humanize(kv.Key)} ({kv.Value:0.00})")
            .ToList();
        var baseText = active.Count == 0 ? "calm" : string.Join(", ", active);
        return $"{baseText} | mood {(MoodValence >= 0 ? "positive" : "negative")} ({MoodValence:+0.00;-0.00})";
    }

    /// <summary>Snapshot of intensities as a plain dictionary (for persistence).</summary>
    public Dictionary<EmotionType, double> Snapshot() => new(_intensities);
}
