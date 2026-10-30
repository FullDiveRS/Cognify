using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Personality;

/// <summary>
/// Big Five (OCEAN) personality traits, each in 0..1.
/// Traits modulate emotional sensitivity, exploration, deliberation depth
/// and response style — the same stimulus "feels" different to different minds.
/// </summary>
public sealed class BigFiveTraits
{
    /// <summary>Curiosity, imagination, preference for novelty. Drives exploration.</summary>
    public double Openness { get; set; } = 0.5;

    /// <summary>Organization, deliberation, goal-directedness. Deepens thinking.</summary>
    public double Conscientiousness { get; set; } = 0.5;

    /// <summary>Sociability, enthusiasm. Amplifies positive affect and warm style.</summary>
    public double Extraversion { get; set; } = 0.5;

    /// <summary>Warmth, trust, cooperation. Amplifies trust, softens anger.</summary>
    public double Agreeableness { get; set; } = 0.5;

    /// <summary>Emotional sensitivity. Amplifies negative affect.</summary>
    public double Neuroticism { get; set; } = 0.5;

    /// <summary>Creates traits with custom values (defaults are all 0.5).</summary>
    public BigFiveTraits() { }

    internal BigFiveTraits(PersonalityOptions options)
    {
        Openness = Math.Clamp(options.Openness, 0, 1);
        Conscientiousness = Math.Clamp(options.Conscientiousness, 0, 1);
        Extraversion = Math.Clamp(options.Extraversion, 0, 1);
        Agreeableness = Math.Clamp(options.Agreeableness, 0, 1);
        Neuroticism = Math.Clamp(options.Neuroticism, 0, 1);
    }

    /// <summary>Multiplier applied to the intensity of a given emotion.</summary>
    public double EmotionFactor(EmotionType emotion) => emotion switch
    {
        EmotionType.Joy => 1.0 + (Extraversion - 0.5) * 0.8,
        EmotionType.Trust => 1.0 + (Agreeableness - 0.5) * 0.8,
        EmotionType.Sadness => 1.0 + (Neuroticism - 0.5) * 0.9,
        EmotionType.Anger => 1.0 + (Neuroticism - 0.5) * 0.6 - (Agreeableness - 0.5) * 0.5,
        EmotionType.Fear => 1.0 + (Neuroticism - 0.5) * 1.0,
        EmotionType.Disgust => 1.0 + (Neuroticism - 0.5) * 0.4,
        EmotionType.Surprise => 1.0 + (Openness - 0.5) * 0.6,
        EmotionType.Anticipation => 1.0 + (Openness - 0.5) * 0.7,

        // Darwin's finer feelings — traits shape them too.
        EmotionType.Love or EmotionType.Tenderness =>
            1.0 + (Agreeableness - 0.5) * 0.6 + (Extraversion - 0.5) * 0.2,
        EmotionType.Devotion => 1.0 + (Agreeableness - 0.5) * 0.7,
        EmotionType.HighSpirits => 1.0 + (Extraversion - 0.5) * 0.7,
        EmotionType.Pride => 1.0 + (Extraversion - 0.5) * 0.5,
        EmotionType.Determination => 1.0 + (Conscientiousness - 0.5) * 0.7,
        EmotionType.Patience => 1.0 + (Agreeableness - 0.5) * 0.4,
        EmotionType.Reflection or EmotionType.Meditation => 1.0 + (Openness - 0.5) * 0.5,
        EmotionType.Anxiety or EmotionType.Weeping or EmotionType.Grief or
            EmotionType.Despair or EmotionType.Suffering or EmotionType.Helplessness or
            EmotionType.Dejection or EmotionType.LowSpirits =>
            1.0 + (Neuroticism - 0.5) * 1.0,
        EmotionType.Hatred or EmotionType.Disdain or EmotionType.Contempt or
            EmotionType.IllTemper or EmotionType.Sulkiness =>
            1.0 + (Neuroticism - 0.5) * 0.6 - (Agreeableness - 0.5) * 0.5,
        EmotionType.Guilt or EmotionType.Shyness or EmotionType.Blushing or
            EmotionType.SelfAttention => 1.0 + (Neuroticism - 0.5) * 0.7,
        _ => 1.0
    };

    /// <summary>
    /// Exploration rate for trying new response styles. Open minds explore more.
    /// </summary>
    public double ExplorationRate(double baseRate) =>
        Math.Clamp(baseRate + (Openness - 0.5) * 0.3, 0.02, 0.5);

    /// <summary>
    /// How many deliberation rounds the thinking module uses. Conscientious
    /// minds think more carefully.
    /// </summary>
    public int DeliberationDepth(int baseDepth) =>
        Math.Max(1, baseDepth + (int)Math.Round((Conscientiousness - 0.5) * 4));

    /// <summary>Tendency to mention goals in responses.</summary>
    public double GoalFocusBias => Math.Clamp(Conscientiousness * 0.8, 0, 1);

    /// <summary>Short human-readable summary, e.g. "curious, warm, organized".</summary>
    public string Describe()
    {
        var words = new List<string>();
        if (Openness > 0.65) words.Add("curious");
        else if (Openness < 0.35) words.Add("conventional");
        if (Conscientiousness > 0.65) words.Add("organized");
        else if (Conscientiousness < 0.35) words.Add("spontaneous");
        if (Extraversion > 0.65) words.Add("outgoing");
        else if (Extraversion < 0.35) words.Add("reserved");
        if (Agreeableness > 0.65) words.Add("warm");
        else if (Agreeableness < 0.35) words.Add("blunt");
        if (Neuroticism > 0.65) words.Add("sensitive");
        else if (Neuroticism < 0.35) words.Add("even-keeled");
        return words.Count == 0 ? "balanced" : string.Join(", ", words);
    }
}
