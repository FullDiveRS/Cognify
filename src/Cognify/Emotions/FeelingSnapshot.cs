using Cognify.Core;

namespace Cognify.Emotions;

/// <summary>
/// A developer-facing snapshot of how the mind feels RIGHT NOW. The
/// programmatic counterpart of what used to be only spoken aloud — chat stays
/// natural by default, and the developer reads the state here instead:
///
/// <code>
/// Console.WriteLine(yume.Feeling?.CurrentStatus);
/// // "feeling anticipation (intensity 0.15) - mood positive (+0.00)"
/// </code>
/// </summary>
public sealed class FeelingSnapshot
{
    /// <summary>Humanized name of the dominant emotion (e.g. "anticipation").</summary>
    public string Name { get; init; } = "steady";

    /// <summary>Intensity of the dominant emotion (0..1).</summary>
    public double Intensity { get; init; }

    /// <summary>Mood valence (-1 gloomy .. +1 bright).</summary>
    public double Valence { get; init; }

    /// <summary>Arousal level (0 calm .. 1 keyed up).</summary>
    public double Arousal { get; init; }

    /// <summary>Mood band word: "positive", "neutral" or "negative".</summary>
    public string Mood { get; init; } = "neutral";

    /// <summary>
    /// One-line, human-readable status of the mind's current feeling —
    /// the "yume.feeling.currentstatus" read.
    /// </summary>
    public string CurrentStatus { get; init; } = "feeling steady";

    /// <summary>Builds a snapshot from the live emotional state.</summary>
    public static FeelingSnapshot From(EmotionalState state)
    {
        var dominant = state.Dominant;
        double intensity = state.DominantIntensity;
        string mood = state.Valence switch
        {
            < -0.15 => "negative",
            > 0.15 => "positive",
            _ => "neutral"
        };
        string name = intensity < 0.05
            ? "steady"
            : EmotionalState.Humanize(dominant).ToLowerInvariant();
        return new FeelingSnapshot
        {
            Name = name,
            Intensity = Math.Round(intensity, 3),
            Valence = Math.Round(state.Valence, 3),
            Arousal = Math.Round(state.Arousal, 3),
            Mood = mood,
            CurrentStatus = intensity < 0.05
                ? $"feeling steady - mood {mood} ({state.Valence:+0.00;-0.00;+0.00})"
                : $"feeling {name} (intensity {intensity:0.00}) - mood {mood} ({state.Valence:+0.00;-0.00;+0.00})"
        };
    }

    /// <summary>Returns <see cref="CurrentStatus"/>.</summary>
    public override string ToString() => CurrentStatus;
}
