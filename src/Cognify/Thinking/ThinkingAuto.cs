using Cognify.Core;

namespace Cognify.Thinking;

/// <summary>
/// The AUTOMATIC thinking resolver: decides HOW MUCH to think about an
/// individual topic or conversation turn — the library's version of a
/// reasoning model spending more or fewer reasoning tokens depending on the
/// question. Host-facing and deterministic on the inputs (no randomness),
/// and always clamped by the model's <see cref="ThinkingCeiling"/>.
///
/// Relationship to the built-in cycle: when <see cref="ThinkingOptions"/>.
/// <c>Automatic</c> is on, <see cref="ThinkingModule"/> runs its own
/// internal per-cycle selection inside the cognitive loop. This class is the
/// same IDEA as a standalone, reusable resolver — hosts, demos and tools
/// that reason outside the loop call <see cref="Choose(string,bool,int,double,bool,ThinkingCeiling)"/>
/// directly.
///
/// The weights, in order of how much they usually matter:
///   questions (+2, thinking exists to answer them well),
///   "why"/"how" probes (+1), long inputs (+1),
///   novel ground — nothing known and nothing recalled (+1),
///   a strong emotional surge (+1), a fresh epistemic gap (+1).
/// </summary>
public static class ThinkingAuto
{
    /// <summary>
    /// Chooses the thinking level for THIS turn from the cycle context.
    /// Returns one of Low..Max, never a flag, always within the ceiling.
    /// </summary>
    public static ThinkingLevel Choose(CognitiveContext ctx, ThinkingCeiling ceiling)
    {
        string text = ctx.EffectiveContent ?? "";
        bool hasAnswer = ctx.Reasoning?.Answer is not null;
        int recalled = ctx.Recalled.Count;
        double emotion = ctx.Emotions?.DominantIntensity ?? 0;
        bool epistemic = ctx.Bag.ContainsKey("emotions.epistemic");
        return Choose(text, hasAnswer, recalled, emotion, epistemic, ceiling);
    }

    /// <summary>
    /// Chooses the thinking level for THIS turn from plain inputs (the
    /// host-facing overload — everything it needs is text and the already
    /// computed signals). Returns one of Low..Max, always within the
    /// ceiling.
    /// </summary>
    public static ThinkingLevel Choose(string text, bool hasDirectAnswer,
        int recalledCount, double emotionIntensity, bool epistemicGap,
        ThinkingCeiling ceiling)
    {
        text ??= "";
        int tier = 0;

        // Questions deserve real deliberation — that is what thinking is for.
        bool question = text.Contains('?') || StartsWithQuestionWord(text);
        if (question) tier += 2;
        if (text.Contains(" why ", StringComparison.OrdinalIgnoreCase) ||
            text.Contains(" how ", StringComparison.OrdinalIgnoreCase) ||
            text.StartsWith("why ", StringComparison.OrdinalIgnoreCase) ||
            text.StartsWith("how ", StringComparison.OrdinalIgnoreCase))
            tier += 1;

        // Longer turns carry more to weigh.
        int words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        if (words > 12) tier += 1;

        // Novel ground: no direct knowledge and nothing recalled — the mind
        // must reason its way forward instead of reciting, so think more.
        // Only for turns that actually seek something: a bare "hi" stays a
        // quick pause.
        if (!hasDirectAnswer && recalledCount == 0 &&
            (text.Contains('?') || words > 3)) tier += 1;

        // Emotion: a strong surge deserves a pause before speaking.
        if (emotionIntensity > 0.5) tier += 1;

        // A fresh epistemic gap (the emotion module's curiosity flag).
        if (epistemicGap) tier += 1;

        tier = Math.Clamp(tier, 0, 4);
        var level = tier switch
        {
            0 => ThinkingLevel.Low,
            1 => ThinkingLevel.Medium,
            2 => ThinkingLevel.High,
            3 => ThinkingLevel.VeryHigh,
            _ => ThinkingLevel.Max,
        };
        return ceiling.Clamp(level);
    }

    private static bool StartsWithQuestionWord(string text)
    {
        var first = text.TrimStart();
        foreach (var w in new[] { "what", "who", "when", "where", "why", "how", "which", "can ", "does ", "do ", "is ", "are " })
            if (first.StartsWith(w, StringComparison.OrdinalIgnoreCase))
                return true;
        return false;
    }
}
