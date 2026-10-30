using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Thinking;

/// <summary>
/// The thought generator — the mind's silent self-talk BEFORE it speaks.
/// When thinking is enabled, each cycle produces a small set of internal
/// thoughts (sized by <see cref="ThinkingLevel"/>: Low=1 ... Max=8, or an
/// explicit <c>ThoughtBudget</c>) flavored by the configured thinking tone.
/// The thoughts are grounded in what the cycle actually gathered — the
/// topic, the current feeling, recalled memories, freshly inferred facts,
/// the private goal and new vocabulary — so the final reply ARRIVES from
/// thought instead of appearing out of nowhere. Everything recorded here
/// stays on the chain of thought; whether the host SHOWS it is controlled
/// by <see cref="ExpressionOptions.ShowThinking"/>.
/// </summary>
internal static class ThoughtGenerator
{
    /// <summary>
    /// Generates the pre-speech thoughts. Returns between 1 and 10 lines;
    /// never empty (the mind always thinks at least once before speaking
    /// when thinking is enabled).
    /// </summary>
    public static IReadOnlyList<(string Thought, double Confidence)> Generate(
        CognitiveContext ctx, ThinkingOptions options, Random rng)
    {
        int rounds = Math.Clamp(options.ThoughtRounds, 1, 10);
        string tone = (options.Tone ?? "balanced").Trim().ToLowerInvariant();
        var topic = ResponseComposer.TopicOf(ctx.EffectiveContent);
        var emotions = ctx.Emotions;
        var dominant = emotions?.Dominant;
        var dominantName = emotions is null || emotions.DominantIntensity < 0.05
            ? "steady"
            : EmotionalState.Humanize(dominant ?? EmotionType.Reflection).ToLowerInvariant();

        // The candidate pool: grounded observations of THIS cycle. Order is
        // shuffled so different levels think different things first, not
        // just "more of the same".
        var pool = new List<string>();

        // 1. Knowledge check — do I know this?
        pool.Add(ctx.Reasoning?.Answer is { } ans
            ? Pick(rng,
                $"I know this one - {Truncate(ans, 60)} Lead with it.",
                $"Solid knowledge here: {Truncate(ans, 60)} Say it plainly.")
            : Pick(rng,
                $"What do I actually know about {topic}? Not enough - say so honestly and offer to learn.",
                $"No solid knowledge about {topic} yet. Do not fake it - invite them to teach me."));

        // 2. Reasoning echo — what did inference just produce?
        if (ctx.Reasoning is { } r && r.Inferred.Count > 0)
            pool.Add(Pick(rng,
                $"My reasoning connected {r.Inferred.Count} new thing(s) - {Truncate(r.Inferred[0].ToString()!, 60)}. Worth sharing.",
                $"Inference ran: {Truncate(r.Inferred[0].ToString()!, 60)} - that can lead the reply."));

        // 3. Feeling check — how should the reply FEEL?
        pool.Add(Pick(rng,
            $"This lands as {dominantName} - let the feeling color the words, not dominate them.",
            $"I feel {dominantName} about this. The reply should sound like that, without announcing it."));

        // 4. Memory check — does anything echo?
        if (ctx.Recalled.Count > 0)
            pool.Add(Pick(rng,
                $"This echoes what they told me about {ResponseComposer.TopicOf(ctx.Recalled[0].Content)} - nod to it obliquely, never quote it back.",
                $"An old thread connects here - mention it lightly, one clause at most."));

        // 5. Shape of the reply.
        pool.Add(tone switch
        {
            "focused" => "Stay on the question. Answer, one follow-up at most, stop.",
            "playful" => "Keep it light - a little bounce in the wording is fine today.",
            "poetic" => "Give the reply one soft image - like light through a window - and no more.",
            "plain" => "Short sentences. No decoration. Just the answer.",
            _ => "Answer first, stay warm, ask at most one question - then stop."
        });

        // 6. New vocabulary — maybe weave the fresh word in.
        var freshWord = ctx.Mind.Learning?.Lexicon?.FreshWordWithAge();
        if (freshWord is { } word)
            pool.Add($"\"{word.Word}\" is still unplaced in my mind - if it fits, ask about it in my own words.");

        // 7. The private goal — felt, never spoken about.
        if (ctx.FocusGoal is { } goal)
            pool.Add($"Quiet motive in view ({goal.Name}) - let it warm the tone without ever naming it.");

        // 8. Epistemic gap — curiosity wants to ask.
        if (ctx.Bag.ContainsKey("emotions.epistemic"))
            pool.Add("A gap just opened - I want to know more. One curious question is natural, two is pestering.");

        // Shuffle, then take the level's share. When the cycle gave the mind
        // less to chew on than its level asks for (a bare "hello" at Max is
        // only three grounded observations), the generator pads the remainder
        // with generic deliberation — the level's promise (its "token set")
        // is always kept.
        for (int i = pool.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (pool[i], pool[j]) = (pool[j], pool[i]);
        }

        var taken = pool.Take(rounds).ToList();
        int filler = 0;
        while (taken.Count < rounds)
            taken.Add(FillerThoughts[filler++ % FillerThoughts.Length]);

        // Tone garnish: a light lead-in word that flavors HOW it is said
        // without changing what is thought.
        if (tone == "playful")
            taken = taken.Select(t => rng.Next(2) == 0 ? "Ooh - " + Lower(t) : t).ToList();
        else if (tone == "focused")
            taken = taken.Select(t => "Focus: " + Lower(t)).ToList();

        return taken.Select(t => (t, 0.45 + rng.NextDouble() * 0.25)).ToList();
    }

    private static string Pick(Random rng, params string[] options) =>
        options[rng.Next(options.Length)];

    // Generic deliberation used to keep the level's promise when this cycle
    // offered fewer grounded observations than the level asks for.
    private static readonly string[] FillerThoughts =
    {
        "Second look - is there anything in this I am missing?",
        "Weighing how much of this is fact and how much is just feeling.",
        "Keep the words simple and honest - say what I actually know.",
        "No need to rush the answer - one clear sentence beats three vague ones.",
        "Check the tone once more before speaking - match theirs, stay warm.",
        "Is there one small useful thing I can offer here? Lead with that.",
        "If I am unsure, say so plainly - honesty reads better than bluffing.",
        "What would make this reply worth their time? Answer, then stop."
    };

    private static string Lower(string text) =>
        string.IsNullOrEmpty(text) ? text : char.ToLowerInvariant(text[0]) + text[1..];

    private static string Truncate(string text, int max) =>
        text.Length <= max ? text : text[..max].TrimEnd() + "...";
}
