using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Tests;

/// <summary>
/// Tests for the dual-process mind: goals are PRIVATE motivation (part of
/// the inner prompt, never recited), the mind thinks before it speaks, and
/// strong emotional surges — blame, dismissal, contradiction — make it answer
/// impulsively, before thinking finishes. Grounded in appraisal research:
/// OCC (anger = blame + unfairness), Zajonc's affect primacy, and
/// information-gap theory (surprise/interest drive questions).
/// </summary>
public class ImpulseTests
{
    private static Mind BuildMind(Action<MindBuilder>? extra = null)
    {
        var builder = MindBuilder.Create("Yume")
            .Describe("Yume (夢, 'dream') — a warm, curious companion AI.")
            .WithEmotions(e => e.Intensity(1.0).Decay(0.0))   // no decay: deterministic
            .WithPersonality(p => p.Traits())                  // neutral traits: factor 1.0
            .WithReasoning()
            .WithThinking(t => t.Steps(5))
            .WithSelfLearning()
            .WithMemory(m => m.Capacity(20));
        extra?.Invoke(builder);
        return builder.Build();
    }

    // ------------------------------------------------------------------
    // Impulse: emotion answers before thinking
    // ------------------------------------------------------------------

    [Fact]
    public void Blame_Makes_The_Mind_Answer_Impulsively()
    {
        using var mind = BuildMind();
        var reply = mind.Process("You are stupid and useless. Shut up.", intensity: 0.9);
        Assert.Equal("impulse", reply.StyleName);
        Assert.False(string.IsNullOrWhiteSpace(reply.Content));
    }

    [Fact]
    public void Dismissal_Is_Reflexive_Too()
    {
        using var mind = BuildMind();
        var reply = mind.Process("Whatever. Shut up already.", intensity: 0.9);
        Assert.Equal("impulse", reply.StyleName);
    }

    [Fact]
    public void Contradicting_Known_Knowledge_Startles()
    {
        using var mind = BuildMind();
        mind.Process("Birds are animals. Animals are living things.");
        var before = mind.EmotionalState!.Intensities[EmotionType.Surprise];
        var reply = mind.Process("Birds are not animals.", intensity: 0.9);
        // The violated expectation both startles (surprise surge)...
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Surprise] > before + 0.2);
        // ...and answers before thinking (reflexive impulse).
        Assert.Equal("impulse", reply.StyleName);
    }

    [Fact]
    public void Calm_Topics_Never_Impulse()
    {
        using var mind = BuildMind();
        mind.Process("Birds are animals. Animals are living things.");
        var reply = mind.Process("I like quiet evenings and reading books.");
        Assert.NotEqual("impulse", reply.StyleName);
    }

    [Fact]
    public void Knowledge_Still_Wins_Over_Impulse()
    {
        using var mind = BuildMind();
        mind.Process("2 + 2 = 4.");
        var reply = mind.Process("You are stupid. Now tell me, what is 2 + 2?", intensity: 0.9);
        // A confident answer is never blurted over — knowing wins; feeling
        // only colors how it is said.
        Assert.NotEqual("impulse", reply.StyleName);
        Assert.Contains("4", reply.Content);
    }

    [Fact]
    public void Impulse_Is_For_The_Surging_Turn_Only()
    {
        using var mind = BuildMind();
        var hot = mind.Process("You are stupid. Shut up.", intensity: 0.9);
        Assert.Equal("impulse", hot.StyleName);

        // The gate reads the FRESH appraisal each cycle, so once the surge
        // passes, the mind returns to deliberate conversation on its own.
        var calm = mind.Process("Anyway, how are you doing today?");
        Assert.NotEqual("impulse", calm.StyleName);
    }

    // ------------------------------------------------------------------
    // Goals are private: part of the prompt, never spoken
    // ------------------------------------------------------------------

    [Fact]
    public void Goal_Is_Never_Spoken_Aloud()
    {
        using var mind = BuildMind(b =>
            b.WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness")));
        string[] turns =
        {
            "I had a long day at work.",
            "The weather is nice today.",
            "I finally finished my project!",
            "tell me about yourself",
            "What should I know about gardening?",
            "just checking in, how are you"
        };
        foreach (var turn in turns)
        {
            var reply = mind.Process(turn);
            Assert.DoesNotContain("moves my goal", reply.Content, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("make the user smile", reply.Content, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void Goal_Still_Steers_The_Inner_Prompt()
    {
        using var mind = BuildMind(b =>
            b.WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness")));
        var reply = mind.Process("I had a long day at work.");
        // The goal IS part of the private chain of thought...
        Assert.Contains("make the user smile", string.Join(" ",
            reply.Chain!.Steps.Select(s => s.Text)), StringComparison.OrdinalIgnoreCase);
        // ...but never part of the spoken answer.
        Assert.DoesNotContain("make the user smile", reply.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void No_Pull_It_Tight_Phrasing()
    {
        using var mind = BuildMind(b =>
            b.WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness")));
        mind.Process("My favorite topic is lighthouse engineering.");
        var reply = mind.Process("So, about lighthouse engineering again.");
        Assert.DoesNotContain("pull it tight", reply.Content, StringComparison.OrdinalIgnoreCase);
    }

    // ------------------------------------------------------------------
    // Social appraisal research: anger, interest, surprise
    // ------------------------------------------------------------------

    [Fact]
    public void Insults_Raise_Anger_More_Than_Other_Negativity()
    {
        using var mind = BuildMind();
        mind.Process("the movie was terrible and boring");
        double observedNegativity = mind.EmotionalState!.Intensities[EmotionType.Anger];
        mind.Module<EmotionModule>()!.Calm();
        mind.Process("you are terrible and boring");
        double directedBlame = mind.EmotionalState!.Intensities[EmotionType.Anger];
        // Same negative words — but blame aimed at the mind provokes far
        // more anger (OCC: other-accountability is what makes anger).
        Assert.True(directedBlame > observedNegativity + 0.2,
            $"directed={directedBlame:0.00} observed={observedNegativity:0.00}");
    }

    [Fact]
    public void Unfairness_Provokes_Anger()
    {
        using var mind = BuildMind();
        mind.Process("that is completely unfair and rigged", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Anger] > 0.25);
    }

    [Fact]
    public void Shown_Interest_Becomes_Anticipation_And_Questions()
    {
        using var mind = BuildMind();
        var reply = mind.Process("tell me more about volcanoes, they seem really interesting");
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Anticipation] > 0.15);
        // Information-gap theory: interest drives the mind to ASK.
        Assert.Contains('?', reply.Content);
    }

    [Fact]
    public void Surprise_Marker_Sets_Up_The_Startle()
    {
        using var mind = BuildMind();
        mind.Process("guess what happened to me today!", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Surprise] > 0.25);
    }

    [Fact]
    public void Negations_Are_Never_Stored_As_Facts()
    {
        using var mind = BuildMind();
        mind.Process("Birds are animals.");
        mind.Process("Birds are not machines.");
        // The denial must not poison the knowledge base with (bird, is-a, not machine).
        Assert.Empty(mind.Knowledge!.Query("bird", "is-a", "not machine"));
        Assert.Single(mind.Knowledge.Query("bird", "is-a", "animal"));
    }

    // ------------------------------------------------------------------
    // Vocabulary: speech glue is no longer "picked up"
    // ------------------------------------------------------------------

    [Fact]
    public void Vocabulary_Ignores_Speech_Glue()
    {
        using var mind = BuildMind();
        mind.Process("okay so lets try getting this good job done, slightly slowly, fore the vocab");
        var picked = mind.Lexicon!.LastPicks;
        Assert.DoesNotContain("okay", picked);
        Assert.DoesNotContain("getting", picked);
        Assert.DoesNotContain("doing", picked);
        Assert.DoesNotContain("good", picked);
        Assert.DoesNotContain("job", picked);
        Assert.DoesNotContain("lets", picked);
        Assert.DoesNotContain("try", picked);
        Assert.DoesNotContain("slightly", picked);
        Assert.DoesNotContain("vocab", picked);
        Assert.DoesNotContain("fore", picked);
        // Real content words are still picked up.
        Assert.Contains("slowly", picked);
    }
}
