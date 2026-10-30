using Cognify.Core;

namespace Cognify.Tests;

/// <summary>
/// Regression tests for the "AI repeats the same thing" bug: memory content
/// is usually the user's own words, so responses must reference recalled
/// memories obliquely instead of quoting them back verbatim.
/// </summary>
public class AntiRepetitionTests
{
    // Distinctive multi-word fragments of the user's inputs — none of these
    // may ever appear inside a Yume-style response (single topic words like
    // "apples" are fine; whole phrases are parroting).
    private static readonly string[] UserPhrases =
    {
        "favorite food",
        "2 + 2",
        "based number system",
        "how many apples",
    };

    private const string TeachingLine =
        "okay, ill teach you so there is 10 based number system. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. " +
        "if i have 1 apple want to buy another one, how many apples would i have?";

    private static Mind BuildMind() =>
        MindBuilder.Create("Yume")
            .Describe("repetition regression mind")
            .WithPersonality(p => p.Traits(openness: 0.9, conscientiousness: 0.6,
                                           extraversion: 0.75, agreeableness: 0.85,
                                           neuroticism: 0.3))
            .WithEmotions(e => e.Intensity(0.85))
            .WithReasoning()
            .WithThinking(t => t.Steps(5))
            .WithSelfLearning(l => l.Rate(0.12))
            .WithMemory(m => m.Capacity(20))
            .WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness"))
            .Build();

    private static void AssertNoParroting(MindResponse reply)
    {
        foreach (var phrase in UserPhrases)
            Assert.DoesNotContain(phrase, reply.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Replayed_Transcript_Never_Echoes_User_Words()
    {
        using var mind = BuildMind();

        // The exact exchange from the bug report, in order.
        AssertNoParroting(mind.Process("what is your favorite food?"));
        AssertNoParroting(mind.Process("what's 2 + 2?"));
        AssertNoParroting(mind.Process(TeachingLine));
        AssertNoParroting(mind.Process(TeachingLine)); // user repeated the line
        AssertNoParroting(mind.Process("2 + 2 is 4"));
    }

    [Fact]
    public void Recalled_Memory_Is_Referenced_Obliquely_Not_Quoted()
    {
        using var mind = BuildMind();
        mind.Process("the golden gate bridge is in san francisco");

        // Follow-up that has no direct answer but shares topic words, so the
        // earlier line surfaces from memory.
        var reply = mind.Process("do you remember the golden gate bridge? tell me more about it");

        Assert.DoesNotContain("golden gate bridge is in san francisco", reply.Content,
                              StringComparison.OrdinalIgnoreCase);
        // The response must still acknowledge the recall happened in some way
        // (topic reference or a recall-aware question) rather than ignoring it.
        // Every oblique-recall phrasing in the composer is covered here.
        string c = reply.Content;
        Assert.True(
            c.Contains("earlier", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("remember", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("recall", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("connect", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("echo", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("stirs", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("memory", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("familiar", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("heard", StringComparison.OrdinalIgnoreCase) ||
            c.Contains("bell", StringComparison.OrdinalIgnoreCase),
            $"recall not acknowledged: {c}");
    }

    [Fact]
    public void Teaching_Is_Acknowledged_Not_Echoed()
    {
        using var mind = BuildMind();
        var before = mind.Knowledge?.FactCount ?? 0;

        var reply = mind.Process("2 + 2 is 4");

        // Knowledge actually grew...
        Assert.True(mind.Knowledge!.FactCount > before, "teaching should add a fact");
        // ...and every response style embeds a learning acknowledgment
        // ("new thing(s)", "new fact(s)" or "new piece(s)") instead of parroting.
        string content = reply.Content;
        Assert.True(
            content.Contains("new thing", StringComparison.OrdinalIgnoreCase) ||
            content.Contains("new fact", StringComparison.OrdinalIgnoreCase) ||
            content.Contains("new piece", StringComparison.OrdinalIgnoreCase),
            $"teaching not acknowledged: {content}");
        AssertNoParroting(reply);
    }

    [Fact]
    public void Same_Message_Twice_Still_No_Echo()
    {
        using var mind = BuildMind();
        var first = mind.Process(TeachingLine);
        var second = mind.Process(TeachingLine);

        // The second time, the first message is sitting in memory — the mind
        // must still not parrot it (this was the worst offender: the reply was
        // literally just the recalled text).
        Assert.DoesNotContain("based number system", second.Content, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("how many apples", second.Content, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("based number system", first.Content, StringComparison.OrdinalIgnoreCase);

        // Deliberation penalizes repeating the previous reply word-for-word,
        // so two consecutive identical replies are effectively impossible.
        Assert.NotEqual(first.Content, second.Content);
    }

    [Fact]
    public void Fallback_Composer_Never_Echoes_Memory()
    {
        // No thinking module: the minimal fallback composer answers instead.
        using var mind = MindBuilder.Create("Fallback")
            .WithMemory()
            .WithReasoning()
            .WithEmotions()
            .Without("thinking")
            .Build();

        mind.Process("the sky is very blue today");
        var reply = mind.Process("tell me about the sky");

        Assert.DoesNotContain("sky is very blue", reply.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("fallback", reply.StyleName);
    }

    [Fact]
    public void Concise_Style_Never_Quotes_Memory()
    {
        // Drive enough cycles with recall present that the concise style is
        // reachable; whatever style wins, no reply may quote memory verbatim.
        using var mind = BuildMind();
        mind.Process("sodium is a metal");
        for (int i = 0; i < 6; i++)
        {
            var reply = mind.Process($"what about sodium, anything else on sodium? ({i})");
            Assert.DoesNotContain("sodium is a metal", reply.Content, StringComparison.OrdinalIgnoreCase);
        }
    }
}
