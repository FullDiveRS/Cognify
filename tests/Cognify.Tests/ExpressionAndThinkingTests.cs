using Cognify.Core;
using Cognify.Thinking;

namespace Cognify.Tests;

/// <summary>
/// Tests for the developer-facing surface: the Feeling snapshot, the
/// Expression visibility switches, the optional follow-up questions, and
/// the thinking levels + thought generator.
/// </summary>
public class ExpressionAndThinkingTests
{
    private static Mind BuildMind(Action<ThinkingOptions>? thinking = null,
                                  Action<ExpressionOptions>? expression = null) =>
        MindBuilder.Create("Feel")
            .WithReasoning()
            .WithThinking(thinking)
            .WithMemory()
            .WithEmotions()
            .WithExpression(expression)
            .Build();

    // ---- the feeling API --------------------------------------------------

    [Fact]
    public void Feeling_CurrentStatus_IsAvailableAfterACycle()
    {
        using var mind = BuildMind();
        mind.Process("hello there");
        Assert.NotNull(mind.Feeling);
        Assert.StartsWith("feeling", mind.Feeling!.CurrentStatus);
        Assert.False(string.IsNullOrWhiteSpace(mind.Feeling.Name));
        Assert.InRange(mind.Feeling.Intensity, 0, 1);
        Assert.InRange(mind.Feeling.Valence, -1, 1);
    }

    [Fact]
    public void Feeling_DefaultChat_DoesNotAnnounceTheRead()
    {
        using var mind = BuildMind();
        var reply = mind.Process("I am learning so much today");
        // The mechanical introspection is opt-in now — feelings color the
        // words instead, and the state stays on mind.Feeling.
        Assert.DoesNotContain("my current read", reply.Content);
    }

    [Fact]
    public void DescribeSelf_ShowsMoodOnlyWhenFeelingReadsEnabled()
    {
        using var quiet = BuildMind(expression: e => e.Feelings(false));
        quiet.Process("hello there");
        Assert.DoesNotContain("Right now", quiet.DescribeSelf());

        using var loud = BuildMind(expression: e => e.Feelings(true));
        loud.Process("hello there");
        Assert.Contains("Right now", loud.DescribeSelf());
    }

    // ---- internals stay hidden by default ---------------------------------

    [Fact]
    public void Internals_DefaultHidden()
    {
        using var mind = BuildMind();
        mind.Process("hello");
        Assert.False(mind.Config.Expression.ShowFeelingReads);
        Assert.False(mind.Config.Expression.ShowThinking);
        Assert.False(mind.Config.Expression.ShowResponseMeta);
    }

    // ---- follow-up questions are optional ----------------------------------

    [Fact]
    public void FollowUps_EnabledByDefault()
    {
        using var mind = BuildMind();
        Assert.True(mind.Config.Expression.AskFollowUpQuestions);
    }

    [Fact]
    public void FollowUps_Disabled_NeverAsksAboutUnknownPeople()
    {
        using var mind = BuildMind(expression: e => e.FollowUps(false));
        for (int i = 0; i < 6; i++)
        {
            var reply = mind.Process("I met someone new today");
            Assert.DoesNotContain("Who are they to you", reply.Content);
            Assert.DoesNotContain("What are they like", reply.Content);
            Assert.DoesNotContain("Tell me more about them", reply.Content);
        }
    }

    // ---- thinking levels + the thought generator ---------------------------

    [Fact]
    public void ThinkingLevels_MapToThoughtRounds()
    {
        var opts = new ThinkingOptions();
        Assert.Equal(2, opts.At(ThinkingLevel.Medium).ThoughtRounds);
        Assert.Equal(1, opts.At(ThinkingLevel.Low).ThoughtRounds);
        Assert.Equal(4, opts.At(ThinkingLevel.High).ThoughtRounds);
        Assert.Equal(6, opts.At(ThinkingLevel.VeryHigh).ThoughtRounds);
        Assert.Equal(8, opts.At(ThinkingLevel.Max).ThoughtRounds);
        // An explicit token budget overrides the level (budget/50 thoughts).
        Assert.Equal(5, opts.Budget(250).ThoughtRounds);
    }

    [Fact]
    public void Thinking_GeneratesThoughtsBeforeSpeaking()
    {
        using var mind = BuildMind(t => t.At(ThinkingLevel.High));
        var reply = mind.Process("hello there");
        var thinker = mind.Thinking;
        Assert.NotNull(thinker);
        // The generator keeps the level's promise (High = 4 thoughts) even
        // on a bare first cycle, and they are part of the explainable chain.
        Assert.Equal(mind.Config.Thinking.ThoughtRounds, thinker!.LastThoughts.Count);
        Assert.All(thinker.LastThoughts, t => Assert.False(string.IsNullOrWhiteSpace(t)));
        Assert.Contains(reply.Chain!.Steps, s => s.Phase == ThoughtPhase.Generate);
    }

    [Fact]
    public void ThinkingDisabled_ShortCircuitsTheGenerator()
    {
        using var mind = BuildMind(t => t.Enable(false));
        var reply = mind.Process("hello there");
        Assert.False(string.IsNullOrWhiteSpace(reply.Content));
        Assert.NotNull(mind.Thinking);
        Assert.Empty(mind.Thinking!.LastThoughts);
        Assert.DoesNotContain(reply.Chain!.Steps, s => s.Phase == ThoughtPhase.Generate);
    }

    [Fact]
    public void ThinkingTone_FlavorsTheThoughts()
    {
        using var mind = BuildMind(t => t.WithTone("focused"));
        mind.Process("hello there");
        var thoughts = mind.Thinking!.LastThoughts;
        Assert.NotEmpty(thoughts);
        Assert.All(thoughts, t => Assert.StartsWith("Focus:", t));
    }

    [Fact]
    public void ThoughtBudget_OverridesLevel()
    {
        using var mind = BuildMind(t => t.Budget(300));
        mind.Process("hello there");
        Assert.Equal(6, mind.Thinking!.LastThoughts.Count);
    }
}
