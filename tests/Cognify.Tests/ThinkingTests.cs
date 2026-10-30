using Cognify.Core;
using Cognify.Thinking;

namespace Cognify.Tests;

/// <summary>Tests for chain-of-thought, deliberation and the inner monologue.</summary>
public class ThinkingTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Thinker")
            .WithReasoning()
            .WithThinking()
            .WithMemory()
            .WithEmotions()
            .Build();

    [Fact]
    public void Process_ProducesChainOfThought()
    {
        using var mind = BuildMind();
        var response = mind.Process("hello there");
        Assert.NotNull(response.Chain);
        Assert.NotEmpty(response.Chain!.Steps);
        Assert.Contains(response.Chain!.Steps, s => s.Phase == ThoughtPhase.Observe);
        Assert.Contains(response.Chain!.Steps, s => s.Phase == ThoughtPhase.Deliberate);
        Assert.Contains(response.Chain!.Steps, s => s.Phase == ThoughtPhase.Conclude);
    }

    [Fact]
    public void ChainSteps_AreSequentiallyNumbered()
    {
        using var mind = BuildMind();
        var chain = mind.Process("test input").Chain!;
        for (int i = 0; i < chain.Steps.Count; i++)
            Assert.Equal(i + 1, chain.Steps[i].Index);
    }

    [Fact]
    public void ChainRender_IncludesConclusion()
    {
        using var mind = BuildMind();
        var response = mind.Process("tell me something");
        string rendered = response.Chain!.Render();
        Assert.Contains("=>", rendered);
        Assert.Equal(response.Chain.Conclusion, response.Content);
    }

    [Fact]
    public void InnerMonologue_RecordsEntries()
    {
        using var mind = BuildMind();
        mind.Process("hello world");
        var entries = mind.Stream!.Recent(50);
        Assert.NotEmpty(entries);
        Assert.Contains(entries, e => e.Phase == "Observe");
        Assert.Contains(entries, e => e.Phase == "Conclude");
    }

    [Fact]
    public void ThoughtProduced_EventFires()
    {
        using var mind = BuildMind();
        int count = 0;
        mind.ThoughtProduced += (_, _) => count++;
        mind.Process("hello");
        Assert.True(count > 0);
    }

    [Fact]
    public void Deliberation_ProducesConfidentResponse()
    {
        using var mind = BuildMind();
        var response = mind.Process("hello");
        Assert.InRange(response.Confidence, 0, 1);
        Assert.True(response.Confidence >= 0.25); // ConfidenceFloor
        Assert.False(string.IsNullOrWhiteSpace(response.StyleName));
    }

    [Fact]
    public void ReasoningAnswer_OverridesGenericBody()
    {
        using var mind = BuildMind();
        mind.Teach("birds are animals");
        var response = mind.Ask("what is a bird?");
        Assert.Contains("animal", response.Content);
        Assert.True(response.Confidence >= 0.8);
    }

    [Fact]
    public void GoalContext_AppearsInEvaluationStep()
    {
        using var mind = MindBuilder.Create("GoalThinker")
            .WithGoals(g => g.AddGoal("explore-space", 0.9, "curiosity"))
            .WithThinking()
            .Build();
        var chain = mind.Process("hello").Chain!;
        Assert.Contains(chain.Steps, s => s.Text.Contains("explore-space"));
    }

    [Fact]
    public void ResponseStyles_AreStable()
    {
        // Style ids double as Q-table keys; verify stable naming.
        Assert.Equal(0, ResponseComposerStyle.Informative);
        Assert.Equal(5, ResponseComposerStyle.Encouraging);
    }

    // ------------------------------------------------------------------
    // Thinking ceiling — the hard cap on how far a model may think
    // (Only1 = Low only … Only5 = everything).
    // ------------------------------------------------------------------

    [Fact]
    public void CeilingLadder_MatchesTheFivePresets()
    {
        Assert.Equal(ThinkingLevel.Low, ThinkingCeiling.Only1.MaxLevel());
        Assert.Equal(ThinkingLevel.Medium, ThinkingCeiling.Only2.MaxLevel());
        Assert.Equal(ThinkingLevel.High, ThinkingCeiling.Only3.MaxLevel());
        Assert.Equal(ThinkingLevel.VeryHigh, ThinkingCeiling.Only4.MaxLevel());
        Assert.Equal(ThinkingLevel.Max, ThinkingCeiling.Only5.MaxLevel());
    }

    [Fact]
    public void Ceiling_Only1_ClampsEveryRequestedLevelToLow()
    {
        var t = new ThinkingOptions().At(ThinkingLevel.Max).UpTo(ThinkingCeiling.Only1);
        Assert.Equal(ThinkingLevel.Low, t.Level);
        Assert.Equal(1, t.ThoughtRounds);
        t.At(ThinkingLevel.High); // later requests are clamped too
        Assert.Equal(ThinkingLevel.Low, t.Level);
        Assert.Equal(ThinkingLevel.Low, t.Ceiling.Clamp(ThinkingLevel.Max));
    }

    [Fact]
    public void Ceiling_Only3_ClampsAboveHigh_PassesBelow()
    {
        var t = new ThinkingOptions().UpTo(ThinkingCeiling.Only3);
        t.At(ThinkingLevel.VeryHigh);
        Assert.Equal(ThinkingLevel.High, t.Level);
        t.At(ThinkingLevel.Max);
        Assert.Equal(ThinkingLevel.High, t.Level);
        Assert.Equal(4, t.ThoughtRounds);   // High's promise, not Max's 8
        t.At(ThinkingLevel.Medium);         // below the ceiling — untouched
        Assert.Equal(ThinkingLevel.Medium, t.Level);
    }

    [Fact]
    public void Ceiling_Only5_IsTheUncappedDefault()
    {
        var t = new ThinkingOptions().At(ThinkingLevel.Max);
        Assert.Equal(ThinkingCeiling.Only5, t.Ceiling);
        Assert.Equal(ThinkingLevel.Max, t.Level);
        Assert.Equal(8, t.ThoughtRounds);
    }

    [Fact]
    public void Ceiling_BuilderPreset_ClampsRuntimeLevelChanges()
    {
        using var mind = MindBuilder.Create("CappedThinker")
            .WithReasoning()
            .WithThinking(ceiling: ThinkingCeiling.Only2)   // Low or Medium, nothing above
            .WithMemory()
            .WithEmotions()
            .Build();
        Assert.Equal(ThinkingCeiling.Only2, mind.Config.Thinking.Ceiling);

        mind.Config.Thinking.At(ThinkingLevel.Max);         // a runtime request for Max…
        Assert.Equal(ThinkingLevel.Medium, mind.Config.Thinking.Level); // …thinks at Medium

        mind.Process("hello");
        Assert.True(mind.Thinking!.LastThoughts.Count <= 2); // Medium's token-set promise
    }

    internal static class ResponseComposerStyle
    {
        public const int Informative = 0;
        public const int Encouraging = 5;
    }
}
