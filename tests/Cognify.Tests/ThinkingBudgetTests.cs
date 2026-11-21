using System.Text.Json;
using Cognify.Core;
using Cognify.Thinking;

namespace Cognify.Tests;

/// <summary>
/// THINKING TOKEN BUDGETS PER LEVEL — the ladder from Low up to Max (or up
/// to the ceiling's top level when a clamp is set): the developer sets how
/// many tokens each level may spend, ThoughtRounds derives the generated
/// thought set (budget/50, 1..10), a global one-size budget overrides, and
/// under AUTOMATIC thinking the resolver only picks the level — the chosen
/// level's budget follows. Mirrors CognifyPy's TestThinkingBudgets.
/// </summary>
public class ThinkingBudgetTests
{
    private static Mind BuildMind(Action<ThinkingOptions>? configure = null,
        ThinkingCeiling ceiling = ThinkingCeiling.Only5) =>
        MindBuilder.Create("Budgeter")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithEmotions()
            .WithThinking(configure, ceiling: ceiling)
            .Build();

    [Fact]
    public void Level_budget_defaults_reproduce_the_classic_ladder()
    {
        var t = new ThinkingOptions();
        Assert.Equal(50, t.TokensFor(ThinkingLevel.Low));
        Assert.Equal(100, t.TokensFor(ThinkingLevel.Medium));
        Assert.Equal(200, t.TokensFor(ThinkingLevel.High));
        Assert.Equal(300, t.TokensFor(ThinkingLevel.VeryHigh));
        Assert.Equal(400, t.TokensFor(ThinkingLevel.Max));
        // the defaults still map to the classic thought counts
        Assert.Equal(1, t.TokensFor(ThinkingLevel.Low) / 50);
        Assert.Equal(8, t.TokensFor(ThinkingLevel.Max) / 50);
    }

    [Fact]
    public void Per_level_budget_sizes_the_thought_set()
    {
        var t = new ThinkingOptions().At(ThinkingLevel.Medium).Budget(ThinkingLevel.Medium, 250);
        Assert.Equal(250, t.TokensFor(ThinkingLevel.Medium));
        Assert.Equal(5, t.ThoughtRounds);          // 250/50 — re-leveled, re-sized
        t.At(ThinkingLevel.Low);
        Assert.Equal(1, t.ThoughtRounds);          // Low untouched: default 50
        Assert.Equal(50, t.TokensFor(ThinkingLevel.Low));
    }

    [Fact]
    public void Ceiling_clamps_budget_writes_and_reads()
    {
        // With the clamp set, the ladder runs from Low up to the CLAMP
        // level: a Max budget on an Only2 model lands on Medium.
        var t = new ThinkingOptions().UpTo(ThinkingCeiling.Only2)
            .Budget(ThinkingLevel.Max, 600);
        Assert.Equal(600, t.TokensFor(ThinkingLevel.Max));   // read clamps to Medium
        Assert.Equal(600, t.TokensFor(ThinkingLevel.Medium));
        Assert.Equal(400, t.LevelBudgets[ThinkingLevel.Max]);   // default, unreachable
        Assert.Equal(50, t.TokensFor(ThinkingLevel.Low));
        // the clamp's top level is Medium — budgets above it are unreachable
        Assert.Equal(ThinkingLevel.Medium, ThinkingCeiling.Only2.MaxLevel());
    }

    [Fact]
    public void Global_budget_overrides_every_per_level_entry()
    {
        var t = new ThinkingOptions()
            .Budget(ThinkingLevel.High, 222)
            .Budget(350);                          // the ONE global budget
        Assert.Equal(350, t.TokensFor(ThinkingLevel.High));
        Assert.Equal(350, t.TokensFor(ThinkingLevel.Low));
        Assert.Equal(7, t.ThoughtRounds);
    }

    [Fact]
    public void Zero_budget_entry_falls_back_to_the_default()
    {
        var t = new ThinkingOptions().Budget(ThinkingLevel.High, 222);
        Assert.Equal(222, t.TokensFor(ThinkingLevel.High));
        t.Budget(ThinkingLevel.High, 0);
        Assert.Equal(200, t.TokensFor(ThinkingLevel.High)); // default ladder again
    }

    [Fact]
    public void For_level_copies_budgets_in_isolation()
    {
        var shared = new ThinkingOptions().Budget(ThinkingLevel.High, 222);
        var perCycle = shared.ForLevel(ThinkingLevel.Medium);
        perCycle.Budget(ThinkingLevel.High, 999);
        Assert.Equal(222, shared.TokensFor(ThinkingLevel.High));   // untouched
        Assert.Equal(999, perCycle.TokensFor(ThinkingLevel.High));
    }

    [Fact]
    public void Budgets_persist_through_config_serialization()
    {
        var mind = BuildMind(t => t.At(ThinkingLevel.High).Budget(ThinkingLevel.High, 260)
                                             .Budget(ThinkingLevel.Max, 640));
        var json = JsonSerializer.Serialize(mind.Config);
        var restored = JsonSerializer.Deserialize<MindConfig>(json)!;
        Assert.Equal(260, restored.Thinking.TokensFor(ThinkingLevel.High));
        Assert.Equal(640, restored.Thinking.TokensFor(ThinkingLevel.Max));
        Assert.Equal(50, restored.Thinking.TokensFor(ThinkingLevel.Low));
    }

    [Fact]
    public void Mind_set_thinking_budget_changes_the_cycle()
    {
        using var mind = BuildMind(t => t.At(ThinkingLevel.Low));
        mind.Teach("Tulips are flowers. Tulips grow in gardens. Gardens smell sweet.");
        mind.Ask("What are tulips?");
        Assert.Equal(1, mind.Thinking!.LastThoughts.Count);   // default Low = 50 tokens

        mind.SetThinkingBudget(ThinkingLevel.Low, 100);
        Assert.Equal(100, mind.ThinkingTokensFor(ThinkingLevel.Low));
        mind.Ask("What are tulips?");
        Assert.Equal(2, mind.Thinking!.LastThoughts.Count);   // 100/50
    }

    [Fact]
    public void Automatic_thinking_uses_the_chosen_levels_budget()
    {
        // The resolver only picks the LEVEL; the level's budget follows.
        // Deep questions resolve to High..Max — with those levels budgeted
        // to 50 (one thought) a deep question thinks SHALLOWER than small
        // talk budgeted to 150 (three thoughts).
        using var mind = BuildMind(t => t.Auto()
            .Budget(ThinkingLevel.Low, 150)
            .Budget(ThinkingLevel.High, 50)
            .Budget(ThinkingLevel.VeryHigh, 50)
            .Budget(ThinkingLevel.Max, 50));
        mind.Teach("Tulips are flowers. Tulips grow in gardens. Gardens smell sweet.");
        mind.Ask("hi");
        int smallTalk = mind.Thinking!.LastThoughts.Count;
        Assert.Equal(3, smallTalk);   // auto chose Low → Low's 150-token budget

        mind.Ask("Why do flowers turn toward the sun, and how does the whole " +
                 "process actually work inside the plant over a full day?");
        int deep = mind.Thinking!.LastThoughts.Count;
        Assert.True(deep <= smallTalk,
            $"the chosen level's budget must govern: deep={deep}, small talk={smallTalk}");
    }
}
