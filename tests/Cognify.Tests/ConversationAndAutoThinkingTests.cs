using System.Text.Json;
using Cognify.Core;
using Cognify.Thinking;
using Cognify.Tools;

namespace Cognify.Tests;

/// <summary>
/// Tests for the conversational dial (WithConversational → rate), the
/// ConversationTracker (topics / open questions / familiarity) and the
/// AUTOMATIC thinking resolution — the mind choosing how much to think per
/// turn, inside the cycle (ThinkingOptions.Automatic) and standalone
/// (ThinkingAuto).
/// </summary>
public class ConversationAndAutoThinkingTests
{
    private static Mind BuildMind(double? rate = null)
    {
        var builder = MindBuilder.Create("Chatty")
            .WithReasoning()
            .WithThinking()             // the conversational dial lives in composition
            .WithSelfLearning()
            .WithMemory()
            .WithEmotions();
        if (rate is { } r)
            builder.WithConversational(c => c.rate(r));
        return builder.Build();
    }

    // ------------------------------------------------------------------ rate

    [Fact]
    public void Rate_Clamps_To_Valid_Range()
    {
        var c = new ConversationalOptions();
        c.rate(99);
        Assert.Equal(3.0, c.Rate);
        c.rate(-5);
        Assert.Equal(0.5, c.Rate);
        c.rate(1.7);
        Assert.Equal(1.7, c.Rate);
    }

    [Fact]
    public void Rate_Two_Speaks_Longer_Than_Rate_One()
    {
        var mind1 = BuildMind();
        mind1.Teach("The ocean is water.");
        var baseReply = mind1.Ask("What is the ocean?").Content;

        var mind2 = BuildMind(2.0);
        mind2.Teach("The ocean is water.");
        var talkativeReply = mind2.Ask("What is the ocean?").Content;

        int Sentences(string t) => t.Count(ch => ch is '.' or '!' or '?');
        // A talkative mind never answers a real question with one bare line:
        // the dial expands whatever candidate wins to roughly double length.
        Assert.True(Sentences(talkativeReply) >= 2,
            $"rate 2.0 should speak at least two sentences: '{talkativeReply}'");
        Assert.NotEqual(baseReply, talkativeReply);
    }

    [Fact]
    public void Rate_Below_One_Trims()
    {
        using var mind = BuildMind(0.5);
        mind.Teach("A rose is a flower. Roses grow in gardens. Roses smell sweet.");
        var reply = mind.Ask("Tell me about roses please.").Content;
        Assert.True(reply.Count(ch => ch is '.' or '!' or '?') <= 3,
            $"rate 0.5 should be terse: '{reply}'");
    }

    [Fact]
    public void Rate_Persists_Through_Config_Serialization()
    {
        using var mind = BuildMind(2.5);
        var json = JsonSerializer.Serialize(mind.Config);
        var restored = JsonSerializer.Deserialize<MindConfig>(json)!;
        Assert.Equal(2.5, restored.Conversational.Rate);
    }

    [Fact]
    public void Understand_Fluent_Switch()
    {
        var c = new ConversationalOptions().understand(false);
        Assert.False(c.Understand);
        c.understand();
        Assert.True(c.Understand);
    }

    // ------------------------------------------------------------- tracker

    [Fact]
    public void Tracker_Collects_Topics_Turns_And_Familiarity()
    {
        using var mind = BuildMind();
        var tracker = new ConversationTracker();
        tracker.Observe(mind, "I love gardening very much.", "that sounds nice!", 1);
        tracker.Observe(mind, "Do tulips need much sun?", "Tulips are flowers that like light.", 2);
        tracker.Observe(mind, "Gardening makes me happy.", "it is a calm hobby", 3);

        Assert.Equal(3, tracker.Turns);
        Assert.Contains(tracker.Topics, t => t.Contains("garden", StringComparison.OrdinalIgnoreCase));
        Assert.True(tracker.Familiarity > 0 && tracker.Familiarity < 1);
        var summary = string.Join(" ", tracker.Summary());
        Assert.Contains("garden", summary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Tracker_Keeps_Unanswered_Questions_As_Open_Threads()
    {
        using var mind = BuildMind();
        var tracker = new ConversationTracker();

        tracker.Observe(mind, "Do tulips need much sun?", "nice weather today", 1);
        Assert.Contains(tracker.PendingQuestions, q => q.Contains("tulips", StringComparison.OrdinalIgnoreCase));

        // A reply that shares content words with the question closes the thread.
        tracker.Observe(mind, "Do roses need water?", "Roses do need water, yes.", 2);
        Assert.DoesNotContain(tracker.PendingQuestions,
            q => q.Contains("roses", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Tracker_Open_Threads_Fade_After_Eight_Cycles()
    {
        using var mind = BuildMind();
        var tracker = new ConversationTracker();
        tracker.Observe(mind, "Who wrote that book about mountains?", "I am not sure!", 1);
        Assert.NotEmpty(tracker.PendingQuestions);
        tracker.Observe(mind, "just chatting a bit here", "cool", 10);
        Assert.Empty(tracker.PendingQuestions);   // cycle 10 - cycle 1 > 8 → faded
    }

    [Fact]
    public void Tracker_Writes_Topics_Into_Long_Term_Memory()
    {
        using var mind = BuildMind();
        var tracker = new ConversationTracker();
        for (int i = 1; i <= 7; i++)
            tracker.Observe(mind, "More about astronomy please.", "space is fascinating", i);
        Assert.Contains(mind.Memory!.LongTermMemories(),
            m => m.Content.Contains("talking about") && m.Source == "conversation");
    }

    // ------------------------------------------------------- thinking auto

    [Fact]
    public void Automatic_Through_The_Cycle_Thinks_Shallow_For_Small_Talk()
    {
        using var mind = MindBuilder.Create("Auto")
            .WithReasoning().WithSelfLearning().WithMemory().WithEmotions()
            .WithThinking(t => t.Auto())
            .Build();
        mind.Ask("hi");
        int thoughtsForBare = mind.Thinking!.LastThoughts.Count;      // a quick pause
        Assert.True(thoughtsForBare <= 2, $"a bare 'hi' should think shallowly, got {thoughtsForBare}");

        mind.Ask("Why do flowers turn toward the sun, and how does the whole " +
                 "process actually work inside the plant over a full day?");
        int thoughtsForDeep = mind.Thinking!.LastThoughts.Count;
        Assert.True(thoughtsForDeep > thoughtsForBare,
            $"a complex why/how question should think deeper ({thoughtsForDeep}) than 'hi' ({thoughtsForBare})");
    }

    [Fact]
    public void Automatic_Resolver_Never_Exceeds_The_Ceiling()
    {
        using var capped = MindBuilder.Create("Capped")
            .WithReasoning().WithSelfLearning().WithMemory().WithEmotions()
            .WithThinking(t => t.Auto(), ceiling: ThinkingCeiling.Only2)
            .Build();
        capped.Ask("Why do flowers turn toward the sun, and how does the whole " +
                   "process actually work inside the plant over a full day?");
        // A model capped at Only2 may never think above Medium — even when
        // the resolver wants Max.
        Assert.True(capped.Thinking!.LastThoughts.Count <= 2,
            $"ceiling Only2 must cap automatic thinking, got {capped.Thinking!.LastThoughts.Count}");
    }

    [Fact]
    public void Automatic_Flag_Persists_Through_Config_Serialization()
    {
        var t = new ThinkingOptions().Auto().UpTo(ThinkingCeiling.Only4);
        var json = JsonSerializer.Serialize(t);
        var restored = JsonSerializer.Deserialize<ThinkingOptions>(json)!;
        Assert.True(restored.Automatic);
        Assert.Equal(ThinkingCeiling.Only4, restored.Ceiling);
    }

    [Fact]
    public void UpTo_Clamps_The_Level_At_The_Ceiling()
    {
        var t = new ThinkingOptions().At(ThinkingLevel.Max).UpTo(ThinkingCeiling.Only2);
        Assert.Equal(ThinkingLevel.Medium, t.Ceiling.Clamp(t.Level));
        Assert.Equal(2, t.ThoughtRounds);   // Medium's promise, for status displays
    }

    // ------------------------------------------------- ThinkingAuto resolver

    [Fact]
    public void Resolver_Bare_Hello_Thinks_Shallow()
    {
        var level = ThinkingAuto.Choose("hi", hasDirectAnswer: false, recalledCount: 0,
            emotionIntensity: 0.1, epistemicGap: false, ceiling: ThinkingCeiling.Only5);
        Assert.Equal(ThinkingLevel.Low, level);
    }

    [Fact]
    public void Resolver_Complex_Why_How_Question_Thinks_Deep()
    {
        var level = ThinkingAuto.Choose(
            "Why do flowers turn toward the sun, and how does the whole process " +
            "actually work inside the plant over a full day, from dawn to dusk?",
            hasDirectAnswer: false, recalledCount: 0, emotionIntensity: 0.2,
            epistemicGap: true, ceiling: ThinkingCeiling.Only5);
        Assert.True(level is ThinkingLevel.VeryHigh or ThinkingLevel.Max,
            $"a heavy why/how probe should think very hard, got {level}");
    }

    [Fact]
    public void Resolver_Statement_With_Known_Answer_Thinks_Light_Questions_Think_Deep()
    {
        // A bare statement whose answer is already known and whose memories
        // are rich stays light — thinking exists for the hard parts.
        var statement = ThinkingAuto.Choose("tulips are my favorite flowers", hasDirectAnswer: true,
            recalledCount: 3, emotionIntensity: 0.2, epistemicGap: false,
            ceiling: ThinkingCeiling.Only5);
        Assert.Equal(ThinkingLevel.Low, statement);

        // The SAME known ground, asked as a question, still deserves real
        // deliberation — questions are what thinking is for (+2).
        var question = ThinkingAuto.Choose("what is a tulip?", hasDirectAnswer: true,
            recalledCount: 3, emotionIntensity: 0.2, epistemicGap: false,
            ceiling: ThinkingCeiling.Only5);
        Assert.True(question is ThinkingLevel.High or ThinkingLevel.VeryHigh,
            $"a question should think deeper than a statement ({question} vs {statement})");
        Assert.True(question > statement);
    }

    [Fact]
    public void Resolver_Never_Exceeds_The_Ceiling()
    {
        foreach (var ceiling in new[] { ThinkingCeiling.Only1, ThinkingCeiling.Only2,
                                        ThinkingCeiling.Only3, ThinkingCeiling.Only4 })
        {
            var level = ThinkingAuto.Choose(
                "Why does the whole mechanism of memory formation actually work " +
                "the way it does over time?", hasDirectAnswer: false, recalledCount: 0,
                emotionIntensity: 0.9, epistemicGap: true, ceiling: ceiling);
            Assert.Equal(ceiling.Clamp(ThinkingLevel.Max), level);
        }
    }

    [Fact]
    public void Resolver_Is_Deterministic()
    {
        var a = ThinkingAuto.Choose("why is the sky blue and how does scattering work?",
            false, 0, 0.3, true, ThinkingCeiling.Only5);
        var b = ThinkingAuto.Choose("why is the sky blue and how does scattering work?",
            false, 0, 0.3, true, ThinkingCeiling.Only5);
        Assert.Equal(a, b);
    }
}
