using Cognify.Core;
using Cognify.Emotions;
using Cognify.Memory;

namespace Cognify.Tests;

/// <summary>Tests for short-term, long-term, episodic memory and associations.</summary>
public class MemoryTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Rememberer")
            .WithMemory(m => m.Capacity(3))
            .Build();

    [Fact]
    public void ShortTerm_EvictsOldest_WhenOverCapacity()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        for (int i = 0; i < 6; i++)
            memory.Remember($"item-{i}", MemoryKind.ShortTerm, importance: 0.1);
        Assert.Equal(3, memory.Counts.ShortTerm);
        var recent = memory.RecentShortTerm().Select(m => m.Content).ToList();
        Assert.DoesNotContain("item-0", recent);
        Assert.Contains("item-5", recent);
    }

    [Fact]
    public void HighImportance_ConsolidatesToLongTerm()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("critical fact", MemoryKind.ShortTerm, importance: 0.9);
        Assert.Equal(1, memory.Counts.LongTerm);
        Assert.Equal(0, memory.Counts.ShortTerm);
    }

    [Fact]
    public void Recall_FindsKeywordMatches()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("the cat sat on the mat", MemoryKind.LongTerm, importance: 0.5);
        memory.Remember("quantum physics is strange", MemoryKind.LongTerm, importance: 0.5);
        var recalled = memory.Recall("tell me about the cat", max: 1);
        Assert.NotEmpty(recalled);
        Assert.Contains("cat", recalled[0].Content);
    }

    [Fact]
    public void Recall_BoostsStrength()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("birds fly south for winter", MemoryKind.LongTerm, importance: 0.5);
        double before = memory.StrengthOf("birds fly south for winter");
        memory.Recall("birds", max: 5);
        double after = memory.StrengthOf("birds fly south for winter");
        Assert.True(after > before);
    }

    [Fact]
    public void RepeatedRecall_ConsolidatesShortTermToLongTerm()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("a moderately important note", MemoryKind.ShortTerm, importance: 0.4);
        memory.Recall("moderately important note", max: 1);
        memory.Recall("moderately important note", max: 1);
        Assert.Equal(1, memory.Counts.LongTerm);
    }

    [Fact]
    public void EmotionalMemories_BecomeEpisodic()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("the concert was loud and bright", MemoryKind.Episodic,
            importance: 0.7, emotionType: EmotionType.Joy, emotionIntensity: 0.8);
        Assert.Equal(1, memory.Counts.Episodic);
        Assert.True(memory.Episodes()[0].IsFlashbulb); // intensity >= 0.75
    }

    [Fact]
    public void FlashbulbMemories_NeverDecay()
    {
        var record = new MemoryRecord("wedding day", MemoryKind.Episodic, 0.8,
            EmotionType.Joy, 0.9, "event");
        record.Decay(0.5);
        Assert.Equal(1.0, record.Strength);
    }

    [Fact]
    public void Tick_DecaysLongTermMemories()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("fading memory", MemoryKind.LongTerm, importance: 0.5);
        double before = memory.StrengthOf("fading memory");
        for (int i = 0; i < 50; i++) mind.Tick();
        double after = memory.StrengthOf("fading memory");
        Assert.True(after < before);
    }

    [Fact]
    public void Associations_LinkCooccurringConcepts()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("coffee morning routine", MemoryKind.ShortTerm, importance: 0.1);
        Assert.True(memory.Associations.NodeCount > 0);
        var related = memory.Associations.Activate(new[] { "coffee" });
        Assert.Contains(related, r => r.Concept == "morning");
    }

    [Fact]
    public void Cycle_StoresStimulusAsMemory()
    {
        using var mind = BuildMind();
        mind.Process("a distinctive purple elephant");
        Assert.True(mind.Memory!.Counts.ShortTerm + mind.Memory.Counts.LongTerm +
                    mind.Memory.Counts.Episodic > 0);
    }

    [Fact]
    public void Cycle_Recall_PopulatesResponse()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("the Eiffel tower is in Paris", MemoryKind.LongTerm, importance: 0.8);
        var response = mind.Process("tell me about Paris");
        Assert.NotEmpty(response.RecalledMemories);
        Assert.Contains("Paris", response.RecalledMemories[0].Content);
    }

    [Fact]
    public void EventStimuli_AreStoredEpisodically()
    {
        using var mind = BuildMind();
        mind.Process(Stimulus.Event("the alarm rang loudly", intensity: 0.9));
        Assert.Equal(1, mind.Memory!.Counts.Episodic);
    }
}
