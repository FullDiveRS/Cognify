using Cognify.Core;
using Cognify.Memory;
using Xunit;

namespace Cognify.Tests;

/// <summary>
/// Regression tests for the "repeats the same thing" bug: recall used to be
/// dominated by fresh-but-unrelated memories (the user's own last message),
/// duplicates piled up in the stores, the current stimulus was echoed back as
/// a "memory", and deliberation kept choosing the same style every turn.
/// </summary>
public class RepetitionTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Echo")
            .WithThinking()
            .WithMemory(m => m.Capacity(20))
            .Build();

    // --------------------------------------------------------------- memory

    [Fact]
    public void DuplicateRemember_StrengthensInsteadOfStacking()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        var first = memory.Remember("the cat sat on the mat", MemoryKind.ShortTerm, importance: 0.4);
        double strengthAfterFirst = first.Strength;

        var second = memory.Remember("The cat sat on the mat!", MemoryKind.ShortTerm, importance: 0.4);
        var third = memory.Remember("the  cat  sat  on  the  mat", MemoryKind.ShortTerm, importance: 0.4);

        Assert.Same(first, second);
        Assert.Same(first, third);
        Assert.Equal(1, memory.Counts.ShortTerm);          // no duplicate copies
        Assert.True(first.Strength > strengthAfterFirst);  // rehearsed instead
    }

    [Fact]
    public void DistinctMemories_AreAllStored()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        for (int i = 0; i < 5; i++)
            memory.Remember($"unique fact number {i}", MemoryKind.ShortTerm, importance: 0.4);
        Assert.Equal(5, memory.Counts.ShortTerm);
    }

    [Fact]
    public void Recall_DoesNotSurfaceFreshUnrelatedMemories()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("apples grow on trees", MemoryKind.LongTerm, importance: 0.9);

        // Completely unrelated query: freshness alone must not win anymore.
        Assert.Empty(memory.Recall("quantum entanglement experiment", max: 3));
    }

    [Fact]
    public void Recall_StillFindsGenuinelyRelatedMemories()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("the Eiffel tower is in Paris", MemoryKind.LongTerm, importance: 0.8);
        var hits = memory.Recall("tell me about Paris", max: 3);
        Assert.NotEmpty(hits);
        Assert.Contains("Paris", hits[0].Content);
    }

    [Fact]
    public void Recall_SelfText_ExcludesEchoOfCurrentInput()
    {
        using var mind = BuildMind();
        var memory = mind.Memory!;
        memory.Remember("hello there my good friend", MemoryKind.LongTerm, importance: 0.8);

        // Cycle path: the stimulus itself must not come back as a "memory".
        var hits = memory.Recall("hello there my good friend", max: 3,
            selfText: "hello there my good friend");
        Assert.Empty(hits);

        // Public search path (no selfText) keeps working for callers that
        // explicitly want to find a stored sentence.
        Assert.NotEmpty(memory.Recall("hello there my good friend", max: 3));
    }

    [Fact]
    public void Cycle_DoesNotParrotBackTheUsersOwnWords()
    {
        using var mind = BuildMind();
        mind.Process("the purple elephant danced through the market square");

        // Saying nearly the same thing again: the stored copy must not be
        // recalled and echoed back as the reply's "memory".
        var response = mind.Process("the purple elephant danced through the market square again");
        Assert.DoesNotContain(response.RecalledMemories,
            m => Jaccard(m.Content, "the purple elephant danced through the market square again") >= 0.85);
    }

    private static double Jaccard(string a, string b)
    {
        var ta = a.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        var tb = b.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        return ta.Count(tb.Contains) / (double)(ta.Count + tb.Count - ta.Count(tb.Contains));
    }

    // ------------------------------------------------------------- thinking

    [Fact]
    public void ConsecutiveReplies_VaryInsteadOfRepeating()
    {
        using var mind = MindBuilder.Create("Varied")
            .WithPersonality(p => p.Traits(openness: 0.9, extraversion: 0.7, agreeableness: 0.8))
            .WithEmotions(e => e.Intensity(0.8))
            .WithThinking()
            .WithSelfLearning()
            .WithMemory(m => m.Capacity(20))
            .Build();

        string[] inputs =
        {
            "tell me about the ocean",
            "what do you think about mountains?",
            "i enjoy quiet rainy afternoons",
            "let's talk about music",
            "how do rivers reach the sea?",
            "books fill my quiet room"
        };

        var replies = inputs.Select(i => mind.Process(i)).ToList();

        // No two consecutive replies are identical…
        for (int i = 1; i < replies.Count; i++)
            Assert.NotEqual(replies[i - 1].Content, replies[i].Content);

        // …and the conversation is not locked onto a single style.
        Assert.True(replies.Select(r => r.StyleName).Distinct().Count() >= 3,
            $"expected style variety, got: {string.Join(", ", replies.Select(r => r.StyleName))}");
    }
}
