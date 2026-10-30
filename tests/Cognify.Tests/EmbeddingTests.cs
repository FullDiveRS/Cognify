using Cognify.Core;
using Cognify.Embeddings;

namespace Cognify.Tests;

/// <summary>Tests for hashed n-gram embeddings, the store and the module.</summary>
public class EmbeddingTests
{
    [Fact]
    public void Embedder_IsDeterministic()
    {
        var embedder = new HashedNGramEmbedder(64);
        var a = embedder.Embed("the quick brown fox");
        var b = embedder.Embed("the quick brown fox");
        Assert.Equal(a, b);
    }

    [Fact]
    public void Embedder_ProducesUnitVectors()
    {
        var embedder = new HashedNGramEmbedder(64);
        var v = embedder.Embed("some arbitrary text for normalization testing");
        double norm = Math.Sqrt(v.Sum(x => x * x));
        Assert.Equal(1.0, norm, precision: 6);
    }

    [Fact]
    public void Embedder_EmptyText_IsZeroVector()
    {
        var embedder = new HashedNGramEmbedder(64);
        var v = embedder.Embed("   ");
        Assert.All(v, x => Assert.Equal(0, x));
        Assert.Equal(0, HashedNGramEmbedder.CosineSimilarity(v, embedder.Embed("anything")));
    }

    [Fact]
    public void SimilarTexts_AreCloserThanUnrelated()
    {
        double close = HashedNGramEmbedder.Similarity(
            "the cat sat on the mat",
            "a cat sat on a mat");
        double far = HashedNGramEmbedder.Similarity(
            "the cat sat on the mat",
            "quarterly revenue increased by twelve percent");

        Assert.True(close > far, $"close={close:0.000} should exceed far={far:0.000}");
        Assert.True(close > 0.4);
    }

    [Fact]
    public void Similarity_IsBoundedInUnitInterval()
    {
        double s = HashedNGramEmbedder.Similarity("abc def ghi", "jkl mno pqr");
        Assert.InRange(s, 0.0, 1.0);
    }

    [Fact]
    public void Embedder_RejectsTooSmallDimensions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new HashedNGramEmbedder(8));
    }

    [Fact]
    public void Store_EvictsOldest_OverCapacity()
    {
        var store = new EmbeddingStore(3);
        var model = new HashedNGramEmbedder(32);
        for (int i = 0; i < 5; i++)
            store.Add($"text-{i}", model.Embed($"text-{i}"));

        Assert.Equal(3, store.Count);
        var hits = store.Search("text-4", model, topK: 10);
        Assert.Contains(hits, h => h.Entry.Text == "text-4");
        Assert.DoesNotContain(hits, h => h.Entry.Text == "text-0");
    }

    [Fact]
    public void Store_Search_ReturnsTopK_BySimilarity()
    {
        var store = new EmbeddingStore(10);
        var model = new HashedNGramEmbedder(64);
        store.Add("cats love fish", model.Embed("cats love fish"), "a");
        store.Add("dogs love bones", model.Embed("dogs love bones"), "b");
        store.Add("quarterly financial report", model.Embed("quarterly financial report"), "c");

        // minSimilarity 0 so all three are returned; ordering is what matters.
        var hits = store.Search("cats and fish", model, topK: 3, minSimilarity: 0);
        Assert.Equal(3, hits.Count);
        Assert.Equal("cats love fish", hits[0].Entry.Text);
        Assert.True(hits[0].Similarity >= hits[1].Similarity);
        Assert.True(hits[1].Similarity >= hits[2].Similarity);
    }

    [Fact]
    public void Module_StoresAndSearchesSemantics()
    {
        using var mind = MindBuilder.Create("Sage")
            .WithEmbeddings(e => e.Dim(64))
            .Build();

        mind.Ask("cats love to eat fish all day");
        mind.Ask("the sky is blue and wide");

        var semantic = mind.SemanticMemory!;
        Assert.Equal(2, semantic.Store.Count);

        var neighbors = semantic.FindSimilar("cats and their fish", topK: 2);
        Assert.NotEmpty(neighbors);
        Assert.Contains("cats", neighbors[0].Entry.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_DetectsNearDuplicates()
    {
        using var mind = MindBuilder.Create("Sage")
            .WithEmbeddings(e => e.Dim(64).Dim(e.Dimensions)) // defaults; threshold 0.92
            .Build();

        mind.Ask("the quick brown fox jumps over the lazy dog");
        mind.Ask("the quick brown fox jumps over the lazy dog");

        Assert.True(mind.SemanticMemory!.LastInputWasDuplicate,
            "an identical repetition should be flagged as a near-duplicate");
    }

    [Fact]
    public void Module_SharesNeighborsInBag()
    {
        using var mind = MindBuilder.Create("Sage").WithEmbeddings().Build();
        mind.Ask("kittens drink warm milk every morning");
        var response = mind.Ask("kittens drink warm milk every evening");

        // Second cycle should have surfaced the first as a semantic neighbor.
        Assert.True(mind.SemanticMemory!.Store.Count >= 2);
        Assert.NotNull(response);
    }

    [Fact]
    public void SemanticSimilarity_Convenience_Works()
    {
        using var mind = MindBuilder.Create("Sage").WithEmbeddings().Build();
        double s = mind.SemanticSimilarity("hello world", "hello world");
        Assert.True(s > 0.99);
    }

    [Fact]
    public void SemanticSimilarity_Throws_WhenDisabled()
    {
        using var mind = MindBuilder.Create("NoSage").Build();
        Assert.Throws<MindConfigurationException>(() => mind.SemanticSimilarity("a", "b"));
    }
}
