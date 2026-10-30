using Cognify.Core;

namespace Cognify.Embeddings;

/// <summary>
/// The embeddings module: gives the mind a semantic sense of "aboutness".
/// Every stimulus is embedded into a vector and stored in a semantic store;
/// the most similar past inputs are surfaced for the rest of the cycle. This
/// lets the mind later notice "this new thing resembles something I've seen"
/// even when they share no keywords.
/// </summary>
public sealed class EmbeddingModule : CognitiveModuleBase
{
    private readonly EmbeddingOptions _options;
    private readonly object _gate = new();
    private double[]? _lastQueryVector;

    /// <summary>Creates the embeddings module with an optional custom model.</summary>
    public EmbeddingModule(EmbeddingOptions options, IEmbeddingModel? model = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Model = model ?? new HashedNGramEmbedder(options.Dimensions);
        Store = new EmbeddingStore(options.Capacity);
    }

    /// <inheritdoc />
    public override string Name => "embeddings";

    /// <inheritdoc />
    public override int Order => 14; // perception band, after vision/audio

    /// <summary>The active embedding model.</summary>
    public IEmbeddingModel Model { get; }

    /// <summary>The semantic vector store.</summary>
    public EmbeddingStore Store { get; }

    /// <summary>Embeds a text with the active model.</summary>
    public double[] Embed(string text) => Model.Embed(text);

    /// <summary>Cosine similarity between two texts via the active model.</summary>
    public double Similarity(string a, string b) =>
        HashedNGramEmbedder.CosineSimilarity(Model.Embed(a), Model.Embed(b));

    /// <summary>
    /// Searches the semantic store for texts closest to the query.
    /// </summary>
    public IReadOnlyList<(SemanticEntry Entry, double Similarity)> FindSimilar(string query, int topK = 3) =>
        Store.Search(query, Model, topK);

    /// <summary>Explicitly adds a text to the semantic store.</summary>
    public void Remember(string text, string tag = "")
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        Store.Add(text, Model.Embed(text), tag);
    }

    /// <summary>True when the last two processed texts were near-duplicates.</summary>
    public bool LastInputWasDuplicate { get; private set; }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var text = context.EffectiveContent;
        if (string.IsNullOrWhiteSpace(text)) return;

        var vector = Model.Embed(text);
        _lastQueryVector = vector;
        context.Bag["embedding.vector"] = vector;

        // Search before inserting so the query never matches itself.
        var neighbors = Store.Search(vector, _options.Neighbors);
        context.Bag["semantic.neighbors"] = neighbors;

        LastInputWasDuplicate = neighbors.Count > 0 &&
                                neighbors[0].Similarity >= _options.DuplicateThreshold;
        context.Bag["semantic.duplicate"] = LastInputWasDuplicate;

        if (neighbors.Count > 0 && neighbors[0].Similarity >= 0.25)
        {
            Host?.Stream?.Add("semantic",
                $"this resembles something known ({neighbors[0].Similarity:P0}): \"{Core.Mind.Truncate(neighbors[0].Entry.Text, 60)}\"");
        }

        Store.Add(text, vector, context.Stimulus.Kind.ToString().ToLowerInvariant());
    }

    /// <summary>The vector of the most recently processed stimulus (for testing/analysis).</summary>
    public double[]? LastQueryVector { get { lock (_gate) return _lastQueryVector; } }
}
