namespace Cognify.Embeddings;

/// <summary>
/// A text embedding model: maps any string to a dense unit vector.
/// Implement this to plug a real embedder (sentence-transformers, an API...)
/// into the semantic pipeline; the built-in one uses deterministic hashing.
/// </summary>
public interface IEmbeddingModel
{
    /// <summary>Vector dimensionality.</summary>
    int Dimensions { get; }

    /// <summary>Embeds a piece of text into a unit-length vector.</summary>
    double[] Embed(string text);
}

/// <summary>
/// A real, dependency-free embedding model based on feature hashing of word
/// unigrams, character 3-grams and word bigrams. Each feature is hashed into
/// two buckets ("one permutation hashing") with a signed weight to reduce
/// collision bias, then the whole vector is L2-normalized. The result is a
/// deterministic, language-agnostic embedding with genuine semantic behavior:
/// texts sharing words or sub-words land close together in cosine space.
/// </summary>
public sealed class HashedNGramEmbedder : IEmbeddingModel
{
    private readonly int _dimensions;

    /// <summary>Creates an embedder with the given vector dimensionality (min 16).</summary>
    public HashedNGramEmbedder(int dimensions = 64)
    {
        if (dimensions < 16)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Embedding dimensions must be at least 16.");
        _dimensions = dimensions;
    }

    /// <inheritdoc />
    public int Dimensions => _dimensions;

    /// <inheritdoc />
    public double[] Embed(string text)
    {
        var vector = new double[_dimensions];
        if (string.IsNullOrWhiteSpace(text)) return vector;

        var normalized = Normalize(text);
        var words = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // Word unigrams ---------------------------------------------------
        foreach (var word in words)
            AddFeature(vector, "w:" + word, 1.0);

        // Word bigrams ----------------------------------------------------
        for (int i = 0; i < words.Length - 1; i++)
            AddFeature(vector, "b:" + words[i] + "_" + words[i + 1], 0.7);

        // Character trigrams (capture sub-word similarity, typos, morphology)
        var joined = "#" + string.Join("#", words) + "#";
        for (int i = 0; i + 3 <= joined.Length; i++)
            AddFeature(vector, "c:" + joined.Substring(i, 3), 0.35);

        // L2 normalize ----------------------------------------------------
        double norm = Math.Sqrt(vector.Sum(v => v * v));
        if (norm > 0)
            for (int i = 0; i < vector.Length; i++)
                vector[i] /= norm;

        return vector;
    }

    private void AddFeature(double[] vector, string feature, double weight)
    {
        // Two independent hashes into the same space: reduces collision bias
        // the way "one permutation hashing" does in sparse projections.
        var h1 = Fnv1a(feature, seed: 2166136261);
        var h2 = Fnv1a(feature, seed: 16777619);
        int bucket1 = (int)(h1 % (uint)_dimensions);
        int bucket2 = (int)(h2 % (uint)_dimensions);
        bool negative = (h1 >> 31) == 1;

        vector[bucket1] += negative ? -weight : weight;
        vector[bucket2] += weight * 0.5;
    }

    private static uint Fnv1a(string text, uint seed)
    {
        uint hash = seed;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= 16777619;
        }
        return hash;
    }

    private static string Normalize(string text) =>
        new string(text.ToLowerInvariant()
                .Select(c => char.IsLetterOrDigit(c) ? c : ' ')
                .ToArray())
            .Replace("  ", " ", StringComparison.Ordinal)
            .Trim();

    /// <summary>Cosine similarity between two texts (0..1; never negative).</summary>
    public static double Similarity(string a, string b, int dimensions = 64) =>
        CosineSimilarity(new HashedNGramEmbedder(dimensions).Embed(a),
                         new HashedNGramEmbedder(dimensions).Embed(b));

    /// <summary>Cosine similarity between two vectors (clamped to 0..1).</summary>
    public static double CosineSimilarity(double[] a, double[] b)
    {
        if (a is null || b is null || a.Length != b.Length || a.Length == 0) return 0;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return Math.Clamp(dot / (Math.Sqrt(na) * Math.Sqrt(nb)), 0, 1);
    }
}

/// <summary>
/// One entry of the semantic store: the original text, its vector and a
/// free-form tag (e.g. the memory kind or the source module).
/// </summary>
public sealed record SemanticEntry(string Text, double[] Vector, string Tag, DateTimeOffset CreatedAt);

/// <summary>
/// An in-memory vector store with cosine-similarity search. Capacity-bounded
/// (oldest entries are evicted) and thread-safe. This is the mind's semantic
/// memory: it can later find "thoughts like this one" even when no keyword matches.
/// </summary>
public sealed class EmbeddingStore
{
    private readonly object _gate = new();
    private readonly Queue<SemanticEntry> _entries;
    private readonly int _capacity;

    /// <summary>Creates a store with the given capacity (min 1).</summary>
    public EmbeddingStore(int capacity = 500)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
        _entries = new Queue<SemanticEntry>(capacity);
    }

    /// <summary>How many entries are currently stored.</summary>
    public int Count { get { lock (_gate) return _entries.Count; } }

    /// <summary>Adds a text with its precomputed vector.</summary>
    public void Add(string text, double[] vector, string tag = "")
    {
        lock (_gate)
        {
            _entries.Enqueue(new SemanticEntry(text, vector, tag, DateTimeOffset.UtcNow));
            while (_entries.Count > _capacity)
                _entries.Dequeue();
        }
    }

    /// <summary>
    /// Finds the k entries most similar to the query vector. Entries whose
    /// similarity is below <paramref name="minSimilarity"/> are skipped.
    /// </summary>
    public IReadOnlyList<(SemanticEntry Entry, double Similarity)> Search(
        double[] query, int topK = 3, double minSimilarity = 0.05)
    {
        List<(SemanticEntry, double)> scored;
        lock (_gate)
        {
            scored = new List<(SemanticEntry, double)>(_entries.Count);
            foreach (var entry in _entries)
            {
                double sim = HashedNGramEmbedder.CosineSimilarity(query, entry.Vector);
                if (sim >= minSimilarity) scored.Add((entry, sim));
            }
        }

        return scored.OrderByDescending(s => s.Item2).Take(topK).ToList();
    }

    /// <summary>Convenience: search by text (embeds the query with the given model).</summary>
    public IReadOnlyList<(SemanticEntry Entry, double Similarity)> Search(
        string query, IEmbeddingModel model, int topK = 3, double minSimilarity = 0.05) =>
        Search(model.Embed(query), topK, minSimilarity);

    /// <summary>Returns all entries, oldest first (for persistence).</summary>
    public IReadOnlyList<SemanticEntry> All()
    {
        lock (_gate) return _entries.ToList();
    }

    /// <summary>Clears the store.</summary>
    public void Clear()
    {
        lock (_gate) _entries.Clear();
    }
}
