namespace Cognify.Core;

/// <summary>
/// A mind's trained vocabulary: sentence-level "sayings" it picked up through
/// training and may speak in its own voice later.
///
/// This is the direct answer to "she keeps repeating the same sentence — is
/// that because she does not have many words?": partly, yes. A mind's spoken
/// variety comes from the sentence shapes it owns. Built-in templates are a
/// fixed starter set; every saying added here widens what the mind can say,
/// and the response composer draws from this pool when it has nothing
/// stronger to answer with — exactly the situation where a small vocabulary
/// used to sound like a broken record.
/// </summary>
public sealed class PhrasePool
{
    private readonly object _gate = new();
    private readonly List<string> _sayings = new();

    /// <summary>How many trained sayings the mind owns.</summary>
    public int Count { get { lock (_gate) return _sayings.Count; } }

    /// <summary>
    /// Registers a saying. Returns true when it was new; duplicates (case /
    /// punctuation-insensitive) are dropped so re-training a corpus never
    /// bloats the pool.
    /// </summary>
    public bool Add(string saying)
    {
        var clean = Normalize(saying);
        if (clean is null) return false;
        lock (_gate)
        {
            if (_sayings.Any(s => string.Equals(s, clean, StringComparison.OrdinalIgnoreCase)))
                return false;
            _sayings.Add(clean);
            return true;
        }
    }

    /// <summary>All sayings, in the order they were trained.</summary>
    public IReadOnlyList<string> All()
    {
        lock (_gate) return _sayings.ToArray();
    }

    /// <summary>
    /// Removes a saying (case/punctuation-insensitive match). Used by the
    /// self-improvement module to retire sayings that consistently drew
    /// disapproval — the mind stops saying what never lands. Returns true
    /// when a saying was removed.
    /// </summary>
    public bool Remove(string saying)
    {
        var clean = Normalize(saying);
        if (clean is null) return false;
        lock (_gate)
        {
            for (int i = 0; i < _sayings.Count; i++)
            {
                if (string.Equals(_sayings[i], clean, StringComparison.OrdinalIgnoreCase))
                {
                    _sayings.RemoveAt(i);
                    return true;
                }
            }
            return false;
        }
    }

    /// <summary>Loads sayings from a snapshot without duplicating existing ones.</summary>
    public void Restore(IEnumerable<string> sayings)
    {
        foreach (var s in sayings) Add(s);
    }

    /// <summary>
    /// Returns up to <paramref name="max"/> sayings for the composer, best
    /// first. Relevance is word overlap with the conversation text; when
    /// nothing overlaps, a rotating random window keeps trained lines fresh
    /// instead of always surfacing the first one.
    /// </summary>
    public IReadOnlyList<string> PickRelevant(string contextText, Random rng, int max)
    {
        if (max <= 0) return Array.Empty<string>();
        var contextWords = ContentWords(contextText ?? "");
        if (contextWords.Count == 0) return RotatedSample(rng, max);

        List<(string Saying, int Score)> scored;
        lock (_gate)
        {
            scored = _sayings.Select(s => (Saying: s,
                                  Score: s.Split(' ').Count(w => contextWords.Contains(w))))
                             .ToList();
        }
        // Relevant first (stable within a score band, shuffled inside ties so
        // the same saying does not always represent its score band).
        var ordered = scored.OrderByDescending(p => p.Score)
                            .ThenBy(_ => rng.Next())
                            .ToList();
        int best = ordered[0].Score;
        if (best == 0) return RotatedSample(rng, max);

        var pool = ordered.Where(p => p.Score >= Math.Max(1, best / 2))
                          .Select(p => p.Saying)
                          .Take(max)
                          .ToList();
        return pool.Count > 0 ? pool : RotatedSample(rng, max);
    }

    // A tiny rotation cursor so "random fallback" does not always start from
    // the same first saying in a big pool.
    private int _cursor;
    private IReadOnlyList<string> RotatedSample(Random rng, int max)
    {
        lock (_gate)
        {
            if (_sayings.Count == 0) return Array.Empty<string>();
            int start = _cursor + rng.Next(Math.Min(3, _sayings.Count));
            _cursor = (start + max) % _sayings.Count;
            return Enumerable.Range(0, Math.Min(max, _sayings.Count))
                .Select(i => _sayings[(start + i) % _sayings.Count])
                .ToList();
        }
    }

    private static HashSet<string> ContentWords(string text)
    {
        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var raw in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var w = new string(raw.Where(char.IsLetter).ToArray()).ToLowerInvariant();
            if (w.Length >= 3) words.Add(w);
        }
        return words;
    }

    internal static string? Normalize(string? saying)
    {
        if (string.IsNullOrWhiteSpace(saying)) return null;
        var t = saying.Trim().TrimEnd();
        // Normalize trailing punctuation + internal whitespace so
        // "Hello there !" and "hello there!" are the same saying.
        t = string.Join(' ', t.Split(' ', StringSplitOptions.RemoveEmptyEntries));
        if (t.Length < 3 || t.Length > 300) return null;
        return t.TrimEnd('.', '!', '"') is { } trimmed && trimmed.Length >= 3 ? trimmed : t;
    }
}
