using System.Text.Json;

namespace Cognify.Learning;

/// <summary>One next-token prediction: the token and its backoff probability.</summary>
public sealed record TokenPrediction(string Token, double Probability);

/// <summary>
/// A compact n-gram language model — the mind's "next token prediction".
///
/// Real trigram language modeling with stupid-backoff (Brants et al. 2007):
/// trigram count ratio, backed off to bigram (×0.4), then to a Laplace-
/// smoothed unigram (×0.16). The mind observes EVERYTHING it reads and says
/// — user turns, its own replies, taught facts, trained sayings, web search
/// results — so its predictions track the conversation it actually lives in.
///
/// Two uses inside the cognitive cycle:
///   1. <see cref="Score"/> — a fluency estimate for candidate replies; the
///      thinking module adds it to deliberation so wordings that match the
///      language the mind has absorbed win close ties.
///   2. <see cref="PredictNext"/> — literal next-token predictions
///      ("after these words, what would I say?").
/// </summary>
public sealed class NextTokenModel
{
    private readonly object _gate = new();
    private readonly Dictionary<string, int> _unigram = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dictionary<string, int>> _bigram = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dictionary<string, int>> _trigram = new(StringComparer.Ordinal);
    private long _totalTokens;


    // Stupid-backoff factors (the classic 0.4, squared for two hops).
    private const double Backoff = 0.4;
    private const double Backoff2 = 0.16;

    /// <summary>Total tokens the model has ever observed.</summary>
    public long TotalTokens { get { lock (_gate) return _totalTokens; } }

    /// <summary>Distinct words known to the model.</summary>
    public int Vocabulary { get { lock (_gate) return _unigram.Count; } }

    /// <summary>True once the model has seen anything at all.</summary>
    public bool IsTrained { get { lock (_gate) return _totalTokens > 0; } }

    // Cap on stored n-gram keys so a long-lived mind cannot grow forever.
    private const int MaxBigramKeys = 120_000;
    private const int MaxTrigramKeys = 120_000;

    /// <summary>
    /// Observes a piece of text: every sentence is tokenized and every
    /// 1/2/3-gram counted. Call it for anything the mind reads or says.
    /// </summary>
    public void Observe(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        foreach (var sentence in SplitSentences(text))
        {
            var tokens = Tokenize(sentence);
            lock (_gate)
            {
                for (int i = 0; i < tokens.Length; i++)
                {
                    _unigram[tokens[i]] = _unigram.TryGetValue(tokens[i], out var c1) ? c1 + 1 : 1;
                    _totalTokens++;
                    if (i >= 1 && _bigram.Count < MaxBigramKeys)
                    {
                        var bk = tokens[i - 1];
                        if (!_bigram.TryGetValue(bk, out var followers))
                            _bigram[bk] = followers = new Dictionary<string, int>(StringComparer.Ordinal);
                        followers[tokens[i]] = followers.TryGetValue(tokens[i], out var c2) ? c2 + 1 : 1;
                    }
                    if (i >= 2 && _trigram.Count < MaxTrigramKeys)
                    {
                        var tk = tokens[i - 2] + "\x1" + tokens[i - 1];
                        if (!_trigram.TryGetValue(tk, out var tri))
                            _trigram[tk] = tri = new Dictionary<string, int>(StringComparer.Ordinal);
                        tri[tokens[i]] = tri.TryGetValue(tokens[i], out var c3) ? c3 + 1 : 1;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Stupid-backoff probability of the next token given the last two
    /// tokens. Returns (token candidates excluded) a raw score in (0, 1].
    /// </summary>
    private double Probability(string? w1, string? w2, string next)
    {
        lock (_gate)
        {
            if (w1 is not null && w2 is not null &&
                _trigram.TryGetValue(w1 + "\x1" + w2, out var tri) &&
                tri.TryGetValue(next, out var triCount))
            {
                long denom = 0;
                foreach (var v in tri.Values) denom += v;
                if (denom > 0) return (double)triCount / denom;
            }
            if (w2 is not null && _bigram.TryGetValue(w2, out var bi) &&
                bi.TryGetValue(next, out var biCount))
            {
                long denom = 0;
                foreach (var v in bi.Values) denom += v;
                if (denom > 0) return Backoff * (double)biCount / denom;
            }
            if (_unigram.TryGetValue(next, out var uni) && _totalTokens > 0)
                return Backoff2 * (uni + 1.0) / (_totalTokens + _unigram.Count + 1.0);
            return Backoff2 * 1.0 / Math.Max(1, _unigram.Count + 1);
        }
    }

    /// <summary>
    /// Predicts the next tokens after the given context, best first.
    /// Candidates come from the trigram table, backed off to bigram, then to
    /// the most frequent unigrams — so there is ALWAYS an answer.
    /// </summary>
    public IReadOnlyList<TokenPrediction> PredictNext(string context, int count = 5)
    {
        if (count < 1) count = 1;
        var tokens = Tokenize(context ?? "");
        string? w1 = tokens.Length >= 2 ? tokens[^2] : null;
        string? w2 = tokens.Length >= 1 ? tokens[^1] : null;

        var candidates = new List<(string Token, double Score, bool Observed)>();
        lock (_gate)
        {
            if (w1 is not null && w2 is not null &&
                _trigram.TryGetValue(w1 + "\x1" + w2, out var tri))
                foreach (var (next, c) in tri)
                    candidates.Add((next, triScore(tri, c), true));
            if (w2 is not null && _bigram.TryGetValue(w2, out var bi))
                foreach (var (next, c) in bi)
                    if (!candidates.Any(cand => cand.Token == next))
                        candidates.Add((next, Backoff * biScore(bi, c), true));
            if (candidates.Count < count)
                foreach (var (next, c) in _unigram.OrderByDescending(kv => kv.Value).Take(40))
                    if (!candidates.Any(cand => cand.Token == next))
                        candidates.Add((next, Backoff2 * (c + 1.0) / (_totalTokens + _unigram.Count + 1.0), false));
        }

        double raw = 0;
        var ranked = candidates
            .Select(c =>
            {
                double p = Probability(w1, w2, c.Token);
                raw += p;
                return new TokenPrediction(c.Token, p);
            })
            .Where(p => p.Probability > 0)
            .OrderByDescending(p => p.Probability)
            .Take(count)
            .ToList();

        // Normalize so the returned probabilities form a friendly 0..1 share.
        if (raw > 0)
            ranked = ranked.Select(p => new TokenPrediction(p.Token, p.Probability / raw)).ToList();
        return ranked;
    }

    private static double triScore(Dictionary<string, int> tri, int c)
    {
        long denom = 0;
        foreach (var v in tri.Values) denom += v;
        return denom > 0 ? (double)c / denom : 0;
    }

    private static double biScore(Dictionary<string, int> bi, int c)
    {
        long denom = 0;
        foreach (var v in bi.Values) denom += v;
        return denom > 0 ? (double)c / denom : 0;
    }

    /// <summary>
    /// Mean per-token stupid-backoff log-probability of a text under the
    /// model — a fluency estimate. 0 means "unseen words everywhere";
    /// closer to 0 from below means the text reads like what the mind has
    /// absorbed. Untrained models return 0 (no opinion).
    /// </summary>
    public double Score(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return 0;
        lock (_gate)
            if (_totalTokens == 0) return 0;

        double sum = 0;
        int n = 0;
        foreach (var sentence in SplitSentences(text))
        {
            var tokens = Tokenize(sentence);
            for (int i = 0; i < tokens.Length; i++)
            {
                string? w1 = i >= 2 ? tokens[i - 2] : null;
                string? w2 = i >= 1 ? tokens[i - 1] : null;
                double p = Probability(w1, w2, tokens[i]);
                sum += Math.Log(Math.Max(p, 1e-8));
                n++;
            }
        }
        return n == 0 ? 0 : sum / n;
    }

    // ------------------------------------------------------------------
    // Tokenization: same normalization the MoodEngine uses (lowercase,
    // non-alphanumerics become spaces), applied per sentence so n-grams
    // never cross sentence boundaries.
    // ------------------------------------------------------------------
    private static IEnumerable<string> SplitSentences(string text)
    {
        foreach (var part in text.Split(new[] { '.', '!', '?', ';', '\n', '\r' },
                     StringSplitOptions.RemoveEmptyEntries))
        {
            var t = part.Trim();
            if (t.Length > 0) yield return t;
        }
    }

    private static string[] Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();
        var clean = new string(text.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : ' ')
            .ToArray());
        return clean.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>
    /// Merges another model's counts into this one (used when restoring a
    /// snapshot into a mind that already observed some text).
    /// </summary>
    public void Merge(NextTokenModel other)
    {
        if (other is null) return;
        lock (_gate)
        lock (other._gate)
        {
            foreach (var (word, count) in other._unigram)
                _unigram[word] = _unigram.TryGetValue(word, out var c1) ? c1 + count : count;
            foreach (var (w1, followers) in other._bigram)
            {
                if (!_bigram.TryGetValue(w1, out var mine))
                    _bigram[w1] = mine = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (var (w2, count) in followers)
                    mine[w2] = mine.TryGetValue(w2, out var c2) ? c2 + count : count;
            }
            foreach (var (key, followers) in other._trigram)
            {
                if (!_trigram.TryGetValue(key, out var mine))
                    _trigram[key] = mine = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (var (w3, count) in followers)
                    mine[w3] = mine.TryGetValue(w3, out var c3) ? c3 + count : count;
            }
            _totalTokens += other._totalTokens;
        }
    }

    // ------------------------------------------------------------------
    // Persistence: JSON of the three count tables. Older count-1 n-grams
    // beyond the cap are dropped so snapshots stay bounded for a mind that
    // talks every day.
    // ------------------------------------------------------------------
    private sealed class ModelDto
    {
        public List<string[]>? Uni { get; set; }
        public List<string[]>? Bi { get; set; }   // [w1, w2, count]
        public List<string[]>? Tri { get; set; }  // [w1, w2, w3, count]
        public long Total { get; set; }
    }

    private const int SerializeCap = 60_000;

    /// <summary>Serializes the model to JSON (for persistence).</summary>
    public string Serialize()
    {
        var dto = new ModelDto { Total = 0 };
        lock (_gate)
        {
            dto.Total = _totalTokens;
            dto.Uni = _unigram.Select(kv => new[] { kv.Key, kv.Value.ToString() }).ToList();
            dto.Bi = new List<string[]>();
            foreach (var (w1, followers) in _bigram)
                foreach (var (w2, c) in followers)
                    dto.Bi.Add(new[] { w1, w2, c.ToString() });
            dto.Tri = new List<string[]>();
            foreach (var (key, followers) in _trigram)
            {
                var parts = key.Split('\x1');
                foreach (var (w3, c) in followers)
                    dto.Tri.Add(new[] { parts[0], parts[1], w3, c.ToString() });
            }
            PruneForSerialization(dto);
        }
        return JsonSerializer.Serialize(dto);
    }

    // Keep every count>=2 entry, then top-up with count-1 entries newest-
    // first until the cap — the frequent structure of the language survives.
    private static void PruneForSerialization(ModelDto dto)
    {
        if (dto.Bi is { Count: > SerializeCap })
            dto.Bi = dto.Bi.Where(r => int.Parse(r[2]) >= 2).Take(SerializeCap).ToList();
        if (dto.Tri is { Count: > SerializeCap })
            dto.Tri = dto.Tri.Where(r => int.Parse(r[3]) >= 2).Take(SerializeCap).ToList();
    }

    /// <summary>Rebuilds a model from serialized JSON.</summary>
    public static NextTokenModel Deserialize(string json)
    {
        var model = new NextTokenModel();
        if (string.IsNullOrWhiteSpace(json)) return model;
        try
        {
            var dto = JsonSerializer.Deserialize<ModelDto>(json);
            if (dto is null) return model;
            lock (model._gate)
            {
                if (dto.Uni is not null)
                    foreach (var row in dto.Uni)
                        if (row.Length == 2 && int.TryParse(row[1], out var uc))
                            model._unigram[row[0]] = uc;
                if (dto.Bi is not null)
                    foreach (var row in dto.Bi)
                        if (row.Length == 3 && int.TryParse(row[2], out var bc))
                        {
                            if (!model._bigram.TryGetValue(row[0], out var bi))
                                model._bigram[row[0]] = bi = new Dictionary<string, int>(StringComparer.Ordinal);
                            bi[row[1]] = bc;
                        }
                if (dto.Tri is not null)
                    foreach (var row in dto.Tri)
                        if (row.Length == 4 && int.TryParse(row[3], out var tc))
                        {
                            var key = row[0] + "\x1" + row[1];
                            if (!model._trigram.TryGetValue(key, out var tri))
                                model._trigram[key] = tri = new Dictionary<string, int>(StringComparer.Ordinal);
                            tri[row[2]] = tc;
                        }
                model._totalTokens = dto.Total;
            }
        }
        catch (JsonException)
        {
            // Corrupt payload: a fresh model is better than a broken one.
        }
        return model;
    }
}
