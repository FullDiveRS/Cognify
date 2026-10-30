namespace Cognify.Learning;

/// <summary>
/// The hybrid of the mind's two language models: the statistical n-gram
/// (<see cref="NextTokenModel"/>) and the neural transformer
/// (<see cref="TransformerLM"/>).
///
/// The two models fail in complementary ways, which is exactly why they
/// belong together:
///
///   • the n-gram is exact about what it has literally seen ("after 'thank
///     you' comes 'so'") but helpless one step beyond its tables;
///   • the transformer generalizes — it can predict a word in a context it
///     never saw verbatim because attention ties contexts together through
///     embeddings — but on tiny data it is noisier than raw counts.
///
/// Blending rule: <c>p = w · p_neural + (1 − w) · p_ngram</c> over the union
/// of both models' candidate tokens, renormalized. When the transformer is
/// untrained the blend degrades gracefully to pure n-gram (and
/// <see cref="Fluency"/> to the n-gram score), so the mind is never worse
/// off for having a neural half.
/// </summary>
public static class HybridNextToken
{
    /// <summary>
    /// Blended next-token predictions after the given context, best first.
    /// </summary>
    /// <param name="ngram">The mind's statistical n-gram model (always present).</param>
    /// <param name="neural">The mind's transformer (may be untrained).</param>
    /// <param name="context">The words to predict after.</param>
    /// <param name="count">How many predictions to return.</param>
    /// <param name="neuralWeight">Share of the neural model when it is trained (0..1).</param>
    public static IReadOnlyList<TokenPrediction> PredictNext(
        NextTokenModel ngram, TransformerLM? neural, string context, int count = 5,
        double neuralWeight = 0.6)
    {
        if (count < 1) count = 1;
        var fromNgram = ngram.PredictNext(context, Math.Max(count * 5, 25));

        if (neural is null || !neural.IsTrained)
            return fromNgram.Take(count).ToList();

        var fromNeural = neural.PredictNext(context, Math.Max(count * 5, 25));
        double w = Math.Clamp(neuralWeight, 0.0, 1.0);

        // Union of candidates, interpolated; the top `count` are then
        // renormalized over the returned set (same convention as the n-gram
        // model: probabilities sum to ~1 across the returned predictions).
        var scores = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var p in fromNgram)
            scores[p.Token] = scores.GetValueOrDefault(p.Token) + (1.0 - w) * p.Probability;
        foreach (var p in fromNeural)
            scores[p.Token] = scores.GetValueOrDefault(p.Token) + w * p.Probability;

        var top = scores.OrderByDescending(kv => kv.Value).Take(count).ToList();
        double total = top.Sum(kv => kv.Value);
        if (total <= 0) return fromNgram.Take(count).ToList();

        return top
            .Select(kv => new TokenPrediction(kv.Key, kv.Value / total))
            .ToList();
    }

    /// <summary>
    /// Blended fluency score for a text — the mean per-token log-probability
    /// under both models (higher, i.e. closer to 0 from below, reads more
    /// natural). Untrained neural half contributes nothing.
    /// </summary>
    public static double Fluency(NextTokenModel ngram, TransformerLM? neural,
        string text, double neuralWeight = 0.5)
    {
        double ngramScore = ngram.Score(text);
        if (neural is null || !neural.IsTrained) return ngramScore;
        double w = Math.Clamp(neuralWeight, 0.0, 1.0);
        double neuralScore = neural.Score(text);
        return w * neuralScore + (1.0 - w) * ngramScore;
    }
}
