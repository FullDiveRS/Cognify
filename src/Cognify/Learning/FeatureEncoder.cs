using System.Security.Cryptography;

namespace Cognify.Learning;

/// <summary>
/// Converts text into a fixed-size feature vector using feature hashing
/// (the "hashing trick"). Same text always maps to the same vector; similar
/// texts share tokens and thus overlap in the vector space. Vectors are
/// L2-normalized so the network sees consistent input magnitudes.
/// </summary>
public static class FeatureEncoder
{
    /// <summary>Encodes text into a normalized feature vector of the given dimension.</summary>
    public static double[] Encode(string text, int dimensions)
    {
        var vector = new double[dimensions];
        if (string.IsNullOrWhiteSpace(text)) return vector;

        var tokens = Tokenize(text);
        foreach (var token in tokens)
        {
            int index = (int)(Fnv1aHash(token) % (ulong)dimensions);
            vector[index] += 1.0;
        }

        // Add bigram features for a little context sensitivity.
        for (int i = 0; i < tokens.Count - 1; i++)
        {
            int index = (int)(Fnv1aHash(tokens[i] + "_" + tokens[i + 1]) % (ulong)dimensions);
            vector[index] += 0.5;
        }

        double norm = Math.Sqrt(vector.Sum(v => v * v));
        if (norm > 0)
            for (int i = 0; i < dimensions; i++) vector[i] /= norm;
        return vector;
    }

    private static IReadOnlyList<string> Tokenize(string text)
    {
        var clean = new string(text.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : ' ')
            .ToArray());
        return clean.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private static ulong Fnv1aHash(string text)
    {
        ulong hash = 14695981039346656037UL;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= 1099511628211UL;
        }
        return hash;
    }
}
