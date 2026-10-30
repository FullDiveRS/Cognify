using System.Text.RegularExpressions;

namespace Cognify.Tools;

/// <summary>
/// Finds an arithmetic expression embedded inside natural language, e.g.
/// "Hello Yume, what is 2 + 2?" -> "2 + 2", or
/// "what is the answer to 12 * 3?" -> "12 * 3".
///
/// A candidate must contain at least one operator strictly between two
/// numbers, so "chapter 3", "I am 25 years old" or "what is a bird?" never
/// match and stay with the normal cognitive cycle.
/// </summary>
internal static class MathSniffer
{
    private static readonly Regex Embedded = new(
        @"\d+(?:\.\d+)?(?:\s*[+\-*/%×÷]\s*\d+(?:\.\d+)?)+",
        RegexOptions.Compiled);

    internal static string? FindExpression(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;

        // Casual multiplication: "2 x 3" / "2 X 3" reads as "2 * 3".
        var normalized = Regex.Replace(text, @"(?<=\d)\s+[xX]\s+(?=\d)", " * ");

        string? best = null;
        foreach (Match m in Embedded.Matches(normalized))
        {
            var candidate = m.Value.Trim();
            if (candidate.Length < 3) continue;
            if (best is null || candidate.Length > best.Length) best = candidate;
        }
        return best;
    }
}
