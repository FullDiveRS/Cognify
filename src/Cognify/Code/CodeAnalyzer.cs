using System.Text.RegularExpressions;

namespace Cognify.Code;

/// <summary>
/// Metrics produced by <see cref="CodeAnalyzer"/>: structural facts measured
/// from the actual source text.
/// </summary>
public sealed class CodeMetrics
{
    /// <summary>Total lines.</summary>
    public int Lines { get; internal set; }

    /// <summary>Non-empty, non-comment lines.</summary>
    public int CodeLines { get; internal set; }

    /// <summary>Comment lines (// or # or -- style).</summary>
    public int CommentLines { get; internal set; }

    /// <summary>Maximum brace/indent nesting depth found.</summary>
    public int MaxNestingDepth { get; internal set; }

    /// <summary>Whether (), [] and {} are all balanced.</summary>
    public bool BracketsBalanced { get; internal set; } = true;

    /// <summary>Best-guess language ("csharp", "python", "javascript", "sql" or "unknown").</summary>
    public string LanguageGuess { get; internal set; } = "unknown";

    /// <summary>Keywords and identifiers seen most often (top 5, for insight).</summary>
    public IReadOnlyList<string> FrequentTokens { get; internal set; } = Array.Empty<string>();

    /// <summary>A rough complexity score: code lines × nesting depth ÷ 10, rounded.</summary>
    public double ComplexityScore => Math.Round(CodeLines * (double)Math.Max(1, MaxNestingDepth) / 10.0, 1);

    /// <summary>Formats the metrics as a one-line human summary.</summary>
    public string Summarize() =>
        $"{LanguageGuess}: {Lines} lines ({CodeLines} code, {CommentLines} comments), " +
        $"nesting {MaxNestingDepth}, complexity {ComplexityScore}, " +
        $"brackets {(BracketsBalanced ? "balanced" : "UNBALANCED")}";
}

/// <summary>
/// Lightweight, real static analysis of source code: line counts, comment
/// ratio, bracket balance, nesting depth and a language heuristic. No
/// compilers, no dependencies — enough for a cognitive agent to "eyeball"
/// a snippet the way a developer skimming it would.
/// </summary>
public static class CodeAnalyzer
{
    /// <summary>Analyzes a snippet of source code.</summary>
    public static CodeMetrics Analyze(string source)
    {
        var metrics = new CodeMetrics();
        if (string.IsNullOrWhiteSpace(source)) return metrics;

        var lines = source.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
        metrics.Lines = lines.Length;

        int paren = 0, square = 0, brace = 0;
        var tokenCounts = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var rawLine in lines)
        {
            var line = rawLine.Trim();
            if (line.Length == 0) continue;

            bool isComment = line.StartsWith("//", StringComparison.Ordinal)
                             || line.StartsWith('#')
                             || line.StartsWith("--", StringComparison.Ordinal)
                             || line.StartsWith("/*", StringComparison.Ordinal)
                             || line.StartsWith("*", StringComparison.Ordinal);
            if (isComment) { metrics.CommentLines++; continue; }

            metrics.CodeLines++;

            // Nesting depth: prefer brace depth; fall back to indentation (4 spaces per level).
            foreach (var c in line)
            {
                switch (c)
                {
                    case '(': paren++; break;
                    case ')': paren--; break;
                    case '[': square++; break;
                    case ']': square--; break;
                    case '{': brace++; break;
                    case '}': brace--; break;
                }
            }

            int indent = rawLine.Length - rawLine.TrimStart().Length;
            int indentDepth = indent == 0 ? 0 : (indent + 3) / 4;
            metrics.MaxNestingDepth = Math.Max(metrics.MaxNestingDepth, Math.Max(brace, indentDepth));

            foreach (Match m in Regex.Matches(line, @"[A-Za-z_][A-Za-z0-9_]*"))
            {
                var token = m.Value;
                tokenCounts[token] = tokenCounts.GetValueOrDefault(token) + 1;
            }
        }

        metrics.BracketsBalanced = paren == 0 && square == 0 && brace == 0;
        metrics.LanguageGuess = GuessLanguage(source);
        metrics.FrequentTokens = tokenCounts
            .Where(kv => kv.Value > 1 && kv.Key.Length > 3)
            .OrderByDescending(kv => kv.Value).Take(5)
            .Select(kv => kv.Key).ToList();

        return metrics;
    }

    /// <summary>
    /// Heuristic language detection from syntax markers — real signal,
    /// simple rules.
    /// </summary>
    public static string GuessLanguage(string source)
    {
        if (string.IsNullOrWhiteSpace(source)) return "unknown";

        if (source.Contains("using System", StringComparison.Ordinal) ||
            Regex.IsMatch(source, @"\b(public|private|internal|protected)\s+(sealed\s+|static\s+|abstract\s+|readonly\s+)*(class|void|int|string|bool|double|var)\b") ||
            source.Contains("Console.WriteLine", StringComparison.Ordinal))
            return "csharp";

        if (Regex.IsMatch(source, @"\bdef \w+\(.*\)\s*:") ||
            source.Contains("elif ", StringComparison.Ordinal) ||
            source.Contains("print(", StringComparison.Ordinal) && Regex.IsMatch(source, @"\bimport \w+\b"))
            return "python";

        if (source.Contains("function ", StringComparison.Ordinal) ||
            source.Contains("console.log", StringComparison.Ordinal) ||
            source.Contains("=>", StringComparison.Ordinal) && (source.Contains("const ", StringComparison.Ordinal) || source.Contains("let ", StringComparison.Ordinal)))
            return "javascript";

        if (Regex.IsMatch(source, @"\bSELECT\b.*\bFROM\b", RegexOptions.IgnoreCase | RegexOptions.Singleline) ||
            Regex.IsMatch(source, @"\bINSERT INTO\b", RegexOptions.IgnoreCase) ||
            Regex.IsMatch(source, @"\bCREATE TABLE\b", RegexOptions.IgnoreCase))
            return "sql";

        return "unknown";
    }
}
