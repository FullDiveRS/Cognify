using System.Globalization;

namespace Cognify.Tools;

/// <summary>
/// Tells the current UTC time and date. A trivial tool, but the canonical
/// "the agent can reach beyond its own mind" example.
/// </summary>
public sealed class ClockTool : ITool
{
    /// <inheritdoc />
    public string Name => "clock";

    /// <inheritdoc />
    public string Description => "Returns the current UTC date and time.";

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Parameters => new Dictionary<string, string>();

    /// <inheritdoc />
    public IReadOnlyList<string> Triggers => new[] { "what time", "what's the time", "current time", "what date", "today's date" };

    /// <inheritdoc />
    public ToolResult Execute(ToolInvocation invocation)
    {
        var now = DateTimeOffset.UtcNow;
        return ToolResult.Ok(
            $"It is currently {now:HH:mm:ss} UTC on {now:yyyy-MM-dd} ({now:dddd}).",
            new Dictionary<string, object>
            {
                ["utc"] = now,
                ["unix"] = now.ToUnixTimeSeconds(),
                ["weekday"] = now.DayOfWeek.ToString()
            });
    }
}

/// <summary>
/// Evaluates arithmetic expressions — implemented with a real tokenizer and
/// recursive-descent parser (no string evaluation, no external dependency).
/// Supports +, -, *, /, %, parentheses, unary minus and decimal numbers.
/// </summary>
public sealed class CalculatorTool : ITool
{
    /// <inheritdoc />
    public string Name => "calculator";

    /// <inheritdoc />
    public string Description => "Evaluates an arithmetic expression, e.g. 2 + 2 * 3 or (1 + 2) / 4.";

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Parameters =>
        new Dictionary<string, string> { ["expression"] = "the arithmetic expression to evaluate" };

    /// <inheritdoc />
    public IReadOnlyList<string> Triggers => new[] { "calculate", "what is", "how much is", "compute" };

    /// <inheritdoc />
    public ToolResult Execute(ToolInvocation invocation)
    {
        var expression = invocation.Arguments.GetValueOrDefault("expression")
                         ?? ExtractExpression(invocation.RawInput);
        if (string.IsNullOrWhiteSpace(expression))
            return ToolResult.Fail("No expression given. Usage: calculate <expression>.");

        try
        {
            var parser = new Parser(expression);
            double value = parser.ParseExpression();
            parser.ExpectEnd();

            string formatted = Math.Abs(value - Math.Round(value)) < 1e-9
                ? Math.Round(value).ToString(CultureInfo.InvariantCulture)
                : value.ToString("0.######", CultureInfo.InvariantCulture);

            return ToolResult.Ok($"{expression.Trim()} = {formatted}",
                new Dictionary<string, object> { ["value"] = value, ["expression"] = expression });
        }
        catch (FormatException ex)
        {
            return ToolResult.Fail($"Could not evaluate '{expression.Trim()}': {ex.Message}");
        }
    }

    private static string ExtractExpression(string raw)
    {
        // Strip common prefixes: "calculate", "what is", "how much is", trailing "?".
        var trimmed = raw.Trim().TrimEnd('?', '!', '.');
        var text = trimmed.ToLowerInvariant();
        foreach (var prefix in new[] { "calculate ", "compute ", "what is ", "what's ", "how much is " })
            if (text.StartsWith(prefix, StringComparison.Ordinal))
                return trimmed[prefix.Length..].Trim();
        // Otherwise look for math anywhere in the sentence
        // ("Hello Yume, what is 2 + 2?" -> "2 + 2").
        return MathSniffer.FindExpression(trimmed) ?? string.Empty;
    }

    // ------------------------------------------------------------------
    // Tokenizer + recursive-descent parser
    // ------------------------------------------------------------------
    private sealed class Parser
    {
        private readonly string _input;
        private int _pos;

        public Parser(string input)
        {
            _input = input.Replace(" ", string.Empty, StringComparison.Ordinal)
                          .Replace(",", string.Empty, StringComparison.Ordinal) // 1,000 → 1000
                          .Replace("×", "*", StringComparison.Ordinal)
                          .Replace("÷", "/", StringComparison.Ordinal);
        }

        public double ParseExpression()
        {
            var left = ParseTerm();
            while (Peek() is '+' or '-')
            {
                char op = Advance();
                var right = ParseTerm();
                left = op == '+' ? left + right : left - right;
            }
            return left;
        }

        private double ParseTerm()
        {
            var left = ParseFactor();
            while (Peek() is '*' or '/' or '%')
            {
                char op = Advance();
                var right = ParseFactor();
                left = op switch
                {
                    '*' => left * right,
                    '/' => right == 0 ? throw new FormatException("division by zero") : left / right,
                    '%' => right == 0 ? throw new FormatException("modulo by zero") : left % right,
                    _ => left
                };
            }
            return left;
        }

        private double ParseFactor()
        {
            if (Peek() == '(')
            {
                Advance();
                var value = ParseExpression();
                if (Peek() != ')') throw new FormatException("missing closing parenthesis");
                Advance();
                return value;
            }
            if (Peek() == '-')
            {
                Advance();
                return -ParseFactor();
            }
            if (Peek() == '+') { Advance(); return ParseFactor(); }

            return ParseNumber();
        }

        private double ParseNumber()
        {
            int start = _pos;
            while (_pos < _input.Length && (char.IsDigit(_input[_pos]) || _input[_pos] == '.'))
            {
                _pos++;
            }
            if (start == _pos)
                throw new FormatException($"unexpected character '{Peek()}' at position {_pos}");
            if (!double.TryParse(_input[start.._pos], NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
                throw new FormatException($"invalid number '{_input[start.._pos]}'");
            return value;
        }

        public void ExpectEnd()
        {
            if (_pos < _input.Length)
                throw new FormatException($"unexpected character '{_input[_pos]}' at position {_pos}");
        }

        private char Peek() => _pos < _input.Length ? _input[_pos] : '\0';
        private char Advance() => _input[_pos++];
    }
}

/// <summary>
/// Basic text statistics: words, characters, sentences, reading time.
/// </summary>
public sealed class TextStatsTool : ITool
{
    /// <inheritdoc />
    public string Name => "text-stats";

    /// <inheritdoc />
    public string Description => "Counts words, characters, sentences and estimated reading time of a text.";

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Parameters =>
        new Dictionary<string, string> { ["text"] = "the text to analyze (defaults to the whole input)" };

    /// <inheritdoc />
    public IReadOnlyList<string> Triggers => new[] { "count words", "how many words", "text stats", "word count" };

    /// <inheritdoc />
    public ToolResult Execute(ToolInvocation invocation)
    {
        var text = invocation.Arguments.GetValueOrDefault("text") ?? invocation.RawInput;
        if (string.IsNullOrWhiteSpace(text))
            return ToolResult.Fail("No text given to analyze.");

        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var sentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        int readingSeconds = (int)Math.Ceiling(words.Length / 200.0 * 60); // ~200 wpm

        return ToolResult.Ok(
            $"{words.Length} words, {text.Length} characters, {Math.Max(sentences.Length, text.Length > 0 ? 1 : 0)} sentence(s), " +
            "~" + Math.Max(1, (int)Math.Ceiling(readingSeconds / 60.0)) + " min reading time.",
            new Dictionary<string, object>
            {
                ["words"] = words.Length,
                ["characters"] = text.Length,
                ["sentences"] = Math.Max(sentences.Length, 1),
                ["readingSeconds"] = readingSeconds
            });
    }
}

/// <summary>
/// Registers the built-in tool set on a registry.
/// </summary>
public static class BuiltinTools
{
    /// <summary>Registers clock, calculator and text-stats on the given registry.</summary>
    public static void RegisterAll(ToolRegistry registry)
    {
        registry.Register(new ClockTool());
        registry.Register(new CalculatorTool());
        registry.Register(new TextStatsTool());
    }
}
