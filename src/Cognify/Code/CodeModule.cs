using System.Text.RegularExpressions;
using Cognify.Core;

namespace Cognify.Code;

/// <summary>
/// The code module: a developer mind. It recognizes two situations —
/// <list type="bullet">
/// <item><b>Code requests</b> ("write a function that...", "generate a class...")
/// → generates a scaffold via the configured <see cref="ICodeBackend"/>.</item>
/// <item><b>Code input</b> (fenced blocks or code-looking text)
/// → runs real static analysis (<see cref="CodeAnalyzer"/>) and reports metrics.</item>
/// </list>
/// </summary>
public sealed class CodeModule : CognitiveModuleBase
{
    private readonly CodeOptions _options;
    private readonly object _gate = new();
    private int _requestsHandled;
    private int _snippetsAnalyzed;

    /// <summary>Creates the code module with an optional custom generation backend.</summary>
    public CodeModule(CodeOptions options, ICodeBackend? backend = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Backend = backend ?? new TemplateCodeBackend { Namespace = options.Namespace };
    }

    /// <inheritdoc />
    public override string Name => "code";

    /// <inheritdoc />
    public override int Order => 46; // action band: response set here is kept by thinking (order 50)

    /// <summary>The active code-generation backend.</summary>
    public ICodeBackend Backend { get; }

    /// <summary>How many code requests have been handled.</summary>
    public int RequestsHandled { get { lock (_gate) return _requestsHandled; } }

    /// <summary>How many snippets have been analyzed.</summary>
    public int SnippetsAnalyzed { get { lock (_gate) return _snippetsAnalyzed; } }

    /// <summary>Generates code programmatically (also used by the cognitive cycle).</summary>
    public CodeResult Generate(string description, CodeRequestKind kind, string? language = null, string? name = null) =>
        Backend.Generate(new CodeRequest(description, kind, language ?? _options.PreferredLanguage, name ?? ""));

    /// <summary>Analyzes a snippet with the static analyzer.</summary>
    public CodeMetrics Analyze(string source) => CodeAnalyzer.Analyze(source);

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        if (context.Stimulus.Kind is StimulusKind.Image or StimulusKind.Audio or StimulusKind.Video or StimulusKind.Feedback)
            return;

        var text = context.EffectiveContent;
        if (string.IsNullOrWhiteSpace(text)) return;

        // ---- 1. Code input? (fenced block or clearly code-like) ---------
        var snippet = ExtractSnippet(context.Stimulus.Content);
        if (snippet is { } code && _options.AnalyzeInput)
        {
            var metrics = CodeAnalyzer.Analyze(code.Source);
            lock (_gate) _snippetsAnalyzed++;
            context.Bag["code.metrics"] = metrics;
            Host?.Stream?.Add("code", $"analyzed {metrics.LanguageGuess} snippet: {metrics.Summarize()}");

            if (context.Response is null)
            {
                string verdict = metrics.BracketsBalanced
                    ? "The brackets look balanced."
                    : "Heads up: the brackets are NOT balanced.";
                context.Response = new ResponseDraft(
                    $"I analyzed the {metrics.LanguageGuess} snippet: {metrics.Summarize()}. {verdict}",
                    0.6 + (metrics.BracketsBalanced ? 0.1 : 0), 860, "code-analysis");
            }
            return;
        }

        // ---- 2. Code request? --------------------------------------------
        var request = ParseRequest(text);
        if (request is null) return;

        var result = Generate(request.Value.description, request.Value.kind, request.Value.language);
        lock (_gate) _requestsHandled++;
        context.Bag["code.generated"] = result;
        Host?.Stream?.Add("code", $"generated {result.Language} {result.Kind} scaffold ({result.Source.CountLines()} lines)");

        if (context.Response is null)
        {
            context.Response = new ResponseDraft(
                $"Here is a {result.Language} scaffold for \"{request.Value.description}\":\n\n{result.Source}\n\n" +
                string.Join(' ', result.Notes),
                0.55, 861, "code-generation");
        }
    }

    private static (string description, CodeRequestKind kind, string? language)? ParseRequest(string text)
    {
        var lower = text.ToLowerInvariant();

        bool isRequest = Regex.IsMatch(lower, @"\b(write|create|generate|make|give me|show me)\b.*\b(code|class|function|method|loop|query|sql|endpoint|script|program)\b")
                         || Regex.IsMatch(lower, @"\b(scaffold|snippet)\b");
        if (!isRequest) return null;

        var language = (string?)null;
        if (lower.Contains("python")) language = "python";
        else if (lower.Contains("javascript") || lower.Contains("typescript")) language = "javascript";
        else if (lower.Contains("sql") || lower.Contains("query")) language = "sql";

        var kind = CodeRequestKind.Generic;
        if (Regex.IsMatch(lower, @"\b(sql|query|select)\b")) kind = CodeRequestKind.Query;
        else if (Regex.IsMatch(lower, @"\bendpoint\b|\bapi\b|\brest\b")) kind = CodeRequestKind.Endpoint;
        else if (Regex.IsMatch(lower, @"\bloop\b|\bforeach\b|\biterate\b")) kind = CodeRequestKind.Loop;
        else if (Regex.IsMatch(lower, @"\bfunction\b|\bmethod\b")) kind = CodeRequestKind.Function;
        else if (Regex.IsMatch(lower, @"\bclass\b")) kind = CodeRequestKind.Class;

        // Description = the request text minus the imperative prefix.
        var description = Regex.Replace(text.Trim(),
            @"^\s*(please\s+)?(can you\s+)?(write|create|generate|make|give me|show me)\s+(me\s+)?(a |an |the )?(c#|csharp|python|javascript|typescript|sql)?\s*",
            string.Empty, RegexOptions.IgnoreCase).Trim();
        if (description.Length == 0) description = text.Trim();

        return (description, kind, language);
    }

    private static (string Source, string? Language)? ExtractSnippet(string text)
    {
        // Markdown fenced block: ```lang ... ```
        var fence = Regex.Match(text, @"```(\w+)?\s*\n(.*?)```", RegexOptions.Singleline);
        if (fence.Success)
            return (fence.Groups[2].Value, fence.Groups[1].Success ? fence.Groups[1].Value : null);

        // Inline code-ish heuristic: several lines, code markers, no prose verbs.
        var lines = text.Split('\n');
        if (lines.Length >= 3)
        {
            bool codeMarkers = text.Contains(';') || text.Contains('{') && text.Contains('}') ||
                               text.Contains("def ") || text.Contains("SELECT", StringComparison.OrdinalIgnoreCase);
            bool noQuestions = !text.Contains('?');
            if (codeMarkers && noQuestions && CodeAnalyzer.GuessLanguage(text) != "unknown")
                return (text, null);
        }
        return null;
    }
}

/// <summary>Tiny extension used for line counts in messages.</summary>
internal static class StringLineExtensions
{
    public static int CountLines(this string s) =>
        s.Length == 0 ? 0 : s.Replace("\r\n", "\n", StringComparison.Ordinal).Count(c => c == '\n') + 1;
}
