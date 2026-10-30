namespace Cognify.Tools;

/// <summary>
/// What a tool needs to run: named string arguments (values come from the
/// parsed user request). Tools validate their own arguments and never throw
/// for invalid input — they return a failed <see cref="ToolResult"/> instead.
/// </summary>
public sealed class ToolInvocation
{
    /// <summary>Name of the tool being invoked.</summary>
    public string ToolName { get; init; } = string.Empty;

    /// <summary>Arguments for the tool (empty when none are needed).</summary>
    public IReadOnlyDictionary<string, string> Arguments { get; init; } =
        new Dictionary<string, string>();

    /// <summary>The raw user text that triggered the invocation (context for the tool).</summary>
    public string RawInput { get; init; } = string.Empty;
}

/// <summary>
/// The outcome of a tool invocation: success flag, a human-readable output
/// and optional structured data for programmatic use.
/// </summary>
public sealed class ToolResult
{
    /// <summary>Creates a successful result.</summary>
    public static ToolResult Ok(string output, IReadOnlyDictionary<string, object>? data = null) =>
        new(true, output, data ?? new Dictionary<string, object>());

    /// <summary>Creates a failed result with an error message.</summary>
    public static ToolResult Fail(string error) =>
        new(false, error, new Dictionary<string, object>());

    /// <summary>Creates a result from an exception thrown by a tool.</summary>
    public static ToolResult FromException(Exception exception) =>
        new(false, $"Tool error: {exception.Message}", new Dictionary<string, object>());

    private ToolResult(bool success, string output, IReadOnlyDictionary<string, object> data)
    {
        Success = success;
        Output = output;
        Data = data;
    }

    /// <summary>Whether the tool ran successfully.</summary>
    public bool Success { get; }

    /// <summary>Human-readable output (or error message).</summary>
    public string Output { get; }

    /// <summary>Structured results for programmatic consumption.</summary>
    public IReadOnlyDictionary<string, object> Data { get; }
}

/// <summary>
/// A capability the mind can use: a named, self-describing unit of work —
/// the classic "tool use / function calling" contract. Register tools with
/// <see cref="ToolRegistry"/> or via <c>mind.Tools.Register(...)</c>.
/// </summary>
public interface ITool
{
    /// <summary>Short lowercase name used to invoke the tool (e.g. "calculator").</summary>
    string Name { get; }

    /// <summary>One-line description of what the tool does (shown when listing capabilities).</summary>
    string Description { get; }

    /// <summary>
    /// Argument names and example/hint values (e.g. "expression" → "2+2*3").
    /// Empty when the tool needs no arguments.
    /// </summary>
    IReadOnlyDictionary<string, string> Parameters { get; }

    /// <summary>
    /// Optional trigger phrases: when a stimulus contains one of these
    /// (case-insensitive), the tool is a strong candidate for invocation.
    /// </summary>
    IReadOnlyList<string> Triggers { get; }

    /// <summary>
    /// True when the tool must NEVER fire on its own during the cognitive
    /// cycle — it only runs when invoked explicitly (a slash command, a
    /// "use tool ..." request, or direct code). Web search is the canonical
    /// manual-only tool: the mind must not go online whenever it feels like it.
    /// </summary>
    bool ManualOnly => false;

    /// <summary>Runs the tool. Implementations should return, not throw, for bad input.</summary>
    ToolResult Execute(ToolInvocation invocation);
}
