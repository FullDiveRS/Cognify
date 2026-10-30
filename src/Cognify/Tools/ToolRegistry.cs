namespace Cognify.Tools;

/// <summary>
/// A name-indexed collection of <see cref="ITool"/> instances. Thread-safe;
/// registering a duplicate name replaces the old tool.
/// </summary>
public sealed class ToolRegistry
{
    private readonly object _gate = new();
    private readonly Dictionary<string, ITool> _tools = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Registers (or replaces) a tool. Null names/tools are rejected.</summary>
    public void Register(ITool tool)
    {
        if (tool is null) throw new ArgumentNullException(nameof(tool));
        if (string.IsNullOrWhiteSpace(tool.Name))
            throw new ArgumentException("A tool must have a name.", nameof(tool));
        lock (_gate) _tools[tool.Name.Trim()] = tool;
    }

    /// <summary>Removes a tool by name. Returns true when it existed.</summary>
    public bool Unregister(string name)
    {
        lock (_gate) return _tools.Remove(name);
    }

    /// <summary>Looks a tool up by name.</summary>
    public ITool? Get(string name)
    {
        lock (_gate) return _tools.GetValueOrDefault(name?.Trim() ?? string.Empty);
    }

    /// <summary>All registered tools.</summary>
    public IReadOnlyList<ITool> All()
    {
        lock (_gate) return _tools.Values.OrderBy(t => t.Name).ToList();
    }

    /// <summary>How many tools are registered.</summary>
    public int Count { get { lock (_gate) return _tools.Count; } }

    /// <summary>
    /// Finds tools whose triggers or name appear in the given text
    /// (case-insensitive), best matches first.
    /// </summary>
    public IReadOnlyList<ITool> Match(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<ITool>();
        var lower = text.ToLowerInvariant();
        lock (_gate)
        {
            var matches = new List<(ITool Tool, int Score)>();
            foreach (var tool in _tools.Values)
            {
                if (tool.ManualOnly)
                    continue; // manual-only tools never auto-fire (e.g. web search)
                int score = 0;
                if (lower.Contains(tool.Name.ToLowerInvariant())) score += 2;
                foreach (var trigger in tool.Triggers)
                    if (lower.Contains(trigger.ToLowerInvariant())) { score += 3; break; }
                if (score > 0) matches.Add((tool, score));
            }
            return matches.OrderByDescending(m => m.Score).Select(m => m.Tool).ToList();
        }
    }
}
