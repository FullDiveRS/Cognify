namespace Cognify.Core;

/// <summary>Entry in a mind's inner monologue.</summary>
public sealed record StreamEntry(DateTimeOffset At, string Phase, string Text, double Confidence);

/// <summary>
/// The mind's inner monologue: a running, inspectable log of everything the
/// mind "says to itself" while processing stimuli.
/// </summary>
public sealed class ThoughtStream
{
    private readonly object _gate = new();
    private readonly List<StreamEntry> _entries = new();

    /// <summary>Raised every time a new entry is added to the stream.</summary>
    public event EventHandler<StreamEntry>? EntryAdded;

    /// <summary>Adds an entry to the stream.</summary>
    public void Add(string phase, string text, double confidence = 1.0)
    {
        var entry = new StreamEntry(DateTimeOffset.UtcNow, phase, text, confidence);
        lock (_gate)
        {
            _entries.Add(entry);
            if (_entries.Count > 512) _entries.RemoveRange(0, 64);
        }
        EntryAdded?.Invoke(this, entry);
    }

    /// <summary>Returns the most recent entries (newest last).</summary>
    public IReadOnlyList<StreamEntry> Recent(int count = 20)
    {
        lock (_gate)
        {
            int take = Math.Min(count, _entries.Count);
            return _entries.Skip(_entries.Count - take).ToList();
        }
    }

    /// <summary>Number of entries currently held.</summary>
    public int Count { get { lock (_gate) return _entries.Count; } }

    /// <summary>Clears the stream.</summary>
    public void Clear() { lock (_gate) _entries.Clear(); }
}
