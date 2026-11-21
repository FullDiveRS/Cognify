using Cognify.Emotions;
using Cognify.Memory;

namespace Cognify.Core;

/// <summary>
/// The conversation tracker — the mind's running UNDERSTANDING of the chat
/// it is in: which topics keep coming back, which of the user's questions
/// were never fully answered, and how familiar the two of you have become.
///
/// It is deliberately lightweight (no separate module, no extra cycle cost):
/// a host feeds each turn's user text and the mind's own reply to
/// <see cref="Observe"/>, and the composer/thought generator may reference
/// the result ("Earlier you mentioned …"). Everything important ALSO flows
/// into long-term memory every few turns, so the understanding survives a
/// save/reload like the rest of the mind.
///
/// Read it any time: <c>tracker.Summary()</c>.
/// </summary>
public sealed class ConversationTracker
{
    private readonly Dictionary<string, int> _topicCounts = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<(string Question, int Cycle)> _pendingQuestions = new();
    private readonly object _gate = new();
    private int _lastMemoryTurn;

    /// <summary>How many turns (user messages) this conversation has had.</summary>
    public int Turns { get; private set; }

    /// <summary>0..1 — how familiar the conversation has become (turns/20, capped).</summary>
    public double Familiarity => Math.Min(1.0, Turns / 20.0);

    /// <summary>The conversation's recurring topics, most frequent first.</summary>
    public IReadOnlyList<string> Topics { get { lock (_gate) return _topicCounts.OrderByDescending(kv => kv.Value).Take(12).Select(kv => kv.Key).ToList(); } }

    /// <summary>Questions the user asked that the mind's replies never clearly addressed.</summary>
    public IReadOnlyList<string> PendingQuestions
    {
        get { lock (_gate) return _pendingQuestions.Select(q => q.Question).ToList(); }
    }

    /// <summary>
    /// Observes one turn. Topics are counted from the user's content words;
    /// a user question whose reply shares none of its content words is kept
    /// as an open thread (max 3, forgotten after 8 more cycles). Every 6th
    /// turn the top topics are written into long-term memory.
    /// </summary>
    public void Observe(Mind mind, string userText, string replyText, int cycle)
    {
        if (string.IsNullOrWhiteSpace(userText)) return;
        lock (_gate)
        {
            Turns++;

            var words = MoodEngine.Tokenize(userText)
                .Where(w => w.Length > 2 && !MoodEngine.IsStopWord(w))
                .Select(w => w.ToLowerInvariant());
            foreach (var w in words)
                _topicCounts[w] = _topicCounts.GetValueOrDefault(w) + 1;
            while (_topicCounts.Count > 40)
            {
                var rarest = _topicCounts.MinBy(kv => kv.Value).Key;
                _topicCounts.Remove(rarest);
            }

            if (userText.Contains('?'))
            {
                var replyWords = new HashSet<string>(
                    MoodEngine.Tokenize(replyText ?? "").Select(w => w.ToLowerInvariant()));
                bool addressed = MoodEngine.Tokenize(userText)
                    .Where(w => w.Length > 3 && !MoodEngine.IsStopWord(w))
                    .Any(w => replyWords.Contains(w.ToLowerInvariant()));
                if (!addressed)
                {
                    _pendingQuestions.Add((Truncate(userText.Trim(), 120), cycle));
                    if (_pendingQuestions.Count > 3) _pendingQuestions.RemoveAt(0);
                }
            }
            // Old open threads fade — they are no longer pending after a while.
            _pendingQuestions.RemoveAll(q => cycle - q.Cycle > 8);
        }

        // Persistence: every 6th turn, the conversation's shape lands in
        // long-term memory ("We have been talking about X and Y") — that is
        // how understanding survives a save/reload without new snapshot code.
        if (Turns - _lastMemoryTurn >= 6)
        {
            _lastMemoryTurn = Turns;
            var top = Topics;
            if (top.Count > 0)
                mind.Memory?.Remember(
                    $"We have been talking about {string.Join(", ", top.Take(2))}.",
                    MemoryKind.LongTerm, importance: 0.45, source: "conversation");
        }
    }

    /// <summary>
    /// A short developer-facing summary of what the mind understands about
    /// this conversation right now.
    /// </summary>
    public IReadOnlyList<string> Summary()
    {
        lock (_gate)
        {
            var lines = new List<string>
            {
                $"we have exchanged {Turns} message(s) (familiarity {Familiarity:P0})"
            };
            var top = _topicCounts.OrderByDescending(kv => kv.Value).Take(5).Select(kv => kv.Key).ToList();
            lines.Add(top.Count > 0
                ? $"recurring topics: {string.Join(", ", top)}"
                : "no recurring topics yet");
            if (_pendingQuestions.Count > 0)
                lines.Add($"still open: {string.Join(" / ", _pendingQuestions.Select(q => q.Question))}");
            return lines;
        }
    }

    /// <summary>The topic to reference when a reply wants to nod at the shared thread, or "".
    /// Ties break alphabetically (same order as <see cref="Topics"/>) so the choice is stable.</summary>
    internal string LatestTopic()
    {
        lock (_gate)
        {
            return _topicCounts.Count == 0
                ? ""
                : _topicCounts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.OrdinalIgnoreCase).First().Key;
        }
    }

    /// <summary>True when a reply may reference an earlier open question.</summary>
    internal bool HasOpenThread()
    {
        lock (_gate) return _pendingQuestions.Count > 0;
    }

    internal string? OldestOpenQuestion()
    {
        lock (_gate) return _pendingQuestions.Count > 0 ? _pendingQuestions[0].Question : null;
    }

    private static string Truncate(string text, int max) =>
        text.Length <= max ? text : text[..max].TrimEnd() + "...";
}
