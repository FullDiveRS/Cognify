using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Memory;

/// <summary>
/// The memory module: a three-store memory system.
/// <list type="bullet">
/// <item><b>Short-term</b> — a fixed-capacity recency buffer.</item>
/// <item><b>Long-term</b> — important or rehearsed memories, with decay and strength-based recall.</item>
/// <item><b>Episodic</b> — experiences tagged with the emotion felt at the time; very intense ones become "flashbulb" memories that never decay.</item>
/// </list>
/// Recall combines keyword overlap, associative activation and salience.
/// </summary>
public sealed class MemoryModule : CognitiveModuleBase
{
    private readonly MemoryOptions _options;
    private readonly object _gate = new();
    private readonly Queue<MemoryRecord> _shortTerm = new();
    private readonly List<MemoryRecord> _longTerm = new();
    private readonly List<MemoryRecord> _episodes = new();

    internal MemoryModule(MemoryOptions options) => _options = options;

    /// <inheritdoc />
    public override string Name => "memory";

    /// <inheritdoc />
    public override int Order => 10;

    /// <summary>The associative concept network used for spreading-activation recall.</summary>
    public AssociativeNetwork Associations { get; } = new();

    /// <summary>Number of memories per store.</summary>
    public (int ShortTerm, int LongTerm, int Episodic) Counts
    {
        get { lock (_gate) return (_shortTerm.Count, _longTerm.Count, _episodes.Count); }
    }

    /// <summary>
    /// Stores a memory. Importance above the consolidation threshold (or high
    /// emotional intensity) promotes it directly to long-term / episodic store.
    /// Near-duplicate content rehearses the existing memory instead of adding
    /// a copy — duplicate pile-up used to dominate recall and made the mind
    /// repeat the same sentences back over and over.
    /// </summary>
    public MemoryRecord Remember(string content, MemoryKind kind = MemoryKind.ShortTerm,
        double importance = 0.4, EmotionType? emotionType = null, double emotionIntensity = 0,
        string? source = null)
    {
        lock (_gate)
        {
            // Repetition guard: restating near-identical content strengthens
            // the existing memory (re-encounter = rehearsal) rather than
            // stacking another copy into the store.
            var existing = FindNearDuplicate(content);
            if (existing is not null)
            {
                existing.Strength = Math.Min(1.0, existing.Strength + 0.08);
                existing.Importance = Math.Max(existing.Importance, Math.Clamp(importance, 0, 1));
                existing.LastRecalledAt = DateTimeOffset.UtcNow;
                LinkConcepts(content);
                return existing;
            }
        }

        var record = new MemoryRecord(content, kind, importance, emotionType, emotionIntensity, source);
        lock (_gate)
        {
            if (emotionIntensity >= 0.45 && record.Importance < 0.7)
                record.Importance = Math.Min(1.0, record.Importance + 0.2); // emotion tags memories stronger

            if (record.Importance >= _options.LongTermThreshold && kind == MemoryKind.ShortTerm)
                record.Kind = MemoryKind.LongTerm;

            switch (record.Kind)
            {
                case MemoryKind.LongTerm:
                    _longTerm.Add(record);
                    break;
                case MemoryKind.Episodic:
                    _episodes.Add(record);
                    break;
                default:
                    _shortTerm.Enqueue(record);
                    while (_shortTerm.Count > _options.ShortTermCapacity)
                        _shortTerm.Dequeue();
                    break;
            }
        }
        LinkConcepts(content);
        return record;
    }

    /// <summary>
    /// Token-set Jaccard similarity (0..1) between two texts. Used to detect
    /// near-duplicate content (repeated phrasing, self-echo).
    /// </summary>
    internal static double TokenJaccard(string a, string b)
    {
        var ta = MoodEngine.Tokenize(a).ToHashSet();
        var tb = MoodEngine.Tokenize(b).ToHashSet();
        if (ta.Count == 0 && tb.Count == 0) return 1;
        if (ta.Count == 0 || tb.Count == 0) return 0;
        int shared = ta.Count(tb.Contains);
        return shared / (double)(ta.Count + tb.Count - shared);
    }

    /// <summary>Finds an existing memory that is a near-duplicate of the given content, if any.</summary>
    private MemoryRecord? FindNearDuplicate(string content)
    {
        foreach (var record in _shortTerm.Concat(_longTerm).Concat(_episodes))
        {
            if (TokenJaccard(record.Content, content) >= 0.9)
                return record;
        }
        return null;
    }

    private void LinkConcepts(string content)
    {
        if (!_options.UseAssociations) return;
        var tokens = MoodEngine.Tokenize(content).Where(t => t.Length > 3).Distinct().Take(8).ToList();
        for (int i = 0; i < tokens.Count; i++)
            for (int j = i + 1; j < tokens.Count; j++)
                Associations.Link(tokens[i], tokens[j], 0.15);
    }

    /// <summary>
    /// Recalls the most relevant memories for a query, combining keyword
    /// overlap, associative activation and salience. Recalled memories get
    /// stronger (rehearsal); the strongest hits may consolidate to long-term.
    /// A memory must actually share content with the query to surface —
    /// freshness alone used to let the previous chat message win recall every
    /// turn, making the mind parrot the user's last sentence back.
    /// Flashbulb memories may still intrude uninvited (that is what makes
    /// them vivid). Pass <paramref name="selfText"/> to exclude near-duplicates
    /// of what is being perceived right now (the present is not a memory).
    /// </summary>
    public IReadOnlyList<MemoryRecord> Recall(string query, int max = 3, string? selfText = null)
    {
        var now = DateTimeOffset.UtcNow;
        var queryTokens = MoodEngine.Tokenize(query).ToHashSet();
        var queryContentWords = queryTokens.Where(t => !MoodEngine.IsStopWord(t)).ToHashSet();

        List<(MemoryRecord Record, double Score)> scored = new();
        lock (_gate)
        {
            var pool = _shortTerm.Concat(_longTerm).Concat(_episodes);
            foreach (var record in pool)
            {
                // Self-echo guard: a record that merely restates the current
                // perception adds no information — recalling it made the mind
                // repeat the user's own words back to them.
                if (selfText is not null &&
                    (TokenJaccard(record.Content, selfText) >= 0.85 ||
                     TokenJaccard(record.Content, query) >= 0.85))
                    continue;

                var contentTokens = MoodEngine.Tokenize(record.Content);
                double overlap = contentTokens.Count(t => queryTokens.Contains(t));
                if (queryTokens.Count == 0) overlap = 0;
                double keywordScore = queryTokens.Count == 0 ? 0 : overlap / (double)queryTokens.Count;
                double salience = record.Salience(now);

                // Relevance gate (content words): two sentences that only share
                // "about"/"the"/"you" are not about the same thing. Without
                // this, one shared stopword was enough to surface an unrelated
                // recent message — and the mind parroted it back.
                bool sharesContentWord = queryContentWords.Count > 0 &&
                    contentTokens.Any(t => !MoodEngine.IsStopWord(t) && queryContentWords.Contains(t));

                // No genuine content-word overlap means no recall, unless the
                // memory is a flashbulb (vivid memories may intrude on their
                // own — that is what makes them vivid).
                if (!sharesContentWord && !record.IsFlashbulb) continue;
                scored.Add((record, keywordScore * 1.6 + salience));
            }
        }

        // Associative spread: boost memories containing related concepts.
        if (_options.UseAssociations && queryTokens.Count > 0)
        {
            var related = Associations.Activate(queryTokens, depth: 2, max: 6);
            if (related.Count > 0)
            {
                var relatedSet = related.ToDictionary(r => r.Concept, r => r.Activation);
                lock (_gate)
                {
                    foreach (var record in _longTerm.Concat(_episodes))
                    {
                        var tokens = MoodEngine.Tokenize(record.Content);
                        double boost = tokens.Where(t => relatedSet.ContainsKey(t))
                            .Select(t => relatedSet[t]).DefaultIfEmpty(0).Max() * 0.4;
                        if (boost > 0)
                        {
                            double previous = 0;
                            foreach (var entry in scored)
                                if (entry.Record == record) { previous = entry.Score; break; }
                            if (previous > 0)
                                scored.RemoveAll(s => s.Record == record);
                            scored.Add((record, (previous > 0 ? previous : 0.15) + boost));
                        }
                    }
                }
            }
        }

        var results = scored
            .Where(s => s.Score > 0.12)
            .OrderByDescending(s => s.Score)
            .Take(max)
            .Select(s => s.Record)
            .ToList();

        foreach (var record in results)
        {
            record.Reinforce(_options.RecallBoost);
            TryConsolidate(record);
        }

        Host?.RaiseMemoryRecalled(results);
        return results;
    }

    private void TryConsolidate(MemoryRecord record)
    {
        if (record.Kind != MemoryKind.ShortTerm) return;
        if (record.RecallCount >= 2 || record.Strength >= 0.9)
        {
            lock (_gate)
            {
                record.Kind = MemoryKind.LongTerm;
                var list = _longTerm;
                list.Add(record);
            }
        }
    }

    /// <summary>Returns recent short-term contents (oldest first).</summary>
    public IReadOnlyList<MemoryRecord> RecentShortTerm()
    {
        lock (_gate) return _shortTerm.ToList();
    }

    /// <summary>Returns long-term memories above a strength filter.</summary>
    public IReadOnlyList<MemoryRecord> LongTermMemories(double minStrength = 0)
    {
        lock (_gate) return _longTerm.Where(m => m.Strength >= minStrength)
            .OrderByDescending(m => m.Strength).ToList();
    }

    /// <summary>Returns episodic memories, newest first.</summary>
    public IReadOnlyList<MemoryRecord> Episodes()
    {
        lock (_gate) return _episodes.OrderByDescending(m => m.CreatedAt).ToList();
    }

    /// <summary>How strongly is this exact content remembered right now (0 if absent)?</summary>
    public double StrengthOf(string content) =>
        LongTermMemories().FirstOrDefault(m => m.Content == content)?.Strength ?? 0;

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        // Recall memories relevant to the stimulus, excluding near-duplicates
        // of the stimulus itself (self-echo).
        var recalled = Recall(context.EffectiveContent, max: 3, selfText: context.Stimulus.Content);
        context.Recalled = recalled;

        // Store the stimulus itself as a memory.
        double intensity = context.Emotions?.DominantIntensity ?? 0;
        var emotion = context.Emotions?.Dominant;
        Remember(context.Stimulus.Content,
            kind: context.Stimulus.Kind == StimulusKind.Event ? MemoryKind.Episodic : MemoryKind.ShortTerm,
            importance: Math.Clamp(0.3 + context.Stimulus.Intensity * 0.4, 0, 1),
            emotionType: intensity > 0.25 ? emotion : null,
            emotionIntensity: intensity,
            source: context.Stimulus.Kind.ToString().ToLowerInvariant());
    }

    /// <inheritdoc />
    public override void Tick()
    {
        lock (_gate)
        {
            foreach (var m in _longTerm) m.Decay(_options.DecayRate);
            foreach (var m in _episodes) m.Decay(_options.DecayRate * 0.5);

            // Forget memories that faded completely (unless flashbulb).
            _longTerm.RemoveAll(m => !m.IsFlashbulb && m.Strength <= 0.06);
        }
    }
}
