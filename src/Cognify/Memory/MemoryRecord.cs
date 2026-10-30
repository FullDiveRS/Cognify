namespace Cognify.Memory;

/// <summary>The three memory stores of the mind.</summary>
public enum MemoryKind
{
    /// <summary>Fleeting buffer — holds the last N items, then evicts the oldest.</summary>
    ShortTerm,

    /// <summary>Durable store — consolidated memories that persist and strengthen on recall.</summary>
    LongTerm,

    /// <summary>Experience records tagged with the emotion felt at the time.</summary>
    Episodic
}

/// <summary>
/// A single memory record: what happened, how important it was, how strong it
/// currently is (memories decay unless rehearsed), and the emotion attached.
/// </summary>
public sealed class MemoryRecord
{
    /// <summary>Creates a memory record. Prefer <see cref="MemoryModule.Remember"/> for normal use.</summary>
    public MemoryRecord(string content, MemoryKind kind, double importance,
                        Emotions.EmotionType? emotionType, double emotionIntensity, string? source)
    {
        Id = Guid.NewGuid();
        Content = content;
        Kind = kind;
        Importance = Math.Clamp(importance, 0, 1);
        IsFlashbulb = emotionIntensity >= 0.75;
        // Fresh memories start below full strength so rehearsal (recall) and
        // decay have room to work. Flashbulb memories start vivid.
        Strength = IsFlashbulb ? 1.0 : Math.Clamp(0.55 + Importance * 0.3, 0.3, 0.9);
        CreatedAt = DateTimeOffset.UtcNow;
        EmotionType = emotionType;
        EmotionIntensity = emotionIntensity;
        Source = source;
    }

    /// <summary>Unique memory id.</summary>
    public Guid Id { get; }

    /// <summary>The remembered content.</summary>
    public string Content { get; internal set; }

    /// <summary>Which store the memory lives in.</summary>
    public MemoryKind Kind { get; internal set; }

    /// <summary>How important the memory was when formed (0..1). Drives consolidation.</summary>
    public double Importance { get; internal set; }

    /// <summary>Current memory strength (0..1). Decays over time, boosted by recall.</summary>
    public double Strength { get; internal set; }

    /// <summary>When the memory was formed.</summary>
    public DateTimeOffset CreatedAt { get; }

    /// <summary>When the memory was last recalled.</summary>
    public DateTimeOffset? LastRecalledAt { get; internal set; }

    /// <summary>How many times this memory has been recalled (rehearsal count).</summary>
    public int RecallCount { get; internal set; }

    /// <summary>Emotion felt when the memory formed (flashbulb memories keep vivid).</summary>
    public Emotions.EmotionType? EmotionType { get; }

    /// <summary>Intensity of the attached emotion (0..1).</summary>
    public double EmotionIntensity { get; }

    /// <summary>True for "flashbulb" memories — emotionally vivid, never decay.</summary>
    public bool IsFlashbulb { get; }

    /// <summary>Where the memory came from ("taught", "observed", "event"...).</summary>
    public string? Source { get; internal set; }

    /// <summary>Call when the memory is recalled: boosts strength (rehearsal effect).</summary>
    internal void Reinforce(double boost)
    {
        Strength = Math.Min(1.0, Strength + boost);
        RecallCount++;
        LastRecalledAt = DateTimeOffset.UtcNow;
    }

    /// <summary>Applies time-based decay unless the memory is a flashbulb.</summary>
    public void Decay(double rate)
    {
        if (IsFlashbulb) return;
        Strength = Math.Max(0.05, Strength - rate * (1.0 - Importance));
    }

    /// <summary>Restores full persisted state (used by the persistence layer).</summary>
    internal void Restore(double strength, int recallCount, DateTimeOffset? lastRecalledAt, string? source)
    {
        Strength = Math.Clamp(strength, 0, 1);
        RecallCount = recallCount;
        LastRecalledAt = lastRecalledAt;
        Source = source;
    }

    /// <summary>Ranking score used by recall: strength x importance x recency.</summary>
    internal double Salience(DateTimeOffset now)
    {
        double ageHours = Math.Max(0, (now - LastRecalledAt ?? now - CreatedAt).TotalHours);
        double recency = 1.0 / (1.0 + ageHours / 24.0);
        return Strength * (0.5 + Importance) * (0.6 + 0.4 * recency);
    }

    /// <summary>Human-readable summary of the memory.</summary>
    public override string ToString()
    {
        string strengthPart = IsFlashbulb ? "flashbulb" : Strength.ToString("0.00");
        return $"[{Kind}/{strengthPart}] {Content}";
    }
}
