using Cognify.Emotions;
using Cognify.Memory;
using Cognify.Reasoning;
using Cognify.Thinking;

namespace Cognify.Core;

/// <summary>
/// A draft response being assembled while cognitive modules run.
/// The first module to set <see cref="Content"/> wins, unless another module
/// replaces it explicitly.
/// </summary>
public sealed class ResponseDraft
{
    public ResponseDraft(string content, double confidence, int styleId, string styleName)
    {
        Content = content;
        Confidence = Math.Clamp(confidence, 0, 1);
        StyleId = styleId;
        StyleName = styleName;
    }

    /// <summary>The response text.</summary>
    public string Content { get; set; }

    /// <summary>Confidence in the response (0..1).</summary>
    public double Confidence { get; set; }

    /// <summary>Identifier of the response style that produced this draft (used by learning).</summary>
    public int StyleId { get; }

    /// <summary>Human-readable style name.</summary>
    public string StyleName { get; }
}

/// <summary>
/// The complete, immutable result of one cognitive cycle: what the mind said,
/// how confident it was, what it thought, felt and recalled.
/// </summary>
public sealed class MindResponse
{
    internal MindResponse(
        int cycle, string content, double confidence, int styleId, string styleName,
        ChainOfThought? chain, ReasoningResult? reasoning,
        EmotionalState? emotions, IReadOnlyList<MemoryRecord> recalled,
        TimeSpan elapsed, DateTimeOffset at)
    {
        CycleNumber = cycle;
        Content = content;
        Confidence = confidence;
        StyleId = styleId;
        StyleName = styleName;
        Chain = chain;
        Reasoning = reasoning;
        Emotions = emotions;
        RecalledMemories = recalled;
        ProcessingTime = elapsed;
        Timestamp = at;
    }

    /// <summary>1-based number of the cognitive cycle that produced this response.</summary>
    public int CycleNumber { get; }

    /// <summary>The final response text.</summary>
    public string Content { get; }

    /// <summary>Confidence in the response (0..1).</summary>
    public double Confidence { get; }

    /// <summary>Identifier of the response style used (useful for reward attribution).</summary>
    public int StyleId { get; }

    /// <summary>Human-readable name of the response style.</summary>
    public string StyleName { get; }

    /// <summary>The explainable chain of thought, if the thinking module is enabled.</summary>
    public ChainOfThought? Chain { get; }

    /// <summary>Reasoning trace (inferred facts, explanations, direct answers), if enabled.</summary>
    public ReasoningResult? Reasoning { get; }

    /// <summary>Emotional state after processing the stimulus, if the emotion module is enabled.</summary>
    public EmotionalState? Emotions { get; }

    /// <summary>Memories that were recalled during the cycle.</summary>
    public IReadOnlyList<MemoryRecord> RecalledMemories { get; }

    /// <summary>How long the cognitive cycle took.</summary>
    public TimeSpan ProcessingTime { get; }

    /// <summary>When the response was produced.</summary>
    public DateTimeOffset Timestamp { get; }

    /// <summary>Returns just the response text — convenient for chat-style usage.</summary>
    public override string ToString() => Content;
}
