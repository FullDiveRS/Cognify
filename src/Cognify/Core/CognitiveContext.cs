using Cognify.Emotions;
using Cognify.Memory;
using Cognify.Reasoning;
using Cognify.Thinking;

namespace Cognify.Core;

/// <summary>
/// Shared workspace for one cognitive cycle. Modules read what earlier modules
/// produced (memories, emotions, inferences...) and contribute their own results.
/// </summary>
public sealed class CognitiveContext
{
    internal CognitiveContext(Mind mind, int cycle, Stimulus stimulus)
    {
        Mind = mind;
        Cycle = cycle;
        Stimulus = stimulus;
        EffectiveContent = stimulus.Content;
    }

    /// <summary>The mind running this cycle.</summary>
    public Mind Mind { get; }

    /// <summary>1-based cycle number.</summary>
    public int Cycle { get; }

    /// <summary>The stimulus being processed.</summary>
    public Stimulus Stimulus { get; }

    /// <summary>
    /// What deeper modules should reason over: the stimulus content itself,
    /// or an effective version substituted by a perception module — e.g. the
    /// transcription of an audio clip, or a translation of foreign text into
    /// the mind's preferred language. Memory still records the raw original.
    /// </summary>
    public string EffectiveContent { get; internal set; }

    /// <summary>Memories recalled this cycle (set by the memory module).</summary>
    public IReadOnlyList<MemoryRecord> Recalled { get; internal set; } = Array.Empty<MemoryRecord>();

    /// <summary>Emotional state for this cycle (set by the emotion module).</summary>
    public EmotionalState? Emotions { get; internal set; }

    /// <summary>Reasoning result for this cycle (set by the reasoning module).</summary>
    public ReasoningResult? Reasoning { get; internal set; }

    /// <summary>Chain of thought being built (set by the thinking module).</summary>
    public ChainOfThought? Chain { get; internal set; }

    /// <summary>The focus goal for this cycle (set by the goal module).</summary>
    public Goals.Goal? FocusGoal { get; internal set; }

    /// <summary>A suggested action coming from the goal planner (may be empty).</summary>
    public string SuggestedAction { get; internal set; } = string.Empty;

    /// <summary>The response draft. Null until a module produces a response.
    /// Custom modules may set this to make the mind respond their way.</summary>
    public ResponseDraft? Response { get; set; }

    /// <summary>
    /// The features extracted for this stimulus (set by the learning module) —
    /// used for neural prediction and credit assignment.
    /// </summary>
    public double[]? Features { get; internal set; }

    /// <summary>
    /// A free-form bag that custom modules can use to exchange data during a cycle.
    /// </summary>
    public Dictionary<string, object> Bag { get; } = new();
}
