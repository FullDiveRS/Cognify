namespace Cognify.Core;

/// <summary>The phase of cognition a thought step belongs to.</summary>
public enum ThoughtPhase
{
    /// <summary>Initial perception of the stimulus.</summary>
    Observe,

    /// <summary>Retrieving relevant memories.</summary>
    Recall,

    /// <summary>Emotional appraisal.</summary>
    Feel,

    /// <summary>Forming hypotheses from reasoning / knowledge.</summary>
    Hypothesize,

    /// <summary>Weighing goals and drives.</summary>
    Evaluate,

    /// <summary>Generated self-talk produced by the thought generator BEFORE
    /// the reply is composed (thinking levels Low..Max control how much).</summary>
    Generate,

    /// <summary>Choosing between candidate responses.</summary>
    Deliberate,

    /// <summary>Forming the final conclusion / response.</summary>
    Conclude
}

/// <summary>A single recorded step inside a chain of thought. Fully explainable.</summary>
public sealed record ThoughtStep(int Index, ThoughtPhase Phase, string Text, double Confidence);

/// <summary>
/// An explainable chain of thought produced by the <c>ThinkingModule</c>.
/// Every step the mind "thought" is recorded here and can be inspected or logged.
/// </summary>
public sealed class ChainOfThought
{
    private readonly List<ThoughtStep> _steps = new();

    /// <summary>The recorded steps, in order.</summary>
    public IReadOnlyList<ThoughtStep> Steps => _steps;

    /// <summary>The final conclusion text.</summary>
    public string Conclusion { get; internal set; } = string.Empty;

    /// <summary>Confidence in the conclusion (0..1).</summary>
    public double Confidence { get; internal set; }

    /// <summary>Adds a step to the chain. Intended for module internals.</summary>
    internal void Add(ThoughtPhase phase, string text, double confidence = 1.0)
    {
        _steps.Add(new ThoughtStep(_steps.Count + 1, phase, text, Math.Clamp(confidence, 0, 1)));
    }

    /// <summary>Renders the chain as an indented, human-readable trace.</summary>
    public string Render()
    {
        var lines = _steps.Select(s => $"{s.Index,2}. [{s.Phase}] {s.Text} ({s.Confidence:0.00})");
        return string.Join(Environment.NewLine, lines) +
               Environment.NewLine + $"   => {Conclusion} (confidence {Confidence:0.00})";
    }
}
