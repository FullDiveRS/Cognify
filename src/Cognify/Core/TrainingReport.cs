namespace Cognify.Core;

/// <summary>
/// What one training pass added to a mind: lines read, facts learned into
/// the knowledge base, sayings added to its spoken vocabulary and number
/// lines taught to its arithmetic. Returned by <see cref="Mind.Train"/>,
/// <see cref="Mind.TrainAll"/> and <see cref="Mind.TrainFile"/>.
/// When <see cref="Mind.TrainFile"/> was called with a validation split, the
/// validation fields carry how the held-out lines score under the trained
/// transformer — the generalization check a re-train loop can't give you.
/// </summary>
public sealed record TrainingReport
{
    /// <summary>Training lines that were not empty/comment lines.</summary>
    public int Lines { get; internal set; }

    /// <summary>New facts added to the knowledge base.</summary>
    public int FactsLearned { get; internal set; }

    /// <summary>New sayings added to the phrase pool (the mind's vocabulary).</summary>
    public int SayingsAdded { get; internal set; }

    /// <summary>Counting sequences that became / extended the number line.</summary>
    public int SequencesTaught { get; internal set; }

    /// <summary>
    /// The validation fraction requested from <see cref="Mind.TrainFile"/>
    /// (0 when no split was asked for). The train/validate split is
    /// deterministic for a given (file, seed): the same file and seed hold
    /// out the same lines, so runs are comparable.
    /// </summary>
    public double ValidationSplit { get; internal set; }

    /// <summary>Held-out (validation) lines; 0 when no split was used.</summary>
    public int ValidationLines { get; internal set; }

    /// <summary>
    /// Mean cross-entropy of the held-out lines under the transformer, in
    /// nats/token — the SAME metric the deep-training loop minimizes (full
    /// softmax, same windows), computed without touching the weights or the
    /// observation buffer. NaN when the model is untrained (nothing to
    /// score with) or no line held anything scoreable.
    /// </summary>
    public double ValidationLoss { get; internal set; } = double.NaN;

    /// <summary>Validation tokens actually scored; 0 when nothing scored.</summary>
    public int ValidationWords { get; internal set; }

    /// <summary>
    /// Human-readable summary, e.g. "10 line(s): 4 fact(s), 5 saying(s),
    /// 1 number line(s)." — with "…; validation 2 line(s): loss 1.234
    /// nats/token" appended when a split was used and produced a score.
    /// </summary>
    public override string ToString()
    {
        var s = $"{Lines} line(s): {FactsLearned} fact(s), {SayingsAdded} saying(s), {SequencesTaught} number line(s).";
        if (ValidationSplit > 0)
        {
            s += ValidationLines > 0 && !double.IsNaN(ValidationLoss)
                ? $" Validation: {ValidationLines} line(s) held out, loss {ValidationLoss:F3} nats/token over {ValidationWords} token(s)."
                : $" Validation: {ValidationLines} line(s) held out (no score — train the deep model first).";
        }
        return s;
    }
}
