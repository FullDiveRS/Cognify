namespace Cognify.Core;

/// <summary>
/// What one training pass added to a mind: lines read, facts learned into
/// the knowledge base, sayings added to its spoken vocabulary and number
/// lines taught to its arithmetic. Returned by <see cref="Mind.Train"/>,
/// <see cref="Mind.TrainAll"/> and <see cref="Mind.TrainFile"/>.
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

    /// <summary>Human-readable summary, e.g. "10 line(s): 4 fact(s), 5 saying(s), 1 number line(s)."</summary>
    public override string ToString() =>
        $"{Lines} line(s): {FactsLearned} fact(s), {SayingsAdded} saying(s), {SequencesTaught} number line(s).";
}
