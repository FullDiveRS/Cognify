namespace Cognify.Goals;

/// <summary>Lifecycle status of a goal.</summary>
public enum GoalStatus
{
    /// <summary>The goal is being pursued.</summary>
    Active,

    /// <summary>The goal was reached.</summary>
    Achieved,

    /// <summary>The goal was abandoned.</summary>
    Abandoned
}

/// <summary>
/// Something the mind wants. Goals have priority, progress toward a target,
/// optional deadline, and tags that tie them to drives (curiosity / social /
/// achievement / calm).
/// </summary>
public sealed class Goal
{
    internal Goal(string name, string description, double priority, IEnumerable<string> tags, double target)
    {
        Id = Guid.NewGuid();
        Name = name;
        Description = description;
        Priority = Math.Clamp(priority, 0, 1);
        Tags = tags.ToList();
        Target = Math.Max(0.01, target);
        Status = GoalStatus.Active;
        CreatedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>Unique goal id.</summary>
    public Guid Id { get; }

    /// <summary>Goal name.</summary>
    public string Name { get; }

    /// <summary>Human-readable description.</summary>
    public string Description { get; }

    /// <summary>Priority 0..1 — higher priority goals win the focus slot.</summary>
    public double Priority { get; internal set; }

    /// <summary>Current progress (0..Target).</summary>
    public double Progress { get; internal set; }

    /// <summary>Progress needed for the goal to be achieved.</summary>
    public double Target { get; }

    /// <summary>Tags linking the goal to drives and planning strategies.</summary>
    public IReadOnlyList<string> Tags { get; }

    /// <summary>Current lifecycle status.</summary>
    public GoalStatus Status { get; internal set; }

    /// <summary>When the goal was created.</summary>
    public DateTimeOffset CreatedAt { get; }

    /// <summary>Optional deadline; overdue goals get an urgency boost.</summary>
    public DateTimeOffset? Deadline { get; internal set; }

    /// <summary>Progress as a fraction of the target (0..1+).</summary>
    public double Completion => Progress / Target;

    /// <summary>
    /// Combined importance used to pick the current focus goal:
    /// priority x urgency (deadline pressure) x incompleteness.
    /// </summary>
    public double Urgency(DateTimeOffset now)
    {
        if (Status != GoalStatus.Active) return 0;
        double incompleteness = Math.Max(0.05, 1.0 - Completion);
        double deadlinePressure = 1.0;
        if (Deadline is { } d)
        {
            double remaining = (d - now).TotalDays;
            deadlinePressure = remaining <= 0 ? 1.5 : 1.0 + Math.Max(0, 0.5 - remaining / 30.0);
        }
        return Priority * incompleteness * deadlinePressure;
    }

    /// <summary>Human-readable description of the goal.</summary>
    public override string ToString() =>
        $"{Name} ({Status}, {Completion:P0} done, priority {Priority:0.00})";

    /// <summary>Restores persisted state (used by the persistence layer).</summary>
    internal void Restore(double progress, GoalStatus status)
    {
        Progress = Math.Max(0, progress);
        Status = status;
    }
}
