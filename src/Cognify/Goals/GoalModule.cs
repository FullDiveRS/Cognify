using Cognify.Core;

namespace Cognify.Goals;

/// <summary>
/// The goal module: gives the mind motivation. It maintains internal drives
/// (curiosity, social belonging, achievement, calm) that grow over time and
/// autonomously spawn goals when they become pressing. Each cycle it selects
/// a focus goal and a suggested action from a simple planner.
/// </summary>
public sealed class GoalModule : CognitiveModuleBase
{
    private readonly GoalOptions _options;
    private readonly object _gate = new();
    private readonly List<Goal> _goals = new();
    private readonly Dictionary<string, Drive> _drives;

    internal GoalModule(GoalOptions options)
    {
        _options = options;
        _drives = new Dictionary<string, Drive>
        {
            ["curiosity"] = new Drive("curiosity", options.DriveGrowth),
            ["social"] = new Drive("social", options.DriveGrowth * 0.7),
            ["achievement"] = new Drive("achievement", options.DriveGrowth * 0.85),
            ["calm"] = new Drive("calm", options.DriveGrowth * 0.5)
        };
        foreach (var spec in options.InitialGoals)
            _goals.Add(new Goal(spec.Name, spec.Description, spec.Priority, spec.Tags, spec.Target));
    }

    /// <inheritdoc />
    public override string Name => "goals";

    /// <inheritdoc />
    public override int Order => 40;

    /// <summary>All drives, by name.</summary>
    public IReadOnlyDictionary<string, Drive> Drives { get { lock (_gate) return _drives; } }

    /// <summary>Adds a goal at runtime and returns it.</summary>
    public Goal AddGoal(string name, string description = "", double priority = 0.5,
                        IEnumerable<string>? tags = null, double target = 1.0,
                        DateTimeOffset? deadline = null)
    {
        var goal = new Goal(name, description, priority, tags ?? Array.Empty<string>(), target)
        {
            Deadline = deadline
        };
        lock (_gate) _goals.Add(goal);
        Host?.RaiseGoalUpdated(goal, "added");
        return goal;
    }

    /// <summary>Reports progress on a goal; achieves it when the target is reached.</summary>
    public void ReportProgress(Goal goal, double amount)
    {
        if (goal.Status != GoalStatus.Active) return;
        goal.Progress += amount;
        if (goal.Completion >= 1.0)
        {
            goal.Status = GoalStatus.Achieved;
            foreach (var tag in goal.Tags.Where(_drives.ContainsKey))
                _drives[tag].Satisfy(0.45);
            Host?.RaiseGoalUpdated(goal, "achieved");
        }
        else
        {
            Host?.RaiseGoalUpdated(goal, "progress");
        }
    }

    /// <summary>Abandons a goal (removes it from focus consideration).</summary>
    public void Abandon(Goal goal)
    {
        goal.Status = GoalStatus.Abandoned;
        Host?.RaiseGoalUpdated(goal, "abandoned");
    }

    /// <summary>Returns active goals sorted by current urgency.</summary>
    public IReadOnlyList<Goal> ActiveGoals()
    {
        var now = DateTimeOffset.UtcNow;
        lock (_gate) return _goals.Where(g => g.Status == GoalStatus.Active)
            .OrderByDescending(g => g.Urgency(now)).ToList();
    }

    /// <summary>All goals including achieved/abandoned.</summary>
    public IReadOnlyList<Goal> AllGoals()
    {
        lock (_gate) return _goals.ToList();
    }

    /// <summary>The currently focused goal (null when nothing is active).</summary>
    public Goal? FocusGoal { get; private set; }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var now = DateTimeOffset.UtcNow;
        lock (_gate) FocusGoal = _goals
            .Where(g => g.Status == GoalStatus.Active)
            .OrderByDescending(g => g.Urgency(now))
            .FirstOrDefault();

        context.FocusGoal = FocusGoal;
        context.SuggestedAction = FocusGoal is null ? string.Empty : SuggestAction(FocusGoal);

        // Goal-relevant stimuli give progress: if the stimulus mentions a tag,
        // the mind feels it is working toward that goal.
        if (FocusGoal is { } goal)
        {
            var tokens = Emotions.MoodEngine.Tokenize(context.EffectiveContent);
            if (goal.Tags.Any(tag => tokens.Contains(tag)))
                ReportProgress(goal, 0.05);
        }
    }

    /// <summary>
    /// Background tick: drives grow; pressing drives autonomously spawn goals
    /// (intrinsic motivation) unless a matching goal already exists.
    /// </summary>
    public override void Tick()
    {
        lock (_gate)
        {
            foreach (var drive in _drives.Values) drive.Grow();

            foreach (var drive in _drives.Values.Where(d => d.Level >= _options.DriveSpawnThreshold))
            {
                if (_goals.Any(g => g.Status == GoalStatus.Active && g.Tags.Contains(drive.Name))) continue;
                var goal = new Goal(
                    $"satisfy-{drive.Name}",
                    $"Autonomously spawned to satisfy the {drive.Name} drive",
                    priority: Math.Min(0.95, 0.5 + drive.Level * 0.4),
                    new[] { drive.Name }, target: 1.0);
                _goals.Add(goal);
                Host?.RaiseGoalUpdated(goal, "auto-spawned");
                drive.Satisfy(0.15); // a bit of relief from committing to act
            }
        }
    }

    private string SuggestAction(Goal goal)
    {
        if (goal.Tags.Contains("curiosity"))
            return "explore something new and connect it to what I already know";
        if (goal.Tags.Contains("social"))
            return "engage warmly and ask about the other's perspective";
        if (goal.Tags.Contains("achievement"))
            return "focus on the task and deliver something concrete";
        if (goal.Tags.Contains("calm"))
            return "keep responses measured and reassuring";
        return "make progress on it with whatever the next turn offers";
    }
}
