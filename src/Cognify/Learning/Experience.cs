namespace Cognify.Learning;

/// <summary>A stored learning experience: features -> chosen action -> reward.</summary>
public sealed class Experience
{
    internal Experience(double[] features, int actionId, double reward, DateTimeOffset at)
    {
        Features = features;
        ActionId = actionId;
        Reward = reward;
        At = at;
    }

    /// <summary>Encoded stimulus features.</summary>
    public double[] Features { get; }

    /// <summary>Response style that was chosen.</summary>
    public int ActionId { get; }

    /// <summary>Reward received for that choice (-1..1).</summary>
    public double Reward { get; }

    /// <summary>When the experience happened.</summary>
    public DateTimeOffset At { get; }
}

/// <summary>Running statistics of the learning system.</summary>
public sealed class LearningStats
{
    /// <summary>Total reinforcement events.</summary>
    public int TotalExperiences { get; internal set; }

    /// <summary>Number of gradient-descent training steps performed.</summary>
    public int TrainingSteps { get; internal set; }

    /// <summary>Exponential moving average of reward (-1..1).</summary>
    public double AverageReward { get; internal set; }

    /// <summary>Last training loss (mean squared error).</summary>
    public double LastLoss { get; internal set; }

    /// <summary>Current exploration rate (epsilon).</summary>
    public double ExplorationRate { get; internal set; }
}
