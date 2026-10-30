namespace Cognify.Goals;

/// <summary>
/// An internal drive (need) that grows over time and can spawn goals when it
/// becomes pressing — this is what gives the mind intrinsic motivation.
/// </summary>
public sealed class Drive
{
    internal Drive(string name, double growth)
    {
        Name = name;
        Growth = growth;
        Level = 0.15;
    }

    /// <summary>Drive name (e.g. "curiosity").</summary>
    public string Name { get; }

    /// <summary>How much the drive grows per tick (0..1).</summary>
    public double Growth { get; internal set; }

    /// <summary>Current drive level (0..1). Higher = more pressing need.</summary>
    public double Level { get; internal set; }

    /// <summary>Advances the drive by one tick.</summary>
    internal void Grow()
    {
        Level = Math.Min(1.0, Level + Growth);
    }

    /// <summary>Satisfies part of the drive (e.g. after achieving a related goal).</summary>
    public void Satisfy(double amount = 0.4)
    {
        Level = Math.Max(0, Level - amount);
    }

    /// <summary>Sets the drive level directly (0..1), e.g. to restore persisted state.</summary>
    public void SetLevel(double level)
    {
        Level = Math.Clamp(level, 0, 1);
    }
}
