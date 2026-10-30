namespace Cognify.Core;

/// <summary>
/// Contract for a pluggable cognitive module. Implement this (or derive from
/// <see cref="CognitiveModuleBase"/>) and register it with
/// <c>MindBuilder.WithModule(...)</c> to extend the mind with new capabilities.
/// </summary>
public interface ICognitiveModule
{
    /// <summary>Unique module name (e.g. "reasoning", "emotions").</summary>
    string Name { get; }

    /// <summary>
    /// Execution order inside the cognitive cycle. Lower runs earlier.
    /// Recommended bands: perception 10-19, affect 20-29, cognition 30-59, action 60+.
    /// </summary>
    int Order { get; }

    /// <summary>Called once when the module is attached to a mind.</summary>
    void Attach(Mind mind);

    /// <summary>Called for every cognitive cycle, in <see cref="Order"/> sequence.</summary>
    void Process(CognitiveContext context);

    /// <summary>
    /// Background housekeeping, independent of any stimulus (memory decay,
    /// emotional decay, drive growth...). Call <see cref="Mind.Tick"/> periodically.
    /// </summary>
    void Tick();
}

/// <summary>Convenience base class for cognitive modules.</summary>
public abstract class CognitiveModuleBase : ICognitiveModule
{
    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract int Order { get; }

    /// <summary>The mind this module is attached to (set after <see cref="Attach"/>).</summary>
    protected Mind? Host { get; private set; }

    /// <inheritdoc />
    public virtual void Attach(Mind mind) => Host = mind;

    /// <inheritdoc />
    public abstract void Process(CognitiveContext context);

    /// <inheritdoc />
    public virtual void Tick() { }
}
