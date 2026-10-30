using Cognify.Emotions;
using Cognify.Generation;
using Cognify.Goals;
using Cognify.Memory;
using Cognify.Tools;

namespace Cognify.Core;

/// <summary>Event args raised after every completed cognitive cycle.</summary>
public sealed class CycleEventArgs : EventArgs
{
    public CycleEventArgs(MindResponse response) => Response = response;

    /// <summary>The completed response.</summary>
    public MindResponse Response { get; }
}

/// <summary>Event args raised when an emotion meaningfully changes.</summary>
public sealed class EmotionChangedEventArgs : EventArgs
{
    public EmotionChangedEventArgs(EmotionType emotion, double intensity, double valence)
    {
        Emotion = emotion;
        Intensity = intensity;
        Valence = valence;
    }

    /// <summary>The emotion that changed.</summary>
    public EmotionType Emotion { get; }

    /// <summary>Its new intensity (0..1).</summary>
    public double Intensity { get; }

    /// <summary>The overall emotional valence after the change (-1..1).</summary>
    public double Valence { get; }
}

/// <summary>Event args raised for every step of the inner monologue.</summary>
public sealed class ThoughtProducedEventArgs : EventArgs
{
    public ThoughtProducedEventArgs(ThoughtStep step) => Step = step;

    /// <summary>The thought step that was produced.</summary>
    public ThoughtStep Step { get; }
}

/// <summary>Event args raised when memories are recalled during a cycle.</summary>
public sealed class MemoryRecalledEventArgs : EventArgs
{
    public MemoryRecalledEventArgs(IReadOnlyList<MemoryRecord> memories) => Memories = memories;

    /// <summary>The recalled memories.</summary>
    public IReadOnlyList<MemoryRecord> Memories { get; }
}

/// <summary>Event args raised when a goal changes (added, progressed, achieved...).</summary>
public sealed class GoalUpdatedEventArgs : EventArgs
{
    public GoalUpdatedEventArgs(Goal goal, string change)
    {
        Goal = goal;
        Change = change;
    }

    /// <summary>The goal that changed.</summary>
    public Goal Goal { get; }

    /// <summary>Short description of what happened ("added", "progress", "achieved"...).</summary>
    public string Change { get; }
}

/// <summary>Event args raised whenever the mind learns something.</summary>
public sealed class LearnedEventArgs : EventArgs
{
    public LearnedEventArgs(double reward, double averageReward, int totalExperiences, string detail)
    {
        Reward = reward;
        AverageReward = averageReward;
        TotalExperiences = totalExperiences;
        Detail = detail;
    }

    /// <summary>The reward that triggered this learning event (-1..1).</summary>
    public double Reward { get; }

    /// <summary>Running average reward across all experiences.</summary>
    public double AverageReward { get; }

    /// <summary>Total experiences stored so far.</summary>
    public int TotalExperiences { get; }

    /// <summary>Human-readable detail about what was learned.</summary>
    public string Detail { get; }
}

/// <summary>Event args raised when a module throws during a cycle.</summary>
public sealed class ModuleErrorEventArgs : EventArgs
{
    public ModuleErrorEventArgs(string moduleName, Exception exception)
    {
        ModuleName = moduleName;
        Exception = exception;
    }

    /// <summary>Name of the failing module.</summary>
    public string ModuleName { get; }

    /// <summary>The thrown exception.</summary>
    public Exception Exception { get; }
}

/// <summary>Event args raised when a tool is invoked by the tool module.</summary>
public sealed class ToolInvokedEventArgs : EventArgs
{
    public ToolInvokedEventArgs(string toolName, IReadOnlyDictionary<string, string> arguments, ToolResult result)
    {
        ToolName = toolName;
        Arguments = arguments;
        Result = result;
    }

    /// <summary>Name of the invoked tool.</summary>
    public string ToolName { get; }

    /// <summary>Arguments passed to the tool.</summary>
    public IReadOnlyDictionary<string, string> Arguments { get; }

    /// <summary>What the tool returned.</summary>
    public ToolResult Result { get; }
}

/// <summary>Event args raised when a media artifact is generated.</summary>
public sealed class MediaGeneratedEventArgs : EventArgs
{
    public MediaGeneratedEventArgs(MediaArtifact artifact) => Artifact = artifact;

    /// <summary>The generated artifact.</summary>
    public Generation.MediaArtifact Artifact { get; }
}
