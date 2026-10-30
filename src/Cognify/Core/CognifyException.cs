namespace Cognify.Core;

/// <summary>Base exception type thrown by the Cognify library.</summary>
public class CognifyException : Exception
{
    public CognifyException(string message) : base(message) { }
    public CognifyException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>Thrown when a mind is built or used in an invalid way.</summary>
public sealed class MindConfigurationException : CognifyException
{
    public MindConfigurationException(string message) : base(message) { }
}

/// <summary>Thrown when persisting or restoring a mind fails.</summary>
public sealed class PersistenceException : CognifyException
{
    public PersistenceException(string message) : base(message) { }
    public PersistenceException(string message, Exception inner) : base(message, inner) { }
}
