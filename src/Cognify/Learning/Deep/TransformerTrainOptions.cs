namespace Cognify.Learning;

/// <summary>
/// How one deep-training run should go — the knobs of the training loop:
/// how many epochs, the learning rate (with per-epoch decay), when to stop
/// early, and who to tell about each epoch's loss.
/// <code>
/// mind.TrainDeepModel(new TransformerTrainOptions {
///     Epochs = 40, Decay = 0.95, Patience = 4, MinDelta = 1e-4,
///     Progress = (epoch, total, loss) => Console.WriteLine($"epoch {epoch}/{total}: {loss:F3}"),
/// });
/// </code>
/// </summary>
public sealed class TransformerTrainOptions
{
    /// <summary>Epochs planned for this run (at least 1).</summary>
    public int Epochs { get; init; } = 12;

    /// <summary>Learning rate for THIS run; null keeps the model config's rate.</summary>
    public double? LearningRate { get; init; }

    /// <summary>The learning rate is multiplied by this after every epoch (1.0 = constant).</summary>
    public double Decay { get; init; } = 1.0;

    /// <summary>An epoch counts as progress when the loss improves by at least this much (nats/token).</summary>
    public double MinDelta { get; init; } = 0.0;

    /// <summary>Consecutive epochs without progress before the run stops early (0 = never stop early).</summary>
    public int Patience { get; init; } = 0;

    /// <summary>Fired after each epoch: (epoch, epochs planned, loss).</summary>
    public Action<int, int, double>? Progress { get; init; }

    /// <summary>A run of the given length with everything else default.</summary>
    public static TransformerTrainOptions For(int epochs) => new() { Epochs = Math.Max(1, epochs) };
}
