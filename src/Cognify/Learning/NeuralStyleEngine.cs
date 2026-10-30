using System.Text.Json;

namespace Cognify.Learning;

/// <summary>
/// The neural half of Cognify's hybrid decision engine: a real feed-forward
/// network (backpropagation + gradient descent) that learns to predict how
/// rewarding a given response STYLE will be in the CURRENT conversational
/// context — text features, which style is on the table, and the mood the
/// mind is in, all feed the prediction.
///
/// Input vector layout:
///   [0 .. FeatureDim)                  hashed text features of the stimulus
///   [FeatureDim .. FeatureDim+Styles)  one-hot style id
///   [last two)                         mood: valence (-1..1), arousal (0..1)
///
/// The symbolic half of the hybrid lives in <see cref="SelfLearningModule"/>
/// (Q-table evidence per style + the reward net's context prediction). The
/// two halves are blended adaptively: the more accurate the network's
/// predictions become, the more the mind trusts the neural side — see
/// <see cref="SelfLearningModule.NeuralWeight"/>.
/// </summary>
public sealed class NeuralStyleEngine
{
    private readonly MicroNet _net;

    /// <summary>Creates the style-conditioned network.</summary>
    public NeuralStyleEngine(int featureDim, int styleCount, int hiddenUnits, int seed = 42)
    {
        if (featureDim < 1) throw new ArgumentOutOfRangeException(nameof(featureDim));
        if (styleCount < 1) throw new ArgumentOutOfRangeException(nameof(styleCount));
        FeatureDim = featureDim;
        StyleCount = styleCount;
        _net = new MicroNet(featureDim + styleCount + 2, Math.Max(4, hiddenUnits), 1, seed);
    }

    private NeuralStyleEngine(MicroNet net, int featureDim, int styleCount)
    {
        _net = net;
        FeatureDim = featureDim;
        StyleCount = styleCount;
    }

    /// <summary>Dimension of the hashed text-feature block.</summary>
    public int FeatureDim { get; }

    /// <summary>How many style slots the one-hot block has.</summary>
    public int StyleCount { get; }

    /// <summary>Total input dimension (features + style one-hot + 2 mood inputs).</summary>
    public int InputDim => FeatureDim + StyleCount + 2;

    /// <summary>
    /// Builds the network input: text features, a one-hot style block, and
    /// the current mood (valence, arousal).
    /// </summary>
    public double[] ComposeInput(double[] features, int styleId, double valence, double arousal)
    {
        if (features.Length != FeatureDim)
            throw new ArgumentException($"Expected {FeatureDim} features, got {features.Length}.", nameof(features));
        var input = new double[InputDim];
        Array.Copy(features, input, FeatureDim);
        int styleSlot = FeatureDim + Math.Clamp(styleId, 0, StyleCount - 1);
        input[styleSlot] = 1.0;
        input[^2] = Math.Clamp(valence, -1, 1);
        input[^1] = Math.Clamp(arousal, 0, 1);
        return input;
    }

    /// <summary>
    /// Predicts the reward (0..1) of choosing <paramref name="styleId"/> in
    /// this context with this mood. Untrained, the network answers ~0.5 for
    /// everything — the symbolic side carries the decision until it learns.
    /// </summary>
    public double Predict(double[] features, int styleId, double valence, double arousal) =>
        _net.Predict(ComposeInput(features, styleId, valence, arousal))[0];

    /// <summary>
    /// One gradient-descent step toward the observed reward. Returns the
    /// squared error before the update — the signal the adaptive blend uses
    /// to decide how much to trust the neural side.
    /// </summary>
    public double Learn(double[] features, int styleId, double valence, double arousal,
        double reward01, double learningRate)
    {
        var input = ComposeInput(features, styleId, valence, arousal);
        return _net.Train(input, new[] { Math.Clamp(reward01, 0, 1) }, learningRate);
    }

    /// <summary>Number of trainable parameters (for stats display).</summary>
    public int ParameterCount =>
        (InputDim * _net.HiddenDim) + _net.HiddenDim + (_net.HiddenDim * 1) + 1;

    /// <summary>Serializes the network for persistence.</summary>
    public string Serialize() => JsonSerializer.Serialize(new Dto
    {
        FeatureDim = FeatureDim,
        StyleCount = StyleCount,
        Net = _net.Serialize()
    });

    /// <summary>Rebuilds the engine from serialized JSON.</summary>
    public static NeuralStyleEngine Deserialize(string json)
    {
        try
        {
            var dto = JsonSerializer.Deserialize<Dto>(json);
            if (dto?.Net is null)
                throw new Cognify.Core.PersistenceException("Style network payload is incomplete.");
            var net = MicroNet.Deserialize(dto.Net);
            int featureDim = dto.FeatureDim.GetValueOrDefault();
            int styleCount = dto.StyleCount.GetValueOrDefault();
            if (featureDim < 1 || styleCount < 1)
                throw new Cognify.Core.PersistenceException("Style network payload is incomplete.");
            if (net.InputDim != featureDim + styleCount + 2)
                throw new Cognify.Core.PersistenceException("Style network shape does not match its header.");
            return new NeuralStyleEngine(net, featureDim, styleCount);
        }
        catch (JsonException ex)
        {
            throw new Cognify.Core.PersistenceException("Failed to deserialize the style network.", ex);
        }
    }

    private sealed class Dto
    {
        public int? FeatureDim { get; set; }
        public int? StyleCount { get; set; }
        public string? Net { get; set; }
    }
}
