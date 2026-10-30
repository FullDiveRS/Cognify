using System.Text.Json;

namespace Cognify.Learning;

/// <summary>Hidden-layer activation functions for <see cref="DeepNet"/>.</summary>
public enum DeepActivation
{
    /// <summary>Rectified linear unit — the classic deep-learning default.</summary>
    ReLU,
    /// <summary>Gaussian Error Linear Unit (tanh approximation) — GPT/BERT's choice.</summary>
    GELU,
    /// <summary>Hyperbolic tangent — squashes to (-1, 1), what <see cref="MicroNet"/> uses.</summary>
    Tanh,
}

/// <summary>Output-layer activation for <see cref="DeepNet"/>.</summary>
public enum DeepOutput
{
    /// <summary>Logistic sigmoid per unit, for multi-label / binary targets in (0, 1).</summary>
    Sigmoid,
    /// <summary>Softmax over all outputs, trained with cross-entropy — classification.</summary>
    Softmax,
    /// <summary>No activation, plain linear regression output.</summary>
    Linear,
}

/// <summary>
/// A DEEP feed-forward neural network — the library's general deep-learning
/// workhorse. Where <see cref="MicroNet"/> is one hidden layer deep, a
/// <see cref="DeepNet"/> stacks an arbitrary number of hidden layers with
/// ReLU/GELU/Tanh activations, He initialization, inverted dropout, L2
/// regularization and full backpropagation trained by the
/// <see cref="AdamOptimizer"/>. This is the classic "deep" in deep learning:
/// depth lets the network compose simple features into more abstract ones,
/// which is exactly what the mind needs when stimulus features get richer.
///
/// Used for classification (softmax + cross-entropy) and regression
/// (sigmoid/linear + MSE) alike; small enough to train inside a cognitive
/// cycle on a CPU, real enough to genuinely learn (XOR, sentiment, intent).
/// </summary>
public sealed class DeepNet
{
    private readonly int _input;
    private readonly int _output;
    private readonly DeepActivation _hiddenAct;
    private readonly DeepOutput _outputAct;
    private readonly double _dropout;      // inverted dropout probability, 0 = off
    private readonly double _l2;           // weight decay
    private readonly double[][][] _weights; // per layer: [units][fanIn]
    private readonly double[][] _biases;    // per layer: [units]
    private readonly AdamOptimizer _adam = new();
    private readonly Random _rng;
    private readonly object _gate = new();

    private DeepNet(double[][][] weights, double[][] biases, int input, int output,
        DeepActivation hiddenAct, DeepOutput outputAct, double dropout, double l2, int seed)
    {
        _weights = weights;
        _biases = biases;
        _input = input;
        _output = output;
        _hiddenAct = hiddenAct;
        _outputAct = outputAct;
        _dropout = dropout;
        _l2 = l2;
        _rng = new Random(seed);
    }

    /// <summary>
    /// Creates the network. <paramref name="hiddenLayers"/> lists the width of
    /// each hidden layer in order (may be empty for a plain linear model).
    /// </summary>
    public DeepNet(int input, int[] hiddenLayers, int output,
        DeepActivation hiddenActivation = DeepActivation.GELU,
        DeepOutput outputActivation = DeepOutput.Sigmoid,
        double dropout = 0.0, double l2 = 0.0, int seed = 42)
    {
        if (input < 1 || output < 1) throw new ArgumentException("input and output must be >= 1");
        if (hiddenLayers.Any(h => h < 1)) throw new ArgumentException("hidden layer widths must be >= 1");

        _input = input;
        _output = output;
        _hiddenAct = hiddenActivation;
        _outputAct = outputActivation;
        _dropout = Math.Clamp(dropout, 0.0, 0.9);
        _l2 = Math.Max(0.0, l2);
        _rng = new Random(seed);

        int fanIn = input;
        var units = new List<int>(hiddenLayers) { output };
        _weights = new double[units.Count][][];
        _biases = new double[units.Count][];
        for (int l = 0; l < units.Count; l++)
        {
            int width = units[l];
            _weights[l] = new double[width][];
            _biases[l] = new double[width];
            bool isOutput = l == units.Count - 1;
            double std = isOutput && _outputAct == DeepOutput.Softmax
                ? 0.01                                  // small final layer for stable softmax starts
                : HeStd(fanIn);
            for (int u = 0; u < width; u++)
            {
                _weights[l][u] = new double[fanIn];
                for (int i = 0; i < fanIn; i++)
                    _weights[l][u][i] = Gaussian(std);
            }
            fanIn = width;
        }
    }

    // He initialization (He et al. 2015): std = sqrt(2 / fanIn) — keeps
    // activation variance stable through deep stacks of ReLU/GELU layers.
    private static double HeStd(int fanIn) => Math.Sqrt(2.0 / Math.Max(1, fanIn));

    private double Gaussian(double std)
    {
        // Box-Muller
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * std;
    }

    /// <summary>Input dimension.</summary>
    public int InputDim { get { lock (_gate) return _input; } }

    /// <summary>Output dimension.</summary>
    public int OutputDim { get { lock (_gate) return _output; } }

    /// <summary>Number of weight matrices (hidden layers + output layer).</summary>
    public int Depth { get { lock (_gate) return _weights.Length; } }

    /// <summary>Widths of every layer, hidden layers first, output layer last.</summary>
    public int[] Units { get { lock (_gate) return _weights.Select(w => w.Length).ToArray(); } }

    /// <summary>Total trainable parameters.</summary>
    public int ParameterCount { get { lock (_gate) return _weights.Sum(w => w.Sum(r => r.Length)) + _biases.Sum(b => b.Length); } }

    /// <summary>How many full training passes the network has completed.</summary>
    public int EpochsTrained { get; private set; }

    /// <summary>How many training samples the network has seen.</summary>
    public long SamplesSeen { get; private set; }

    // ------------------------------------------------------------------
    // Activations
    // ------------------------------------------------------------------

    private static double Activate(DeepActivation act, double x) => act switch
    {
        DeepActivation.ReLU => x > 0 ? x : 0,
        DeepActivation.GELU => Gelu(x),
        _ => Math.Tanh(x),
    };

    private static double ActivateGrad(DeepActivation act, double z, double a) => act switch
    {
        DeepActivation.ReLU => z > 0 ? 1 : 0,
        DeepActivation.GELU => GeluGrad(z),
        _ => 1 - a * a, // tanh'
    };

    /// <summary>GELU (tanh approximation), the transformer-era activation.</summary>
    public static double Gelu(double x)
    {
        const double c = 0.7978845608028654; // sqrt(2/pi)
        double inner = c * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + Math.Tanh(inner));
    }

    /// <summary>Derivative of the GELU tanh-approximation.</summary>
    public static double GeluGrad(double x)
    {
        const double c = 0.7978845608028654;
        double x3 = x * x * x;
        double inner = c * (x + 0.044715 * x3);
        double t = Math.Tanh(inner);
        double dt = (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
        return 0.5 * (1 + t) + 0.5 * x * dt;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -20, 20)));

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /// <summary>Forward pass for one sample. Returns the output activations.</summary>
    public double[] Predict(double[] input)
    {
        if (input.Length != _input)
            throw new ArgumentException($"Expected {_input} features, got {input.Length}.", nameof(input));
        lock (_gate)
            return Forward(input, training: false, scratchMask: null).Output;
    }

    /// <summary>Forward pass for a batch. Returns one output row per input row.</summary>
    public double[][] PredictBatch(double[][] inputs)
    {
        lock (_gate)
        {
            var result = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
                result[i] = Forward(inputs[i], training: false, scratchMask: null).Output;
            return result;
        }
    }

    private double[]?[] NewMask() => new double[_weights.Length][];

    // Forward pass keeping per-layer pre-activations and activations.
    // When training and dropout > 0, scratchMask receives one mask row per
    // hidden layer (scaled by 1/(1-p)) so backward can reuse it.
    private (double[] Output, double[][] Activations, double[][] PreActs, double[]?[] Masks) Forward(
        double[] input, bool training, double[]?[]? scratchMask)
    {
        var activations = new double[_weights.Length][];
        var preActs = new double[_weights.Length][];
        var masks = new double[_weights.Length][];

        double[] a = input;
        for (int l = 0; l < _weights.Length; l++)
        {
            bool isOutput = l == _weights.Length - 1;
            int width = _weights[l].Length;
            var z = new double[width];
            var outA = new double[width];
            for (int u = 0; u < width; u++)
            {
                double sum = _biases[l][u];
                var row = _weights[l][u];
                for (int i = 0; i < a.Length; i++) sum += row[i] * a[i];
                z[u] = sum;
                outA[u] = isOutput
                    ? _outputAct switch
                    {
                        DeepOutput.Sigmoid => Sigmoid(sum),
                        DeepOutput.Softmax => sum,          // normalized after the loop
                        _ => sum,                            // Linear
                    }
                    : Activate(_hiddenAct, sum);
            }

            if (isOutput && _outputAct == DeepOutput.Softmax)
            {
                double max = outA.Length > 0 ? outA.Max() : 0;
                double sumExp = 0;
                for (int u = 0; u < width; u++) { outA[u] = Math.Exp(z[u] - max); sumExp += outA[u]; }
                for (int u = 0; u < width; u++) outA[u] /= sumExp;
            }

            if (!isOutput && training && _dropout > 0)
            {
                // Inverted dropout: zero with probability p, scale survivors.
                var mask = new double[width];
                double keep = 1.0 - _dropout;
                for (int u = 0; u < width; u++)
                {
                    mask[u] = _rng.NextDouble() < _dropout ? 0.0 : 1.0 / keep;
                    outA[u] *= mask[u];
                }
                if (scratchMask is not null) scratchMask[l] = mask;
                masks[l] = mask;
            }

            activations[l] = outA;
            preActs[l] = z;
            a = outA;
        }
        return (a, activations, preActs, masks);
    }

    // ------------------------------------------------------------------
    // Training — mini-batch backpropagation with Adam
    // ------------------------------------------------------------------

    /// <summary>
    /// Trains for <paramref name="epochs"/> full passes over the data in
    /// mini-batches, shuffling each epoch. Targets are vectors matching
    /// <see cref="OutputDim"/> (one-hot rows for a softmax head). Returns the
    /// mean loss of the final epoch.
    /// </summary>
    public double Train(double[][] X, double[][] Y, int epochs, double learningRate, int batchSize = 32)
    {
        return TrainVectors(X, Y, epochs, learningRate, batchSize);
    }

    private double TrainVectors(double[][] X, double[][] Y, int epochs, double learningRate, int batchSize)
    {
        if (X.Length != Y.Length)
            throw new ArgumentException("X and Y must have the same length.");
        if (X.Length == 0) return 0;

        lock (_gate)
        {
            var order = Enumerable.Range(0, X.Length).ToArray();
            double loss = 0;
            for (int e = 0; e < Math.Max(1, epochs); e++)
            {
                Shuffle(order);
                loss = 0;
                int samples = 0;
                for (int start = 0; start < X.Length; start += batchSize)
                {
                    int end = Math.Min(X.Length, start + batchSize);
                    loss += TrainBatchInner(X, Y, order, start, end, learningRate);
                    samples += end - start;
                }
                EpochsTrained++;
                SamplesSeen += X.Length;
                loss /= Math.Max(1, samples);
            }
            return loss;
        }
    }

    /// <summary>Convenience overload: class-index targets for softmax heads.</summary>
    public double Train(double[][] X, int[] classTargets, int epochs, double learningRate, int batchSize = 32)
    {
        if (X.Length != classTargets.Length)
            throw new ArgumentException("X and targets must have the same length.");
        var Y = new double[classTargets.Length][];
        for (int i = 0; i < classTargets.Length; i++)
        {
            if (_outputAct != DeepOutput.Softmax)
                throw new InvalidOperationException("Class-index targets require a Softmax output head.");
            var onehot = new double[_output];
            if (classTargets[i] < 0 || classTargets[i] >= _output)
                throw new ArgumentException($"Class index {classTargets[i]} out of range.");
            onehot[classTargets[i]] = 1.0;
            Y[i] = onehot;
        }
        return TrainVectors(X, Y, epochs, learningRate, batchSize);
    }

    private static void Shuffle(int[] order)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = Random.Shared.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    // One mini-batch: forward every sample, backprop, Adam step.
    private double TrainBatchInner(double[][] X, double[][] Y, int[] order, int start, int end, double lr)
    {
        int depth = _weights.Length;
        var masks = NewMask();
        var gradW = _weights.Select(w => new double[w.Length][]).ToArray();
        var gradB = _biases.Select(b => new double[b.Length]).ToArray();
        for (int l = 0; l < depth; l++)
            for (int u = 0; u < _weights[l].Length; u++)
                gradW[l][u] = new double[_weights[l][u].Length];

        double batchLoss = 0;

        for (int s = start; s < end; s++)
        {
            Array.Clear(masks, 0, masks.Length);
            var (output, activations, preActs, _) = Forward(X[order[s]], training: true, scratchMask: masks);

            // Loss gradient at the output layer.
            var delta = new double[_output];
            double n = 1.0 / _output;
            if (_outputAct == DeepOutput.Softmax)
            {
                // Cross-entropy + softmax: dL/dz = (p - y). Loss in nats.
                for (int o = 0; o < _output; o++)
                {
                    delta[o] = output[o] - Y[order[s]][o];
                    if (Y[order[s]][o] > 0) batchLoss -= Math.Log(Math.Max(output[o], 1e-12));
                }
            }
            else
            {
                // MSE (optionally through a sigmoid). Loss = mean squared error.
                for (int o = 0; o < _output; o++)
                {
                    double err = output[o] - Y[order[s]][o];
                    batchLoss += err * err * n;
                    double dAct = _outputAct == DeepOutput.Sigmoid
                        ? output[o] * (1 - output[o])
                        : 1.0;
                    delta[o] = err * dAct * n;
                }
            }

            // Backprop through the layers.
            double[] grad = delta;
            for (int l = depth - 1; l >= 0; l--)
            {
                double[] lower = l == 0 ? X[order[s]] : activations[l - 1];
                bool isOutput = l == depth - 1;

                for (int u = 0; u < _weights[l].Length; u++)
                {
                    double d = grad[u];
                    var gw = gradW[l][u];
                    for (int i = 0; i < lower.Length; i++)
                        gw[i] += d * lower[i];
                    gradB[l][u] += d;
                }

                if (l > 0)
                {
                    var next = new double[_weights[l - 1].Length];
                    for (int u = 0; u < _weights[l].Length; u++)
                    {
                        double d = grad[u];
                        var row = _weights[l][u];
                        for (int i = 0; i < row.Length; i++)
                            next[i] += d * row[i];
                    }
                    // Hidden-layer activation gradient (undo dropout first —
                    // the mask scales forward activation AND its gradient).
                    var mask = masks[l - 1];
                    for (int i = 0; i < next.Length; i++)
                    {
                        if (mask is not null) next[i] *= mask[i];
                        next[i] *= ActivateGrad(_hiddenAct, preActs[l - 1][i], activations[l - 1][i]);
                    }
                    grad = next;
                }
                _ = isOutput; // (output-layer flag kept for readability)
            }
        }

        // L2 weight decay + Adam step per layer.
        int batch = end - start;
        for (int l = 0; l < depth; l++)
        {
            for (int u = 0; u < _weights[l].Length; u++)
            {
                var gw = gradW[l][u];
                for (int i = 0; i < gw.Length; i++)
                {
                    gw[i] /= batch;
                    if (_l2 > 0) gw[i] += _l2 * _weights[l][u][i];
                }
            }
            for (int u = 0; u < gradB[l].Length; u++) gradB[l][u] /= batch;

            _adam.Step($"w{l}", _weights[l], gradW[l], lr);
            _adam.StepVec($"b{l}", _biases[l], gradB[l], lr);
        }

        return batchLoss;
    }

    // ------------------------------------------------------------------
    // Serialization — weights, biases, architecture and hyperparameters.
    // Optimizer moments are not persisted (a fresh Adam restarts cleanly).
    // ------------------------------------------------------------------

    private sealed class NetDto
    {
        public int Input { get; set; }
        public int Output { get; set; }
        public List<int>? Hidden { get; set; }
        public string HiddenAct { get; set; } = "GELU";
        public string OutputAct { get; set; } = "Sigmoid";
        public double Dropout { get; set; }
        public double L2 { get; set; }
        public int Seed { get; set; }
        public List<double[]>? Biases { get; set; }
        public List<List<double[]>>? Weights { get; set; }
        public int EpochsTrained { get; set; }
        public long SamplesSeen { get; set; }
    }

    /// <summary>Serializes the network to JSON (for persistence).</summary>
    public string Serialize()
    {
        lock (_gate)
        {
            var dto = new NetDto
            {
                Input = _input,
                Output = _output,
                Hidden = _weights.Length <= 1
                    ? new List<int>()
                    : _weights.Take(_weights.Length - 1).Select(w => w.Length).ToList(),
                HiddenAct = _hiddenAct.ToString(),
                OutputAct = _outputAct.ToString(),
                Dropout = _dropout,
                L2 = _l2,
                Seed = 42,
                Biases = _biases.Select(b => b.ToArray()).ToList(),
                Weights = _weights.Select(w => w.Select(r => r.ToArray()).ToList()).ToList(),
                EpochsTrained = EpochsTrained,
                SamplesSeen = SamplesSeen,
            };
            return JsonSerializer.Serialize(dto);
        }
    }

    /// <summary>Rebuilds a network from serialized JSON.</summary>
    public static DeepNet Deserialize(string json)
    {
        try
        {
            var dto = JsonSerializer.Deserialize<NetDto>(json)
                ?? throw new Cognify.Core.PersistenceException("Network payload is empty.");
            if (dto.Weights is null || dto.Biases is null)
                throw new Cognify.Core.PersistenceException("Network payload is incomplete.");

            var weights = dto.Weights.Select(w => w.Select(r => r.ToArray()).ToArray()).ToArray();
            var biases = dto.Biases.Select(b => b.ToArray()).ToArray();

            var net = new DeepNet(weights, biases, dto.Input, dto.Output,
                Enum.TryParse<DeepActivation>(dto.HiddenAct, out var ha) ? ha : DeepActivation.GELU,
                Enum.TryParse<DeepOutput>(dto.OutputAct, out var oa) ? oa : DeepOutput.Sigmoid,
                dto.Dropout, dto.L2, dto.Seed);
            net.EpochsTrained = dto.EpochsTrained;
            net.SamplesSeen = dto.SamplesSeen;
            return net;
        }
        catch (JsonException ex)
        {
            throw new Cognify.Core.PersistenceException("Failed to deserialize deep network weights.", ex);
        }
    }
}
