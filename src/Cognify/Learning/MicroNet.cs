using System.Text.Json;

namespace Cognify.Learning;

/// <summary>
/// A tiny feed-forward neural network with real backpropagation and
/// gradient descent. Input -> hidden (tanh) -> output (sigmoid).
/// Used by the self-learning module to predict reward/valence from stimulus
/// features, so the mind gets better at choosing responses over time.
/// </summary>
public sealed class MicroNet
{
    private readonly double[][] _w1; // hidden x input
    private readonly double[][] _w2; // output x hidden
    private readonly double[] _b1;
    private readonly double[] _b2;
    private readonly int _input, _hidden, _output;

    /// <summary>Creates the network with Xavier-style random initialization.</summary>
    public MicroNet(int input, int hidden, int output, int seed = 42)
    {
        _input = input;
        _hidden = hidden;
        _output = output;
        var rng = new Random(seed);

        _w1 = Matrix(hidden, input, () => Xavier(input, rng));
        _w2 = Matrix(output, hidden, () => Xavier(hidden, rng));
        _b1 = new double[hidden];
        _b2 = new double[output];
    }

    private MicroNet(double[][] w1, double[][] w2, double[] b1, double[] b2)
    {
        _w1 = w1;
        _w2 = w2;
        _b1 = b1;
        _b2 = b2;
        _input = w1[0].Length;
        _hidden = w1.Length;
        _output = w2.Length;
    }

    private static double Xavier(int fanIn, Random rng) =>
        (rng.NextDouble() * 2 - 1) * Math.Sqrt(2.0 / (fanIn + 1));

    private static double[][] Matrix(int rows, int cols, Func<double> init) =>
        Enumerable.Range(0, rows).Select(_ =>
            Enumerable.Range(0, cols).Select(__ => init()).ToArray()).ToArray();

    /// <summary>Input dimension.</summary>
    public int InputDim => _input;

    /// <summary>Hidden dimension.</summary>
    public int HiddenDim => _hidden;

    /// <summary>Output dimension.</summary>
    public int OutputDim => _output;

    /// <summary>First-layer weights [hidden x input] (defensive copy, for export).</summary>
    public double[][] GetW1() => _w1.Select(r => (double[])r.Clone()).ToArray();

    /// <summary>Second-layer weights [output x hidden] (defensive copy, for export).</summary>
    public double[][] GetW2() => _w2.Select(r => (double[])r.Clone()).ToArray();

    /// <summary>First-layer biases [hidden] (defensive copy, for export).</summary>
    public double[] GetB1() => (double[])_b1.Clone();

    /// <summary>Second-layer biases [output] (defensive copy, for export).</summary>
    public double[] GetB2() => (double[])_b2.Clone();

    /// <summary>
    /// Forward pass. Returns output activations (0..1 per unit, sigmoid).
    /// </summary>
    public double[] Predict(double[] input)
    {
        if (input.Length != _input)
            throw new ArgumentException($"Expected {_input} features, got {input.Length}.", nameof(input));

        var hidden = new double[_hidden];
        for (int h = 0; h < _hidden; h++)
        {
            double sum = _b1[h];
            for (int i = 0; i < _input; i++) sum += _w1[h][i] * input[i];
            hidden[h] = Math.Tanh(sum);
        }

        var output = new double[_output];
        for (int o = 0; o < _output; o++)
        {
            double sum = _b2[o];
            for (int h = 0; h < _hidden; h++) sum += _w2[o][h] * hidden[h];
            output[o] = Sigmoid(sum);
        }
        return output;
    }

    /// <summary>
    /// One gradient-descent step on mean-squared-error between prediction and
    /// target. Returns the squared error before the update.
    /// </summary>
    public double Train(double[] input, double[] target, double learningRate)
    {
        // Forward pass, keeping activations.
        var hidden = new double[_hidden];
        for (int h = 0; h < _hidden; h++)
        {
            double sum = _b1[h];
            for (int i = 0; i < _input; i++) sum += _w1[h][i] * input[i];
            hidden[h] = Math.Tanh(sum);
        }

        var preOut = new double[_output];
        var output = new double[_output];
        for (int o = 0; o < _output; o++)
        {
            double sum = _b2[o];
            for (int h = 0; h < _hidden; h++) sum += _w2[o][h] * hidden[h];
            preOut[o] = sum;
            output[o] = Sigmoid(sum);
        }

        // Output layer error (sigmoid + MSE -> (out - target) * out * (1 - out)).
        var deltaOut = new double[_output];
        double loss = 0;
        for (int o = 0; o < _output; o++)
        {
            double err = output[o] - target[o];
            loss += err * err;
            deltaOut[o] = err * output[o] * (1 - output[o]);
        }

        // Hidden layer error.
        var deltaHidden = new double[_hidden];
        for (int h = 0; h < _hidden; h++)
        {
            double sum = 0;
            for (int o = 0; o < _output; o++) sum += _w2[o][h] * deltaOut[o];
            deltaHidden[h] = sum * (1 - hidden[h] * hidden[h]); // tanh'
        }

        // Update weights and biases.
        for (int o = 0; o < _output; o++)
        {
            for (int h = 0; h < _hidden; h++)
                _w2[o][h] -= learningRate * deltaOut[o] * hidden[h];
            _b2[o] -= learningRate * deltaOut[o];
        }
        for (int h = 0; h < _hidden; h++)
        {
            for (int i = 0; i < _input; i++)
                _w1[h][i] -= learningRate * deltaHidden[h] * input[i];
            _b1[h] -= learningRate * deltaHidden[h];
        }

        return loss / _output;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-Math.Clamp(x, -20, 20)));

    /// <summary>Serializes all weights/biases to JSON (for persistence).</summary>
    public string Serialize()
    {
        var dto = new NetDto { W1 = _w1, W2 = _w2, B1 = _b1, B2 = _b2 };
        return JsonSerializer.Serialize(dto);
    }

    /// <summary>Rebuilds a network from serialized JSON.</summary>
    public static MicroNet Deserialize(string json)
    {
        try
        {
            var dto = JsonSerializer.Deserialize<NetDto>(json);
            if (dto?.W1 is null || dto.W2 is null || dto.B1 is null || dto.B2 is null)
                throw new Cognify.Core.PersistenceException("Network payload is incomplete.");
            return new MicroNet(dto.W1, dto.W2, dto.B1, dto.B2);
        }
        catch (JsonException ex)
        {
            throw new Cognify.Core.PersistenceException("Failed to deserialize network weights.", ex);
        }
    }

    private sealed class NetDto
    {
        public double[][]? W1 { get; set; }
        public double[][]? W2 { get; set; }
        public double[]? B1 { get; set; }
        public double[]? B2 { get; set; }
    }
}
