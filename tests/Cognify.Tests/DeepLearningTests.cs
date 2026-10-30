using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>
/// Tests for the deep-learning core: the Adam optimizer and
/// <see cref="DeepNet"/> — a genuinely deep feed-forward network with
/// backpropagation, dropout, L2 and mini-batch Adam training.
/// </summary>
public class DeepLearningTests
{
    [Fact]
    public void DeepNet_LearnsXOR()
    {
        // The classic deep-learning sanity check: XOR is not linearly
        // separable, so only a network with real hidden layers solves it.
        var net = new DeepNet(2, [8, 8], 1, DeepActivation.GELU,
            DeepOutput.Sigmoid, dropout: 0.0, l2: 0.0, seed: 11);

        double[][] x = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] y = [[0], [1], [1], [0]];

        net.Train(x, y, epochs: 600, learningRate: 0.05, batchSize: 4);

        Assert.True(net.Predict(x[0])[0] < 0.3, $"XOR(0,0) should be ~0, got {net.Predict(x[0])[0]:F3}");
        Assert.True(net.Predict(x[1])[0] > 0.7, $"XOR(0,1) should be ~1, got {net.Predict(x[1])[0]:F3}");
        Assert.True(net.Predict(x[2])[0] > 0.7, $"XOR(1,0) should be ~1, got {net.Predict(x[2])[0]:F3}");
        Assert.True(net.Predict(x[3])[0] < 0.3, $"XOR(1,1) should be ~0, got {net.Predict(x[3])[0]:F3}");
    }

    [Fact]
    public void DeepNet_SoftmaxHead_Classifies()
    {
        // Three one-hot-pattern classes, learned with a softmax + CE head.
        var net = new DeepNet(6, [12], 3, DeepActivation.ReLU,
            DeepOutput.Softmax, seed: 5);

        double[][] x =
        [
            [1, 0, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0], [0, 0.9, 0.1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0], [0, 0, 0.9, 0.1, 0, 0],
        ];
        int[] classes = [0, 0, 1, 1, 2, 2];

        double loss = net.Train(x, classes, epochs: 300, learningRate: 0.03, batchSize: 6);
        Assert.True(loss < 0.3, $"final cross-entropy should drop, got {loss:F3}");

        for (int i = 0; i < x.Length; i++)
        {
            var p = net.Predict(x[i]);
            int argmax = 0;
            for (int c = 1; c < p.Length; c++) if (p[c] > p[argmax]) argmax = c;
            Assert.Equal(classes[i], argmax);
        }
    }

    [Fact]
    public void DeepNet_TrainingReducesLoss()
    {
        var net = new DeepNet(4, [6, 6], 2, DeepActivation.Tanh, DeepOutput.Sigmoid, seed: 3);
        double[][] x = [[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0.5, 0, 1, 0], [0, 0.5, 0, 1]];
        double[][] y = [[1, 0], [0, 1], [1, 0], [0, 1]];

        double first = net.Train(x, y, epochs: 1, learningRate: 0.05);
        double last = net.Train(x, y, epochs: 200, learningRate: 0.05);
        Assert.True(last < first, $"loss should decrease: {first:F4} -> {last:F4}");
        Assert.Equal(201, net.EpochsTrained);
    }

    [Fact]
    public void DeepNet_ParameterCount_IsExact()
    {
        // 3x4 weights + 3 biases + 2x3 weights + 2 biases = 12 + 3 + 6 + 2 = 23
        var net = new DeepNet(4, [3], 2);
        Assert.Equal(23, net.ParameterCount);
        Assert.Equal(2, net.Depth);
        Assert.Equal(new[] { 3, 2 }, net.Units);
    }

    [Fact]
    public void DeepNet_Predict_IsDeterministic()
    {
        // Dropout is training-only; inference must be stable.
        var net = new DeepNet(4, [8, 8], 2, DeepActivation.GELU, DeepOutput.Sigmoid, dropout: 0.5, seed: 9);
        double[] input = [0.2, 0.8, 0.5, 0.1];
        Assert.Equal(net.Predict(input), net.Predict(input));
    }

    [Fact]
    public void DeepNet_Serialization_Roundtrip()
    {
        var net = new DeepNet(5, [7, 4], 3, DeepActivation.GELU, DeepOutput.Sigmoid, dropout: 0.2, seed: 21);
        double[][] x = [[1, 2, 3, 4, 5]];
        net.Train(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
            [[1, 0, 0], [0, 0, 1]], epochs: 30, learningRate: 0.05);

        var restored = DeepNet.Deserialize(net.Serialize());
        Assert.Equal(net.Predict(x[0]), restored.Predict(x[0]));
        Assert.Equal(net.ParameterCount, restored.ParameterCount);
        Assert.Equal(net.Depth, restored.Depth);
    }

    [Fact]
    public void DeepNet_RejectsInvalidShapes()
    {
        Assert.Throws<ArgumentException>(() => new DeepNet(0, [4], 1));
        Assert.Throws<ArgumentException>(() => new DeepNet(3, [0], 1));
        var net = new DeepNet(3, [4], 2);
        Assert.Throws<ArgumentException>(() => net.Predict(new double[4]));
    }

    [Fact]
    public void AdamOptimizer_StepsAndResets()
    {
        var adam = new AdamOptimizer();
        var p = new[] { 1.0, 1.0 };
        var g = new[] { 0.5, -0.5 };
        adam.StepVec("p", p, g, 0.1);
        Assert.Equal(1, adam.Steps);
        Assert.NotEqual(1.0, p[0]); // the step actually moved the parameter
        adam.Reset();
        Assert.Equal(0, adam.Steps);
    }

    [Fact]
    public void Gelu_MatchesKnownValues()
    {
        Assert.InRange(DeepNet.Gelu(0.0), -1e-9, 1e-9);
        Assert.InRange(DeepNet.Gelu(1.0), 0.841, 0.842);   // ≈0.8412
        Assert.InRange(DeepNet.Gelu(-1.0), -0.159, -0.158); // ≈-0.1588
        Assert.True(DeepNet.Gelu(10.0) > 9.9);              // ≈ identity for large x
    }
}
