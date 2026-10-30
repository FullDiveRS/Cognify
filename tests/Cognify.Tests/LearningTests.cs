using Cognify.Core;
using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>Tests for the micro neural network and self-learning module.</summary>
public class LearningTests
{
    [Fact]
    public void MicroNet_Predict_ReturnsValidRange()
    {
        var net = new MicroNet(16, 8, 3, seed: 7);
        var output = net.Predict(new double[16]);
        Assert.Equal(3, output.Length);
        Assert.All(output, v => Assert.InRange(v, 0, 1));
    }

    [Fact]
    public void MicroNet_Train_ReducesLossOnRepeatedTarget()
    {
        var net = new MicroNet(8, 10, 1, seed: 3);
        var input = new double[8];
        input[2] = 1.0;
        double firstLoss = 0, lastLoss = 0;
        for (int i = 0; i < 200; i++)
        {
            double loss = net.Train(input, new[] { 0.95 }, 0.1);
            if (i == 0) firstLoss = loss;
            if (i == 199) lastLoss = loss;
        }
        Assert.True(lastLoss < firstLoss, $"loss should drop: {firstLoss} -> {lastLoss}");
    }

    [Fact]
    public void MicroNet_Train_LearnsToPredictTarget()
    {
        var net = new MicroNet(8, 12, 1, seed: 5);
        var input = new double[8];
        input[0] = 0.9;
        for (int i = 0; i < 400; i++)
            net.Train(input, new[] { 1.0 }, 0.15);
        double prediction = net.Predict(input)[0];
        Assert.True(prediction > 0.85, $"expected high prediction, got {prediction}");
    }

    [Fact]
    public void MicroNet_Serialization_Roundtrip()
    {
        var net = new MicroNet(10, 6, 3, seed: 11);
        var json = net.Serialize();
        var restored = MicroNet.Deserialize(json);

        var input = new double[10];
        input[3] = 0.5;
        var original = net.Predict(input);
        var copy = restored.Predict(input);
        Assert.Equal(original, copy);
    }

    [Fact]
    public void FeatureEncoder_Deterministic()
    {
        var a = FeatureEncoder.Encode("hello world", 32);
        var b = FeatureEncoder.Encode("hello world", 32);
        Assert.Equal(a, b);
        Assert.All(a, v => Assert.InRange(v, -1, 1));
    }

    [Fact]
    public void FeatureEncoder_SimilarTexts_Overlap()
    {
        var a = FeatureEncoder.Encode("the cat sat", 64);
        var b = FeatureEncoder.Encode("a cat sat", 64);
        double dot = a.Zip(b, (x, y) => x * y).Sum();
        Assert.True(dot > 0.2, $"expected overlap, got {dot}");
    }

    [Fact]
    public void FeatureEncoder_Normalized()
    {
        var v = FeatureEncoder.Encode("some sample text with words", 32);
        double norm = Math.Sqrt(v.Sum(x => x * x));
        Assert.Equal(1.0, norm, precision: 3);
    }

    [Fact]
    public void Reinforce_UpdatesQTable()
    {
        using var mind = MindBuilder.Create("Learner")
            .WithThinking()
            .WithSelfLearning()
            .Build();
        mind.Process("hello there");
        mind.Reinforce(1.0);
        Assert.Equal(1, mind.Learning!.Stats.TotalExperiences);
        Assert.True(mind.Learning.QTable.Count > 0);
    }

    [Fact]
    public void Reinforce_Positive_RaisesAverageReward()
    {
        using var mind = MindBuilder.Create("Learner").WithThinking().WithSelfLearning().Build();
        mind.Process("hello");
        for (int i = 0; i < 10; i++)
            mind.Reinforce(1.0); // EMA average climbs toward 1.0
        Assert.True(mind.Learning!.Stats.AverageReward > 0.5);
    }

    [Fact]
    public void Reinforce_TrainsNetwork()
    {
        using var mind = MindBuilder.Create("Learner").WithThinking().WithSelfLearning().Build();
        int stepsBefore = mind.Learning!.Stats.TrainingSteps;
        mind.Process("train me");
        mind.Reinforce(0.5);
        Assert.True(mind.Learning.Stats.TrainingSteps >= stepsBefore + 1);
        Assert.Single(mind.Learning.Experiences);
    }

    [Fact]
    public void Reinforce_WithoutThinking_StillLearns()
    {
        // Features are encoded by the learning module itself every cycle.
        using var mind = MindBuilder.Create("SoloLearner").WithSelfLearning().Build();
        mind.Process("learn directly");
        mind.Reinforce(0.8);
        Assert.Equal(1, mind.Learning!.Stats.TotalExperiences);
    }

    [Fact]
    public void Learned_EventFires()
    {
        using var mind = MindBuilder.Create("Events").WithThinking().WithSelfLearning().Build();
        bool learned = false;
        mind.Learned += (_, _) => learned = true;
        mind.Process("hello");
        mind.Reinforce(0.6);
        Assert.True(learned);
    }

    [Fact]
    public void Replay_RunsOnStoredExperiences()
    {
        using var mind = MindBuilder.Create("Replay").WithThinking().WithSelfLearning().Build();
        for (int i = 0; i < 5; i++)
        {
            mind.Process($"experience number {i}");
            mind.Reinforce(0.5);
        }
        double loss = mind.Learning!.Replay(epochs: 2);
        Assert.True(loss >= 0);
    }

    [Fact]
    public void Exploration_RespectsOpenness()
    {
        var open = new Cognify.Personality.BigFiveTraits { Openness = 0.95 };
        var closed = new Cognify.Personality.BigFiveTraits { Openness = 0.05 };
        Assert.True(open.ExplorationRate(0.15) > closed.ExplorationRate(0.15));
        Assert.True(open.ExplorationRate(0.15) > 0.15);
        Assert.True(closed.ExplorationRate(0.15) < 0.15);
    }
}
