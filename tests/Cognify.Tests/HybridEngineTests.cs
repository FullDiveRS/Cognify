using Cognify.Core;
using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>
/// Tests for the hybrid decision engine: the style-conditioned neural
/// network (<see cref="NeuralStyleEngine"/>) learning (context, style, mood)
/// → reward, the adaptive symbolic/neural blend, and their persistence.
/// </summary>
public class HybridEngineTests
{
    private static double[] Features(int dim, double fill = 0.5) =>
        Enumerable.Range(0, dim).Select(i => fill * ((i % 3) + 1) / 3.0).ToArray();

    // ------------------------------------------------------------------
    // NeuralStyleEngine basics
    // ------------------------------------------------------------------

    [Fact]
    public void ComposeInput_HasStyleOneHotAndMood()
    {
        var engine = new NeuralStyleEngine(featureDim: 8, styleCount: 9, hiddenUnits: 8, seed: 1);
        var features = Features(8);
        var input = engine.ComposeInput(features, styleId: 3, valence: -0.5, arousal: 0.9);

        Assert.Equal(8 + 9 + 2, input.Length);
        Assert.Equal(1.0, input[8 + 3]);           // style one-hot set
        Assert.Equal(0.0, input.Where((v, i) => i >= 8 && i < 17 && i != 11).Sum()); // rest zero
        Assert.Equal(-0.5, input[^2]);             // valence
        Assert.Equal(0.9, input[^1]);              // arousal
    }

    [Fact]
    public void Predict_IsInUnitRange_BeforeTraining()
    {
        var engine = new NeuralStyleEngine(8, 9, 8, seed: 2);
        var features = Features(8);
        for (int style = 0; style < 9; style++)
        {
            double p = engine.Predict(features, style, valence: 0, arousal: 0.5);
            Assert.InRange(p, 0.0, 1.0);
        }
    }

    [Fact]
    public void StyleNet_LearnsPreferredStyle()
    {
        // Style 2 keeps paying off, style 5 keeps disappointing: after
        // training the network must rank style 2 above style 5 for the SAME
        // features and mood — a learned preference, not a rule.
        var engine = new NeuralStyleEngine(8, 9, 12, seed: 7);
        var features = Features(8);
        for (int step = 0; step < 60; step++)
        {
            engine.Learn(features, 2, 0.2, 0.6, reward01: 0.95, learningRate: 0.25);
            engine.Learn(features, 5, 0.2, 0.6, reward01: 0.05, learningRate: 0.25);
        }
        Assert.True(engine.Predict(features, 2, 0.2, 0.6) > engine.Predict(features, 5, 0.2, 0.6) + 0.2,
            "the style-conditioned net should learn which style pays off");
    }

    [Fact]
    public void Serialize_RoundTrips()
    {
        var engine = new NeuralStyleEngine(8, 9, 12, seed: 3);
        var features = Features(8);
        for (int i = 0; i < 10; i++)
            engine.Learn(features, 1, 0.1, 0.4, reward01: 0.8, learningRate: 0.2);

        var restored = NeuralStyleEngine.Deserialize(engine.Serialize());
        Assert.Equal(engine.FeatureDim, restored.FeatureDim);
        Assert.Equal(engine.StyleCount, restored.StyleCount);
        Assert.Equal(engine.Predict(features, 1, 0.1, 0.4),
                     restored.Predict(features, 1, 0.1, 0.4), precision: 10);
    }

    // ------------------------------------------------------------------
    // SelfLearningModule: hybrid blend
    // ------------------------------------------------------------------

    [Fact]
    public void NeuralWeight_StaysInBounds_AndAdapts()
    {
        using var mind = MindBuilderForHybrid();
        var learner = mind.Learning!;
        Assert.InRange(learner.NeuralWeight, 0.25, 0.80);

        // Feed consistent rewards: predictions improve, weight drifts up.
        var features = FeatureEncoder.Encode("this is wonderful and delightful", learner.Net.InputDim);
        learner.ObserveAction(features, 2, 0.4, 0.6);
        for (int i = 0; i < 30; i++)
            learner.Reinforce(0.9);
        Assert.InRange(learner.NeuralWeight, 0.25, 0.80);
    }

    [Fact]
    public void HybridScore_RanksReinforcedStyleHigher()
    {
        using var mind = MindBuilderForHybrid();
        var learner = mind.Learning!;
        var features = FeatureEncoder.Encode("let us talk about training", learner.Net.InputDim);

        learner.ObserveAction(features, 4, 0.1, 0.5);
        for (int i = 0; i < 40; i++)
            learner.Reinforce(1.0);
        learner.ObserveAction(features, 1, 0.1, 0.5);
        for (int i = 0; i < 40; i++)
            learner.Reinforce(-1.0);

        double good = learner.Score(features, 4, 0.1, 0.5);
        double bad = learner.Score(features, 1, 0.1, 0.5);
        Assert.True(good > bad, $"hybrid score should favor the rewarded style ({good:0.000} vs {bad:0.000})");
    }

    [Fact]
    public void HybridEngine_SurvivesSaveAndRestore()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-hybrid-{Guid.NewGuid():N}.db");
        try
        {
            double trainedScore;
            using (var mind = MindBuilderForHybrid(path))
            {
                var learner = mind.Learning!;
                var features = FeatureEncoder.Encode("restore me please", learner.Net.InputDim);
                learner.ObserveAction(features, 3, 0.2, 0.5);
                for (int i = 0; i < 25; i++)
                    learner.Reinforce(0.8);
                trainedScore = learner.Score(features, 3, 0.2, 0.5);
                mind.Store!.Save(Persistence.MindSnapshot.Capture(mind));
            }

            using var restored = MindBuilder.Restore(path);
            var rLearner = restored.Learning!;
            var rFeatures = FeatureEncoder.Encode("restore me please", rLearner.Net.InputDim);
            double restoredScore = rLearner.Score(rFeatures, 3, 0.2, 0.5);
            Assert.Equal(trainedScore, restoredScore, precision: 5);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------------
    // Mind-level: deliberation stays healthy with the hybrid engine
    // ------------------------------------------------------------------

    [Fact]
    public void Mind_DeliberatesWithHybridEngine_AndMentionsBlend()
    {
        using var mind = MindBuilderForHybrid();
        mind.Process("hello there, how are you doing today?");
        Assert.NotNull(mind.LastResponse);
        Assert.False(string.IsNullOrWhiteSpace(mind.LastResponse!.Content));

        // The chain of thought explains the symbolic/neural blend.
        var deliberate = mind.Stream!.Recent(50).FirstOrDefault(t => t.Phase == "Deliberate");
        Assert.NotNull(deliberate);
        Assert.Contains("hybrid blend", deliberate!.Text);
    }

    private static Core.Mind MindBuilderForHybrid(string? path = null) =>
        MindBuilder.Create("Hybrid")
            .WithPersonality(p => p.Traits(openness: 0.8, extraversion: 0.6))
            .WithEmotions()
            .WithThinking()
            .WithSelfLearning()
            .WithMemory()
            .WithPersistence(path ?? Path.Combine(Path.GetTempPath(), $"cognify-hybrid-{Guid.NewGuid():N}.db"))
            .Build();
}
