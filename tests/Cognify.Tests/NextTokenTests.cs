using Cognify.Core;
using Cognify.Learning;
using Cognify.Persistence;

namespace Cognify.Tests;

/// <summary>
/// Tests for the n-gram next-token prediction model — the mind's language
/// engine: observation, prediction, fluency scoring, persistence.
/// </summary>
public class NextTokenTests
{
    [Fact]
    public void Observe_ThenPredict_TopsTheTaughtContinuation()
    {
        var model = new NextTokenModel();
        for (int i = 0; i < 10; i++)
            model.Observe("birds are animals. birds can fly. birds have wings.");

        var predictions = model.PredictNext("birds are", 5);
        Assert.NotEmpty(predictions);
        Assert.Equal("animals", predictions[0].Token);
        Assert.True(predictions[0].Probability > 0.5);
    }

    [Fact]
    public void Predict_FallsBackToBigram_WhenTrigramUnknown()
    {
        var model = new NextTokenModel();
        for (int i = 0; i < 10; i++)
            model.Observe("the cat sat on the mat");

        // "sat on" was observed, but never "zebra sat on" — bigram backoff
        // still proposes "the".
        var predictions = model.PredictNext("zebra sat", 5);
        Assert.NotEmpty(predictions);
        Assert.Contains(predictions, p => p.Token == "on");
    }

    [Fact]
    public void Predict_AlwaysReturnsSomething_EvenForUnknownContext()
    {
        var model = new NextTokenModel();
        model.Observe("hello world");

        var predictions = model.PredictNext("completely unseen xylophone context", 3);
        Assert.NotEmpty(predictions); // unigram fallback — never empty
    }

    [Fact]
    public void Score_PrefersTextSimilarToWhatItObserved()
    {
        var model = new NextTokenModel();
        for (int i = 0; i < 10; i++)
            model.Observe("birds are animals. birds can fly.");

        double seen = model.Score("birds are animals");
        double unseen = model.Score("quantum kaleidoscopes negotiate ferociously");
        Assert.True(seen > unseen,
            $"fluent text should outscore gibberish: {seen} vs {unseen}");
    }

    [Fact]
    public void Score_UntrainedModelReturnsZero()
    {
        var model = new NextTokenModel();
        Assert.Equal(0, model.Score("anything at all"));
        Assert.False(model.IsTrained);
        Assert.Empty(model.PredictNext("anything", 3)); // nothing observed: no candidates
    }

    [Fact]
    public void Serialize_RoundTripsCounts()
    {
        var model = new NextTokenModel();
        for (int i = 0; i < 6; i++)
            model.Observe("birds are animals. birds can fly.");

        var restored = NextTokenModel.Deserialize(model.Serialize());
        Assert.Equal(model.TotalTokens, restored.TotalTokens);
        Assert.Equal(model.Vocabulary, restored.Vocabulary);
        Assert.Equal(model.PredictNext("birds are", 3)[0].Token,
                     restored.PredictNext("birds are", 3)[0].Token);
    }

    [Fact]
    public void Merge_AddsCounts()
    {
        var a = new NextTokenModel();
        var b = new NextTokenModel();
        a.Observe("birds are animals");
        b.Observe("birds are animals");
        a.Merge(b);

        Assert.Equal(2 * 3, a.TotalTokens); // both observe the same 3 tokens
        var predictions = a.PredictNext("birds are", 1);
        Assert.Equal("animals", predictions[0].Token);
    }

    [Fact]
    public void Mind_ObservesConversationAndTraining()
    {
        using var mind = MindBuilder.Create("Talker")
            .WithReasoning().WithSelfLearning().Build();

        Assert.False(mind.LanguageModel.IsTrained);
        mind.Train("birds are animals. birds can fly. birds have wings.");
        mind.Ask("what are birds?");
        Assert.True(mind.LanguageModel.IsTrained);
        Assert.True(mind.LanguageModel.TotalTokens > 0);

        // The taught trigram is predictable through the mind's public API.
        var predictions = mind.PredictNext("birds are", 3);
        Assert.Equal("animals", predictions[0].Token);
    }

    [Fact]
    public void LanguageModel_PersistsAcrossSnapshot()
    {
        var path = Path.Combine(Path.GetTempPath(), $"cognify-lm-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilder.Create("Sleepy")
                       .WithReasoning().WithSelfLearning()
                       .WithPersistence(path).Build())
            {
                mind.Train("birds are animals. birds can fly.");
                mind.SaveAsync(path).Wait();
            }

            using (var restored = MindBuilder.Restore(path))
            {
                Assert.True(restored.LanguageModel.IsTrained);
                Assert.Equal("animals", restored.PredictNext("birds are", 3)[0].Token);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
