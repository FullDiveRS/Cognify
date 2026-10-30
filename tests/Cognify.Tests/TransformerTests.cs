using Cognify.Core;
using Cognify.Export;
using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>
/// Tests for the transformer language model, the hybrid next-token blend,
/// the mind-level wiring, persistence and tensor export.
/// </summary>
public class TransformerTests
{
    private static TransformerConfig TinyConfig(int seed = 7) => new()
    {
        Dim = 16,
        Heads = 2,
        Layers = 1,
        ContextLen = 8,
        FfnMult = 2,
        Dropout = 0.0,     // deterministic tests
        LearningRate = 0.02,
        Seed = seed,
        BatchSize = 8,
        MaxWindowsPerEpoch = 128,
    };

    // Deliberately repetitive: the model MUST learn these transitions.
    private static readonly string[] Corpus =
    [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat ran to the mat",
        "the dog ran to the rug",
        "the cat ran to the mat",
        "the dog ran to the rug",
    ];

    private static TransformerLM TrainedModel(int epochs = 60, int seed = 7)
    {
        var lm = new TransformerLM(TinyConfig(seed));
        foreach (var line in Corpus) lm.Observe(line);
        lm.Train(epochs);
        return lm;
    }

    [Fact]
    public void Untrained_ReturnsNoOpinions()
    {
        var lm = new TransformerLM(TinyConfig());
        Assert.False(lm.IsTrained);
        Assert.Empty(lm.PredictNext("the cat"));
        Assert.Equal(0, lm.Score("the cat sat"));
        Assert.Equal("", lm.Generate("the"));
    }

    [Fact]
    public void Training_ReducesLoss()
    {
        var lm = new TransformerLM(TinyConfig());
        foreach (var line in Corpus) lm.Observe(line);
        var report = lm.Train(60);
        Assert.True(report.FinalLoss < report.InitialLoss,
            $"loss must drop: {report.InitialLoss:F3} -> {report.FinalLoss:F3}");
        Assert.True(report.FinalLoss < 2.5, $"a repetitive corpus should reach low loss, got {report.FinalLoss:F3}");
        Assert.Equal(60, report.Epochs);
        Assert.True(report.Parameters > 0);
    }

    [Fact]
    public void PredictNext_LearnsCorpusTransitions()
    {
        var lm = TrainedModel();

        // After a bare "the" both continuations exist in this corpus
        // (sentence-start "the cat/dog", mid-sentence "on the mat/rug").
        var afterThe = lm.PredictNext("the", 2);
        Assert.All(afterThe, p => Assert.True(
            p.Token is "cat" or "dog" or "mat" or "rug", $"unexpected token {p.Token}"));

        // Unambiguous mid-sentence context: after "sat on the" only mat/rug.
        var afterSeat = lm.PredictNext("sat on the", 2);
        Assert.All(afterSeat, p => Assert.True(p.Token is "mat" or "rug", $"unexpected token {p.Token}"));

        // Unambiguous subject context: after "the cat" only sat/ran.
        var afterCat = lm.PredictNext("the cat", 2);
        Assert.All(afterCat, p => Assert.True(p.Token is "sat" or "ran", $"unexpected token {p.Token}"));

        // Probabilities are a normalized share of the returned set.
        Assert.InRange(afterSeat.Sum(p => p.Probability), 0.99, 1.01);
    }

    [Fact]
    public void PredictNext_ExcludesSpecialTokens()
    {
        var lm = TrainedModel();
        for (int i = 0; i < 10; i++)
        {
            var preds = lm.PredictNext(Corpus[i % Corpus.Length], 5);
            Assert.All(preds, p => Assert.DoesNotContain("<", p.Token));
        }
    }

    [Fact]
    public void Score_TrainedTextBeatsGibberish()
    {
        var lm = TrainedModel();
        double real = lm.Score("the cat sat on the mat");
        double gibberish = lm.Score("zqx wvv kkk jjj qqq");
        Assert.True(real > gibberish,
            $"corpus text ({real:F3}) should score above out-of-vocabulary noise ({gibberish:F3})");
        Assert.True(real < 0); // log-probabilities are negative
    }

    [Fact]
    public void Generate_UsesOnlyKnownWords()
    {
        var lm = TrainedModel();
        var generated = lm.Generate("the cat", maxTokens: 6);
        if (generated.Length == 0) return; // sampling may hit <eos> immediately
        foreach (var word in generated.Split(' '))
            Assert.DoesNotContain("<", word);
    }

    [Fact]
    public void Vocabulary_GrowsOnObservation()
    {
        var lm = new TransformerLM(TinyConfig());
        int before = lm.VocabularySize; // 4 special tokens
        lm.Observe("the cat sat");
        Assert.True(lm.VocabularySize > before);
        int after = lm.VocabularySize;
        lm.Observe("quantum entanglement sparkles");
        Assert.True(lm.VocabularySize > after);
        Assert.Equal(6, lm.TokensObserved); // the+cat+sat = 3, quantum+entanglement+sparkles = 3
    }

    [Fact]
    public void Serialization_Roundtrip_PreservesPredictions()
    {
        var lm = TrainedModel(epochs: 30);
        var before = lm.PredictNext("the cat", 5);

        var restored = new TransformerLM(TinyConfig());
        Assert.True(restored.TryRestore(lm.Serialize()));
        var after = restored.PredictNext("the cat", 5);

        Assert.Equal(before.Select(p => p.Token), after.Select(p => p.Token));
        for (int i = 0; i < before.Count; i++)
            Assert.InRange(after[i].Probability, before[i].Probability - 0.01, before[i].Probability + 0.01);
        Assert.True(restored.IsTrained);
        Assert.Equal(lm.BufferedSentences, restored.BufferedSentences);
    }

    [Fact]
    public void Serialization_RejectsWrongArchitecture()
    {
        var lm = TrainedModel(epochs: 5);
        var other = new TransformerLM(TinyConfig() with { Layers = 3 });
        Assert.False(other.TryRestore(lm.Serialize()));
        Assert.False(other.TryRestore("not json at all"));
        Assert.False(other.TryRestore(""));
    }

    [Fact]
    public void Hybrid_UntrainedNeural_FallsBackToNgram()
    {
        var ngram = new NextTokenModel();
        foreach (var line in Corpus) ngram.Observe(line);
        var neural = new TransformerLM(TinyConfig()); // never trained

        var hybrid = HybridNextToken.PredictNext(ngram, neural, "the cat", 5);
        var plain = ngram.PredictNext("the cat", 5);
        Assert.Equal(plain.Select(p => p.Token), hybrid.Select(p => p.Token));
        Assert.Equal(plain.Select(p => p.Probability), hybrid.Select(p => p.Probability));

        // Fluency likewise degrades to the n-gram score alone.
        Assert.Equal(ngram.Score("the cat sat"), HybridNextToken.Fluency(ngram, neural, "the cat sat"), 10);
    }

    [Fact]
    public void Hybrid_TrainedNeural_BlendsBothModels()
    {
        var ngram = new NextTokenModel();
        foreach (var line in Corpus) ngram.Observe(line);
        var neural = TrainedModel(epochs: 40);

        var hybrid = HybridNextToken.PredictNext(ngram, neural, "the cat", 5);
        Assert.NotEmpty(hybrid);
        Assert.InRange(hybrid.Sum(p => p.Probability), 0.99, 1.01);
        Assert.All(hybrid, p => Assert.DoesNotContain("<", p.Token));

        // A blend of two negative log-prob scores stays negative and finite.
        double fluency = HybridNextToken.Fluency(ngram, neural, "the cat sat on the mat");
        Assert.True(fluency < 0 && double.IsFinite(fluency));
    }

    // --------------------------------------------------------------
    // Mind-level wiring
    // --------------------------------------------------------------

    private static Mind BuildMind() =>
        MindBuilder.Create("DeepTest")
            .WithAllModules()
            .Build();

    [Fact]
    public void Mind_TrainDeepModel_PowersHybridPrediction()
    {
        using var mind = BuildMind();
        Assert.False(mind.NeuralLanguageModel.IsTrained);

        foreach (var line in Corpus)
            mind.Train(line);

        // The training text alone clears the auto-train threshold? No —
        // TrainDeepModel is the explicit deep pass.
        var report = mind.TrainDeepModel(40);
        Assert.Equal(40, report.Epochs);
        Assert.True(mind.NeuralLanguageModel.IsTrained);

        var predictions = mind.PredictNext("the cat", 5);
        Assert.NotEmpty(predictions);
        Assert.Contains(predictions, p => p.Token is "sat" or "ran");

        // Fluency blending is finite and negative on seen text.
        Assert.True(mind.ScoreFluency("the cat sat on the mat") < 0);
    }

    [Fact]
    public void Mind_PredictNext_BeforeDeepTraining_UsesNgram()
    {
        using var mind = BuildMind();
        foreach (var line in Corpus)
            mind.Train(line);

        // Untrained transformer: hybrid == pure n-gram.
        var fromMind = mind.PredictNext("the cat", 5);
        var fromNgram = mind.LanguageModel.PredictNext("the cat", 5);
        Assert.Equal(fromNgram.Select(p => p.Token), fromMind.Select(p => p.Token));
    }

    [Fact]
    public async Task Mind_Snapshot_KeepsTransformer()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-deep-{Guid.NewGuid():N}.db");
        try
        {
            IReadOnlyList<TokenPrediction> before;
            using (var mind = BuildMind())
            {
                foreach (var line in Corpus)
                    mind.Train(line);
                mind.TrainDeepModel(30);
                before = mind.NeuralLanguageModel.PredictNext("the cat", 3);
                await mind.SaveAsync(path);
            }

            using var restored = MindBuilder.Restore(path);
            Assert.True(restored.NeuralLanguageModel.IsTrained);
            var after = restored.NeuralLanguageModel.PredictNext("the cat", 3);
            Assert.Equal(before.Select(p => p.Token), after.Select(p => p.Token));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Exporter_IncludesTransformerTensors()
    {
        using var mind = BuildMind();
        foreach (var line in Corpus)
            mind.Train(line);
        mind.TrainDeepModel(20);

        var names = MindExporter.ExtractTensors(mind).Select(t => t.Name).ToList();
        Assert.Contains("cognify.transformer.wte", names);
        Assert.Contains("cognify.transformer.l0.attn.wq", names);
        Assert.Contains("cognify.transformer.lnf.g", names);
    }
}
