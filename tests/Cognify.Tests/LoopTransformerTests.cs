using Cognify.Core;
using Cognify.Export;
using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>
/// Tests for the Loop Transformer (<see cref="TransformerConfig.Loop"/>):
/// one block stack applied N times per pass — weight-tied recurrent depth
/// (Universal Transformer / ALBERT style). Looping adds NO parameters,
/// trains like a deeper stack with tied weights, round-trips through
/// snapshots, and the llama.cpp export unrolls it into layers × loop
/// physical blocks so the exported GGUF stays loadable by stock llama.cpp.
/// </summary>
public class LoopTransformerTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), $"cognify-loop-{Guid.NewGuid():N}");

    public LoopTransformerTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, true);
    }

    private static TransformerConfig Base() => new()
    {
        Dim = 8, Heads = 2, Layers = 1, ContextLen = 6, FfnMult = 2,
        Dropout = 0.0, LearningRate = 0.02, Seed = 7, BatchSize = 8,
        MaxWindowsPerEpoch = 64,
    };

    private static readonly string[] Corpus =
    [
        "the cat sat on the mat", "the dog sat on the rug",
        "the cat ran to the mat", "the dog ran to the rug",
        "the cat sat on the mat", "the dog sat on the rug",
    ];

    [Fact]
    public void Loop_Rejects_Out_Of_Range_Values()
    {
        Assert.Throws<ArgumentException>(() => new TransformerLM(Base() with { Loop = 0 }));
        Assert.Throws<ArgumentException>(() => new TransformerLM(Base() with { Loop = 9 }));
        // Default stays the classic architecture.
        Assert.Equal(1, new TransformerLM().Config.Loop);
    }

    [Fact]
    public void Looping_Adds_No_Parameters_And_Trains()
    {
        var plain = new TransformerLM(Base());
        var looped = new TransformerLM(Base() with { Loop = 2 });
        foreach (var line in Corpus) { plain.Observe(line); looped.Observe(line); }

        var p1 = plain.Train(20);
        var p2 = looped.Train(20);
        Assert.Equal(p1.Parameters, p2.Parameters);   // weight tying: same parameter count
        Assert.True(p2.FinalLoss < p2.InitialLoss);   // and looping learns
        Assert.True(looped.IsTrained);

        // Same behavior guarantees as the classic model: known words only.
        foreach (var line in Corpus)
            foreach (var pred in looped.PredictNext(line, 5))
                Assert.DoesNotContain('<', pred.Token);
        var reply = looped.Generate("the cat", 6);
        Assert.True(reply.Length > 0);
        foreach (var word in reply.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            Assert.DoesNotContain('<', word);
    }

    [Fact]
    public void Loop_RoundTrips_Through_Snapshots()
    {
        var looped = new TransformerLM(Base() with { Loop = 3 });
        foreach (var line in Corpus) looped.Observe(line);
        looped.Train(10);

        var restored = new TransformerLM(Base() with { Loop = 3 });
        Assert.True(restored.TryRestore(looped.Serialize()));
        Assert.Equal(3, restored.Config.Loop);
        Assert.True(restored.IsTrained);
        Assert.NotEqual("", restored.Generate("the cat", 6));

        // Architecture mismatch: a one-pass model refuses a looped snapshot.
        Assert.False(new TransformerLM(Base()).TryRestore(looped.Serialize()));
    }

    [Fact]
    public void Builder_WithTransformer_Configures_The_Minds_Model()
    {
        using var mind = MindBuilder.Create("Looped")
            .WithMemory()
            .WithTransformer(c => c with { Loop = 2, Layers = 1, Dim = 16, Heads = 2, ContextLen = 12 })
            .Build();
        Assert.Equal(2, mind.NeuralLanguageModel.Config.Loop);
        Assert.Equal(1, mind.NeuralLanguageModel.Config.Layers);

        var corpus = Path.Combine(_dir, "corpus.txt");
        File.WriteAllLines(corpus, Corpus);
        mind.TrainFile(corpus);
        mind.TrainDeepModel(1);
        Assert.True(mind.NeuralLanguageModel.IsTrained);
    }

    [Fact]
    public void LlamaCpp_Export_Unrolls_The_Loop()
    {
        using var mind = MindBuilder.Create("Looped")
            .WithMemory()
            .WithTransformer(c => c with { Loop = 2, Layers = 1, Dim = 16, Heads = 2, ContextLen = 12 })
            .Build();
        var corpus = Path.Combine(_dir, "corpus.txt");
        File.WriteAllLines(corpus, Corpus);
        mind.TrainFile(corpus);
        mind.TrainDeepModel(1);

        var result = MindExporter.ExportGgufForLlamaCpp(mind, Path.Combine(_dir, "looped.gguf"));
        Assert.Equal(2, result.Layers);   // physical blocks: 1 logical × 2 loop

        var gguf = new GgufReader(Path.Combine(_dir, "looped.gguf"));
        Assert.Equal((uint)2, gguf.Metadata["gpt2.block_count"].AsUInt32());
        Assert.Equal((uint)1, gguf.Metadata["cognify.transformer.layers"].AsUInt32());
        Assert.Equal((uint)2, gguf.Metadata["cognify.transformer.loop"].AsUInt32());

        // Unrolled weights: physical layer 1 carries logical block 0's tensors.
        var blk0 = gguf.ReadTensor("blk.0.attn_norm.weight");
        var blk1 = gguf.ReadTensor("blk.1.attn_norm.weight");
        Assert.Equal(blk0.Length, blk1.Length);
        for (int i = 0; i < blk0.Length; i++) Assert.Equal(blk0[i], blk1[i]);
    }
}
