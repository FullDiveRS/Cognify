using System.Text.Json;
using Cognify.Core;
using Cognify.Export;
using Cognify.Learning;
using Xunit;

namespace Cognify.Tests;

/// <summary>
/// Tests for optional capability toggling (MindBuilder.Without, minimal builds,
/// runtime DetachModule) and the Cognify.Export namespace (GGUF v3, safetensors,
/// tokenizer.json, llama.cpp config bundle).
/// </summary>
public class ExportTests : IDisposable
{
    private readonly string _dir;

    public ExportTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"cognify-export-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ------------------------------------------------------------------
    // Test minds
    // ------------------------------------------------------------------

    /// <summary>The exact minimal build from the feature request.</summary>
    private static Mind MinimalMind() =>
        MindBuilder.Create("ExampleAI")
            .WithThinking()
            .WithSelfLearning()
            .WithMemory()
            .Build();

    /// <summary>A rich mind with taught facts and feedback (produces non-trivial exports).</summary>
    private static Mind TrainedMind()
    {
        var mind = MindBuilder.Create("Yume")
            .Describe("A warm, curious companion AI.")
            .WithPersonality(p => p.Traits(openness: 0.9, extraversion: 0.75))
            .WithEmotions(e => e.Intensity(0.85))
            .WithReasoning()
            .WithThinking()
            .WithSelfLearning(l => l.Rate(0.12))
            .WithMemory()
            .WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness"))
            .Build();

        mind.Process(Stimulus.Text("My name is Alex.", intensity: 0.9));
        mind.Process(Stimulus.Text("Birds are animals. Animals are living things."));
        mind.Process(Stimulus.Text("The sky is blue."));
        mind.Reinforce(0.8);
        mind.Reinforce(-0.2);
        return mind;
    }

    // ------------------------------------------------------------------
    // Capability toggling
    // ------------------------------------------------------------------

    [Fact]
    public void Minimal_build_attaches_only_requested_modules()
    {
        using var mind = MinimalMind();

        Assert.True(mind.HasModule<Cognify.Thinking.ThinkingModule>());
        Assert.True(mind.HasModule<SelfLearningModule>());
        Assert.True(mind.HasModule<Memory.MemoryModule>());

        Assert.False(mind.HasModule<Cognify.Audio.AudioModule>());
        Assert.False(mind.HasModule<Cognify.Tools.ToolModule>());
        Assert.False(mind.HasModule<Cognify.Perception.VisionModule>());
        Assert.False(mind.HasModule<Cognify.Code.CodeModule>());
        Assert.False(mind.HasModule<Cognify.Emotions.EmotionModule>());
        Assert.False(mind.HasModule<Cognify.Reasoning.ReasoningModule>());
        Assert.False(mind.HasModule<Cognify.Generation.MediaGenerationModule>());
        Assert.False(mind.HasModule<Cognify.Embeddings.EmbeddingModule>());
        Assert.False(mind.HasModule<Cognify.Multilingual.MultilingualModule>());
        Assert.Equal(4, mind.Modules.Count); // thinking + learning + self-improvement + memory
    }

    [Fact]
    public void Minimal_mind_processes_stimuli_without_other_modules()
    {
        using var mind = MinimalMind();
        var reply = mind.Process("hello world");
        Assert.False(string.IsNullOrWhiteSpace(reply.Content));

        var question = mind.Ask("how are you today?");
        Assert.False(string.IsNullOrWhiteSpace(question.Content));

        mind.Reinforce(0.5); // learning works standalone
        Assert.True(mind.Module<SelfLearningModule>()!.Stats.TrainingSteps >= 0);
    }

    [Fact]
    public void Without_removes_previously_enabled_capabilities()
    {
        using var mind = MindBuilder.Create("Slim")
            .WithAllModules()
            .Without("audio", "tools", "generation", "code")
            .Build();

        Assert.False(mind.HasModule<Cognify.Audio.AudioModule>());
        Assert.False(mind.HasModule<Cognify.Tools.ToolModule>());
        Assert.False(mind.HasModule<Cognify.Generation.MediaGenerationModule>());
        Assert.False(mind.HasModule<Cognify.Code.CodeModule>());

        // Everything else stays attached.
        Assert.True(mind.HasModule<Cognify.Thinking.ThinkingModule>());
        Assert.True(mind.HasModule<Cognify.Reasoning.ReasoningModule>());
        Assert.True(mind.HasModule<Cognify.Emotions.EmotionModule>());
        Assert.True(mind.HasModule<Cognify.Perception.VisionModule>());
        Assert.Equal(14 - 4 + 1, mind.Modules.Count); // capabilities minus removed + self-improvement (rides on learning)
    }

    [Fact]
    public void Without_is_case_insensitive_and_chainable()
    {
        using var mind = MindBuilder.Create("Case")
            .WithAllModules()
            .Without("AUDIO").Without("Tools ")
            .Build();

        Assert.False(mind.HasModule<Cognify.Audio.AudioModule>());
        Assert.False(mind.HasModule<Cognify.Tools.ToolModule>());
    }

    [Fact]
    public void Without_unknown_capability_throws()
    {
        var builder = MindBuilder.Create("X").WithThinking();
        Assert.Throws<MindConfigurationException>(() => builder.Without("smell"));
    }

    [Fact]
    public void Without_empty_list_throws()
    {
        var builder = MindBuilder.Create("X").WithThinking();
        Assert.Throws<MindConfigurationException>(() => builder.Without());
    }

    [Fact]
    public void Runtime_detach_module_removes_capability()
    {
        using var mind = MindBuilder.Create("Detacher")
            .WithThinking()
            .WithMemory()
            .Build();

        Assert.True(mind.DetachModule<Memory.MemoryModule>());
        Assert.False(mind.HasModule<Memory.MemoryModule>());
        Assert.True(mind.HasModule<Cognify.Thinking.ThinkingModule>());

        // Still functional after detach.
        var reply = mind.Process("still alive?");
        Assert.False(string.IsNullOrWhiteSpace(reply.Content));
    }

    // ------------------------------------------------------------------
    // GGUF v3
    // ------------------------------------------------------------------

    [Fact]
    public void Gguf_file_has_valid_spec_header()
    {
        using var mind = TrainedMind();
        string path = Path.Combine(_dir, "yume.gguf");
        MindExporter.ExportGguf(mind, path);

        var bytes = File.ReadAllBytes(path);
        Assert.True(bytes.Length > 64);

        // Magic "GGUF" as u32 LE = 0x46554747.
        uint magic = BitConverter.ToUInt32(bytes, 0);
        Assert.Equal(0x46554747u, magic);
        Assert.Equal((byte)0x47, bytes[0]); // 'G'
        Assert.Equal((byte)0x47, bytes[1]); // 'G'
        Assert.Equal((byte)0x55, bytes[2]); // 'U'
        Assert.Equal((byte)0x46, bytes[3]); // 'F'

        uint version = BitConverter.ToUInt32(bytes, 4);
        Assert.Equal(3u, version);
    }

    [Fact]
    public void Gguf_walk_returns_all_kvs_and_tensors_with_aligned_offsets()
    {
        using var mind = TrainedMind();
        string path = Path.Combine(_dir, "yume.gguf");
        MindExporter.ExportGguf(mind, path);

        // Fully walk the file like gguf-py would.
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);

        Assert.Equal(0x46554747u, r.ReadUInt32());
        Assert.Equal(3u, r.ReadUInt32());
        ulong tensorCount = r.ReadUInt64();
        ulong kvCount = r.ReadUInt64();

        Assert.Equal(6, (int)tensorCount); // w1, b1, w2, b2, q_values, q_counts
        Assert.True(kvCount >= 10);

        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadString(r);
            var type = (GgufValueType)r.ReadUInt32();
            Assert.True(Enum.IsDefined(type), $"KV '{key}' has undefined type {type}.");
            ReadGgufValue(r, type); // must consume exactly one value
        }

        const uint alignment = 32;
        long dataStart = Align(fs.Position, alignment);
        long expected = dataStart;

        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString(r);
            uint nDims = r.ReadUInt32();
            Assert.InRange(nDims, 1u, 4u);
            long elements = 1;
            for (uint d = 0; d < nDims; d++) elements *= (long)r.ReadUInt64();
            Assert.Equal(GgmlType.F32, r.ReadUInt32());
            long offset = (long)r.ReadUInt64();

            Assert.True(offset >= expected, $"Tensor '{name}' offset {offset} overlaps previous tensor (expected >= {expected}).");
            Assert.Equal(0, offset % alignment);           // every tensor start is 32-byte aligned (GGUF spec)
            expected = offset + elements * 4;              // next free byte (before alignment)
        }

        // File must be at least as long as the last tensor's end.
        Assert.True(fs.Length >= expected);
    }

    [Fact]
    public void Gguf_tensor_values_match_live_network()
    {
        using var mind = TrainedMind();
        var net = mind.Module<SelfLearningModule>()!.Net;

        string path = Path.Combine(_dir, "yume.gguf");
        MindExporter.ExportGguf(mind, path);

        var (names, shapes, blobs) = ParseGgufTensors(path);

        int w1 = names.IndexOf("cognify.net.w1");
        Assert.True(w1 >= 0, "w1 tensor missing.");
        Assert.Equal(new long[] { net.HiddenDim, net.InputDim }, shapes[w1]);
        Assert.Equal(net.GetW1().SelectMany(r => r).Select(v => (float)v), blobs[w1]);
    }

    [Fact]
    public void Gguf_metadata_carries_mind_identity()
    {
        using var mind = TrainedMind();
        string path = Path.Combine(_dir, "yume.gguf");
        MindExporter.ExportGguf(mind, path);

        var kvs = ParseGgufMetadata(path);
        Assert.Equal("cognify", kvs["general.architecture"]);
        Assert.Equal("yume", kvs["general.name"]);
        Assert.Equal(32u, kvs["general.alignment"]);
        Assert.Contains("cognify.personality.openness", kvs.Keys);
    }

    [Fact]
    public void Gguf_export_works_for_minimal_mind()
    {
        using var mind = MinimalMind();
        string path = Path.Combine(_dir, "example.gguf");
        MindExporter.ExportGguf(mind, path);

        // Minimal mind still carries learning (thinking + learning + memory),
        // so its 4 MicroNet tensors (w1, b1, w2, b2) must be present.
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        Assert.Equal(0x46554747u, r.ReadUInt32());
        Assert.Equal(3u, r.ReadUInt32());
        Assert.Equal(4ul, r.ReadUInt64());
        Assert.True(r.ReadUInt64() > 0);
    }

    // ------------------------------------------------------------------
    // safetensors
    // ------------------------------------------------------------------

    [Fact]
    public void Safetensors_header_is_valid_and_offsets_contiguous()
    {
        using var mind = TrainedMind();
        string path = Path.Combine(_dir, "yume.safetensors");
        MindExporter.ExportSafetensors(mind, path);

        var bytes = File.ReadAllBytes(path);
        ulong headerLength = BitConverter.ToUInt64(bytes, 0);
        Assert.True(headerLength > 0);
        Assert.Equal(0ul, headerLength % 8);           // padded to multiple of 8
        Assert.True((long)headerLength < bytes.Length);

        string headerJson = System.Text.Encoding.UTF8.GetString(bytes, 8, (int)headerLength);
        using var doc = JsonDocument.Parse(headerJson);   // throws if invalid JSON
        var root = doc.RootElement;

        Assert.True(root.TryGetProperty("__metadata__", out _));
        Assert.Equal(6, root.EnumerateObject().Count(o => o.Name != "__metadata__"));

        // Offsets must be contiguous, ascending, starting at 0.
        var offsets = new List<(long Start, long End)>();
        foreach (var prop in root.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;
            var descriptor = prop.Value;
            Assert.Equal("F32", descriptor.GetProperty("dtype").GetString());
            offsets.Add((
                descriptor.GetProperty("data_offsets")[0].GetInt64(),
                descriptor.GetProperty("data_offsets")[1].GetInt64()));
        }
        offsets = offsets.OrderBy(o => o.Start).ToList();
        Assert.Equal(0, offsets[0].Start);
        for (int i = 1; i < offsets.Count; i++)
            Assert.Equal(offsets[i - 1].End, offsets[i].Start);

        // File length = 8 + header + data region.
        Assert.Equal(8 + (long)headerLength + offsets[^1].End, bytes.Length);
    }

    [Fact]
    public void Safetensors_tensor_payload_round_trips()
    {
        using var mind = TrainedMind();
        var net = mind.Module<SelfLearningModule>()!.Net;
        string path = Path.Combine(_dir, "yume.safetensors");
        MindExporter.ExportSafetensors(mind, path);

        var bytes = File.ReadAllBytes(path);
        ulong headerLength = BitConverter.ToUInt64(bytes, 0);
        using var doc = JsonDocument.Parse(System.Text.Encoding.UTF8.GetString(bytes, 8, (int)headerLength));

        var b2 = doc.RootElement.GetProperty("cognify.net.b2");
        long start = b2.GetProperty("data_offsets")[0].GetInt64();
        long end = b2.GetProperty("data_offsets")[1].GetInt64();
        int dataStart = 8 + (int)headerLength;

        var expected = net.GetB2().Select(v => (float)v).ToArray();
        int floatCount = (int)((end - start) / 4);
        var actual = new float[floatCount];
        for (int i = 0; i < actual.Length; i++)
            actual[i] = BitConverter.ToSingle(bytes, dataStart + (int)start + i * 4);
        Assert.Equal(expected, actual);
    }

    // ------------------------------------------------------------------
    // Sharded safetensors ({base}-00001-of-{N}.safetensors + index.json)
    // ------------------------------------------------------------------

    [Fact]
    public void Safetensors_split_writes_huggingface_shards_and_index()
    {
        using var mind = TrainedMind();
        var result = MindExporter.ExportSafetensors(mind, Path.Combine(_dir, "yume.safetensors"), 3);

        Assert.True(result.IsSharded);
        Assert.Equal(3, result.WrittenShards);
        Assert.Equal(3, result.ShardFiles.Count);
        Assert.Equal(6, result.TensorCount);

        // HF naming convention, 5-digit zero padded.
        string[] expectedNames =
        {
            "yume-00001-of-00003.safetensors",
            "yume-00002-of-00003.safetensors",
            "yume-00003-of-00003.safetensors"
        };
        for (int i = 0; i < expectedNames.Length; i++)
        {
            Assert.EndsWith(expectedNames[i], result.ShardFiles[i]);
            Assert.True(File.Exists(Path.Combine(_dir, expectedNames[i])), $"Missing shard {expectedNames[i]}");
        }

        // Index file follows the HF convention and carries total_size + weight_map.
        Assert.NotNull(result.IndexPath);
        Assert.Equal("yume.safetensors.index.json", Path.GetFileName(result.IndexPath));
        using var doc = JsonDocument.Parse(File.ReadAllText(result.IndexPath!));
        var root = doc.RootElement;
        Assert.Equal(result.TotalTensorBytes, root.GetProperty("metadata").GetProperty("total_size").GetInt64());

        var weightMap = root.GetProperty("weight_map");
        Assert.Equal(6, weightMap.EnumerateObject().Count());
        var shardSet = expectedNames.ToHashSet();
        foreach (var prop in weightMap.EnumerateObject())
            Assert.True(shardSet.Contains(prop.Value.GetString()!), $"weight_map points at unknown shard: {prop.Value}");
    }

    [Fact]
    public void Safetensors_shard_payloads_reassemble_to_single_file()
    {
        using var mind = TrainedMind();
        string singlePath = Path.Combine(_dir, "single.safetensors");
        MindExporter.ExportSafetensors(mind, singlePath);
        var sharded = MindExporter.ExportSafetensors(mind, Path.Combine(_dir, "shard.safetensors"), 3);

        var single = ParseSafetensors(singlePath);
        Assert.Equal(6, single.Count);

        var reassembled = new Dictionary<string, (long[] Shape, byte[] Payload)>();
        foreach (string shardPath in sharded.ShardFiles)
            foreach (var (name, entry) in ParseSafetensors(shardPath))
            {
                Assert.False(reassembled.ContainsKey(name), $"tensor {name} appears in two shards");
                reassembled[name] = entry;
            }

        Assert.Equal(single.Keys.OrderBy(k => k), reassembled.Keys.OrderBy(k => k));
        foreach (var (name, expected) in single)
        {
            Assert.Equal(expected.Shape, reassembled[name].Shape);
            Assert.Equal(expected.Payload, reassembled[name].Payload);   // byte-for-byte identical
        }
    }

    [Fact]
    public void Safetensors_split_larger_than_tensor_count_clamps()
    {
        using var mind = TrainedMind();
        // The feature-request example: split into 62 — a mind only has whole
        // tensors, so the split clamps to the tensor count (6 here).
        var result = MindExporter.ExportSafetensors(mind, Path.Combine(_dir, "yume.safetensors"), 62);

        Assert.Equal(62, result.RequestedShards);
        Assert.Equal(result.TensorCount, result.WrittenShards);
        Assert.Equal(6, result.WrittenShards);
        Assert.True(File.Exists(Path.Combine(_dir, "yume-00001-of-00006.safetensors")));
        Assert.True(File.Exists(Path.Combine(_dir, "yume-00006-of-00006.safetensors")));
        Assert.False(File.Exists(Path.Combine(_dir, "yume-00007-of-00006.safetensors")));

        // No empty shards.
        foreach (string shard in result.ShardFiles)
            Assert.True(ParseSafetensors(shard).Count > 0);
    }

    [Fact]
    public void Safetensors_split_one_writes_exact_file_without_index()
    {
        using var mind = TrainedMind();
        string path = Path.Combine(_dir, "plain.safetensors");
        var result = MindExporter.ExportSafetensors(mind, path, 1);

        Assert.False(result.IsSharded);
        Assert.Null(result.IndexPath);
        Assert.Single(result.Files);
        Assert.Equal(path, result.PrimaryFile);
        Assert.True(File.Exists(path));
        Assert.False(File.Exists(Path.Combine(_dir, "plain.safetensors.index.json")));
    }

    [Fact]
    public void Safetensors_invalid_split_throws()
    {
        using var mind = TrainedMind();
        string path = Path.Combine(_dir, "yume.safetensors");
        Assert.Throws<ArgumentOutOfRangeException>(() => MindExporter.ExportSafetensors(mind, path, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => MindExporter.ExportSafetensors(mind, path, -5));
    }

    [Fact]
    public void Safetensors_split_mind_without_tensors_throws()
    {
        using var mind = MindBuilder.Create("Bare").WithThinking().WithMemory().Build();
        string path = Path.Combine(_dir, "bare.safetensors");
        Assert.Throws<InvalidOperationException>(() => MindExporter.ExportSafetensors(mind, path, 3));

        // Unsplit export of a tensor-less mind still works (metadata-only file).
        var result = MindExporter.ExportSafetensors(mind, path, 1);
        Assert.True(File.Exists(path));
        Assert.Equal(0, result.TensorCount);
    }

    [Fact]
    public void Export_bundle_with_safetensors_split_writes_shards()
    {
        using var mind = TrainedMind();
        string outDir = Path.Combine(_dir, "bundle-sharded");
        var files = MindExporter.ExportBundle(mind, outDir, new ExportOptions { SafetensorsSplit = 3 });

        Assert.Contains("yume-00001-of-00003.safetensors", files);
        Assert.Contains("yume.safetensors.index.json", files);
        Assert.DoesNotContain("yume.safetensors", files);
        Assert.True(File.Exists(Path.Combine(outDir, "yume-00002-of-00003.safetensors")));
        Assert.True(File.Exists(Path.Combine(outDir, "yume.safetensors.index.json")));
    }

    // ------------------------------------------------------------------
    // tokenizer.json
    // ------------------------------------------------------------------

    [Fact]
    public void Tokenizer_json_follows_hf_schema_and_contains_taught_words()
    {
        using var mind = TrainedMind();
        string tokDir = Path.Combine(_dir, "tokenizer");
        int vocabSize = MindExporter.ExportTokenizer(mind, tokDir);

        Assert.True(vocabSize > TokenizerExporter.SpecialTokens.Length);
        Assert.True(File.Exists(Path.Combine(tokDir, "tokenizer.json")));
        Assert.True(File.Exists(Path.Combine(tokDir, "vocab.json")));
        Assert.True(File.Exists(Path.Combine(tokDir, "special_tokens_map.json")));

        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(tokDir, "tokenizer.json")));
        var root = doc.RootElement;

        Assert.Equal("1.0", root.GetProperty("version").GetString());
        Assert.Equal("Lowercase", root.GetProperty("normalizer").GetProperty("type").GetString());
        Assert.Equal("Whitespace", root.GetProperty("pre_tokenizer").GetProperty("type").GetString());

        var model = root.GetProperty("model");
        Assert.Equal("WordPiece", model.GetProperty("type").GetString());
        Assert.Equal("[UNK]", model.GetProperty("unk_token").GetString());

        var vocab = model.GetProperty("vocab");
        // Specials occupy ids 0..4.
        Assert.Equal(0, vocab.GetProperty("[PAD]").GetInt32());
        Assert.Equal(1, vocab.GetProperty("[UNK]").GetInt32());
        // Words from taught facts must be present ("birds", "animals", "living", "things").
        Assert.True(vocab.TryGetProperty("birds", out _), "corpus word 'birds' missing from vocab.");
        Assert.True(vocab.TryGetProperty("animals", out _));
        Assert.True(vocab.TryGetProperty("living", out _));

        // Ids unique and dense.
        var ids = vocab.EnumerateObject().Select(p => p.Value.GetInt32()).OrderBy(i => i).ToList();
        Assert.Equal(ids.Count, ids.Distinct().Count());
        Assert.Equal(0, ids[0]);
        Assert.Equal(ids.Count - 1, ids[^1]);
    }

    // ------------------------------------------------------------------
    // Bundle
    // ------------------------------------------------------------------

    [Fact]
    public void Export_bundle_contains_complete_llamacpp_toolchain()
    {
        using var mind = TrainedMind();
        string outDir = Path.Combine(_dir, "bundle");
        var files = MindExporter.ExportBundle(mind, outDir, new ExportOptions
        {
            ModelName = "Yume",
            QuantizationType = "Q4_K_M",
            Epochs = 2
        });

        Assert.Contains("yume.gguf", files);
        Assert.Contains("yume.safetensors", files);
        Assert.Contains("config.json", files);
        Assert.Contains("README.md", files);
        Assert.Contains("train_config.json", files);

        foreach (var expected in new[]
                 {
                     "yume.gguf", "yume.safetensors", "config.json", "README.md",
                     "train_config.json", "finetune.sh", "quantize.sh",
                     "training_data.txt", "val_data.txt",
                     "tokenizer/tokenizer.json", "tokenizer/vocab.json", "tokenizer/special_tokens_map.json"
                 })
            Assert.True(File.Exists(Path.Combine(outDir, expected)), $"Missing bundle file: {expected}");

        // finetune.sh references the standard llama.cpp trainer and our artifacts.
        string finetune = File.ReadAllText(Path.Combine(outDir, "finetune.sh"));
        Assert.Contains("llama-finetune", finetune);
        Assert.Contains("yume.gguf", finetune);
        Assert.Contains("--lora-out", finetune);

        // quantize.sh targets the standard quant types.
        string quantize = File.ReadAllText(Path.Combine(outDir, "quantize.sh"));
        Assert.Contains("llama-quantize", quantize);
        Assert.Contains("Q4_K_M", quantize);
        Assert.Contains("Q8_0", quantize);

        // train_config.json is valid JSON with the usual keys.
        using var cfg = JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "train_config.json")));
        Assert.Equal("yume.gguf", cfg.RootElement.GetProperty("model").GetString());
        Assert.Equal("training_data.txt", cfg.RootElement.GetProperty("train_data").GetString());
        Assert.Equal(2, cfg.RootElement.GetProperty("epochs").GetInt32());
        Assert.True(cfg.RootElement.TryGetProperty("quantization", out _));

        // Training corpus contains the taught facts (corpus keeps original casing).
        string training = File.ReadAllText(Path.Combine(outDir, "training_data.txt")).ToLowerInvariant();
        Assert.Contains("birds", training);
        Assert.Contains("animals", training);
    }

    [Fact]
    public void Export_bundle_works_for_minimal_mind()
    {
        using var mind = MinimalMind();
        string outDir = Path.Combine(_dir, "bundle-minimal");
        var files = MindExporter.ExportBundle(mind, outDir);

        Assert.True(File.Exists(Path.Combine(outDir, "exampleai.gguf")));
        Assert.True(File.Exists(Path.Combine(outDir, "exampleai.safetensors")));
        Assert.True(files.Count >= 10);
    }

    [Fact]
    public void Sanitize_produces_safe_model_names()
    {
        Assert.Equal("yume", MindExporter.Sanitize("Yume!"));
        Assert.Equal("my-ai-v2", MindExporter.Sanitize("My AI  V2"));
        Assert.Equal("cognify-model", MindExporter.Sanitize("   "));
    }

    // ------------------------------------------------------------------
    // GGUF binary-walk helpers (shared by several tests)
    // ------------------------------------------------------------------

    private static string ReadString(BinaryReader r)
    {
        ulong length = r.ReadUInt64();
        var bytes = r.ReadBytes((int)length);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static object? ReadGgufValue(BinaryReader r, GgufValueType type) => type switch
    {
        GgufValueType.UInt8 => r.ReadByte(),
        GgufValueType.Int8 => r.ReadSByte(),
        GgufValueType.UInt16 => r.ReadUInt16(),
        GgufValueType.Int16 => r.ReadInt16(),
        GgufValueType.UInt32 => r.ReadUInt32(),
        GgufValueType.Int32 => r.ReadInt32(),
        GgufValueType.Float32 => r.ReadSingle(),
        GgufValueType.Bool => r.ReadBoolean(),
        GgufValueType.String => ReadString(r),
        GgufValueType.UInt64 => r.ReadUInt64(),
        GgufValueType.Int64 => r.ReadInt64(),
        GgufValueType.Float64 => r.ReadDouble(),
        GgufValueType.Array => ReadArray(r),
        _ => throw new InvalidOperationException($"Undefined type {type}.")
    };

    private static object?[] ReadArray(BinaryReader r)
    {
        var elemType = (GgufValueType)r.ReadUInt32();
        ulong count = r.ReadUInt64();
        var items = new object?[count];
        for (ulong i = 0; i < count; i++) items[i] = ReadGgufValue(r, elemType);
        return items;
    }

    private static Dictionary<string, object?> ParseGgufMetadata(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        Assert.Equal(0x46554747u, r.ReadUInt32());
        Assert.Equal(3u, r.ReadUInt32());
        ulong tensorCount = r.ReadUInt64();
        ulong kvCount = r.ReadUInt64();

        var kvs = new Dictionary<string, object?>();
        for (ulong i = 0; i < kvCount; i++)
        {
            string key = ReadString(r);
            var type = (GgufValueType)r.ReadUInt32();
            kvs[key] = ReadGgufValue(r, type);
        }

        // Skip tensor infos so callers could keep reading data.
        const uint alignment = 32;
        long dataStart = Align(fs.Position, alignment);
        long offset = dataStart;
        for (ulong i = 0; i < tensorCount; i++)
        {
            ReadString(r);
            uint nDims = r.ReadUInt32();
            long elements = 1;
            for (uint d = 0; d < nDims; d++) elements *= (long)r.ReadUInt64();
            r.ReadUInt32(); // type
            r.ReadUInt64(); // offset
            offset += elements * 4;
            offset = Align(offset, alignment);
        }
        return kvs;
    }

    private static (List<string> Names, List<long[]> Shapes, List<float[]> Blobs) ParseGgufTensors(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        Assert.Equal(0x46554747u, r.ReadUInt32());
        Assert.Equal(3u, r.ReadUInt32());
        ulong tensorCount = r.ReadUInt64();
        ulong kvCount = r.ReadUInt64();

        for (ulong i = 0; i < kvCount; i++)
        {
            ReadString(r);
            var type = (GgufValueType)r.ReadUInt32();
            ReadGgufValue(r, type);
        }

        // Read tensor infos, keeping each tensor's STORED offset.
        var infos = new List<(string Name, long[] Shape, long Offset)>();
        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadString(r);
            uint nDims = r.ReadUInt32();
            var shape = new long[nDims];
            for (uint d = 0; d < nDims; d++) shape[d] = (long)r.ReadUInt64();
            Assert.Equal(GgmlType.F32, r.ReadUInt32());
            long offset = (long)r.ReadUInt64();
            infos.Add((name, shape, offset));
        }

        // Read each tensor's payload AT ITS STORED OFFSET — this verifies the
        // offsets actually point at the data (would catch layout bugs).
        var names = new List<string>();
        var shapes = new List<long[]>();
        var blobs = new List<float[]>();
        foreach (var (name, shape, offset) in infos)
        {
            long elements = shape.Aggregate(1L, (a, b) => a * b);
            r.BaseStream.Seek(offset, SeekOrigin.Begin);
            var data = new float[elements];
            for (long e = 0; e < elements; e++) data[e] = r.ReadSingle();
            names.Add(name);
            shapes.Add(shape);
            blobs.Add(data);
        }

        return (names, shapes, blobs);
    }

    /// <summary>
    /// Parses a safetensors file: tensor name → (shape, raw payload bytes),
    /// reading each payload at the offset the header declares.
    /// </summary>
    private static Dictionary<string, (long[] Shape, byte[] Payload)> ParseSafetensors(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        ulong headerLength = BitConverter.ToUInt64(bytes, 0);
        using var doc = JsonDocument.Parse(System.Text.Encoding.UTF8.GetString(bytes, 8, (int)headerLength));
        int dataStart = 8 + (int)headerLength;

        var result = new Dictionary<string, (long[], byte[])>();
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;
            var d = prop.Value;
            long start = d.GetProperty("data_offsets")[0].GetInt64();
            long end = d.GetProperty("data_offsets")[1].GetInt64();
            var payload = new byte[end - start];
            Array.Copy(bytes, dataStart + start, payload, 0, payload.Length);
            var shape = d.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            result[prop.Name] = (shape, payload);
        }
        return result;
    }

    private static long Align(long value, long alignment) => (value + alignment - 1) / alignment * alignment;
}
