using Cognify.Core;
using Cognify.Learning;
using System.Text;
using System.Text.Json;

namespace Cognify.Export;

/// <summary>
/// Knobs for the export bundle (hyperparameters used by the llama.cpp config set).
/// </summary>
public sealed class ExportOptions
{
    /// <summary>Model file base name. Defaults to the mind's name, sanitized.</summary>
    public string? ModelName { get; set; }

    /// <summary>Human-readable description embedded in GGUF metadata.</summary>
    public string? Description { get; set; }

    /// <summary>Maximum vocabulary size for tokenizer.json (including specials).</summary>
    public int VocabSize { get; set; } = 4096;

    /// <summary>Primary quantization type written into the config set.</summary>
    public string QuantizationType { get; set; } = "Q4_K_M";

    /// <summary>Context length for fine-tuning.</summary>
    public int ContextLength { get; set; } = 512;

    /// <summary>Batch size for fine-tuning.</summary>
    public int BatchSize { get; set; } = 8;

    /// <summary>Fine-tuning learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Number of fine-tuning epochs.</summary>
    public int Epochs { get; set; } = 3;

    /// <summary>Threads for llama.cpp training tools.</summary>
    public int Threads { get; set; } = Math.Max(1, Environment.ProcessorCount);

    /// <summary>Checkpoint cadence (batches) for fine-tuning.</summary>
    public int SaveEvery { get; set; } = 20;

    /// <summary>
    /// Number of safetensors shards to write (1 = one plain file). Values above
    /// the tensor count are clamped — shards always hold whole tensors.
    /// </summary>
    public int SafetensorsSplit { get; set; } = 1;
}

/// <summary>
/// Exports a trained <see cref="Mind"/> into standard ML interchange formats:
///
/// <list type="bullet">
/// <item><b>GGUF v3</b> — spec-compliant container with llama.cpp-style metadata
/// (<c>general.architecture</c>, <c>general.name</c>, …) and the mind's tensors
/// (<see cref="MicroNet"/> weights, RL Q-table) as F32. Inspectable with
/// <c>gguf-dump</c>, requantizable with <c>llama-quantize</c>.</item>
/// <item><b>safetensors</b> — Hugging Face container with the same tensors plus
/// metadata; loadable with <c>safetensors.numpy.load_file</c>.</item>
/// <item><b>tokenizer.json</b> — Hugging Face WordPiece tokenizer built from the
/// mind's own corpus (facts + memories), plus <c>vocab.json</c> /
/// <c>special_tokens_map.json</c>.</item>
/// <item><b>llama.cpp config set</b> — <c>train_config.json</c>, <c>finetune.sh</c>,
/// <c>quantize.sh</c>, and training/val corpora for Training, FineTuning and
/// Quantization pipelines.</item>
/// </list>
///
/// Note on semantics: Cognify minds are cognitive architectures (micro-net +
/// reinforcement + knowledge graph, plus a tiny internal transformer), not
/// LLM-scale models. The GGUF/safetensors payloads are the mind's learned
/// weights and metadata in open formats — useful as training artifacts,
/// initialization data, or archives — but a llama.cpp LLM runner cannot chat
/// with them as if they were a 7B transformer.
/// </summary>
public static class MindExporter
{
    /// <summary>Tensor names produced from the micro neural network.</summary>
    public static readonly string[] NetTensorNames =
        { "cognify.net.w1", "cognify.net.b1", "cognify.net.w2", "cognify.net.b2" };

    /// <summary>Extracts the mind's tensors (transformer tensors appear once
    /// the neural language model has been trained; the micro-net/Q-table
    /// appear when self-learning is enabled).</summary>
    public static List<TensorSpec> ExtractTensors(Mind mind)
    {
        var tensors = new List<TensorSpec>();
        var learning = mind.Module<SelfLearningModule>();
        if (learning is not null)
        {
            var net = learning.Net;
            tensors.Add(new TensorSpec(NetTensorNames[0], ToF32(net.GetW1()), net.HiddenDim, net.InputDim));
            tensors.Add(new TensorSpec(NetTensorNames[1], ToF32(net.GetB1()), net.HiddenDim));
            tensors.Add(new TensorSpec(NetTensorNames[2], ToF32(net.GetW2()), net.OutputDim, net.HiddenDim));
            tensors.Add(new TensorSpec(NetTensorNames[3], ToF32(net.GetB2()), net.OutputDim));

            var qTable = learning.SnapshotQTable();
            if (qTable.Count > 0)
            {
                var keys = qTable.Keys.OrderBy(k => k).ToList();
                int max = keys.Max();
                var q = new float[max + 1];
                var counts = new float[max + 1];
                foreach (var key in keys)
                {
                    q[key] = (float)qTable[key].Q;
                    counts[key] = qTable[key].Count;
                }
                tensors.Add(new TensorSpec("cognify.rl.q_values", q, max + 1));
                tensors.Add(new TensorSpec("cognify.rl.q_counts", counts, max + 1));
            }
        }

        // Neural language model (transformer) — embeddings, attention
        // projections, LayerNorms and FFN weights, exported when trained.
        var neural = mind.NeuralLanguageModel;
        if (neural.IsTrained)
        {
            foreach (var (name, matrix, vector) in neural.ExportTensors())
            {
                if (matrix.Length > 0)
                    tensors.Add(new TensorSpec(name, ToF32(matrix), matrix.Length, matrix[0].Length));
                else if (vector.Length > 0)
                    tensors.Add(new TensorSpec(name, ToF32(vector), vector.Length));
            }
        }

        return tensors;
    }

    /// <summary>Builds GGUF metadata (llama.cpp-style keys) describing the mind.</summary>
    public static GgufWriter BuildGguf(Mind mind, ExportOptions options, IReadOnlyList<TensorSpec>? tensors = null)
    {
        tensors ??= ExtractTensors(mind);
        var w = new GgufWriter();

        string name = Sanitize(options.ModelName ?? mind.Name);
        w.AddString("general.architecture", "cognify");
        w.AddString("general.name", name);
        w.AddString("general.description", options.Description ?? mind.Config.Description ?? string.Empty);
        w.AddUInt32("general.file_type", 0);            // 0 = ALL_F32
        w.AddUInt32("general.alignment", GgufWriter.DefaultAlignment);
        w.AddString("general.license", "mit");

        // Architecture metadata.
        w.AddUInt32("cognify.module_count", (uint)mind.Modules.Count);
        w.AddBool("cognify.has_reasoning", mind.HasModule<Reasoning.ReasoningModule>());
        w.AddBool("cognify.has_thinking", mind.HasModule<Thinking.ThinkingModule>());
        w.AddBool("cognify.has_memory", mind.HasModule<Memory.MemoryModule>());
        w.AddBool("cognify.has_emotions", mind.HasModule<Emotions.EmotionModule>());
        w.AddBool("cognify.has_learning", mind.HasModule<SelfLearningModule>());

        // Thinking budget metadata — travels with the weights so serving
        // stacks (llama.cpp / Unsloth) know this model's thinking switch,
        // its operating level and the hard cap it was built with.
        w.AddBool("cognify.thinking.enabled", mind.Config.Thinking.Enabled);
        w.AddString("cognify.thinking.level",
            mind.Config.Thinking.Level.ToString().ToLowerInvariant());
        w.AddString("cognify.thinking.ceiling",
            mind.Config.Thinking.Ceiling.ToString().ToLowerInvariant());

        var personality = mind.Module<Personality.PersonalityModule>();
        if (personality is not null)
        {
            w.AddFloat32("cognify.personality.openness", (float)personality.Traits.Openness);
            w.AddFloat32("cognify.personality.conscientiousness", (float)personality.Traits.Conscientiousness);
            w.AddFloat32("cognify.personality.extraversion", (float)personality.Traits.Extraversion);
            w.AddFloat32("cognify.personality.agreeableness", (float)personality.Traits.Agreeableness);
            w.AddFloat32("cognify.personality.neuroticism", (float)personality.Traits.Neuroticism);
        }

        var learning = mind.Module<SelfLearningModule>();
        if (learning is not null)
        {
            w.AddUInt32("cognify.net.input_dim", (uint)learning.Net.InputDim);
            w.AddUInt32("cognify.net.hidden_dim", (uint)learning.Net.HiddenDim);
            w.AddUInt32("cognify.net.output_dim", (uint)learning.Net.OutputDim);
            w.AddFloat32("cognify.net.learning_rate", (float)mind.Config.Learning.LearningRate);
            w.AddUInt32("cognify.training.steps", (uint)learning.Stats.TrainingSteps);
            w.AddFloat32("cognify.training.avg_reward", (float)learning.Stats.AverageReward);
        }

        w.AddUInt32("cognify.cycles", (uint)mind.CycleCount);

        foreach (var tensor in tensors) w.AddTensor(tensor);
        return w;
    }

    /// <summary>Exports the mind as a GGUF v3 file.</summary>
    public static void ExportGguf(Mind mind, string path, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        BuildGguf(mind, options).Write(path);
    }

    /// <summary>
    /// Exports the mind's tensors as a single Hugging Face safetensors file.
    /// </summary>
    public static SafetensorsExportResult ExportSafetensors(Mind mind, string path, ExportOptions? options = null)
        => ExportSafetensors(mind, path, 1, options);

    /// <summary>
    /// Exports the mind's tensors as Hugging Face safetensors, split into
    /// <paramref name="splitAmount"/> shards using the standard checkpoint
    /// naming convention:
    /// <list type="bullet">
    /// <item>Shards: <c>{base}-00001-of-{N}.safetensors</c>, <c>{base}-00002-of-{N}.safetensors</c>, …</item>
    /// <item>Index: <c>{base}.safetensors.index.json</c> with <c>metadata.total_size</c>
    /// and the <c>weight_map</c> from every tensor to its shard — the exact layout
    /// <c>safetensors.torch.load_file</c> and <c>transformers</c> expect.</item>
    /// </list>
    /// <paramref name="splitAmount"/> = 1 writes one plain file and no index.
    /// Because shards always hold whole tensors, the split is clamped to the
    /// tensor count (the result's <see cref="SafetensorsExportResult.RequestedShards"/>/
    /// <see cref="SafetensorsExportResult.WrittenShards"/> report both numbers).
    /// Shards are balanced by payload size (largest tensor first).
    /// </summary>
    public static SafetensorsExportResult ExportSafetensors(Mind mind, string path, int splitAmount, ExportOptions? options = null)
    {
        if (splitAmount <= 0)
            throw new ArgumentOutOfRangeException(nameof(splitAmount), splitAmount,
                "Split amount must be at least 1 (use 1 for a single unsplit file).");

        options ??= new ExportOptions();
        var tensors = ExtractTensors(mind);
        if (splitAmount > 1 && tensors.Count == 0)
            throw new InvalidOperationException(
                "Cannot split a mind with no tensors into shards — attach WithSelfLearning() so the mind has trained weights to export.");

        string fullPath = Path.GetFullPath(path);
        string directory = Path.GetDirectoryName(fullPath) ?? Environment.CurrentDirectory;
        Directory.CreateDirectory(directory);

        string fileName = Path.GetFileName(fullPath);
        string baseName = fileName.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            ? fileName[..^".safetensors".Length]
            : fileName;

        long totalBytes = tensors.Sum(t => t.ByteLength);
        var weightMap = new Dictionary<string, string>();
        string? indexPath = null;
        var shardFiles = new List<string>();
        int written;

        if (splitAmount <= 1)
        {
            // Unsplit: write exactly the file that was asked for, no index.
            var writer = new SafetensorsWriter();
            writer.AddMetadata("format", "cognify-mind");
            writer.AddMetadata("name", mind.Name);
            foreach (var tensor in tensors) writer.AddTensor(tensor);
            writer.Write(fullPath);
            foreach (var tensor in tensors) weightMap[tensor.Name] = fileName;
            shardFiles.Add(fullPath);
            written = 1;
        }
        else
        {
            written = Math.Min(splitAmount, tensors.Count);
            var shards = SafetensorsSharding.Distribute(tensors, written);
            for (int s = 0; s < written; s++)
            {
                string shardName = SafetensorsSharding.ShardFileName(baseName, s, written);
                var writer = new SafetensorsWriter();
                writer.AddMetadata("format", "cognify-mind");
                writer.AddMetadata("name", mind.Name);
                foreach (var tensor in shards[s]) writer.AddTensor(tensor);
                writer.Write(Path.Combine(directory, shardName));
                shardFiles.Add(Path.Combine(directory, shardName));
                foreach (var tensor in shards[s]) weightMap[tensor.Name] = shardName;
            }

            // HF-style weight map: {"metadata": {"total_size": N}, "weight_map": {...}}.
            var index = new Dictionary<string, object?>
            {
                ["metadata"] = new Dictionary<string, object?> { ["total_size"] = totalBytes },
                ["weight_map"] = weightMap
            };
            indexPath = Path.Combine(directory, SafetensorsSharding.IndexFileName(baseName));
            File.WriteAllText(indexPath, JsonSerializer.Serialize(index,
                new JsonSerializerOptions { WriteIndented = true }));
        }

        return new SafetensorsExportResult(directory, shardFiles, indexPath,
            tensors.Count, splitAmount, written, totalBytes, weightMap);
    }

    /// <summary>Exports tokenizer.json (+ vocab.json, special_tokens_map.json) from the mind's corpus.</summary>
    public static int ExportTokenizer(Mind mind, string directory, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        var corpus = TokenizerExporter.CollectCorpus(mind);
        var vocab = TokenizerExporter.BuildVocab(corpus, options.VocabSize);
        return TokenizerExporter.Write(directory, vocab, mind.Name);
    }

    /// <summary>
    /// One-call export: writes the full bundle into <paramref name="directory"/>
    /// and returns the list of files produced (relative paths).
    /// </summary>
    public static List<string> ExportBundle(Mind mind, string directory, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(directory);

        string name = Sanitize(options.ModelName ?? mind.Name);
        var produced = new List<string>();

        // 1. GGUF model.
        string ggufPath = Path.Combine(directory, $"{name}.gguf");
        ExportGguf(mind, ggufPath, options);
        produced.Add($"{name}.gguf");

        // 2. safetensors weights (optionally split HF-style across shards).
        var st = ExportSafetensors(mind, Path.Combine(directory, $"{name}.safetensors"),
            Math.Max(1, options.SafetensorsSplit), options);
        produced.AddRange(st.Files.Select(Path.GetFileName)!);

        // 3. Tokenizer set.
        string tokDir = Path.Combine(directory, "tokenizer");
        int vocabSize = ExportTokenizer(mind, tokDir, options);
        produced.Add("tokenizer/tokenizer.json");
        produced.Add("tokenizer/vocab.json");
        produced.Add("tokenizer/special_tokens_map.json");

        // 4. llama.cpp training / fine-tuning / quantization set.
        var corpus = TokenizerExporter.CollectCorpus(mind);
        var configFiles = LlamaCppConfigWriter.Write(directory, corpus, name, options);
        produced.AddRange(configFiles.Select(f => Path.GetFileName(f)));

        // 5. config.json describing the architecture (HF-style).
        var arch = new Dictionary<string, object?>
        {
            ["model_type"] = "cognify-mind",
            ["architecture"] = "cognify",
            ["name"] = mind.Name,
            ["modules"] = mind.Modules.Select(m => m.Name).ToList(),
            ["vocab_size"] = vocabSize,
            ["tensor_count"] = ExtractTensors(mind).Count,
            ["cycles_trained"] = mind.CycleCount,
            ["export_options"] = new Dictionary<string, object?>
            {
                ["quantization"] = options.QuantizationType,
                ["context_length"] = options.ContextLength,
                ["batch_size"] = options.BatchSize,
                ["epochs"] = options.Epochs,
                ["learning_rate"] = options.LearningRate,
                ["safetensors_shards"] = st.WrittenShards
            }
        };
        string archPath = Path.Combine(directory, "config.json");
        File.WriteAllText(archPath, JsonSerializer.Serialize(arch, new JsonSerializerOptions { WriteIndented = true }));
        produced.Add("config.json");

        // 6. README explaining the bundle.
        string stRow = st.IsSharded
            ? $"| `{name}-00001-of-{st.WrittenShards:00000}.safetensors` (+ `{name}.safetensors.index.json`) | Hugging Face SHARDED safetensors ({st.WrittenShards} shards, whole tensors, size-balanced). Load via the index weight_map with `transformers` |"
            : $"| `{name}.safetensors` | Hugging Face safetensors with the same tensors. Load: `safetensors.numpy.load_file(\"{name}.safetensors\")` |";

        string readme = $@"# {mind.Name} — exported model bundle

This bundle was produced by Cognify.MindExporter from a live cognitive mind.

## Files

| File | Purpose |
|------|---------|
| `{name}.gguf` | GGUF v3 container: mind metadata + MicroNet weights + RL Q-table (F32, aligned). Inspect: `gguf-dump {name}.gguf` |
{stRow}
| `tokenizer/tokenizer.json` | Hugging Face WordPiece tokenizer built from the mind's corpus ({vocabSize} entries) |
| `tokenizer/vocab.json` | Plain word→id vocabulary |
| `tokenizer/special_tokens_map.json` | Special token map ([PAD] [UNK] [BOS] [EOS] [MASK]) |
| `train_config.json` | llama.cpp finetune-style hyperparameter config |
| `finetune.sh` | Ready-to-run `llama-finetune` (LoRA) command |
| `quantize.sh` | `llama-quantize` pipeline → Q4_K_M / Q5_K_M / Q8_0 |
| `training_data.txt` / `val_data.txt` | Corpus harvested from the mind (~90/10 split) |
| `config.json` | Architecture summary |

## Usage notes

- The GGUF file is spec-compliant and readable by the llama.cpp tooling ecosystem
  (gguf-py, llama-quantize). Cognify minds are cognitive architectures rather than
  LLM transformers, so the GGML graphs that llama.cpp ships cannot run the model
  directly; treat these files as training/initialization/archival artifacts, e.g.
  as LoRA training seeds with `finetune.sh`, or as weight snapshots.
- Quantization: run `./quantize.sh` after building llama.cpp. Q4_K_M is the usual
  quality/size trade-off; Q8_0 is near-lossless.
";
        string readmePath = Path.Combine(directory, "README.md");
        File.WriteAllText(readmePath, readme);
        produced.Add("README.md");

        return produced;
    }

    /// <summary>Flattens a 2-D array into row-major F32.</summary>
    private static float[] ToF32(double[][] matrix)
    {
        int rows = matrix.Length, cols = rows > 0 ? matrix[0].Length : 0;
        var flat = new float[rows * cols];
        int i = 0;
        foreach (var row in matrix)
            foreach (var value in row)
                flat[i++] = (float)value;
        return flat;
    }

    /// <summary>Flattens a 1-D array into F32.</summary>
    private static float[] ToF32(double[] vector) => vector.Select(v => (float)v).ToArray();

    /// <summary>Sanitizes a mind name into a filesystem/GGUF-safe model name.</summary>
    public static string Sanitize(string name)
    {
        var chars = name.Trim().Select(c => char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '-').ToArray();
        var collapsed = new string(chars);
        while (collapsed.Contains("--")) collapsed = collapsed.Replace("--", "-");
        collapsed = collapsed.Trim('-');
        return collapsed.Length == 0 ? "cognify-model" : collapsed.ToLowerInvariant();
    }
}
