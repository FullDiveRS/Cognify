using Cognify.Core;
using System.Text;
using System.Text.Json;

namespace Cognify.Export;

/// <summary>Result of a llama.cpp-runnable GGUF export.</summary>
public sealed class LlamaCppGgufResult
{
    /// <summary>Creates the result.</summary>
    public LlamaCppGgufResult(string path, int tensorCount, int vocabSize,
        int dim, int heads, int layers, int contextLength, int feedForwardLength, long fileLength)
    {
        Path = path; TensorCount = tensorCount; VocabSize = vocabSize;
        Dim = dim; Heads = heads; Layers = layers;
        ContextLength = contextLength; FeedForwardLength = feedForwardLength;
        FileLength = fileLength;
    }

    /// <summary>GGUF file that was written.</summary>
    public string Path { get; }

    /// <summary>Number of tensors written.</summary>
    public int TensorCount { get; }

    /// <summary>Vocabulary size (== token embedding rows).</summary>
    public int VocabSize { get; }

    /// <summary>Embedding dimension.</summary>
    public int Dim { get; }

    /// <summary>Attention head count.</summary>
    public int Heads { get; }

    /// <summary>Transformer block count.</summary>
    public int Layers { get; }

    /// <summary>Context length written to the GGUF (the trained window, or
    /// the position-interpolated extension from ExportOptions).</summary>
    public int ContextLength { get; }

    /// <summary>FFN hidden dimension.</summary>
    public int FeedForwardLength { get; }

    /// <summary>File size in bytes.</summary>
    public long FileLength { get; }
}

/// <summary>
/// Exports the mind's neural language model (a GPT-style transformer: token +
/// learned positional embeddings, pre-LayerNorm blocks with GELU FFN, weight-tied
/// output head) as a GGUF file that <b>llama.cpp can actually load and run</b> —
/// the stock <c>gpt2</c> architecture:
/// <list type="bullet">
/// <item>Tensors use llama.cpp's gpt2 names (<c>token_embd.weight</c>,
/// <c>position_embd.weight</c>, <c>blk.N.attn_norm</c>, <c>blk.N.attn_qkv</c>
/// (the separate Q/K/V projections fused exactly as GPT-2's <c>c_attn</c>),
/// <c>blk.N.attn_output</c>, <c>blk.N.ffn_up</c>, <c>blk.N.ffn_down</c>,
/// <c>output_norm</c>); the output head is weight-tied (omitted → llama.cpp
/// duplicates <c>token_embd</c>), matching how the model was trained.</item>
/// <item>Mandatory hyperparameters: <c>gpt2.context_length</c>,
/// <c>gpt2.embedding_length</c>, <c>gpt2.block_count</c>,
/// <c>gpt2.feed_forward_length</c>, <c>gpt2.attention.head_count</c>,
/// <c>gpt2.attention.layer_norm_epsilon</c> (1e-5, the trained epsilon).</item>
/// <item>A full <c>tokenizer.ggml</c> set: tokens (byte-unicode encoded, Ġ for
/// spaces — the GPT-2 convention llama.cpp and transformers both expect),
/// per-token types (specials as CONTROL/UNKNOWN), BOS/EOS/UNK/PAD ids, and BPE
/// merges that re-assemble every vocabulary word from its characters so prompt
/// encoding reaches the word ids the model was trained with.</item>
/// </list>
/// Honest caveats: the FFN activation is the tanh GELU approximation while
/// llama.cpp evaluates exact GELU (a sub-0.003 delta per activation), and the
/// word-level vocabulary means out-of-vocabulary words fall back to &lt;unk&gt;.
/// The mind's cognitive metadata (thinking switch, personality, …) travels in
/// <c>cognify.*</c> keys — unknown keys are ignored by llama.cpp.
/// </summary>
public static class LlamaCppGgufExporter
{
    private const string Prefix = "cognify.transformer.";

    /// <summary>GGML token types (llama.cpp's llama_token_type values, i32).</summary>
    private const int TokenNormal = 1, TokenUnknown = 2, TokenControl = 3;

    /// <summary>
    /// Exports the trained transformer as a llama.cpp-runnable gpt2 GGUF.
    /// <paramref name="path"/> is the <c>.gguf</c> FILE to write; an existing
    /// directory target writes <c>&lt;name&gt;-llamacpp.gguf</c> inside it,
    /// and missing parent folders are created.
    /// Throws <see cref="InvalidOperationException"/> when the neural language
    /// model has not been trained yet.
    /// </summary>
    public static LlamaCppGgufResult Export(Mind mind, string path, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        var neural = mind.NeuralLanguageModel;
        if (!neural.IsTrained)
            throw new InvalidOperationException(
                "The neural language model is not trained yet — call mind.TrainDeepModel(epochs) explicitly " +
                "before exporting for llama.cpp (train_file only feeds the training buffer; the automatic pass " +
                "fires by itself only after ~" + Cognify.Learning.TransformerLM.AutoTrainTokens + " new tokens). " +
                "The gpt2 GGUF needs real transformer weights.");

        var cfg = neural.Config;
        var vocab = neural.ExportVocab();
        var tensors = MindExporter.ExtractTensors(mind)
            .Where(t => t.Name.StartsWith(Prefix, StringComparison.Ordinal))
            .ToDictionary(t => t.Name, t => t, StringComparer.Ordinal);

        int dim = cfg.Dim;
        int layers = cfg.Layers;
        int heads = cfg.Heads;

        var wte = tensors[$"{Prefix}wte"];                       // [V][D]
        var wpe = tensors[$"{Prefix}wpe"];                       // [C][D]
        int vocabSize = (int)wte.Shape[0];
        int ffn = (int)tensors[$"{Prefix}l0.ffn.w1"].Shape[0];   // F1 rows = FFN width

        // Context extension (opt-in): the transformer trained with learned
        // absolute position embeddings for cfg.ContextLen tokens, so that
        // window is baked into the weights. A longer GGUF window is possible
        // by linearly interpolating (pi-style) the trained rows onto the
        // requested range — llama.cpp then accepts -c up to the new length.
        int context = (int)wpe.Shape[0];
        float[] wpeData = wpe.Data;
        long[] wpeShape = wpe.Shape;
        if (options.LlamaCppContextLength is int wanted && wanted != context)
        {
            if (wanted < context)
                throw new ArgumentOutOfRangeException(nameof(options),
                    $"LlamaCppContextLength ({wanted}) cannot be below the trained context ({context}).");
            wpeData = InterpolatePositions(wpe.Data, context, wanted, dim);
            wpeShape = new long[] { wanted, dim };
            context = wanted;
        }

        // Every tensor we write, kept for the post-write self-verification.
        var writer = new GgufWriter();
        var written = new List<TensorSpec>();
        void Add(TensorSpec spec)
        {
            writer.AddTensor(spec);
            written.Add(spec);
        }

        string name = MindExporter.Sanitize(options.ModelName ?? mind.Name);
        writer.AddString("general.architecture", "gpt2");
        writer.AddString("general.name", name);
        writer.AddString("general.description", options.Description ??
            $"Cognify mind '{mind.Name}' neural language model, exported for llama.cpp");
        writer.AddUInt32("general.file_type", 0);                // ALL_F32
        writer.AddUInt32("general.alignment", GgufWriter.DefaultAlignment);
        writer.AddUInt32("general.quantization_version", 2);
        writer.AddString("general.license", "mit");

        writer.AddUInt32("gpt2.context_length", (uint)context);
        writer.AddUInt32("gpt2.embedding_length", (uint)dim);
        writer.AddUInt32("gpt2.block_count", (uint)layers);
        writer.AddUInt32("gpt2.feed_forward_length", (uint)ffn);
        writer.AddUInt32("gpt2.attention.head_count", (uint)heads);
        writer.AddFloat32("gpt2.attention.layer_norm_epsilon", 1e-5f);

        WriteTokenizer(writer, vocab);

        // Capability metadata — what the mind carries, tagged on the weights.
        WriteCapabilityMetadata(writer, mind);

        // Identity metadata — travels with the runnable weights. Level names
        // follow llama.cpp's --reasoning-effort ladder (low, medium, high,
        // xhigh, max) so serving stacks can map them 1:1.
        writer.AddBool("cognify.thinking.enabled",
            mind.Modules.Any(m => m.Name == "thinking"));
        writer.AddString("cognify.thinking.level",
            EffortName(mind.Config.Thinking.Level));
        writer.AddString("cognify.thinking.ceiling",
            mind.Config.Thinking.Ceiling.ToString().ToLowerInvariant());
        writer.AddUInt32("cognify.thinking.budget",
            (uint)mind.Config.Thinking.TokensFor(mind.Config.Thinking.Level));
        writer.AddBool("cognify.reasoning.enabled",
            mind.Modules.Any(m => m.Name == "reasoning") && mind.Config.Reasoning.Enabled);
        writer.AddString("cognify.reasoning.level",
            EffortName(mind.Config.Reasoning.Level));
        writer.AddString("cognify.reasoning.ceiling",
            mind.Config.Reasoning.Ceiling.ToString().ToLowerInvariant());
        writer.AddUInt32("cognify.reasoning.budget",
            (uint)mind.Config.Reasoning.TokensFor(mind.Config.Reasoning.Level));

        // Recommended sampling profile — sized to the vocabulary. Tiny
        // word-level models degenerate into "what you is what you" loops at
        // hot temperatures with no repetition penalty, and have no learned
        // sense of when to stop, so the profile caps the reply and tames
        // the sampling. Serving stacks can read these keys instead of
        // guessing (Unsloth: set the same values in its sampling panel).
        var sampling = SamplingFor(vocabSize);
        writer.AddFloat32("cognify.sampling.temperature", sampling.Temperature);
        writer.AddFloat32("cognify.sampling.top_p", sampling.TopP);
        writer.AddUInt32("cognify.sampling.top_k", sampling.TopK);
        writer.AddFloat32("cognify.sampling.repeat_penalty", sampling.RepeatPenalty);
        writer.AddUInt32("cognify.sampling.max_tokens", sampling.MaxTokens);
        writer.AddUInt32("cognify.cycles", (uint)mind.CycleCount);

        Add(new TensorSpec("token_embd.weight", wte.Data, wte.Shape));      // [V][D], tied output head
        Add(new TensorSpec("position_embd.weight", wpeData, wpeShape));   // [C][D], interpolated when requested

        for (int l = 0; l < layers; l++)
        {
            string p = $"{Prefix}l{l}.";
            Add(new TensorSpec($"blk.{l}.attn_norm.weight", tensors[$"{p}ln1.g"].Data, dim));
            Add(new TensorSpec($"blk.{l}.attn_norm.bias", tensors[$"{p}ln1.b"].Data, dim));
            Add(FuseQkv($"blk.{l}.attn_qkv.weight",
                tensors[$"{p}attn.wq"], tensors[$"{p}attn.wk"], tensors[$"{p}attn.wv"]));
            Add(FuseQkvBias($"blk.{l}.attn_qkv.bias",
                tensors[$"{p}attn.bq"], tensors[$"{p}attn.bk"], tensors[$"{p}attn.bv"]));
            Add(new TensorSpec($"blk.{l}.attn_output.weight", tensors[$"{p}attn.wo"].Data,
                tensors[$"{p}attn.wo"].Shape));
            Add(new TensorSpec($"blk.{l}.attn_output.bias", tensors[$"{p}attn.bo"].Data,
                tensors[$"{p}attn.bo"].Shape));
            Add(new TensorSpec($"blk.{l}.ffn_norm.weight", tensors[$"{p}ln2.g"].Data, dim));
            Add(new TensorSpec($"blk.{l}.ffn_norm.bias", tensors[$"{p}ln2.b"].Data, dim));
            Add(new TensorSpec($"blk.{l}.ffn_up.weight", tensors[$"{p}ffn.w1"].Data,
                tensors[$"{p}ffn.w1"].Shape));                    // [ff, dim]
            Add(new TensorSpec($"blk.{l}.ffn_up.bias", tensors[$"{p}ffn.b1"].Data, ffn));
            Add(new TensorSpec($"blk.{l}.ffn_down.weight", tensors[$"{p}ffn.w2"].Data,
                tensors[$"{p}ffn.w2"].Shape));                    // [dim, ff]
            Add(new TensorSpec($"blk.{l}.ffn_down.bias", tensors[$"{p}ffn.b2"].Data, dim));
        }

        Add(new TensorSpec("output_norm.weight", tensors[$"{Prefix}lnf.g"].Data, dim));
        Add(new TensorSpec("output_norm.bias", tensors[$"{Prefix}lnf.b"].Data, dim));

        string fullPath = MindExporter.ResolveOutputFile(path,
            MindExporter.Sanitize(options.ModelName ?? mind.Name) + "-llamacpp", ".gguf");
        writer.Write(fullPath);

        // ---- post-write self-verification ---------------------------------
        // 1. Full GGUF round-trip: re-read the file and compare every tensor
        //    value against what we just wrote. A corrupt or misplaced payload
        //    (wrong offsets, bad alignment) fails HERE, not inside llama.cpp.
        var reader = new GgufReader(fullPath);
        foreach (var spec in written)
        {
            float[] back = reader.ReadTensor(spec.Name);
            if (back.Length != spec.Data.Length)
                throw new InvalidOperationException(
                    $"export self-check failed: '{spec.Name}' read back {back.Length} values (wrote {spec.Data.Length})");
            for (int i = 0; i < back.Length; i++)
                if (back[i] != spec.Data[i])
                    throw new InvalidOperationException(
                        $"export self-check failed: '{spec.Name}' value {i} read back {back[i]}, wrote {spec.Data[i]}");
        }

        // 2. llama.cpp conformance: hyperparameters, tensor names and shapes,
        //    tokenizer metadata — the exact checks the stock gpt2 loader does.
        var report = GgufDiagnostics.Inspect(fullPath);
        if (!report.LoadableByStockLlamaCpp || report.Issues.Count > 0)
            throw new InvalidOperationException(
                "export self-check failed: the produced GGUF would not load in stock llama.cpp — " +
                report.Render());

        int tensorCount = written.Count;
        if (tensorCount != 2 + layers * 12 + 2)
            throw new InvalidOperationException(
                $"export self-check failed: wrote {tensorCount} tensors, expected {2 + layers * 12 + 2}");
        return new LlamaCppGgufResult(fullPath, tensorCount, vocabSize, dim, heads,
            layers, context, ffn, new FileInfo(fullPath).Length);
    }

    /// <summary>
    /// Resamples the trained absolute position embeddings ([srcCtx][dim],
    /// row-major) onto [dstCtx][dim] by linear interpolation (pi-style):
    /// new position i reads trained position i*(src-1)/(dst-1), so the
    /// trained window stretches over the extended context and both end rows
    /// are preserved exactly.
    /// </summary>
    private static float[] InterpolatePositions(float[] trained, int srcCtx, int dstCtx, int dim)
    {
        var result = new float[dstCtx * dim];
        for (int i = 0; i < dstCtx; i++)
        {
            double s = (double)i * (srcCtx - 1) / (dstCtx - 1);
            int i0 = (int)s;
            int i1 = Math.Min(i0 + 1, srcCtx - 1);
            double t = s - i0;
            int b0 = i0 * dim, b1 = i1 * dim, b = i * dim;
            for (int d = 0; d < dim; d++)
                result[b + d] = (float)((1 - t) * trained[b0 + d] + t * trained[b1 + d]);
        }
        return result;
    }

    private static TensorSpec FuseQkv(string name, TensorSpec wq, TensorSpec wk, TensorSpec wv)
    {
        int rows = (int)wq.Shape[0], cols = (int)wq.Shape[1];
        var fused = new float[rows * 3 * cols];
        int o = 0;
        foreach (var part in new[] { wq, wk, wv })
        {
            if (part.Shape.Length != 2 || part.Shape[0] != rows || part.Shape[1] != cols)
                throw new InvalidOperationException($"cannot fuse QKV: shape mismatch on '{part.Name}'");
            Array.Copy(part.Data, 0, fused, o, part.Data.Length);
            o += part.Data.Length;
        }
        return new TensorSpec(name, fused, rows * 3, cols);
    }

    private static TensorSpec FuseQkvBias(string name, TensorSpec bq, TensorSpec bk, TensorSpec bv)
    {
        int n = (int)bq.Shape[0];
        var fused = new float[n * 3];
        int o = 0;
        foreach (var part in new[] { bq, bk, bv })
        {
            Array.Copy(part.Data, 0, fused, o, part.Data.Length);
            o += part.Data.Length;
        }
        return new TensorSpec(name, fused, n * 3);
    }

    /// <summary>
    /// Capability metadata — mirrors the MindBuilder switches. The registered
    /// module list IS the capability set (WithAudio/WithCode/WithTools/…
    /// attach modules at build time); llama.cpp ignores unknown keys, but any
    /// harness reading the file can see what the mind carries. Alongside it,
    /// <c>tokenizer.chat_template</c> gives llama-server <c>--jinja</c> /
    /// Unsloth a working chat format — the one capability key llama.cpp
    /// actually honors at request time.
    /// </summary>
    private static void WriteCapabilityMetadata(GgufWriter writer, Mind mind)
    {
        var modules = mind.Modules.Select(m => m.Name).ToList();
        bool Has(string module) => modules.Contains(module);

        var thinking = new Dictionary<string, object?>
        {
            // Module presence is the truth — a mind built without WithThinking
            // has no thinking module regardless of option defaults.
            ["enabled"] = Has("thinking"),
            ["level"] = EffortName(mind.Config.Thinking.Level),
            ["ceiling"] = mind.Config.Thinking.Ceiling.ToString().ToLowerInvariant(),
            // The effort ladder this mind implements (lowest→highest, the
            // level names llama.cpp's --reasoning-effort expects) and the
            // token budget each level buys — clients can implement the
            // levels directly from /props.
            ["levels"] = EffortLevels(mind.Config.Thinking.Ceiling),
            ["budget"] = mind.Config.Thinking.TokensFor(mind.Config.Thinking.Level),
            ["budgets"] = BudgetLadder(mind.Config.Thinking.Ceiling,
                mind.Config.Thinking.TokensFor),
        };
        var reasoning = new Dictionary<string, object?>
        {
            ["enabled"] = Has("reasoning") && mind.Config.Reasoning.Enabled,
            ["level"] = EffortName(mind.Config.Reasoning.Level),
            ["ceiling"] = mind.Config.Reasoning.Ceiling.ToString().ToLowerInvariant(),
            ["levels"] = EffortLevels(mind.Config.Reasoning.Ceiling),
            ["budget"] = mind.Config.Reasoning.TokensFor(mind.Config.Reasoning.Level),
            ["budgets"] = BudgetLadder(mind.Config.Reasoning.Ceiling,
                mind.Config.Reasoning.TokensFor),
        };
        var capabilities = new Dictionary<string, object?>
        {
            ["reasoning"] = Has("reasoning"),
            ["memory"] = Has("memory"),
            ["emotions"] = Has("emotions"),
            ["personality"] = Has("personality"),
            ["goals"] = Has("goals"),
            ["audio"] = Has("audio"),
            ["vision"] = Has("vision"),
            ["code"] = Has("code"),
            ["embeddings"] = Has("embeddings"),
            ["multilingual"] = Has("multilingual"),
            ["self_learning"] = Has("learning"),
            ["media_generation"] = Has("generation"),
            ["tools"] = Has("tools"),
            ["web_search"] = mind.Config.Tools.AllowWebSearch,
            ["web_fetch"] = mind.Config.Tools.AutoFetchWeb,
        };
        var toolNames = mind.Tools?.Registry.All().Select(t => t.Name).ToList()
                        ?? new List<string>();

        writer.AddStringArray("cognify.modules", modules);
        foreach (var (cap, on) in capabilities)
            writer.AddBool($"cognify.{cap}.enabled", Convert.ToBoolean(on));
        if (toolNames.Count > 0)
        {
            writer.AddUInt32("cognify.tools.count", (uint)toolNames.Count);
            writer.AddStringArray("cognify.tools.names", toolNames);
        }
        if (!string.IsNullOrEmpty(mind.Config.Description))
            writer.AddString("cognify.description", mind.Config.Description);

        // The manifest also rides inside the chat template as a Jinja `set`
        // (renders no output): llama-server exposes the template via
        // GET /props, so any client — cognify-chat.html, an agent harness,
        // Unsloth-style UIs — can read what this mind carries and gray out
        // what is missing. Single source of truth with the KV keys above.
        var manifest = JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["version"] = 1,
            ["description"] = mind.Config.Description ?? string.Empty,
            ["thinking"] = thinking,
            ["reasoning"] = reasoning,
            ["capabilities"] = capabilities,
            ["tools"] = new Dictionary<string, object?>
            {
                ["enabled"] = Convert.ToBoolean(capabilities["tools"]),
                ["count"] = toolNames.Count,
                ["names"] = toolNames,
            },
        });
        // The template itself is GENERATED from the mind's live state (see
        // Mind.GenerateChatTemplate) and self-verified against the built-in
        // engine before it ships: a template that cannot render the mind's
        // own format never reaches a GGUF.
        var template = mind.GenerateChatTemplate(manifest.Replace("'", "\\u0027"));
        ChatTemplateGenerator.VerifyExportedTemplate(template,
            Convert.ToBoolean(thinking["enabled"]));
        writer.AddString("cognify.chat_template.generator", ChatTemplateMeta.Generator);
        writer.AddString("tokenizer.chat_template", template);
    }

    // ---- effort ladders (llama.cpp-aligned) ---------------------------------

    /// <summary>
    /// The level name llama.cpp's <c>--reasoning-effort</c> ladder uses for
    /// this level: low, medium, high, xhigh (VeryHigh), max. Cognify's
    /// thinking and reasoning levels map 1:1 onto it, so a client can pass
    /// the exported level straight to <c>--reasoning-effort</c>.
    /// </summary>
    internal static string EffortName(ThinkingLevel level) => level switch
    {
        ThinkingLevel.Low => "low",
        ThinkingLevel.Medium => "medium",
        ThinkingLevel.High => "high",
        ThinkingLevel.VeryHigh => "xhigh",
        _ => "max"
    };

    /// <summary>
    /// The effort levels this mind implements, lowest→highest, cut at the
    /// ceiling — the same shape as llama.cpp's
    /// <c>chat_template_caps.reasoning_effort_levels</c> in GET /props.
    /// </summary>
    internal static List<string> EffortLevels(ThinkingCeiling ceiling)
    {
        var levels = new List<string>();
        foreach (ThinkingLevel level in Enum.GetValues<ThinkingLevel>())
        {
            if (level > ceiling.MaxLevel()) break;
            levels.Add(EffortName(level));
        }
        return levels;
    }

    /// <summary>
    /// The token budget each implemented level buys — the ladder a client
    /// needs to reproduce the mind's effort levels with
    /// <c>--reasoning-budget</c> semantics (N>0 = token cap per cycle).
    /// </summary>
    internal static Dictionary<string, int> BudgetLadder(ThinkingCeiling ceiling,
        Func<ThinkingLevel, int> tokensFor)
    {
        var ladder = new Dictionary<string, int>();
        foreach (ThinkingLevel level in Enum.GetValues<ThinkingLevel>())
        {
            if (level > ceiling.MaxLevel()) break;
            ladder[EffortName(level)] = tokensFor(level);
        }
        return ladder;
    }

    /// <summary>
    /// The level name llama.cpp's <c>--reasoning-effort</c> ladder uses for
    /// this <see cref="ReasoningLevel"/> — identical rungs to the thinking
    /// mapping (low, medium, high, xhigh for VeryHigh, max).
    /// </summary>
    internal static string EffortName(ReasoningLevel level) => level switch
    {
        ReasoningLevel.Low => "low",
        ReasoningLevel.Medium => "medium",
        ReasoningLevel.High => "high",
        ReasoningLevel.VeryHigh => "xhigh",
        _ => "max"
    };

    /// <summary>
    /// The reasoning effort levels this mind implements, lowest→highest, cut
    /// at the reasoning ceiling — llama.cpp's
    /// <c>chat_template_caps.reasoning_effort_levels</c> shape.
    /// </summary>
    internal static List<string> EffortLevels(ReasoningCeiling ceiling)
    {
        var levels = new List<string>();
        foreach (ReasoningLevel level in Enum.GetValues<ReasoningLevel>())
        {
            if (level > ceiling.MaxLevel()) break;
            levels.Add(EffortName(level));
        }
        return levels;
    }

    /// <summary>
    /// The reasoning token budget each implemented level buys — the ladder a
    /// client needs to reproduce the mind's effort levels with
    /// <c>--reasoning-budget</c> semantics (N>0 = token cap per cycle).
    /// </summary>
    internal static Dictionary<string, int> BudgetLadder(ReasoningCeiling ceiling,
        Func<ReasoningLevel, int> tokensFor)
    {
        var ladder = new Dictionary<string, int>();
        foreach (ReasoningLevel level in Enum.GetValues<ReasoningLevel>())
        {
            if (level > ceiling.MaxLevel()) break;
            ladder[EffortName(level)] = tokensFor(level);
        }
        return ladder;
    }

    /// <summary>A serving-stack sampling profile shipped as GGUF metadata.</summary>
    internal sealed record SamplingProfile(
        float Temperature, float TopP, uint TopK, float RepeatPenalty, uint MaxTokens);

    /// <summary>
    /// The sampling settings this model needs to produce clean output —
    /// sized to the vocabulary, because a word-level model trained on a
    /// small corpus degenerates into "what you is what you" loops at hot
    /// temperatures with no repetition penalty, and has no learned sense of
    /// when to stop:
    /// <list type="bullet">
    /// <item>&lt; 1k vocab pieces (greeting / toy scale): temp 0.7, repeat penalty 1.20, 128 max tokens.</item>
    /// <item>&lt; 32k pieces (small chat model): temp 0.7, repeat penalty 1.15, 256 max tokens.</item>
    /// <item>anything bigger: temp 0.8, repeat penalty 1.10, 512 max tokens.</item>
    /// </list>
    /// Top-k 40 / top-p 0.9 everywhere — wide enough for variety, narrow
    /// enough to keep the sampling out of the long tail. Emitted as
    /// <c>cognify.sampling.*</c> GGUF metadata so a serving stack can adopt
    /// the profile instead of guessing.
    /// </summary>
    internal static SamplingProfile SamplingFor(int vocabSize) => vocabSize switch
    {
        < 1_000 => new SamplingProfile(0.7f, 0.9f, 40, 1.20f, 128),
        < 32_768 => new SamplingProfile(0.7f, 0.9f, 40, 1.15f, 256),
        _ => new SamplingProfile(0.8f, 0.9f, 40, 1.10f, 512)
    };

    // ---- tokenizer ---------------------------------------------------------

    /// <summary>
    /// Writes the tokenizer.ggml metadata set: tokens in GPT-2 byte-unicode
    /// form (spaces become Ġ — the convention llama.cpp and transformers both
    /// decode), per-token types, special-token ids, and BPE merges that
    /// re-assemble every vocabulary word character by character so prompt
    /// encoding reaches the word ids the model was trained with.
    /// </summary>
    private static void WriteTokenizer(GgufWriter writer, IReadOnlyList<string> vocab)
    {
        var pieces = new List<string>(vocab.Count);
        var types = new List<int>(vocab.Count);
        for (int id = 0; id < vocab.Count; id++)
        {
            var word = vocab[id];
            bool special = word is "<pad>" or "<unk>" or "<bos>" or "<eos>";
            pieces.Add(special ? word : ByteEncode(word, leadingSpace: true));
            types.Add(!special ? TokenNormal : id == 1 ? TokenUnknown : TokenControl);
        }

        writer.AddString("tokenizer.ggml.model", "gpt2");
        writer.AddString("tokenizer.ggml.pre", "default");
        writer.AddValue("tokenizer.ggml.tokens", GgufValueType.Array, pieces);
        writer.AddValue("tokenizer.ggml.token_type", GgufValueType.Array, types);
        writer.AddValue("tokenizer.ggml.merges", GgufValueType.Array, BuildBpeMerges(vocab, pieces));
        writer.AddUInt32("tokenizer.ggml.bos_token_id", 2);
        writer.AddUInt32("tokenizer.ggml.eos_token_id", 3);
        writer.AddUInt32("tokenizer.ggml.unknown_token_id", 1);
        writer.AddUInt32("tokenizer.ggml.padding_token_id", 0);
        writer.AddBool("tokenizer.ggml.add_bos_token", false);
        writer.AddBool("tokenizer.ggml.add_eos_token", false);
    }

    /// <summary>
    /// BPE merge ranks that rebuild every vocabulary piece from single
    /// characters: for each piece the chain ("Ġ","h"), ("Ġh","e"), … merges
    /// the accumulated prefix with the next character, so BPE re-assembles
    /// the whole piece and the tokenizer looks up the exact word id (the
    /// merged symbol is immediately re-paired — repeated letters assemble
    /// correctly). Pairs are deduplicated (first word wins; distinct words
    /// only ever diverge at distinct next characters) and ranks follow word
    /// order.
    /// </summary>
    internal static List<string> BuildBpeMerges(IReadOnlyList<string> vocab, IReadOnlyList<string> pieces)
    {
        var ranks = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var piece in pieces)
        {
            var chars = new List<string>(piece.Select(c => c.ToString()));
            int i = 0;
            // No i++ after a merge — see doc.
            while (i + 1 < chars.Count)
            {
                var entry = chars[i] + " " + chars[i + 1];
                if (seen.Add(entry)) ranks.Add(entry);
                chars[i] = chars[i] + chars[i + 1];
                chars.RemoveAt(i + 1);
            }
        }
        return ranks;
    }

    /// <summary>GPT-2 byte-unicode encoder: UTF-8 bytes → the printable
    /// characters llama.cpp expects; spaces become Ġ (U+0120).
    /// <paramref name="leadingSpace"/> prepends one synthetic Ġ for vocabulary
    /// words stored without spaces (skipped when the text already starts
    /// with one).</summary>
    internal static string ByteEncode(string text, bool leadingSpace)
    {
        var map = ByteMap();
        if (leadingSpace && !text.StartsWith(" ", StringComparison.Ordinal))
            text = " " + text;
        var source = Encoding.UTF8.GetBytes(text);
        var sb = new StringBuilder(source.Length);
        foreach (var b in source) sb.Append(map[b]);
        return sb.ToString();
    }

    private static string[]? _byteMap;

    private static string[] ByteMap()
    {
        if (_byteMap is not null) return _byteMap;
        var map = new string[256];
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            map[b] = printable ? ((char)b).ToString() : new string((char)(256 + n++), 1);
        }
        _byteMap = map;
        return map;
    }
}
