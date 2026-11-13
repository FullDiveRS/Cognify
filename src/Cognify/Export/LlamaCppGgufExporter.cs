using Cognify.Core;
using System.Text;

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

    /// <summary>Trained context length.</summary>
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
    /// Throws <see cref="InvalidOperationException"/> when the neural language
    /// model has not been trained yet.
    /// </summary>
    public static LlamaCppGgufResult Export(Mind mind, string path, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        var neural = mind.NeuralLanguageModel;
        if (!neural.IsTrained)
            throw new InvalidOperationException(
                "The neural language model is not trained yet — run training (e.g. mind.Train(...) or " +
                "train_file) before exporting for llama.cpp. The gpt2 GGUF needs real transformer weights.");

        var cfg = neural.Config;
        var vocab = neural.ExportVocab();
        var tensors = MindExporter.ExtractTensors(mind)
            .Where(t => t.Name.StartsWith(Prefix, StringComparison.Ordinal))
            .ToDictionary(t => t.Name, t => t, StringComparer.Ordinal);

        int dim = cfg.Dim;
        int layers = cfg.Layers;
        int heads = cfg.Heads;
        int context = cfg.ContextLen;

        var wte = tensors[$"{Prefix}wte"];                       // [V][D]
        var wpe = tensors[$"{Prefix}wpe"];                       // [C][D]
        int vocabSize = (int)wte.Shape[0];
        int ffn = (int)tensors[$"{Prefix}l0.ffn.w1"].Shape[0];   // F1 rows = FFN width

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

        // Identity metadata — travels with the runnable weights.
        writer.AddBool("cognify.thinking.enabled", mind.Config.Thinking.Enabled);
        writer.AddString("cognify.thinking.level",
            mind.Config.Thinking.Level.ToString().ToLowerInvariant());
        writer.AddString("cognify.thinking.ceiling",
            mind.Config.Thinking.Ceiling.ToString().ToLowerInvariant());
        writer.AddUInt32("cognify.cycles", (uint)mind.CycleCount);

        Add(new TensorSpec("token_embd.weight", wte.Data, wte.Shape));      // [V][D], tied output head
        Add(new TensorSpec("position_embd.weight", wpe.Data, wpe.Shape));   // [C][D]

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

        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath) ?? Environment.CurrentDirectory);
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
