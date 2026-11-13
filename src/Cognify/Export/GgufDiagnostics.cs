using System.Text;

namespace Cognify.Export;

/// <summary>
/// One llama.cpp-compatibility inspection of a GGUF file. Answers, BEFORE a
/// runner fails, the questions that matter:
/// <list type="bullet">
/// <item>Is the file structurally valid GGUF (spec level)?</item>
/// <item>Which tensor quantization types does it actually contain?</item>
/// <item>Can a STOCK llama.cpp build load it — and if not, why not, and what
/// should the user do next?</item>
/// </list>
/// This is the honest diagnosis for the classic load failure "the installed
/// llama.cpp cannot read this GGUF's weights (its tensor layout does not
/// match this build's quantization types)" — usually a file produced by a
/// NEWER llama.cpp, or by a publisher's own fork (e.g. dynamic quants), that
/// the local build simply does not know how to decode.
/// </summary>
public sealed class GgufDiagnosticsReport
{
    /// <summary>File that was inspected.</summary>
    public string Path { get; init; } = "";

    /// <summary>GGUF version (2 or 3); 0 when the file is not GGUF at all.</summary>
    public uint Version { get; init; }

    /// <summary><c>general.architecture</c> (empty when absent).</summary>
    public string Architecture { get; init; } = "";

    /// <summary>Data-section alignment from <c>general.alignment</c> (default 32).</summary>
    public uint Alignment { get; init; } = 32;

    /// <summary>Number of tensors in the file.</summary>
    public int TensorCount { get; init; }

    /// <summary>Number of metadata key-value pairs.</summary>
    public long MetadataCount { get; init; }

    /// <summary>Tensor quantization types actually present: (ggml type id, name, tensor count).</summary>
    public IReadOnlyList<(uint TypeId, string TypeName, int Count)> TensorTypes { get; init; }
        = Array.Empty<(uint, string, int)>();

    /// <summary>True when the file parses cleanly at the spec level.</summary>
    public bool SpecValid { get; init; }

    /// <summary>
    /// True when a stock llama.cpp build should be able to LOAD and RUN the
    /// file: valid spec, a supported architecture, conformance checks pass,
    /// and every tensor type is universally decodable (F32 / F16).
    /// </summary>
    public bool LoadableByStockLlamaCpp { get; init; }

    /// <summary>Spec-level or conformance problems found (empty when clean).</summary>
    public IReadOnlyList<string> Issues { get; init; } = Array.Empty<string>();

    /// <summary>Concrete next steps for the user (empty when nothing to say).</summary>
    public IReadOnlyList<string> Advice { get; init; } = Array.Empty<string>();

    /// <summary>One-line human verdict.</summary>
    public string Verdict { get; init; } = "";

    /// <summary>Renders the report as readable text (for logs and CLIs).</summary>
    public string Render()
    {
        var sb = new StringBuilder();
        sb.Append($"GGUF {Path}: version {Version}, architecture '{Architecture}', " +
                  $"{TensorCount} tensor(s), {MetadataCount} metadata key(s), alignment {Alignment}");
        if (TensorTypes.Count > 0)
            sb.Append(" — tensors: ").Append(string.Join(", ",
                TensorTypes.Select(t => $"{t.Count}× {t.TypeName}")));
        sb.AppendLine();
        foreach (var issue in Issues) sb.AppendLine("  [issue]  " + issue);
        foreach (var hint in Advice) sb.AppendLine("  [advice] " + hint);
        sb.Append("  verdict: ").Append(Verdict);
        return sb.ToString();
    }
}

/// <summary>
/// Inspects GGUF files for llama.cpp loadability. Tolerant on purpose: it
/// parses ANY GGUF (v2/v3, any tensor type id, including quants this build
/// cannot decode) instead of throwing — the point is to report what is inside
/// and what will happen when a runner loads it. Also carries a conformance
/// lint for the stock <c>gpt2</c> architecture (the architecture
/// <see cref="LlamaCppGgufExporter"/> writes): required hyperparameters,
/// required tensor names and shapes, tokenizer metadata.
/// </summary>
public static class GgufDiagnostics
{
    /// <summary>GGML tensor type names. Ids are stable across llama.cpp; the
    /// highest ids grow over time, so unknown ids are reported as such
    /// (that is exactly the "newer build than yours" case).</summary>
    public static string TypeName(uint typeId) => typeId switch
    {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 4 => "Q4_2 (removed)",
        5 => "Q4_3 (removed)", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K",
        15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS",
        19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S",
        23 => "IQ4_XS", 24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64",
        28 => "F64", 29 => "IQ1_M", 30 => "BF16", 31 => "Q4_0_4_4 (removed)",
        32 => "Q4_0_4_8 (removed)", 33 => "Q4_0_8_8 (removed)",
        34 => "TQ1_0", 35 => "TQ2_0", 36 => "IQ4_NL_4_4 (removed)",
        37 => "IQ4_NL_4_8 (removed)", 38 => "IQ4_NL_8_8 (removed)",
        39 => "MXFP4", 40 => "NVFP4", 41 => "Q1_0", 42 => "Q2_0", 43 => "F8_E4M3",
        _ => $"type-{typeId} (unknown here — likely a newer llama.cpp quant)",
    };

    private static readonly HashSet<uint> UniversalTypes = new() { 0, 1 };  // F32, F16

    /// <summary>Inspects a GGUF file and returns the full report. Never throws
    /// for spec problems — everything lands in <see cref="GgufDiagnosticsReport.Issues"/>.</summary>
    public static GgufDiagnosticsReport Inspect(string path)
    {
        var issues = new List<string>();
        var advice = new List<string>();

        uint version = 0;
        string arch = "";
        uint alignment = 32;
        var typeCounts = new SortedDictionary<uint, int>();
        int tensorCount = 0;
        long metadataCount = 0;
        bool specValid = true;
        Dictionary<string, (long[] Ne, uint Type, long Offset)> tensors = new();
        var metadata = new Dictionary<string, object>();

        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            int pos = 0;
            uint magic = ReadU32(bytes, ref pos);
            if (magic != 0x46554747u)
            {
                issues.Add("not a GGUF file (bad magic)");
                specValid = false;
            }
            else
            {
                version = ReadU32(bytes, ref pos);
                if (version < 2 || version > 3)
                {
                    issues.Add($"GGUF version {version} — stock builds read v2 and v3");
                    advice.Add("Re-export the file with GGUF v2/v3 (Cognify and CognifyPy both write v3).");
                    specValid = false;
                }
                ulong nTensors = ReadU64(bytes, ref pos);
                ulong nKv = ReadU64(bytes, ref pos);
                metadataCount = (long)nKv;

                for (ulong i = 0; i < nKv; i++)
                {
                    string key = ReadString(bytes, ref pos);
                    var vtype = (GgufValueType)ReadU32(bytes, ref pos);
                    object value = ReadValue(bytes, ref pos, vtype);
                    metadata[key] = value;
                }

                if (metadata.TryGetValue("general.architecture", out var archObj))
                    arch = archObj?.ToString() ?? "";
                if (metadata.TryGetValue("general.alignment", out var alignObj) &&
                    alignObj is uint a && a > 0)
                    alignment = a;

                for (ulong i = 0; i < nTensors; i++)
                {
                    string name = ReadString(bytes, ref pos);
                    uint nDims = ReadU32(bytes, ref pos);
                    if (nDims == 0 || nDims > 4)
                    {
                        issues.Add($"tensor '{name}': {nDims} dimensions (spec allows 1..4)");
                        specValid = false;
                        break;
                    }
                    var ne = new long[nDims];
                    for (int d = 0; d < nDims; d++) ne[d] = (long)ReadU64(bytes, ref pos);
                    uint type = ReadU32(bytes, ref pos);
                    long offset = (long)ReadU64(bytes, ref pos);
                    tensors[name] = (ne, type, offset);
                    typeCounts[type] = typeCounts.TryGetValue(type, out int c) ? c + 1 : 1;
                    if (offset % alignment != 0)
                    {
                        issues.Add($"tensor '{name}': offset {offset} not aligned to {alignment}");
                        specValid = false;
                    }
                    tensorCount++;
                }

                // Data section must fit: aligned end of infos + the furthest
                // tensor's data end (offset + payload size, relative to it).
                long dataStart = Align(pos, alignment);
                long lastEnd = 0;
                foreach (var (_, (ne, type, off)) in tensors)
                    lastEnd = Math.Max(lastEnd, off + TensorBytes(ne, type));
                if (dataStart + lastEnd > bytes.Length)
                {
                    issues.Add($"tensor data runs past the end of the file ({dataStart + lastEnd} > {bytes.Length})");
                    advice.Add("The file is truncated or was written with absolute instead of data-relative offsets — export it again.");
                    specValid = false;
                }
            }
        }
        catch (Exception ex)
        {
            issues.Add($"could not parse the file: {ex.Message}");
            specValid = false;
        }

        bool loadable = specValid;
        var typeList = typeCounts.Select(kv => (kv.Key, TypeName(kv.Key), kv.Value)).ToList();

        // Tensor types: F32/F16 decode everywhere; anything else depends on
        // the runner's build and is the classic "tensor layout does not match
        // this build's quantization types" failure.
        foreach (var (typeId, name, count) in typeList)
        {
            if (UniversalTypes.Contains(typeId)) continue;
            loadable = false;
            issues.Add($"{count} tensor(s) use {name} (ggml type {typeId}) — decodable only by llama.cpp builds that know this quant");
            advice.Add("This file was likely written by a newer llama.cpp, or a publisher's own fork. " +
                       "Pick another quant of the model, or use the llama.cpp build the model card names. " +
                       "If the model card says it runs on stock llama.cpp, the file may be damaged: download it again. " +
                       "Re-quantizing an F32 source with llama-quantize (Q8_0, Q5_K_M or Q4_K_M) also produces universally loadable weights.");
        }

        // Architecture: llama.cpp can only run architectures it has graphs for.
        // Cognify's archive GGUF uses the custom 'cognify' architecture on purpose.
        if (specValid && arch.Length == 0)
        {
            issues.Add("no general.architecture metadata — llama.cpp cannot pick a graph");
            loadable = false;
        }
        else if (specValid && arch != "gpt2" && arch != "cognify")
        {
            issues.Add($"architecture '{arch}' — stock llama.cpp has no graph for it");
            advice.Add("Use a standard architecture (gpt2) for runnable exports; " +
                       "Cognify's LlamaCppGgufExporter does this out of the box.");
            loadable = false;
        }
        else if (specValid && arch == "cognify")
        {
            advice.Add("This is Cognify's archive GGUF (custom 'cognify' architecture with the micro-net and RL Q-table). " +
                       "It is spec-valid and readable by gguf tooling, but llama.cpp has no graph for it — " +
                       "hand llama.cpp the '*-llamacpp.gguf' (gpt2) file from the bundle instead.");
            loadable = false; // not RUNNABLE, even though the file is fine
        }

        // gpt2 conformance lint — the runnable export must match what
        // llama.cpp's gpt2 loader expects, or the load fails later anyway.
        if (specValid && arch == "gpt2")
            LintGpt2(tensors, metadata, issues, advice, ref loadable);

        string verdict = !specValid
            ? "structurally invalid GGUF — repair or re-export"
            : loadable
                ? "spec-valid and loadable by a stock llama.cpp build"
                : arch == "cognify"
                    ? "spec-valid Cognify archive GGUF (not runnable by llama.cpp — by design)"
                    : "not loadable by a stock llama.cpp build (see issues and advice)";

        return new GgufDiagnosticsReport
        {
            Path = path,
            Version = version,
            Architecture = arch,
            Alignment = alignment,
            TensorCount = tensorCount,
            MetadataCount = metadataCount,
            TensorTypes = typeList,
            SpecValid = specValid,
            LoadableByStockLlamaCpp = loadable,
            Issues = issues,
            Advice = advice,
            Verdict = verdict,
        };
    }

    // ---- gpt2 conformance --------------------------------------------------

    private static void LintGpt2(Dictionary<string, (long[] Ne, uint Type, long Offset)> tensors,
        Dictionary<string, object> metadata, List<string> issues, List<string> advice,
        ref bool loadable)
    {
        foreach (var key in new[]
        {
            "gpt2.context_length", "gpt2.embedding_length", "gpt2.block_count",
            "gpt2.feed_forward_length", "gpt2.attention.head_count",
            "gpt2.attention.layer_norm_epsilon",
        })
            if (!metadata.ContainsKey(key))
            {
                issues.Add($"gpt2 model is missing required hyperparameter '{key}'");
                loadable = false;
            }

        foreach (var key in new[] { "tokenizer.ggml.tokens", "tokenizer.ggml.merges" })
            if (!metadata.ContainsKey(key))
            {
                issues.Add($"gpt2 model is missing tokenizer metadata '{key}'");
                loadable = false;
            }

        if (!tensors.ContainsKey("token_embd.weight"))
        {
            issues.Add("gpt2 model has no 'token_embd.weight'");
            loadable = false;
        }

        long emb = Value(metadata, "gpt2.embedding_length");
        long ctx = Value(metadata, "gpt2.context_length");
        long blocks = Value(metadata, "gpt2.block_count");
        long ffn = Value(metadata, "gpt2.feed_forward_length");

        foreach (var (name, (ne, _, _)) in tensors)
        {
            string problem = name switch
            {
                "token_embd.weight" when emb > 0 && (ne.Length != 2 || ne[0] != emb) =>
                    $"token_embd.weight ne=[{string.Join(",", ne)}] but embedding_length={emb}",
                "position_embd.weight" when ctx > 0 && (ne.Length != 2 || ne[0] != emb) =>
                    $"position_embd.weight ne=[{string.Join(",", ne)}] but embedding_length={emb}",
                "output_norm.weight" when ne.Length != 1 || ne[0] != emb =>
                    $"output_norm.weight ne=[{string.Join(",", ne)}] but embedding_length={emb}",
                _ => null,
            };
            // Per-block tensors: blk.N.attn_qkv.weight must be [emb, 3*emb], etc.
            if (problem is null && name.StartsWith("blk.", StringComparison.Ordinal))
            {
                var parts = name.Split('.');
                if (parts.Length >= 3 && long.TryParse(parts[1], out long b))
                {
                    if (b < 0 || b >= blocks)
                        problem = $"tensor '{name}' refers to block {b} but block_count={blocks}";
                    else if (emb > 0)
                        problem = parts[^2] switch
                        {
                            "attn_qkv" when ne.Length == 2 && ne[0] != emb =>
                                $"{name} ne=[{string.Join(",", ne)}] but embedding_length={emb}",
                            "ffn_up" when ne.Length == 2 && ffn > 0 && ne[0] != emb =>
                                $"{name} ne=[{string.Join(",", ne)}] but embedding_length={emb}",
                            "ffn_down" when ne.Length == 2 && ffn > 0 && ne[0] != ffn =>
                                $"{name} ne=[{string.Join(",", ne)}] but feed_forward_length={ffn}",
                            _ => null,
                        };
                }
            }
            if (problem is not null)
            {
                issues.Add(problem);
                advice.Add("Tensor shapes must agree with the gpt2 hyperparameters, or llama.cpp " +
                           "rejects the model at load time. Re-export with LlamaCppGgufExporter.");
                loadable = false;
            }
        }
    }

    private static long Value(Dictionary<string, object> metadata, string key) =>
        metadata.TryGetValue(key, out var v) && v is uint u ? u : 0;

    // ---- tolerant binary parsing ------------------------------------------

    /// <summary>Rough payload size of one tensor (exact for F32/F16/ints; for
    /// any quant type an upper bound: the densest real quant is Q8_0/Q8_1 at
    /// 1.125 bytes per element, so 1.25 B/elem + headroom never false-positives
    /// on legit quant files while still catching gross truncation).</summary>
    private static long TensorBytes(long[] ne, uint type)
    {
        long elements = 1;
        foreach (var d in ne) elements *= d;
        return type switch
        {
            0 => elements * 4,                       // F32
            1 or 30 => elements * 2,                 // F16, BF16
            24 => elements, 25 => elements * 2,      // I8, I16
            26 => elements * 4, 27 => elements * 8,  // I32, I64
            28 => elements * 8,                      // F64
            _ => (long)Math.Ceiling(elements * 1.25) + 64,   // quant upper bound
        };
    }

    private static long Align(long value, long alignment) =>
        (value + alignment - 1) / alignment * alignment;

    private static uint ReadU32(byte[] bytes, ref int pos)
    {
        uint v = BitConverter.ToUInt32(bytes, pos);
        pos += 4;
        return v;
    }

    private static ulong ReadU64(byte[] bytes, ref int pos)
    {
        ulong v = BitConverter.ToUInt64(bytes, pos);
        pos += 8;
        return v;
    }

    private static string ReadString(byte[] bytes, ref int pos)
    {
        ulong len = ReadU64(bytes, ref pos);
        string s = Encoding.UTF8.GetString(bytes, pos, (int)len);
        pos += (int)len;
        return s;
    }

    private static object ReadValue(byte[] bytes, ref int pos, GgufValueType type) => type switch
    {
        GgufValueType.UInt8 => bytes[pos++],
        GgufValueType.Int8 => (sbyte)bytes[pos++],
        GgufValueType.UInt16 => ReadU32(bytes, ref pos) & 0xFFFF,
        GgufValueType.Int16 => (short)(ReadU32(bytes, ref pos) & 0xFFFF),
        GgufValueType.UInt32 => ReadU32(bytes, ref pos),
        GgufValueType.Int32 => (int)ReadU32(bytes, ref pos),
        GgufValueType.Float32 => BitConverter.Int32BitsToSingle((int)ReadU32(bytes, ref pos)),
        GgufValueType.Bool => bytes[pos++] != 0,
        GgufValueType.String => ReadString(bytes, ref pos),
        GgufValueType.UInt64 => ReadU64(bytes, ref pos),
        GgufValueType.Int64 => (long)ReadU64(bytes, ref pos),
        GgufValueType.Float64 => BitConverter.Int64BitsToDouble((long)ReadU64(bytes, ref pos)),
        GgufValueType.Array => ReadArray(bytes, ref pos),
        _ => ReadString(bytes, ref pos),   // best effort — diagnostics never hard-fails
    };

    private static List<object> ReadArray(byte[] bytes, ref int pos)
    {
        var elemType = (GgufValueType)ReadU32(bytes, ref pos);
        ulong count = ReadU64(bytes, ref pos);
        var items = new List<object>();
        for (ulong i = 0; i < count; i++) items.Add(ReadValue(bytes, ref pos, elemType));
        return items;
    }
}
