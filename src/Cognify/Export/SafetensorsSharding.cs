namespace Cognify.Export;

/// <summary>
/// Describes the files written by a safetensors export. When the model is
/// split, shards follow the Hugging Face convention
/// (<c>{base}-00001-of-{N}.safetensors</c>) plus a HF weight-map index at
/// <c>{base}.safetensors.index.json</c> — exactly how transformers'
/// <c>save_pretrained</c> names sharded checkpoints.
/// </summary>
public sealed class SafetensorsExportResult
{
    internal SafetensorsExportResult(
        string directory,
        IReadOnlyList<string> shardFiles,
        string? indexPath,
        int tensorCount,
        int requestedShards,
        int writtenShards,
        long totalTensorBytes,
        IReadOnlyDictionary<string, string> weightMap)
    {
        Directory = directory;
        ShardFiles = shardFiles;
        IndexPath = indexPath;
        Files = indexPath is null
            ? shardFiles
            : shardFiles.Concat(new[] { indexPath }).ToList();
        TensorCount = tensorCount;
        RequestedShards = requestedShards;
        WrittenShards = writtenShards;
        TotalTensorBytes = totalTensorBytes;
        WeightMap = weightMap;
    }

    /// <summary>Directory that contains the written files.</summary>
    public string Directory { get; }

    /// <summary>Shard files in order (a single file when the model was not split).</summary>
    public IReadOnlyList<string> ShardFiles { get; }

    /// <summary>
    /// The Hugging Face index file (<c>{base}.safetensors.index.json</c>),
    /// or <c>null</c> when everything fit into a single unsplit file.
    /// </summary>
    public string? IndexPath { get; }

    /// <summary>Every file written (shards, then the index if present) — full paths.</summary>
    public IReadOnlyList<string> Files { get; }

    /// <summary>Number of tensors distributed across the shards.</summary>
    public int TensorCount { get; }

    /// <summary>The split amount the caller asked for (before clamping).</summary>
    public int RequestedShards { get; }

    /// <summary>
    /// Shards actually written. A model cannot be split into more non-empty
    /// shards than it has tensors, so this is
    /// <c>min(RequestedShards, TensorCount)</c> (never below 1).
    /// </summary>
    public int WrittenShards { get; }

    /// <summary>Sum of raw tensor payload bytes across all shards (the index's <c>total_size</c>).</summary>
    public long TotalTensorBytes { get; }

    /// <summary>HF-style weight map: tensor name → shard file name (no directory part).</summary>
    public IReadOnlyDictionary<string, string> WeightMap { get; }

    /// <summary>True when the model was written as multiple shards plus an index.</summary>
    public bool IsSharded => IndexPath is not null;

    /// <summary>First shard — convenient as "the" model file.</summary>
    public string PrimaryFile => ShardFiles[0];
}

/// <summary>
/// Naming and tensor-distribution rules for sharded safetensors exports.
/// Mirrors the Hugging Face convention used by <c>model-00001-of-00062.safetensors</c>.
/// </summary>
public static class SafetensorsSharding
{
    /// <summary>Shard file name, e.g. <c>yume-00001-of-00006.safetensors</c> (5-digit zero padding).</summary>
    public static string ShardFileName(string baseName, int zeroBasedIndex, int totalShards)
    {
        int width = Math.Max(5, Math.Abs(totalShards).ToString().Length);
        string pad = new('0', width);
        return $"{baseName}-{(zeroBasedIndex + 1).ToString(pad)}-of-{totalShards.ToString(pad)}.safetensors";
    }

    /// <summary>Index file name, e.g. <c>yume.safetensors.index.json</c>.</summary>
    public static string IndexFileName(string baseName) => $"{baseName}.safetensors.index.json";

    /// <summary>
    /// Distributes tensors across exactly <paramref name="shards"/> shards,
    /// balancing payload bytes with a largest-tensor-first greedy pass. Each
    /// shard keeps its tensors in registration order. Tensors are never split
    /// in half — a shard always holds whole tensors, so the number of shards
    /// is capped at the tensor count by the caller.
    /// </summary>
    public static List<List<TensorSpec>> Distribute(IReadOnlyList<TensorSpec> tensors, int shards)
    {
        if (shards <= 0) throw new ArgumentOutOfRangeException(nameof(shards), "At least one shard is required.");
        if (tensors.Count < shards) throw new ArgumentOutOfRangeException(nameof(shards),
            $"Cannot fill {shards} shards from {tensors.Count} tensors.");

        var buckets = new List<(int Order, TensorSpec Tensor)>[shards];
        var loads = new long[shards];
        for (int s = 0; s < shards; s++) buckets[s] = new List<(int, TensorSpec)>();

        // Largest tensors first; ties keep registration order. Assign each to
        // the currently lightest shard (ties → lowest shard index).
        var order = Enumerable.Range(0, tensors.Count)
            .OrderByDescending(i => tensors[i].ByteLength)
            .ThenBy(i => i);
        foreach (int i in order)
        {
            int target = 0;
            for (int s = 1; s < shards; s++)
                if (loads[s] < loads[target]) target = s;
            buckets[target].Add((i, tensors[i]));
            loads[target] += tensors[i].ByteLength;
        }

        // Restore registration order inside each shard.
        return buckets.Select(b => b.OrderBy(p => p.Order).Select(p => p.Tensor).ToList()).ToList();
    }
}
