using System.Text;

namespace Cognify.Export;

/// <summary>A named F32 tensor to embed in GGUF / safetensors files.</summary>
public sealed class TensorSpec
{
    /// <summary>Creates a tensor. Dimensions are row-major (GGML convention).</summary>
    public TensorSpec(string name, float[] data, params long[] shape)
    {
        Name = name;
        Data = data;
        Shape = shape;
    }

    /// <summary>Tensor name, e.g. <c>cognify.net.w1</c>.</summary>
    public string Name { get; }

    /// <summary>Flat row-major payload.</summary>
    public float[] Data { get; }

    /// <summary>Dimensions, e.g. [hidden, input] for a 2-D weight matrix.</summary>
    public long[] Shape { get; }

    /// <summary>Total element count.</summary>
    public long ElementCount => Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>Size of the raw payload in bytes (F32 = 4 bytes per element).</summary>
    public long ByteLength => ElementCount * 4;
}

/// <summary>
/// GGUF value types (GGUF spec v3, section "Value types").
/// </summary>
public enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5,
    Float32 = 6, Bool = 7, String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12
}

/// <summary>GGML tensor element types (subset — the writer emits F32 only).</summary>
public static class GgmlType
{
    /// <summary>32-bit IEEE float — GGML type id 0.</summary>
    public const uint F32 = 0;
}

/// <summary>
/// A spec-compliant <b>GGUF v3</b> writer (little-endian) following the format
/// defined by llama.cpp's gguf spec:
/// <list type="bullet">
/// <item>Magic <c>GGUF</c> (u32), version (u32) = 3, tensor_count (u64), kv_count (u64).</item>
/// <item>Key-value metadata: each entry is a string key + value-type (u32) + value.</item>
/// <item>Tensor infos: name, n_dims (u32), shape (u64 each), ggml type (u32), offset (u64).</item>
/// <item>Tensor data region aligned to <c>general.alignment</c> (default 32 bytes).</item>
/// </list>
/// The produced files can be inspected with <c>gguf-dump</c> (gguf-py) and
/// requantized with <c>llama-quantize</c>.
/// </summary>
public sealed class GgufWriter
{
    private readonly Dictionary<string, (GgufValueType Type, object Value)> _kv = new();
    private readonly List<TensorSpec> _tensors = new();

    /// <summary>Default data alignment required by the spec when written to metadata.</summary>
    public const uint DefaultAlignment = 32;

    /// <summary>Adds a metadata key-value pair. Duplicate keys overwrite.</summary>
    public void AddValue(string key, GgufValueType type, object value) => _kv[key] = (type, value);

    /// <summary>Adds a string metadata pair.</summary>
    public void AddString(string key, string value) => AddValue(key, GgufValueType.String, value ?? string.Empty);

    /// <summary>Adds a u32 metadata pair.</summary>
    public void AddUInt32(string key, uint value) => AddValue(key, GgufValueType.UInt32, value);

    /// <summary>Adds an f32 metadata pair.</summary>
    public void AddFloat32(string key, float value) => AddValue(key, GgufValueType.Float32, value);

    /// <summary>Adds a bool metadata pair.</summary>
    public void AddBool(string key, bool value) => AddValue(key, GgufValueType.Bool, value);

    /// <summary>Adds a string-array metadata pair.</summary>
    public void AddStringArray(string key, IReadOnlyList<string> values) =>
        AddValue(key, GgufValueType.Array, values?.ToList() ?? new List<string>());

    /// <summary>Registers an F32 tensor. Tensors are written in registration order.</summary>
    public void AddTensor(TensorSpec tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Data.Length != tensor.ElementCount)
            throw new ArgumentException(
                $"Tensor '{tensor.Name}': data has {tensor.Data.Length} elements but shape implies {tensor.ElementCount}.");
        _tensors.Add(tensor);
    }

    /// <summary>Number of registered tensors.</summary>
    public int TensorCount => _tensors.Count;

    /// <summary>Number of metadata entries.</summary>
    public int MetadataCount => _kv.Count;

    /// <summary>Writes a complete GGUF v3 file to <paramref name="path"/>.</summary>
    public void Write(string path)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);

        uint alignment = DefaultAlignment;
        if (_kv.TryGetValue("general.alignment", out var alignEntry) &&
            alignEntry.Type == GgufValueType.UInt32 && alignEntry.Value is uint a)
            alignment = a;

        // ---- Header -------------------------------------------------------
        w.Write(0x46554747u);          // magic "GGUF"
        w.Write(3u);                   // version
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)_kv.Count);

        // ---- Key-value metadata ------------------------------------------
        foreach (var (key, (type, value)) in _kv)
        {
            WriteString(w, key);
            w.Write((uint)type);
            switch (type)
            {
                case GgufValueType.UInt8: w.Write(Convert.ToByte(value)); break;
                case GgufValueType.Int8: w.Write(Convert.ToSByte(value)); break;
                case GgufValueType.UInt16: w.Write(Convert.ToUInt16(value)); break;
                case GgufValueType.Int16: w.Write(Convert.ToInt16(value)); break;
                case GgufValueType.UInt32: w.Write(Convert.ToUInt32(value)); break;
                case GgufValueType.Int32: w.Write(Convert.ToInt32(value)); break;
                case GgufValueType.Float32: w.Write(Convert.ToSingle(value)); break;
                case GgufValueType.Bool: w.Write(Convert.ToBoolean(value)); break;
                case GgufValueType.String: WriteString(w, (string)value); break;
                case GgufValueType.UInt64: w.Write(Convert.ToUInt64(value)); break;
                case GgufValueType.Int64: w.Write(Convert.ToInt64(value)); break;
                case GgufValueType.Float64: w.Write(Convert.ToDouble(value)); break;
                case GgufValueType.Array:
                {
                    var items = ((System.Collections.IEnumerable)value).Cast<object>().ToList();
                    var elemType = items.Count == 0 ? GgufValueType.String : ValueTypeOf(items[0]);
                    w.Write((uint)elemType);
                    w.Write((ulong)items.Count);
                    foreach (var item in items) WriteScalar(w, elemType, item);
                    break;
                }
                default:
                    throw new InvalidOperationException($"Unsupported GGUF value type {type}.");
            }
        }

        // ---- Tensor infos with pre-computed aligned offsets ---------------
        // The data region starts AFTER the whole info block, so its size must
        // be computed upfront (name len + n_dims + shape + type + offset).
        long infoSize = 0;
        foreach (var tensor in _tensors)
        {
            infoSize += 8 + Encoding.UTF8.GetByteCount(tensor.Name); // name
            infoSize += 4;                                           // n_dims
            infoSize += 8 * tensor.Shape.Length;                     // shape
            infoSize += 4;                                           // ggml type
            infoSize += 8;                                           // offset
        }
        long dataStart = Align(w.BaseStream.Position + infoSize, alignment);
        long offset = dataStart;
        foreach (var tensor in _tensors)
        {
            WriteString(w, tensor.Name);
            w.Write((uint)tensor.Shape.Length);
            foreach (var dim in tensor.Shape) w.Write((ulong)dim);
            w.Write(GgmlType.F32);
            w.Write((ulong)offset);
            offset = Align(offset + tensor.ByteLength, alignment);
        }

        // ---- Padding to alignment, then tensor data -----------------------
        long afterInfos = w.BaseStream.Position;
        WritePadding(w, afterInfos, alignment);

        foreach (var tensor in _tensors)
        {
            long current = w.BaseStream.Position;
            WritePadding(w, current, alignment);
            foreach (float value in tensor.Data) w.Write(value);
        }

        w.Flush();
    }

    private static GgufValueType ValueTypeOf(object value) => value switch
    {
        byte => GgufValueType.UInt8,
        sbyte => GgufValueType.Int8,
        ushort => GgufValueType.UInt16,
        short => GgufValueType.Int16,
        uint => GgufValueType.UInt32,
        int => GgufValueType.Int32,
        float => GgufValueType.Float32,
        bool => GgufValueType.Bool,
        string => GgufValueType.String,
        ulong => GgufValueType.UInt64,
        long => GgufValueType.Int64,
        double => GgufValueType.Float64,
        _ => throw new InvalidOperationException($"Unsupported array element type {value.GetType()}.")
    };

    private static void WriteScalar(BinaryWriter w, GgufValueType type, object value)
    {
        switch (type)
        {
            case GgufValueType.UInt8: w.Write(Convert.ToByte(value)); break;
            case GgufValueType.Int8: w.Write(Convert.ToSByte(value)); break;
            case GgufValueType.UInt16: w.Write(Convert.ToUInt16(value)); break;
            case GgufValueType.Int16: w.Write(Convert.ToInt16(value)); break;
            case GgufValueType.UInt32: w.Write(Convert.ToUInt32(value)); break;
            case GgufValueType.Int32: w.Write(Convert.ToInt32(value)); break;
            case GgufValueType.Float32: w.Write(Convert.ToSingle(value)); break;
            case GgufValueType.Bool: w.Write(Convert.ToBoolean(value)); break;
            case GgufValueType.String: WriteString(w, (string)value); break;
            case GgufValueType.UInt64: w.Write(Convert.ToUInt64(value)); break;
            case GgufValueType.Int64: w.Write(Convert.ToInt64(value)); break;
            case GgufValueType.Float64: w.Write(Convert.ToDouble(value)); break;
        }
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static long Align(long value, long alignment) =>
        (value + alignment - 1) / alignment * alignment;

    private static void WritePadding(BinaryWriter w, long current, long alignment)
    {
        long remainder = current % alignment;
        if (remainder == 0) return;
        for (long i = 0; i < alignment - remainder; i++) w.Write((byte)0);
    }
}
