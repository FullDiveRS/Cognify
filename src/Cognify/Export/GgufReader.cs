using System.Text;

namespace Cognify.Export;

/// <summary>One tensor read back from a GGUF file.</summary>
public sealed class GgufTensorInfo
{
    /// <summary>Tensor name.</summary>
    public string Name { get; init; } = "";

    /// <summary>GGML element type id (0 = F32).</summary>
    public uint GgmlType { get; init; }

    /// <summary>Logical shape, outer dimension first (ne reversed back).</summary>
    public long[] Shape { get; init; } = Array.Empty<long>();

    /// <summary>Offset of the data relative to the tensor data section.</summary>
    public long Offset { get; init; }

    /// <summary>Payload size in bytes.</summary>
    public long ByteLength { get; init; }

    /// <summary>Total element count.</summary>
    public long ElementCount => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>A GGUF metadata value, tagged with its value type.</summary>
public sealed class GgufValue
{
    /// <summary>Creates the value.</summary>
    public GgufValue(GgufValueType type, object value) { Type = type; Value = value; }

    /// <summary>The GGUF value type.</summary>
    public GgufValueType Type { get; }

    /// <summary>The raw value (scalars boxed, arrays as List&lt;object&gt;).</summary>
    public object Value { get; }

    /// <summary>Reads the value as a string.</summary>
    public string AsString() => (string)Value;

    /// <summary>Reads the value as a u32.</summary>
    public uint AsUInt32() => Convert.ToUInt32(Value);

    /// <summary>Reads the value as an f32.</summary>
    public float AsFloat32() => Convert.ToSingle(Value);

    /// <summary>Reads the value as a bool.</summary>
    public bool AsBool() => Convert.ToBoolean(Value);

    /// <summary>Reads the value as a string array.</summary>
    public IReadOnlyList<string> AsStringArray() =>
        ((System.Collections.IEnumerable)Value).Cast<object>().Select(v => (string)v).ToList();
}

/// <summary>Raised when a GGUF file violates the spec (bad magic, misaligned
/// or out-of-range tensor data, truncated file...).</summary>
public sealed class InvalidGgufException : Exception
{
    /// <summary>Creates the exception.</summary>
    public InvalidGgufException(string message) : base(message) { }
}

/// <summary>
/// A strict GGUF v3 reader used to VERIFY what the writer produced — the same
/// checks llama.cpp performs on load: magic + version, metadata value types,
/// tensor offsets relative to the aligned data section, every tensor fully
/// inside the file, and F32 payloads that decode to the exact values written.
/// </summary>
public sealed class GgufReader
{
    private readonly byte[] _bytes;
    private readonly long _dataStart;

    /// <summary>Parses a GGUF file; throws <see cref="InvalidGgufException"/> on any spec violation.</summary>
    public GgufReader(string path)
    {
        _bytes = File.ReadAllBytes(path);
        int pos = 0;

        uint magic = ReadU32(ref pos);
        if (magic != 0x46554747u) throw new InvalidGgufException($"bad magic 0x{magic:X} (expected GGUF)");
        Version = ReadU32(ref pos);
        if (Version < 2 || Version > 3) throw new InvalidGgufException($"unsupported GGUF version {Version}");
        ulong tensorCount = ReadU64(ref pos);
        ulong kvCount = ReadU64(ref pos);

        for (ulong i = 0; i < kvCount; i++)
        {
            var key = ReadString(ref pos);
            var type = (GgufValueType)ReadU32(ref pos);
            Metadata[key] = new GgufValue(type, type switch
            {
                GgufValueType.Array => ReadArray(ref pos),
                _ => ReadScalar(type, ref pos),
            });
        }

        uint alignment = GgufWriter.DefaultAlignment;
        if (Metadata.TryGetValue("general.alignment", out var a) && a.Type == GgufValueType.UInt32)
            alignment = a.AsUInt32();
        if (alignment == 0) throw new InvalidGgufException("general.alignment is zero");

        // Tensor infos — sizes and offsets verified after all of them are read.
        long offset = 0;
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = ReadString(ref pos);
            uint nDims = ReadU32(ref pos);
            if (nDims == 0 || nDims > 4) throw new InvalidGgufException($"tensor '{name}': {nDims} dimensions");
            var ne = new long[nDims];
            long elements = 1;
            for (int d = 0; d < nDims; d++)
            {
                ne[d] = (long)ReadU64(ref pos);
                if (ne[d] <= 0) throw new InvalidGgufException($"tensor '{name}': non-positive dimension");
                elements *= ne[d];
            }
            uint type = ReadU32(ref pos);
            long rel = (long)ReadU64(ref pos);
            if (rel % alignment != 0)
                throw new InvalidGgufException($"tensor '{name}': offset {rel} not aligned to {alignment}");
            if (type != GgmlType.F32)
                throw new InvalidGgufException($"tensor '{name}': only F32 is supported by this reader (type {type})");
            long bytes = elements * 4;
            Tensors.Add(new GgufTensorInfo
            {
                Name = name,
                GgmlType = type,
                Shape = ne.Reverse().ToArray(),   // back to logical, outer first
                Offset = rel,
                ByteLength = bytes,
            });
            offset = Math.Max(offset, rel + bytes);
        }

        _dataStart = Align(pos, alignment);
        long end = _dataStart + offset;
        if (end > _bytes.Length)
            throw new InvalidGgufException(
                $"tensor data runs past the end of the file ({end} > {_bytes.Length}) — " +
                "offsets must be relative to the data section");
    }

    /// <summary>GGUF version (2 or 3).</summary>
    public uint Version { get; }

    /// <summary>All metadata key-values.</summary>
    public Dictionary<string, GgufValue> Metadata { get; } = new();

    /// <summary>All tensor infos, in file order.</summary>
    public List<GgufTensorInfo> Tensors { get; } = new();

    /// <summary>File position where the tensor data section starts.</summary>
    public long DataStart => _dataStart;

    /// <summary>Reads one F32 tensor's payload and returns it as floats.</summary>
    public float[] ReadTensor(string name)
    {
        var info = Tensors.FirstOrDefault(t => t.Name == name)
            ?? throw new InvalidGgufException($"no tensor named '{name}'");
        var floats = new float[info.ElementCount];
        long start = _dataStart + info.Offset;
        for (long i = 0; i < info.ElementCount; i++)
        {
            int p = (int)(start + i * 4);
            floats[i] = BitConverter.ToSingle(_bytes, p);
        }
        return floats;
    }

    private uint ReadU32(ref int pos)
    {
        if (pos + 4 > _bytes.Length) throw new InvalidGgufException("truncated file (u32)");
        uint v = BitConverter.ToUInt32(_bytes, pos);
        pos += 4;
        return v;
    }

    private ulong ReadU64(ref int pos)
    {
        if (pos + 8 > _bytes.Length) throw new InvalidGgufException("truncated file (u64)");
        ulong v = BitConverter.ToUInt64(_bytes, pos);
        pos += 8;
        return v;
    }

    private string ReadString(ref int pos)
    {
        ulong len = ReadU64(ref pos);
        if (pos + (long)len > _bytes.Length) throw new InvalidGgufException("truncated file (string)");
        var s = Encoding.UTF8.GetString(_bytes, pos, (int)len);
        pos += (int)len;
        return s;
    }

    private object ReadScalar(GgufValueType type, ref int pos) => type switch
    {
        GgufValueType.UInt8 => ReadByte(ref pos),
        GgufValueType.Int8 => (sbyte)ReadByte(ref pos),
        GgufValueType.UInt16 => ReadU16(ref pos),
        GgufValueType.Int16 => (short)ReadU16(ref pos),
        GgufValueType.UInt32 => ReadU32(ref pos),
        GgufValueType.Int32 => (int)ReadU32(ref pos),
        GgufValueType.Float32 => BitConverter.Int32BitsToSingle((int)ReadU32(ref pos)),
        GgufValueType.Bool => ReadByte(ref pos) != 0,
        GgufValueType.String => ReadString(ref pos),
        GgufValueType.UInt64 => ReadU64(ref pos),
        GgufValueType.Int64 => (long)ReadU64(ref pos),
        GgufValueType.Float64 => BitConverter.Int64BitsToDouble((long)ReadU64(ref pos)),
        _ => throw new InvalidGgufException($"unsupported scalar type {type}"),
    };

    private List<object> ReadArray(ref int pos)
    {
        var elemType = (GgufValueType)ReadU32(ref pos);
        ulong count = ReadU64(ref pos);
        var items = new List<object>();
        for (ulong i = 0; i < count; i++) items.Add(ReadScalar(elemType, ref pos));
        return items;
    }

    private byte ReadByte(ref int pos)
    {
        if (pos + 1 > _bytes.Length) throw new InvalidGgufException("truncated file (byte)");
        return _bytes[pos++];
    }

    private ushort ReadU16(ref int pos)
    {
        if (pos + 2 > _bytes.Length) throw new InvalidGgufException("truncated file (u16)");
        ushort v = BitConverter.ToUInt16(_bytes, pos);
        pos += 2;
        return v;
    }

    private static long Align(long value, long alignment) =>
        (value + alignment - 1) / alignment * alignment;
}
