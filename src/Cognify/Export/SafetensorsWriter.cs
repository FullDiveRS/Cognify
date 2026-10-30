using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace Cognify.Export;

/// <summary>
/// A spec-compliant <b>safetensors</b> writer (the Hugging Face serialization
/// format). Layout: 8-byte little-endian header length N, then N bytes of JSON
/// header (tensor descriptors + optional "__metadata__"), then the raw tensor
/// byte region. The header is padded with spaces so that N is a multiple of 8.
/// Data offsets are relative to the start of the byte region and must be
/// contiguous, non-overlapping, in ascending order — this writer enforces that.
/// </summary>
public sealed class SafetensorsWriter
{
    private readonly List<TensorSpec> _tensors = new();
    private readonly Dictionary<string, string> _metadata = new();

    /// <summary>Registers an F32 tensor.</summary>
    public void AddTensor(TensorSpec tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Data.Length != tensor.ElementCount)
            throw new ArgumentException(
                $"Tensor '{tensor.Name}': data has {tensor.Data.Length} elements but shape implies {tensor.ElementCount}.");
        _tensors.Add(tensor);
    }

    /// <summary>Adds a metadata entry serialized under "__metadata__".</summary>
    public void AddMetadata(string key, string value) => _metadata[key] = value;

    /// <summary>Number of registered tensors.</summary>
    public int TensorCount => _tensors.Count;

    /// <summary>Writes a complete safetensors file to <paramref name="path"/>.</summary>
    public void Write(string path)
    {
        // Ordered by registration; offsets must be contiguous and ascending.
        var header = new Dictionary<string, object?>();
        if (_metadata.Count > 0)
            header["__metadata__"] = _metadata;

        long offset = 0;
        foreach (var tensor in _tensors)
        {
            var descriptor = new Dictionary<string, object?>
            {
                ["dtype"] = "F32",
                ["shape"] = tensor.Shape.Select(d => (long)d).ToArray(),
                ["data_offsets"] = new[] { offset, offset + tensor.ByteLength }
            };
            header[tensor.Name] = descriptor;
            offset += tensor.ByteLength;
        }

        var jsonOptions = new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        byte[] headerBytes = JsonSerializer.SerializeToUtf8Bytes(header, jsonOptions);

        // Pad with spaces so header length is a multiple of 8.
        int padding = (8 - headerBytes.Length % 8) % 8;
        byte[] padded = new byte[headerBytes.Length + padding];
        Array.Copy(headerBytes, padded, headerBytes.Length);
        for (int i = headerBytes.Length; i < padded.Length; i++) padded[i] = (byte)' ';

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> lengthPrefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(lengthPrefix, (ulong)padded.Length);
        stream.Write(lengthPrefix);
        stream.Write(padded, 0, padded.Length);

        Span<byte> buffer = stackalloc byte[4];
        foreach (var tensor in _tensors)
        {
            foreach (float value in tensor.Data)
            {
                BinaryPrimitives.WriteSingleLittleEndian(buffer, value);
                stream.Write(buffer);
            }
        }
        stream.Flush();
    }
}
