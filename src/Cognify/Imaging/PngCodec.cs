using System.IO.Compression;
using System.Text;

namespace Cognify.Imaging;

/// <summary>
/// Pure-C# PNG codec: encodes RGB buffers and decodes 8-bit non-interlaced
/// PNGs (gray, RGB, palette, gray+alpha, RGBA) — the formats screenshots and
/// text-image tools produce. Uses <see cref="ZLibStream"/> for the zlib
/// wrapper around the IDAT payload.
/// </summary>
public static class PngCodec
{
    /// <summary>The 8-byte PNG signature.</summary>
    public static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>Encodes a buffer as an 8-bit RGB PNG.</summary>
    public static byte[] Encode(PixelBuffer image)
    {
        using var outStream = new MemoryStream();
        outStream.Write(Signature);
        WriteChunk(outStream, "IHDR", BuildIhdr(image.Width, image.Height));
        outStream.Write(Chunk("IDAT", Compress(Scanlines(image))));
        WriteChunk(outStream, "IEND", Array.Empty<byte>());
        return outStream.ToArray();
    }

    /// <summary>Decodes a PNG (or the first frame of an APNG) into RGB.</summary>
    public static PixelBuffer Decode(byte[] data)
    {
        if (data.Length < 8 || !data.AsSpan(0, 8).SequenceEqual(Signature))
            throw new ImageDecodeException("not a PNG stream");
        int width = 0, height = 0, colorType = -1, bitDepth = 0, interlace = 0;
        byte[]? palette = null;
        var idat = new List<byte>();

        int pos = 8;
        while (pos + 8 <= data.Length)
        {
            int length = (data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3];
            var type = Encoding.ASCII.GetString(data, pos + 4, 4);
            var body = new byte[length];
            if (pos + 12 + length > data.Length)
                throw new ImageDecodeException("truncated PNG chunk");
            Array.Copy(data, pos + 8, body, 0, length);

            switch (type)
            {
                case "IHDR":
                    width = ReadInt(body, 0);
                    height = ReadInt(body, 4);
                    bitDepth = body[8];
                    colorType = body[9];
                    interlace = body[12];
                    break;
                case "PLTE":
                    palette = body;
                    break;
                case "IDAT":
                    idat.AddRange(body);
                    break;
                case "IEND":
                    goto done;
            }
            pos += 12 + length;
        }
    done:
        if (width <= 0 || height <= 0)
            throw new ImageDecodeException("missing IHDR chunk");
        if (bitDepth != 8)
            throw new ImageDecodeException($"unsupported PNG bit depth {bitDepth} (only 8-bit)");
        if (interlace != 0)
            throw new ImageDecodeException("interlaced PNG is not supported");
        int channels = colorType switch
        {
            0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4,
            _ => throw new ImageDecodeException($"unsupported PNG color type {colorType}")
        };

        var raw = Decompress(idat.ToArray());
        int stride = width * channels;
        if (raw.Length < height * (stride + 1))
            throw new ImageDecodeException("truncated PNG pixel data");

        // Undo per-scanline filters.
        var lines = new byte[height][];
        var prev = new byte[stride];
        int cursor = 0;
        for (int y = 0; y < height; y++)
        {
            int filter = raw[cursor++];
            var line = new byte[stride];
            Array.Copy(raw, cursor, line, 0, stride);
            cursor += stride;
            switch (filter)
            {
                case 0: break;
                case 1: // Sub
                    for (int x = channels; x < stride; x++)
                        line[x] = (byte)(line[x] + line[x - channels]);
                    break;
                case 2: // Up
                    for (int x = 0; x < stride; x++)
                        line[x] = (byte)(line[x] + prev[x]);
                    break;
                case 3: // Average
                    for (int x = 0; x < stride; x++)
                    {
                        int left = x >= channels ? line[x - channels] : 0;
                        line[x] = (byte)(line[x] + ((left + prev[x]) >> 1));
                    }
                    break;
                case 4: // Paeth
                    for (int x = 0; x < stride; x++)
                    {
                        int a = x >= channels ? line[x - channels] : 0;
                        int b = prev[x];
                        int c = x >= channels ? prev[x - channels] : 0;
                        line[x] = (byte)(line[x] + Paeth(a, b, c));
                    }
                    break;
                default:
                    throw new ImageDecodeException($"unknown PNG filter {filter}");
            }
            lines[y] = line;
            prev = line;
        }

        var image = new PixelBuffer(width, height);
        for (int y = 0; y < height; y++)
        {
            var line = lines[y];
            for (int x = 0; x < width; x++)
            {
                byte r, g, b;
                switch (colorType)
                {
                    case 2:
                        r = line[3 * x]; g = line[3 * x + 1]; b = line[3 * x + 2];
                        break;
                    case 6:
                        r = line[4 * x]; g = line[4 * x + 1]; b = line[4 * x + 2];
                        break;
                    case 0:
                        r = g = b = line[x];
                        break;
                    case 4:
                        r = g = b = line[2 * x];
                        break;
                    default: // palette
                        int index = line[x];
                        if (palette is null || (index + 1) * 3 > palette.Length)
                            r = g = b = 0;
                        else
                        {
                            r = palette[3 * index]; g = palette[3 * index + 1]; b = palette[3 * index + 2];
                        }
                        break;
                }
                image.SetPixel(x, y, r, g, b);
            }
        }
        return image;
    }

    // ------------------------------------------------------------------

    private static byte[] BuildIhdr(int width, int height)
    {
        var ihdr = new byte[13];
        WriteInt(ihdr, 0, width);
        WriteInt(ihdr, 4, height);
        ihdr[8] = 8;  // bit depth
        ihdr[9] = 2;  // color type: truecolor RGB
        return ihdr;
    }

    private static byte[] Scanlines(PixelBuffer image)
    {
        var raw = new byte[(image.Width * 3 + 1) * image.Height];
        int o = 0;
        for (int y = 0; y < image.Height; y++)
        {
            raw[o++] = 0; // filter: None
            var src = y * image.Width * 3;
            Array.Copy(image.Rgb, src, raw, o, image.Width * 3);
            o += image.Width * 3;
        }
        return raw;
    }

    private static byte[] Compress(byte[] raw)
    {
        using var outStream = new MemoryStream();
        // ZLibStream emits the zlib wrapper (header + adler32) for us.
        using (var deflate = new ZLibStream(outStream, CompressionLevel.Fastest, leaveOpen: true))
            deflate.Write(raw);
        return outStream.ToArray();
    }

    private static byte[] Decompress(byte[] idat)
    {
        using var inStream = new MemoryStream(idat);
        using var deflate = new ZLibStream(inStream, CompressionMode.Decompress);
        using var outStream = new MemoryStream();
        deflate.CopyTo(outStream);
        return outStream.ToArray();
    }

    private static void WriteChunk(Stream stream, string type, byte[] body)
    {
        var typeBytes = Encoding.ASCII.GetBytes(type);
        stream.Write(Chunk(type, body));
    }

    private static byte[] Chunk(string type, byte[] body)
    {
        var typeBytes = Encoding.ASCII.GetBytes(type);
        var chunk = new byte[12 + body.Length];
        WriteInt(chunk, 0, body.Length);
        Array.Copy(typeBytes, 0, chunk, 4, 4);
        Array.Copy(body, 0, chunk, 8, body.Length);
        WriteInt(chunk, 8 + body.Length, Crc32(typeBytes, body));
        return chunk;
    }

    private static int ReadInt(byte[] b, int offset) =>
        (b[offset] << 24) | (b[offset + 1] << 16) | (b[offset + 2] << 8) | b[offset + 3];

    private static void WriteInt(byte[] b, int offset, int value)
    {
        b[offset] = (byte)(value >> 24);
        b[offset + 1] = (byte)(value >> 16);
        b[offset + 2] = (byte)(value >> 8);
        b[offset + 3] = (byte)value;
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        if (pb <= pc) return b;
        return c;
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }

    private static int Crc32(byte[] a, byte[] b)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (var t in new[] { a, b })
            foreach (var byteValue in t)
                crc = CrcTable[(crc ^ byteValue) & 0xFF] ^ (crc >> 8);
        return (int)(crc ^ 0xFFFFFFFFu);
    }
}
