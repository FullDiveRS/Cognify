namespace Cognify.Imaging;

/// <summary>One decoded GIF frame: its RGB pixels and render timing.</summary>
public sealed class GifFrame
{
    /// <summary>Creates a frame record.</summary>
    public GifFrame(PixelBuffer pixels, int delayCentiseconds)
    {
        Pixels = pixels;
        DelayCentiseconds = delayCentiseconds;
    }

    /// <summary>The frame's pixels.</summary>
    public PixelBuffer Pixels { get; }

    /// <summary>How long the frame is shown, in 1/100 s (GIF units).</summary>
    public int DelayCentiseconds { get; }
}

/// <summary>
/// Pure-C# GIF codec. Encoding: animated GIF89a over a fixed 216-color
/// web-safe palette (nearest-match per pixel) with the NETSCAPE looping
/// extension — plays everywhere without external libraries. Decoding: the
/// first frame of any standard GIF (global or local color table).
/// </summary>
public static class GifCodec
{
    private const int MinCodeSize = 8;           // 256-entry palette
    private const int ClearCode = 256;
    private const int EndCode = 257;

    /// <summary>True when the bytes look like a GIF.</summary>
    public static bool LooksLikeGif(byte[] data) =>
        data.Length > 6 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F';

    /// <summary>
    /// Encodes frames into a looping animated GIF. All frames must share one
    /// size; the delay is in 1/100 s per frame.
    /// </summary>
    public static byte[] EncodeAnimation(IReadOnlyList<PixelBuffer> frames, int delayCentiseconds = 5)
    {
        if (frames.Count == 0)
            throw new ArgumentException("need at least one frame", nameof(frames));
        var width = frames[0].Width;
        var height = frames[0].Height;
        if (frames.Any(f => f.Width != width || f.Height != height))
            throw new ArgumentException("all frames must share the same size", nameof(frames));

        using var gif = new MemoryStream();
        gif.Write("GIF89a"u8);
        WriteShort(gif, width);
        WriteShort(gif, height);
        gif.WriteByte(0xF7);   // GCT present, 256 entries
        gif.WriteByte(0);      // background color index
        gif.WriteByte(0);      // aspect
        gif.Write(WebPalette());
        WriteNetscapeLoop(gif);

        foreach (var frame in frames)
        {
            // Graphic Control Extension: disposal "keep", per-frame delay.
            gif.WriteByte(0x21);
            gif.WriteByte(0xF9);
            gif.WriteByte(0x04);
            gif.WriteByte(0x04);
            WriteShort(gif, Math.Clamp(delayCentiseconds, 1, 65535));
            gif.WriteByte(0);
            gif.WriteByte(0x00);
            // Image Descriptor
            gif.WriteByte(0x2C);
            WriteShort(gif, 0);
            WriteShort(gif, 0);
            WriteShort(gif, width);
            WriteShort(gif, height);
            gif.WriteByte(0x00); // no local color table, no interlace
            LzwCompressFrame(gif, frame);
        }
        gif.WriteByte(0x3B);   // trailer
        return gif.ToArray();
    }

    /// <summary>Decodes the first frame of a GIF into RGB.</summary>
    public static GifFrame DecodeFirstFrame(byte[] data)
    {
        if (!LooksLikeGif(data))
            throw new ImageDecodeException("not a GIF stream");
        int pos = 6;
        if (pos + 7 > data.Length) throw new ImageDecodeException("truncated GIF header");
        int width = data[pos] | (data[pos + 1] << 8);
        int height = data[pos + 2] | (data[pos + 3] << 8);
        int packed = data[pos + 4];
        pos += 7;
        byte[]? globalTable = null;
        if ((packed & 0x80) != 0)
        {
            int tableSize = 2 << (packed & 0x07);
            globalTable = Slice(data, pos, tableSize * 3);
            pos += tableSize * 3;
        }

        while (pos < data.Length)
        {
            int block = data[pos++];
            if (block == 0x3B)
                throw new ImageDecodeException("GIF ended before any image data");
            if (block == 0x21) // extension: skip sub-blocks (parse GCE delay)
            {
                int label = data[pos++];
                int delay = 0;
                if (label == 0xF9)
                {
                    int size = data[pos++];
                    if (size >= 4 && pos + 3 < data.Length)
                        delay = data[pos + 1] | (data[pos + 2] << 8);
                    pos += size;
                }
                pos = SkipSubBlocks(data, pos);
                if (label == 0xF9) _ = delay; // first-frame delay not needed by callers yet
                continue;
            }
            if (block == 0x2C) // image descriptor
            {
                if (pos + 9 > data.Length) throw new ImageDecodeException("truncated GIF image descriptor");
                int fx = data[pos] | (data[pos + 1] << 8);
                int fy = data[pos + 2] | (data[pos + 3] << 8);
                int fw = data[pos + 4] | (data[pos + 5] << 8);
                int fh = data[pos + 6] | (data[pos + 7] << 8);
                int fp = data[pos + 8];
                pos += 9;
                byte[]? localTable = null;
                if ((fp & 0x80) != 0)
                {
                    int tableSize = 2 << (fp & 0x07);
                    localTable = Slice(data, pos, tableSize * 3);
                    pos += tableSize * 3;
                }
                if (pos >= data.Length) throw new ImageDecodeException("truncated GIF image data");
                int minCodeSize = data[pos++];
                var indices = LzwDecompress(data, ref pos, fw * fh, minCodeSize);
                var palette = localTable ?? globalTable
                    ?? throw new ImageDecodeException("GIF has no color table");
                var frame = new PixelBuffer(fw, fh);
                for (int y = 0; y < fh; y++)
                {
                    for (int x = 0; x < fw; x++)
                    {
                        int index = indices[y * fw + x];
                        int p = index * 3;
                        if (p + 2 < palette.Length)
                            frame.SetPixel(fx + x, fy + y, palette[p], palette[p + 1], palette[p + 2]);
                    }
                }
                return new GifFrame(frame, 0);
            }
            throw new ImageDecodeException($"unknown GIF block 0x{block:X2}");
        }
        throw new ImageDecodeException("GIF ended before any image data");
    }

    // ------------------------------------------------------------------
    // LZW encode (one frame)
    // ------------------------------------------------------------------

    private static void LzwCompressFrame(Stream gif, PixelBuffer frame)
    {
        // 1. Map pixels to the fixed web palette.
        var indices = new byte[frame.Width * frame.Height];
        for (int y = 0; y < frame.Height; y++)
        {
            for (int x = 0; x < frame.Width; x++)
            {
                var (r, g, b) = frame.GetPixel(x, y);
                indices[y * frame.Width + x] =
                    (byte)(Math.Min(5, (r + 25) / 51) * 36 + Math.Min(5, (g + 25) / 51) * 6 +
                           Math.Min(5, (b + 25) / 51));
            }
        }

        // 2. LZW-compress the index stream (GIF variant, LSB-first).
        var bits = new List<byte>();
        int bitBuffer = 0, bitCount = 0;
        int codeSize = MinCodeSize + 1;
        var dictionary = new Dictionary<(int Prefix, int Byte), int>(4096);

        void Emit(int code)
        {
            bitBuffer |= code << bitCount;
            bitCount += codeSize;
            while (bitCount >= 8)
            {
                bits.Add((byte)(bitBuffer & 0xFF));
                bitBuffer >>= 8;
                bitCount -= 8;
            }
        }

        void ResetDictionary()
        {
            dictionary.Clear();
            codeSize = MinCodeSize + 1;
        }

        Emit(ClearCode);
        int prefix = indices[0];
        int next = EndCode + 1;
        for (int i = 1; i < indices.Length; i++)
        {
            int b = indices[i];
            var key = (prefix, b);
            if (dictionary.TryGetValue(key, out var existing))
            {
                prefix = existing;
                continue;
            }
            Emit(prefix);
            if (next < 4096)
            {
                dictionary[key] = next;
                next++;
                // GIF grows the code width as soon as an assigned code no
                // longer fits: decoders bump when their (one-behind) table
                // reaches (1 << codeSize), which lands exactly here.
                if (next - 1 == (1 << codeSize) && codeSize < 12)
                    codeSize++;
            }
            if (next >= 4096)
            {
                Emit(ClearCode);
                ResetDictionary();
                next = EndCode + 1;
            }
            prefix = b;
        }
        Emit(prefix);
        Emit(EndCode);
        if (bitCount > 0)
            bits.Add((byte)(bitBuffer & 0xFF));

        // 3. Pack into GIF sub-blocks.
        gif.WriteByte(MinCodeSize);
        int offset = 0;
        while (offset < bits.Count)
        {
            int take = Math.Min(255, bits.Count - offset);
            gif.WriteByte((byte)take);
            gif.Write(bits.GetRange(offset, take).ToArray());
            offset += take;
        }
        gif.WriteByte(0x00);
    }

    // ------------------------------------------------------------------
    // LZW decode
    // ------------------------------------------------------------------

    private static byte[] LzwDecompress(byte[] data, ref int pos, int pixelCount, int minCodeSize)
    {
        int codeSize = minCodeSize + 1;
        int clearCode = 1 << minCodeSize;
        int endCode = clearCode + 1;
        var table = new List<int[]>();
        void ResetTable()
        {
            table.Clear();
            for (int i = 0; i < clearCode; i++)
                table.Add([i]);
            // Reserve the CLEAR and EOI slots so the first dictionary code
            // lands at index clearCode + 2 — same numbering as the encoder.
            table.Add(Array.Empty<int>());
            table.Add(Array.Empty<int>());
            codeSize = minCodeSize + 1;
        }
        ResetTable();

        var output = new byte[pixelCount];
        int written = 0;
        int bitBuffer = 0, bitCount = 0;
        int prevCode = -1;
        int subRemaining = 0; // bytes left in the current LZW sub-block

        while (written < pixelCount)
        {
            // Refill the bit reader one byte at a time (never more than
            // codeSize-1+8 = 19 bits in the buffer, so 32-bit shifts are safe).
            while (bitCount < codeSize)
            {
                if (pos >= data.Length)
                    throw new ImageDecodeException("truncated GIF LZW data");
                if (subRemaining == 0)
                {
                    subRemaining = data[pos++];
                    if (subRemaining == 0)
                        throw new ImageDecodeException("GIF LZW stream ended early");
                    if (pos + subRemaining > data.Length)
                        throw new ImageDecodeException("truncated GIF sub-block");
                }
                bitBuffer |= data[pos++] << bitCount;
                bitCount += 8;
                subRemaining--;
            }
            int code = bitBuffer & ((1 << codeSize) - 1);
            bitBuffer >>= codeSize;
            bitCount -= codeSize;

            if (code == clearCode)
            {
                ResetTable();
                prevCode = -1;
                continue;
            }
            if (code == endCode)
                break;

            int[] entry;
            if (code < table.Count && code >= clearCode + 2)
                entry = table[code];
            else if (code < clearCode)
                entry = table[code];
            else if (prevCode >= 0) // KwKwK case: code == next
                entry = [.. table[prevCode], table[prevCode][0]];
            else
                throw new ImageDecodeException($"corrupt GIF LZW code {code}");

            foreach (var value in entry)
            {
                if (written >= pixelCount) break;
                output[written++] = (byte)value;
            }

            if (prevCode >= 0 && table.Count < 4096)
            {
                table.Add([.. table[prevCode], entry[0]]);
                if (table.Count == (1 << codeSize) && codeSize < 12)
                    codeSize++;
            }
            prevCode = code;
        }
        return output;
    }

    // ------------------------------------------------------------------
    // Shared bits
    // ------------------------------------------------------------------

    private static void WriteNetscapeLoop(Stream gif)
    {
        gif.WriteByte(0x21);
        gif.WriteByte(0xFF);
        gif.WriteByte(0x0B);
        gif.Write("NETSCAPE2.0"u8);
        gif.WriteByte(0x03);
        gif.WriteByte(0x01);
        WriteShort(gif, 0);      // loop forever
        gif.WriteByte(0x00);
    }

    private static byte[] WebPalette()
    {
        var palette = new byte[256 * 3];
        int i = 0;
        for (int r = 0; r < 6; r++)
            for (int g = 0; g < 6; g++)
                for (int b = 0; b < 6; b++)
                {
                    palette[i++] = (byte)(r * 51);
                    palette[i++] = (byte)(g * 51);
                    palette[i++] = (byte)(b * 51);
                }
        // Pad the last 40 slots with black (already zero).
        return palette;
    }

    private static void WriteShort(Stream s, int value)
    {
        s.WriteByte((byte)value);
        s.WriteByte((byte)(value >> 8));
    }

    private static int SkipSubBlocks(byte[] data, int pos)
    {
        while (pos < data.Length)
        {
            int length = data[pos++];
            if (length == 0) break;
            pos += length;
        }
        return pos;
    }

    private static byte[] Slice(byte[] data, int start, int length)
    {
        var slice = new byte[length];
        if (start + length > data.Length)
            throw new ImageDecodeException("truncated GIF color table");
        Array.Copy(data, start, slice, 0, length);
        return slice;
    }
}
