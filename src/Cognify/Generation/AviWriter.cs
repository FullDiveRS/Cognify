using Cognify.Imaging;
using System.Text;

namespace Cognify.Generation;

/// <summary>
/// Writes uncompressed 24-bit AVI video (RIFF container with '00db' DIB
/// frames plus an index). Plays in Windows Media Player, VLC and ffplay —
/// a dependency-free "real file you can double-click" video format. File
/// sizes are honest (uncompressed): keep frames small.
/// </summary>
public static class AviWriter
{
    /// <summary>Encodes same-size RGB frames into an AVI byte stream.</summary>
    public static byte[] Encode(IReadOnlyList<PixelBuffer> frames, int framesPerSecond = 10)
    {
        if (frames.Count == 0)
            throw new ArgumentException("need at least one frame", nameof(frames));
        int width = frames[0].Width;
        int height = frames[0].Height;
        if (frames.Any(f => f.Width != width || f.Height != height))
            throw new ArgumentException("all frames must share the same size", nameof(frames));
        if (width % 4 != 0)
            throw new ArgumentException("AVI width must be a multiple of 4 (uncompressed DIB rows)", nameof(frames));

        var frameBytes = frames.Select(FrameData).ToList();
        int frameSize = frameBytes[0].Length;
        int moviSize = frameBytes.Sum(f => f.Length + 8);
        int usecPerFrame = 1_000_000 / Math.Clamp(framesPerSecond, 1, 60);

        // Exact byte accounting: RIFF size = everything after the size field.
        int hdrlSize = 4                                  // "hdrl"
                     + 8 + 56                             // avih
                     + 8 + 4                              // LIST hdr + "strl"
                     + 8 + 48                             // strh
                     + 8 + 40;                            // strf
        int idxSize = 8 + frames.Count * 16;
        int riffSize = 4                                  // "AVI "
                     + 8 + hdrlSize                       // hdrl LIST
                     + 8 + moviSize + 4                   // movi LIST ("movi" + frames)
                     + idxSize;

        using var avi = new MemoryStream();
        WriteAscii(avi, "RIFF");
        WriteInt(avi, riffSize);
        WriteAscii(avi, "AVI ");

        // hdrl
        WriteAscii(avi, "LIST");
        WriteInt(avi, hdrlSize);
        WriteAscii(avi, "hdrl");
        WriteAscii(avi, "avih");
        WriteInt(avi, 56);
        WriteInt(avi, usecPerFrame);        // dwMicroSecPerFrame
        WriteInt(avi, frameSize * framesPerSecond); // dwMaxBytesPerSec
        WriteInt(avi, 0);                   // dwPaddingGranularity
        WriteInt(avi, 0x10);                // dwFlags (AVIF_HASINDEX)
        WriteInt(avi, frames.Count);        // dwTotalFrames
        WriteInt(avi, 0);                   // dwInitialFrames
        WriteInt(avi, 1);                   // dwStreams
        WriteInt(avi, frameSize);           // dwSuggestedBufferSize
        WriteInt(avi, width);               // dwWidth
        WriteInt(avi, height);              // dwHeight
        WriteInt(avi, 0); WriteInt(avi, 0); WriteInt(avi, 0); WriteInt(avi, 0); // reserved

        // strl: stream header + format
        WriteAscii(avi, "LIST");
        WriteInt(avi, 4 + 8 + 48 + 8 + 40);
        WriteAscii(avi, "strl");
        WriteAscii(avi, "strh");
        WriteInt(avi, 48);
        WriteAscii(avi, "vids");
        WriteAscii(avi, "DIB ");
        WriteInt(avi, 0); WriteInt(avi, 0); WriteInt(avi, 0);
        WriteInt(avi, 1);                   // dwScale
        WriteInt(avi, framesPerSecond);     // dwRate
        WriteInt(avi, 0);                   // dwStart
        WriteInt(avi, frames.Count);        // dwLength
        WriteInt(avi, frameSize);           // dwSuggestedBufferSize
        WriteInt(avi, -1);                  // dwQuality
        WriteInt(avi, 0);                   // dwSampleSize
        WriteShort(avi, 0); WriteShort(avi, 0); // rcFrame (left, top)
        WriteShort(avi, (short)width); WriteShort(avi, (short)height);

        WriteAscii(avi, "strf");
        WriteInt(avi, 40);                  // BITMAPINFOHEADER
        WriteInt(avi, 40);
        WriteInt(avi, width);
        WriteInt(avi, height);
        WriteShort(avi, 1);                 // planes
        WriteShort(avi, 24);                // bpp
        WriteInt(avi, 0);                   // compression = BI_RGB
        WriteInt(avi, frameSize);
        WriteInt(avi, 0); WriteInt(avi, 0); // ppm
        WriteInt(avi, 0); WriteInt(avi, 0); // colors
        WriteInt(avi, 0);

        // movi
        WriteAscii(avi, "LIST");
        WriteInt(avi, moviSize + 4);
        WriteAscii(avi, "movi");
        foreach (var frame in frameBytes)
        {
            WriteAscii(avi, "00db");
            WriteInt(avi, frame.Length);
            avi.Write(frame);
        }

        // idx1
        WriteAscii(avi, "idx1");
        WriteInt(avi, frames.Count * 16);
        int offset = 4;
        foreach (var frame in frameBytes)
        {
            WriteAscii(avi, "00db");
            WriteInt(avi, 0x10);            // AVIIF_KEYFRAME
            WriteInt(avi, offset);
            WriteInt(avi, frame.Length);
            offset += frame.Length + 8;
        }

        return avi.ToArray();
    }

    /// <summary>One frame as a bottom-up BGR DIB (rows padded to 4 bytes).</summary>
    private static byte[] FrameData(PixelBuffer frame)
    {
        int stride = frame.Width * 3; // width % 4 == 0 → stride already aligned
        var data = new byte[stride * frame.Height];
        for (int y = 0; y < frame.Height; y++)
        {
            var srcRow = y * frame.Width * 3;
            var dstRow = (frame.Height - 1 - y) * stride;
            for (int x = 0; x < frame.Width; x++)
            {
                var s = srcRow + x * 3;
                var d = dstRow + x * 3;
                data[d] = frame.Rgb[s + 2];
                data[d + 1] = frame.Rgb[s + 1];
                data[d + 2] = frame.Rgb[s];
            }
        }
        return data;
    }

    private static void WriteAscii(Stream stream, string text) =>
        stream.Write(Encoding.ASCII.GetBytes(text));

    private static void WriteInt(Stream stream, int value)
    {
        stream.WriteByte((byte)value);
        stream.WriteByte((byte)(value >> 8));
        stream.WriteByte((byte)(value >> 16));
        stream.WriteByte((byte)(value >> 24));
    }

    private static void WriteShort(Stream stream, short value)
    {
        stream.WriteByte((byte)value);
        stream.WriteByte((byte)(value >> 8));
    }
}
