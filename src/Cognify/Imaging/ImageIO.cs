using System.Globalization;
using System.Text;

namespace Cognify.Imaging;

/// <summary>
/// One-stop image loading: sniffs the byte stream and dispatches to the
/// right codec. Supported: PNG (8-bit non-interlaced), BMP (24/32-bit), GIF
/// (first frame), binary PPM/PGM (P6/P5). Everything else is refused with a
/// clear message instead of a mystery error.
/// </summary>
public static class ImageIO
{
    /// <summary>Decodes a still image from its bytes.</summary>
    public static PixelBuffer Decode(byte[] data)
    {
        if (data is null || data.Length < 8)
            throw new ImageDecodeException("no image data");
        if (data[0] == 0x89 && data[1] == 0x50)
            return PngCodec.Decode(data);
        if (data[0] == (byte)'B' && data[1] == (byte)'M')
            return BmpCodec.Decode(data);
        if (data[0] == (byte)'G' && data[1] == (byte)'I' && data[2] == (byte)'F')
            return GifCodec.DecodeFirstFrame(data).Pixels;
        if (data[0] == (byte)'P' && (data[1] == '5' || data[1] == '6'))
            return DecodePnm(data);
        throw new ImageDecodeException(
            $"unsupported image format (starts {DescribeMagic(data)}); " +
            "supported: PNG, BMP, GIF, PPM/PGM. JPEG/WebP are not built in.");
    }

    /// <summary>Loads and decodes an image file.</summary>
    public static PixelBuffer Load(string path) =>
        Decode(File.ReadAllBytes(path));

    /// <summary>Encodes a buffer in the given format.</summary>
    public static byte[] Encode(PixelBuffer image, ImageFormat format) => format switch
    {
        ImageFormat.Png => PngCodec.Encode(image),
        ImageFormat.Bmp => BmpCodec.Encode(image),
        ImageFormat.Ppm => EncodePpm(image),
        _ => throw new ArgumentOutOfRangeException(nameof(format))
    };

    /// <summary>The canonical file extension for a format (with dot).</summary>
    public static string ExtensionOf(ImageFormat format) => format switch
    {
        ImageFormat.Png => ".png",
        ImageFormat.Bmp => ".bmp",
        ImageFormat.Ppm => ".ppm",
        _ => ".bin"
    };

    private static byte[] EncodePpm(PixelBuffer image)
    {
        var header = Encoding.ASCII.GetBytes($"P6\n{image.Width} {image.Height}\n255\n");
        var ppm = new byte[header.Length + image.Rgb.Length];
        Array.Copy(header, ppm, header.Length);
        Array.Copy(image.Rgb, 0, ppm, header.Length, image.Rgb.Length);
        return ppm;
    }

    private static PixelBuffer DecodePnm(byte[] data)
    {
        // P6 = RGB, P5 = gray; header is ASCII tokens separated by whitespace,
        // possibly with '#' comments, ending after the maxval token.
        int channels = data[1] == '6' ? 3 : 1;
        int pos = 2;
        var values = new List<int>(3);
        while (values.Count < 3 && pos < data.Length)
        {
            while (pos < data.Length && char.IsWhiteSpace((char)data[pos])) pos++;
            if (pos < data.Length && data[pos] == '#')
            {
                while (pos < data.Length && data[pos] != '\n') pos++;
                continue;
            }
            int start = pos;
            while (pos < data.Length && !char.IsWhiteSpace((char)data[pos])) pos++;
            if (pos > start)
                values.Add(int.Parse(data[start..pos], CultureInfo.InvariantCulture));
        }
        if (values.Count < 3)
            throw new ImageDecodeException("malformed PNM header");
        pos++; // single whitespace after maxval
        int width = values[0], height = values[1], maxval = values[2];
        if (maxval != 255)
            throw new ImageDecodeException($"unsupported PNM maxval {maxval} (only 255)");
        if (width < 1 || height < 1 || pos + width * height * channels > data.Length)
            throw new ImageDecodeException("truncated PNM pixel data");
        var image = new PixelBuffer(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int p = pos + (y * width + x) * channels;
                if (channels == 3)
                    image.SetPixel(x, y, data[p], data[p + 1], data[p + 2]);
                else
                    image.SetPixel(x, y, data[p], data[p], data[p]);
            }
        }
        return image;
    }

    private static string DescribeMagic(byte[] data) =>
        string.Join(" ", data.Take(4).Select(b => b.ToString("X2", CultureInfo.InvariantCulture)));
}
