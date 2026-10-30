namespace Cognify.Imaging;

/// <summary>Raised when an image cannot be decoded (unknown format, truncated, unsupported variant).</summary>
public sealed class ImageDecodeException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ImageDecodeException(string message) : base(message) { }
}

/// <summary>Supported still-image encoders.</summary>
public enum ImageFormat
{
    /// <summary>Portable Network Graphics (lossless, the default).</summary>
    Png,
    /// <summary>Windows Bitmap, 24-bit.</summary>
    Bmp,
    /// <summary>Portable Anymap, binary PPM (P6).</summary>
    Ppm
}

/// <summary>
/// A simple RGB image buffer: 8 bits per channel, tightly packed
/// (3 bytes per pixel). Every codec and the OCR engine work on this.
/// </summary>
public sealed class PixelBuffer
{
    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Packed RGB bytes, row-major, 3 bytes per pixel.</summary>
    public byte[] Rgb { get; }

    /// <summary>Creates a buffer of the given size (zero-filled = black).</summary>
    public PixelBuffer(int width, int height)
    {
        if (width < 1 || width > 8192 || height < 1 || height > 8192)
            throw new ArgumentOutOfRangeException(nameof(width),
                $"image size {width}x{height} out of range (1..8192)");
        Width = width;
        Height = height;
        Rgb = new byte[width * height * 3];
    }

    /// <summary>Creates a buffer over existing packed RGB bytes (length must match exactly).</summary>
    public PixelBuffer(int width, int height, byte[] rgb)
    {
        if (width < 1 || height < 1)
            throw new ArgumentOutOfRangeException(nameof(width), "image too small");
        if (rgb.Length != width * height * 3)
            throw new ArgumentException("rgb buffer length must be width*height*3", nameof(rgb));
        Width = width;
        Height = height;
        Rgb = rgb;
    }

    /// <summary>Gets the pixel at (x, y) as (r, g, b).</summary>
    public (byte R, byte G, byte B) GetPixel(int x, int y)
    {
        var i = (y * Width + x) * 3;
        return (Rgb[i], Rgb[i + 1], Rgb[i + 2]);
    }

    /// <summary>Sets the pixel at (x, y).</summary>
    public void SetPixel(int x, int y, byte r, byte g, byte b)
    {
        var i = (y * Width + x) * 3;
        Rgb[i] = r;
        Rgb[i + 1] = g;
        Rgb[i + 2] = b;
    }

    /// <summary>Luma (Rec.601) grayscale, one byte per pixel.</summary>
    public byte[] ToGrayscale()
    {
        var gray = new byte[Width * Height];
        for (int i = 0, j = 0; i < gray.Length; i++, j += 3)
            gray[i] = (byte)((Rgb[j] * 299 + Rgb[j + 1] * 587 + Rgb[j + 2] * 114) / 1000);
        return gray;
    }

    /// <summary>True when every pixel is exactly the given color (blank-page detection).</summary>
    public bool IsUniform(byte r, byte g, byte b)
    {
        for (int i = 0; i < Rgb.Length; i += 3)
            if (Rgb[i] != r || Rgb[i + 1] != g || Rgb[i + 2] != b)
                return false;
        return true;
    }

    /// <summary>Crops a rectangle (clamped to the image).</summary>
    public PixelBuffer Crop(int x, int y, int w, int h)
    {
        x = Math.Clamp(x, 0, Width - 1);
        y = Math.Clamp(y, 0, Height - 1);
        w = Math.Clamp(w, 1, Width - x);
        h = Math.Clamp(h, 1, Height - y);
        var outBuffer = new PixelBuffer(w, h);
        for (int row = 0; row < h; row++)
        {
            var src = ((y + row) * Width + x) * 3;
            var dst = row * w * 3;
            Array.Copy(Rgb, src, outBuffer.Rgb, dst, w * 3);
        }
        return outBuffer;
    }
}
