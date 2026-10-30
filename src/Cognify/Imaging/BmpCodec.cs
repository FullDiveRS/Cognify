namespace Cognify.Imaging;

/// <summary>
/// Windows Bitmap (BMP) codec: encodes 24-bit and decodes 24/32-bit
/// uncompressed bottom-up (and top-down) BMPs.
/// </summary>
public static class BmpCodec
{
    /// <summary>Encodes a buffer as a 24-bit uncompressed BMP.</summary>
    public static byte[] Encode(PixelBuffer image)
    {
        int stride = (image.Width * 3 + 3) & ~3; // rows padded to 4 bytes
        int dataSize = stride * image.Height;
        int fileSize = 54 + dataSize;
        var bmp = new byte[fileSize];

        // BITMAPFILEHEADER
        bmp[0] = (byte)'B';
        bmp[1] = (byte)'M';
        WriteInt(bmp, 2, fileSize);
        WriteInt(bmp, 10, 54);
        // BITMAPINFOHEADER
        WriteInt(bmp, 14, 40);
        WriteInt(bmp, 18, image.Width);
        WriteInt(bmp, 22, image.Height);
        bmp[26] = 1;                    // planes
        bmp[28] = 24;                   // bits per pixel
        WriteInt(bmp, 34, dataSize);

        for (int y = 0; y < image.Height; y++)
        {
            var srcRow = y * image.Width * 3;
            var dstRow = 54 + (image.Height - 1 - y) * stride;
            for (int x = 0; x < image.Width; x++)
            {
                var s = srcRow + x * 3;
                var d = dstRow + x * 3;
                bmp[d] = image.Rgb[s + 2];     // BGR order
                bmp[d + 1] = image.Rgb[s + 1];
                bmp[d + 2] = image.Rgb[s];
            }
        }
        return bmp;
    }

    /// <summary>Decodes an uncompressed 24/32-bit BMP into RGB.</summary>
    public static PixelBuffer Decode(byte[] data)
    {
        if (data.Length < 54 || data[0] != (byte)'B' || data[1] != (byte)'M')
            throw new ImageDecodeException("not a BMP stream");
        int pixelOffset = ReadInt(data, 10);
        int headerSize = ReadInt(data, 14);
        if (headerSize < 40)
            throw new ImageDecodeException("unsupported BMP header (need BITMAPINFOHEADER)");
        int width = ReadInt(data, 18);
        int rawHeight = ReadInt(data, 22);
        bool topDown = rawHeight < 0;
        int height = Math.Abs(rawHeight);
        int planes = data[26] | (data[27] << 8);
        int bpp = data[28] | (data[29] << 8);
        int compression = ReadInt(data, 30);
        if (width < 1 || height < 1)
            throw new ImageDecodeException("empty BMP");
        if (planes != 1 || (bpp != 24 && bpp != 32) || compression != 0)
            throw new ImageDecodeException(
                $"unsupported BMP (bpp {bpp}, compression {compression}) — only uncompressed 24/32-bit");

        int stride = ((width * bpp / 8) + 3) & ~3;
        var image = new PixelBuffer(width, height);
        for (int y = 0; y < height; y++)
        {
            int srcRow = pixelOffset + y * stride;
            if (srcRow + width * (bpp / 8) > data.Length)
                throw new ImageDecodeException("truncated BMP pixel data");
            int dstY = topDown ? y : height - 1 - y;
            for (int x = 0; x < width; x++)
            {
                var s = srcRow + x * (bpp / 8);
                image.SetPixel(x, dstY, data[s + 2], data[s + 1], data[s]);
            }
        }
        return image;
    }

    private static int ReadInt(byte[] b, int offset) =>
        b[offset] | (b[offset + 1] << 8) | (b[offset + 2] << 16) | (b[offset + 3] << 24);

    private static void WriteInt(byte[] b, int offset, int value)
    {
        b[offset] = (byte)value;
        b[offset + 1] = (byte)(value >> 8);
        b[offset + 2] = (byte)(value >> 16);
        b[offset + 3] = (byte)(value >> 24);
    }
}
