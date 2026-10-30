using System.Text;

namespace Cognify.Generation;

/// <summary>A decoded audio clip: mono samples in [-1, 1] plus the sample rate.</summary>
public sealed record AudioClip(double[] Samples, int SampleRate)
{
    /// <summary>Duration in seconds.</summary>
    public double Duration => Samples.Length / (double)SampleRate;
}

/// <summary>
/// Pure-C# RIFF/WAVE codec. Reads 8/16/24-bit PCM and 32-bit float WAVs
/// (mono or stereo, downmixed to mono for analysis) and writes 16-bit PCM.
/// </summary>
public static class WavCodec
{
    /// <summary>Decodes a WAV file into mono samples.</summary>
    public static AudioClip Decode(byte[] data)
    {
        if (data.Length < 44 || data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F')
            throw new InvalidDataException("not a RIFF/WAVE stream");
        int pos = 12;
        int sampleRate = 8000, channels = 1, bits = 16;
        bool floatFormat = false;
        byte[]? samplesChunk = null;

        while (pos + 8 <= data.Length)
        {
            var id = Encoding.ASCII.GetString(data, pos, 4);
            int size = data[pos + 4] | (data[pos + 5] << 8) | (data[pos + 6] << 16) | (data[pos + 7] << 24);
            var body = pos + 8;
            if (id == "fmt " && size >= 16)
            {
                ushort format = (ushort)(data[body] | (data[body + 1] << 8));
                channels = (ushort)(data[body + 2] | (data[body + 3] << 8));
                sampleRate = data[body + 4] | (data[body + 5] << 8) | (data[body + 6] << 16) | (data[body + 7] << 24);
                bits = (ushort)(data[body + 14] | (data[body + 15] << 8));
                floatFormat = format == 3;
            }
            else if (id == "data")
            {
                samplesChunk = data[body..Math.Min(body + size, data.Length)];
            }
            pos = body + size + (size & 1); // chunks are word-aligned
        }

        if (samplesChunk is null)
            throw new InvalidDataException("WAV has no data chunk");
        if (sampleRate < 1000 || sampleRate > 192000)
            throw new InvalidDataException($"implausible WAV sample rate {sampleRate}");

        int bytesPer = bits / 8;
        int frameCount = samplesChunk.Length / (bytesPer * Math.Max(1, channels));
        var samples = new double[frameCount];
        for (int frame = 0; frame < frameCount; frame++)
        {
            double sum = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                int o = (frame * channels + ch) * bytesPer;
                sum += ReadSample(samplesChunk, o, bits, floatFormat);
            }
            samples[frame] = Clamp(sum / Math.Max(1, channels));
        }
        return new AudioClip(samples, sampleRate);
    }

    /// <summary>Encodes mono samples as a 16-bit PCM WAV.</summary>
    public static byte[] Encode(double[] samples, int sampleRate, int channels = 1)
    {
        int dataLength = samples.Length * 2 * channels;
        using var stream = new MemoryStream(44 + dataLength);
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write("RIFF"u8);
            writer.Write(36 + dataLength);
            writer.Write("WAVE"u8);
            writer.Write("fmt "u8);
            writer.Write(16);
            writer.Write((short)1);             // PCM
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(sampleRate * 2 * channels);
            writer.Write((short)(2 * channels));
            writer.Write((short)16);
            writer.Write("data"u8);
            writer.Write(dataLength);
            foreach (var sample in samples)
            {
                short value = (short)Math.Round(Math.Clamp(sample, -1, 1) * short.MaxValue);
                writer.Write(value);
                if (channels == 2) writer.Write(value);
            }
        }
        return stream.ToArray();
    }

    private static double ReadSample(byte[] data, int offset, int bits, bool isFloat)
    {
        if (isFloat && bits == 32 && offset + 4 <= data.Length)
        {
            float value = BitConverter.Int32BitsToSingle(
                data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24));
            return Clamp(value);
        }
        if (bits == 8) return (data[offset] - 128) / 128.0;
        if (bits == 16 && offset + 2 <= data.Length)
            return (short)(data[offset] | (data[offset + 1] << 8)) / 32768.0;
        if (bits == 24 && offset + 3 <= data.Length)
        {
            int value = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16);
            if ((value & 0x800000) != 0) value |= unchecked((int)0xFF000000);
            return value / 8388608.0;
        }
        if (bits == 32 && offset + 4 <= data.Length)
        {
            int value = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24);
            return value / 2147483648.0;
        }
        return 0;
    }

    private static double Clamp(double value) => Math.Clamp(value, -1, 1);
}
