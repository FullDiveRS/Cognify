using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Cognify.Generation;

/// <summary>The kind of media to generate.</summary>
public enum MediaType
{
    /// <summary>A still image.</summary>
    Image,
    /// <summary>A video clip.</summary>
    Video,
    /// <summary>An animated GIF.</summary>
    Gif,
    /// <summary>Spoken words.</summary>
    Speech,
    /// <summary>A non-speech sound (effects, ambience, music).</summary>
    Audio
}

/// <summary>A request to generate media.</summary>
/// <param name="Prompt">What to generate, in plain language.</param>
/// <param name="Type">The media type.</param>
/// <param name="Seconds">Desired duration for video/speech/audio (ignored for images).</param>
/// <param name="Style">Optional style hint (e.g. "watercolor", "deep-voice").</param>
public sealed record GenerationRequest(string Prompt, MediaType Type, double Seconds = 0, string Style = "");

/// <summary>
/// The result of a generation: full specifications plus — for audio types —
/// real, playable WAV bytes when the backend supports synthesis.
/// </summary>
public sealed class MediaArtifact
{
    /// <summary>Unique artifact id.</summary>
    public string Id { get; set; } = $"media-{Guid.NewGuid():N}";

    /// <summary>The media type.</summary>
    public MediaType Type { get; set; }

    /// <summary>The prompt it was generated from.</summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>Width in pixels (images/video; 0 for audio).</summary>
    public int Width { get; set; }

    /// <summary>Height in pixels (images/video; 0 for audio).</summary>
    public int Height { get; set; }

    /// <summary>Duration in seconds (audio/video; 0 for images).</summary>
    public double DurationSeconds { get; set; }

    /// <summary>Frames per second (video only).</summary>
    public int FramesPerSecond { get; set; }

    /// <summary>Total frame count (video only).</summary>
    public int FrameCount { get; set; }

    /// <summary>Approximate size in bytes.</summary>
    public long SizeBytes { get; set; }

    /// <summary>Deterministic seed derived from the prompt (reproducible output).</summary>
    public int Seed { get; set; }

    /// <summary>Real bytes for every media type: PNG/GIF/AVI/WAV content.
    /// Null only when the backend produces specifications without data.</summary>
    public byte[]? Data { get; set; }

    /// <summary>File extension of the artifact bytes (e.g. ".png").</summary>
    public string Extension { get; set; } = "";

    /// <summary>Full path where the artifact was saved, or "" when not saved.</summary>
    public string FileName { get; set; } = "";

    /// <summary>Human-readable notes about the artifact.</summary>
    public IReadOnlyList<string> Notes { get; set; } = Array.Empty<string>();

    /// <summary>One-line description.</summary>
    public string Describe()
    {
        var spec = Type switch
        {
            MediaType.Image => $"{Width}x{Height} image",
            MediaType.Gif => $"{Width}x{Height} looping gif, {DurationSeconds:0.0}s @ {FramesPerSecond} fps ({FrameCount} frames)",
            MediaType.Video => $"{Width}x{Height} video, {DurationSeconds:0.0}s @ {FramesPerSecond} fps ({FrameCount} frames)",
            MediaType.Speech => $"speech clip, {DurationSeconds:0.0}s @ {SampleRateHz} Hz",
            _ => $"sound clip, {DurationSeconds:0.0}s @ {SampleRateHz} Hz"
        };
        var where = FileName.Length > 0 ? $", saved as {FileName}" : "";
        return $"[{Id}] {Type}: \"{Prompt}\" — {spec}, ~{SizeBytes / 1024.0:0} KB, seed {Seed}{where}";
    }

    /// <summary>Sample rate of the generated audio (0 when not audio).</summary>
    public int SampleRateHz { get; set; }
}

/// <summary>
/// Pluggable media-generation backend. Implement this to wire in a real
/// video/audio generator (diffusion API, TTS engine...); the built-in one
/// simulates specs and synthesizes real WAV tones for audio.
/// </summary>
public interface IMediaBackend : Core.IModuleBackend
{
    /// <summary>Generates one artifact. Implementations must be deterministic per prompt.</summary>
    MediaArtifact Generate(GenerationRequest request);
}

/// <summary>
/// Simulated media backend with real audio synthesis. Image and video
/// requests produce deterministic specifications derived from a hash of the
/// prompt (seed, dimensions, frame counts, sizes). Speech and audio requests
/// additionally synthesize a genuinely playable 16-bit PCM WAV: the prompt
/// text is turned into a sequence of tone bursts (one per syllable-ish
/// chunk) with amplitude envelopes — audible "speech rhythm".
/// </summary>
public sealed class SimulatedMediaBackend : IMediaBackend
{
    private readonly int _sampleRate;
    private readonly bool _emitWav;

    /// <summary>Creates the backend.</summary>
    public SimulatedMediaBackend(int sampleRate = 8000, bool emitWav = true)
    {
        if (sampleRate < 1000) throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate too low.");
        _sampleRate = sampleRate;
        _emitWav = emitWav;
    }

    /// <inheritdoc />
    public MediaArtifact Generate(GenerationRequest request)
    {
        int seed = SeedOf(request.Prompt);
        var random = new Random(seed);

        var artifact = new MediaArtifact
        {
            Type = request.Type,
            Prompt = request.Prompt,
            Seed = seed
        };

        switch (request.Type)
        {
            case MediaType.Image:
                artifact.Width = 512 + (seed % 5) * 128;   // 512..1024
                artifact.Height = 384 + (seed % 4) * 128;  // 384..768
                artifact.SizeBytes = (long)artifact.Width * artifact.Height * 3;
                artifact.Notes = new[]
                {
                    "Simulated image (no pixels rendered — assign a real IMediaBackend for that).",
                    $"Style hint: {(string.IsNullOrWhiteSpace(request.Style) ? "default" : request.Style)}."
                };
                break;

            case MediaType.Video:
            {
                double seconds = request.Seconds > 0 ? request.Seconds : 5;
                artifact.Width = 640 + (seed % 3) * 320;   // 640..1280
                artifact.Height = 360 + (seed % 3) * 120;  // 360..600
                artifact.FramesPerSecond = 24 + seed % 3 * 12; // 24, 36, 48
                artifact.DurationSeconds = seconds;
                artifact.FrameCount = (int)Math.Ceiling(seconds * artifact.FramesPerSecond);
                artifact.SizeBytes = (long)artifact.FrameCount * artifact.Width * artifact.Height / 8; // rough
                artifact.Notes = new[]
                {
                    $"Simulated video ({artifact.FrameCount} frames at {artifact.FramesPerSecond} fps).",
                    "No frames rendered — assign a real IMediaBackend for actual footage."
                };
                break;
            }

            case MediaType.Speech:
            case MediaType.Audio:
            {
                double seconds = request.Seconds > 0 ? request.Seconds : EstimateDuration(request.Prompt, request.Type);
                artifact.DurationSeconds = seconds;
                artifact.SampleRateHz = _sampleRate;
                if (_emitWav)
                {
                    artifact.Data = SynthesizeWav(request, seconds, seed);
                    artifact.SizeBytes = artifact.Data.Length;
                }
                else
                {
                    artifact.SizeBytes = (long)(seconds * _sampleRate) * 2; // 16-bit mono
                }
                artifact.Notes = new[]
                {
                    _emitWav
                        ? "Contains real WAV bytes (16-bit PCM) — a tone-sketch of the requested audio."
                        : "WAV synthesis disabled (specs only).",
                    request.Type == MediaType.Speech
                        ? "Speech is rendered as rhythmic tones, not real words."
                        : "Sound is rendered as a tone texture."
                };
                break;
            }
        }

        return artifact;
    }

    // ------------------------------------------------------------------
    // Deterministic seed from prompt
    // ------------------------------------------------------------------

    private static int SeedOf(string prompt)
    {
        var bytes = MD5.HashData(Encoding.UTF8.GetBytes(prompt.Trim().ToLowerInvariant()));
        return BitConverter.ToInt32(bytes, 0);
    }

    /// <summary>Process-independent, stable small hash for a word.</summary>
    private static int StableHash(string text)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (char c in text)
            {
                hash ^= c;
                hash *= 16777619;
            }
            return (int)(hash & 0x7FFFFFFF);
        }
    }

    private static double EstimateDuration(string prompt, MediaType type)
    {
        int words = prompt.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        return type == MediaType.Speech
            ? Math.Clamp(words / 2.5, 1.0, 30.0) // ~2.5 words per second
            : Math.Clamp(2.0 + words * 0.2, 2.0, 20.0);
    }

    // ------------------------------------------------------------------
    // Real WAV synthesis
    // ------------------------------------------------------------------

    /// <summary>
    /// Synthesizes a real WAV file: the prompt's characters are chunked into
    /// "syllables"; each syllable becomes a short tone burst whose frequency
    /// comes from the characters (deterministic), with a soft attack/decay
    /// envelope so it sounds rhythmic rather than harsh.
    /// </summary>
    private byte[] SynthesizeWav(GenerationRequest request, double seconds, int seed)
    {
        int totalSamples = (int)(seconds * _sampleRate);
        var samples = new short[totalSamples];

        var random = new Random(seed);
        bool speech = request.Type == MediaType.Speech;

        // Base pitch: speech ~140-220 Hz, sound effects lower.
        double basePitch = speech ? 150 + random.Next(0, 60) : 80 + random.Next(0, 50);
        var words = request.Prompt.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int syllables = Math.Max(1, Math.Min(words.Sum(w => Math.Max(1, w.Length / 2)), totalSamples / (_sampleRate / 5)));
        double burstLength = totalSamples / (double)syllables;

        for (int s = 0; s < syllables; s++)
        {
            int start = (int)(s * burstLength);
            int end = Math.Min(totalSamples, (int)((s + 1) * burstLength * 0.82)); // small gaps between bursts
            var chunk = words.Length > 0 ? words[s % words.Length] : "a";
            double pitch = basePitch * (1.0 + (StableHash(chunk) % 7 - 3) / 12.0); // ±3 semitone-ish steps

            for (int i = start; i < end; i++)
            {
                double t = (i - start) / (double)_sampleRate;
                double phase = 2 * Math.PI * pitch * t;
                // Fundamental + soft harmonics
                double wave = Math.Sin(phase) * 0.7 + Math.Sin(2 * phase) * 0.2 + Math.Sin(3 * phase) * 0.1;
                // Attack/decay envelope over the burst
                double position = (i - start) / (double)Math.Max(1, end - start);
                double envelope = Math.Sin(Math.PI * Math.Clamp(position, 0, 1));
                double amplitude = speech ? 0.5 : 0.4;
                samples[i] = (short)Math.Clamp(wave * envelope * amplitude * short.MaxValue,
                    short.MinValue, short.MaxValue);
            }
        }

        // Gentle global fade-in/out to avoid clicks.
        int fade = Math.Min(_sampleRate / 10, totalSamples / 4);
        for (int i = 0; i < fade; i++)
        {
            double gain = i / (double)fade;
            samples[i] = (short)(samples[i] * gain);
            samples[totalSamples - 1 - i] = (short)(samples[totalSamples - 1 - i] * gain);
        }

        return BuildWav(samples, _sampleRate);
    }

    /// <summary>Wraps PCM samples in a standard 44-byte RIFF/WAVE header.</summary>
    private static byte[] BuildWav(short[] samples, int sampleRate)
    {
        int dataLength = samples.Length * 2;
        using var stream = new MemoryStream(44 + dataLength);
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write("RIFF"u8);
            writer.Write(36 + dataLength);
            writer.Write("WAVE"u8);
            writer.Write("fmt "u8);
            writer.Write(16);                       // fmt chunk size
            writer.Write((short)1);                 // PCM
            writer.Write((short)1);                 // mono
            writer.Write(sampleRate);
            writer.Write(sampleRate * 2);           // byte rate (16-bit mono)
            writer.Write((short)2);                 // block align
            writer.Write((short)16);                // bits per sample
            writer.Write("data"u8);
            writer.Write(dataLength);
            foreach (var sample in samples)
                writer.Write(sample);
        }
        return stream.ToArray();
    }
}
