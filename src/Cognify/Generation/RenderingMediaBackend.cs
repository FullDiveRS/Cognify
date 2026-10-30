using Cognify.Imaging;

namespace Cognify.Generation;

/// <summary>
/// The real rendering media backend: every request produces actual playable
/// bytes — PNG stills with caption stamps, looping GIF animations, AVI
/// videos, formant-speech WAV, and composed music WAV. Everything is
/// deterministic per prompt, saved into the <see cref="Store"/> when the
/// options enable file output, and registered with the <see cref="Phrasebook"/>
/// transcriber so she can transcribe her own speech back.
/// </summary>
public sealed class RenderingMediaBackend : IMediaBackend
{
    /// <summary>Default working size for rendered stills.</summary>
    public const int StillWidth = 320;
    /// <summary>Still height.</summary>
    public const int StillHeight = 240;
    /// <summary>Frame size used for animations and video.</summary>
    public const int AnimWidth = 160;
    /// <summary>Animation height.</summary>
    public const int AnimHeight = 120;

    private readonly int _audioRate;
    private readonly bool _saveFiles;
    private readonly SpeechSynthesizer _speech;
    private readonly MusicSynthesizer _music;
    private readonly object _gate = new();

    /// <summary>Where generated files land.</summary>
    public MediaFileStore Store { get; }

    /// <summary>The transcriber that learns every spoken clip.</summary>
    public Transcriber Phrasebook { get; } = new();

    /// <summary>The voice currently used for speech (replaced by cloning).</summary>
    public VoiceProfile Voice { get; set; } = new();

    /// <summary>Creates the backend.</summary>
    public RenderingMediaBackend(string mediaDirectory = "cognify_media",
        int audioRate = 16000, bool saveFiles = true)
    {
        _audioRate = audioRate;
        _saveFiles = saveFiles;
        Store = new MediaFileStore(mediaDirectory);
        _speech = new SpeechSynthesizer(audioRate);
        _music = new MusicSynthesizer(22050);
    }

    /// <inheritdoc />
    public MediaArtifact Generate(GenerationRequest request)
    {
        var artifact = new MediaArtifact
        {
            Type = request.Type,
            Prompt = request.Prompt,
            Seed = SeedOf(request.Prompt),
        };

        switch (request.Type)
        {
            case MediaType.Image:
            {
                var renderer = new SceneRenderer(request.Prompt, artifact.Seed);
                var image = renderer.RenderStill(request.Prompt, StillWidth, StillHeight);
                var bytes = PngCodec.Encode(image);
                artifact.Width = image.Width;
                artifact.Height = image.Height;
                artifact.Data = bytes;
                artifact.SizeBytes = bytes.Length;
                artifact.Extension = ".png";
                artifact.FileName = MaybeSave("image", artifact.Extension, bytes);
                artifact.Notes = new[]
                {
                    $"PNG still rendered from a {renderer.Mood} '{renderer.Motion}' scene, caption stamped.",
                    _saveFiles && artifact.FileName.Length > 0
                        ? $"Saved to {artifact.FileName}."
                        : "File saving is disabled.",
                };
                break;
            }
            case MediaType.Gif:
            {
                var renderer = new SceneRenderer(request.Prompt, artifact.Seed);
                double seconds = request.Seconds > 0 ? request.Seconds : 2.4;
                int frameCount = Math.Clamp((int)(seconds * 10), 4, 48);
                var frames = Enumerable.Range(0, frameCount)
                    .Select(i => renderer.RenderFrame(i / (double)frameCount, AnimWidth, AnimHeight))
                    .ToList();
                var bytes = GifCodec.EncodeAnimation(frames, delayCentiseconds: 10);
                artifact.Width = AnimWidth;
                artifact.Height = AnimHeight;
                artifact.DurationSeconds = frameCount / 10.0;
                artifact.FramesPerSecond = 10;
                artifact.FrameCount = frameCount;
                artifact.Data = bytes;
                artifact.SizeBytes = bytes.Length;
                artifact.Extension = ".gif";
                artifact.FileName = MaybeSave("animation", artifact.Extension, bytes);
                artifact.Notes = new[]
                {
                    $"Looping animated GIF: {frameCount} frames of a {renderer.Mood} '{renderer.Motion}' scene.",
                    _saveFiles && artifact.FileName.Length > 0
                        ? $"Saved to {artifact.FileName}."
                        : "File saving is disabled.",
                };
                break;
            }
            case MediaType.Video:
            {
                var renderer = new SceneRenderer(request.Prompt, artifact.Seed);
                double seconds = request.Seconds > 0 ? request.Seconds : 4;
                int frameCount = Math.Clamp((int)(seconds * 10), 4, 60);
                var frames = Enumerable.Range(0, frameCount)
                    .Select(i => renderer.RenderFrame(i / (double)frameCount, AnimWidth, AnimHeight))
                    .ToList();
                var bytes = AviWriter.Encode(frames, framesPerSecond: 10);
                artifact.Width = AnimWidth;
                artifact.Height = AnimHeight;
                artifact.DurationSeconds = frameCount / 10.0;
                artifact.FramesPerSecond = 10;
                artifact.FrameCount = frameCount;
                artifact.Data = bytes;
                artifact.SizeBytes = bytes.Length;
                artifact.Extension = ".avi";
                artifact.FileName = MaybeSave("video", artifact.Extension, bytes);
                artifact.Notes = new[]
                {
                    $"AVI video (uncompressed 24-bit): {frameCount} frames @ 10 fps of a {renderer.Mood} '{renderer.Motion}' scene.",
                    _saveFiles && artifact.FileName.Length > 0
                        ? $"Saved to {artifact.FileName}."
                        : "File saving is disabled.",
                };
                break;
            }
            case MediaType.Speech:
            {
                var clip = _speech.Synthesize(request.Prompt, Voice);
                var bytes = WavCodec.Encode(clip.Samples, clip.SampleRate);
                Phrasebook.RegisterTemplate(request.Prompt, clip); // she learns her own voice
                artifact.DurationSeconds = clip.Duration;
                artifact.SampleRateHz = clip.SampleRate;
                artifact.Data = bytes;
                artifact.SizeBytes = bytes.Length;
                artifact.Extension = ".wav";
                artifact.FileName = MaybeSave("speech", artifact.Extension, bytes);
                artifact.Notes = new[]
                {
                    $"Formant speech (source-filter TTS) with {Voice.Describe()}.",
                    "The phrase was added to her transcriber phrasebook.",
                };
                break;
            }
            case MediaType.Audio:
            default:
            {
                var wantsMusic = Regexes.MusicWords.IsMatch(request.Prompt.ToLowerInvariant()) ||
                                 request.Style.Equals("music", StringComparison.OrdinalIgnoreCase);
                var clip = wantsMusic
                    ? _music.Compose(request.Prompt)
                    : Soundscape(request.Prompt, artifact.Seed);
                var bytes = WavCodec.Encode(clip.Samples, clip.SampleRate);
                artifact.DurationSeconds = clip.Duration;
                artifact.SampleRateHz = clip.SampleRate;
                artifact.Data = bytes;
                artifact.SizeBytes = bytes.Length;
                artifact.Extension = ".wav";
                artifact.FileName = MaybeSave(wantsMusic ? "melody" : "sound", artifact.Extension, bytes);
                artifact.Notes = new[]
                {
                    wantsMusic
                        ? $"Composed {MusicSynthesizer.Describe(request.Prompt)} — melody, chord pad and bass."
                        : "Procedural soundscape (filtered noise swells shaped by the prompt).",
                };
                break;
            }
        }
        return artifact;
    }

    /// <summary>Generates speech and returns the raw clip (used by voice flows).</summary>
    public AudioClip SynthesizeSpeech(string text, VoiceProfile? profile = null) =>
        _speech.Synthesize(text, profile ?? Voice);

    /// <summary>Analyzes a recording and switches her voice to the cloned profile.</summary>
    public VoiceProfile CloneVoiceFrom(AudioClip sample)
    {
        var profile = VoiceCloneEngine.CloneFrom(sample);
        Voice = profile;
        return profile;
    }

    /// <summary>Transcribes a clip against her phrasebook.</summary>
    public TranscriptionResult Transcribe(AudioClip clip) => Phrasebook.Transcribe(clip);

    /// <summary>Composes a music piece from a prompt.</summary>
    public AudioClip ComposeMusic(string prompt) => _music.Compose(prompt);

    private string? MaybeSave(string kind, string extension, byte[] bytes) =>
        _saveFiles ? Store.Save(kind, extension, bytes) : null;

    private static AudioClip Soundscape(string prompt, int seed)
    {
        int rate = 22050;
        double seconds = Math.Clamp(2.0 + prompt.Length * 0.05, 2, 12);
        var samples = new double[(int)(seconds * rate)];
        var random = new Random(seed);
        for (int i = 0; i < samples.Length; i++)
        {
            double t = i / (double)rate;
            double swell = 0.5 + 0.5 * Math.Sin(2 * Math.PI * t / seconds);
            double low = Math.Sin(2 * Math.PI * (90 + 20 * Math.Sin(2 * Math.PI * t * 0.7)) * t) * 0.5;
            double mid = Math.Sin(2 * Math.PI * 420 * t + Math.Sin(2 * Math.PI * 3 * t) * 2) * 0.2;
            double hiss = (random.NextDouble() * 2 - 1) * 0.06;
            samples[i] = Math.Clamp((low + mid + hiss) * swell * 0.8, -1, 1);
        }
        return Dsp.Fade(new AudioClip(samples, rate), 0.05, 0.15);
    }

    private static int SeedOf(string prompt)
    {
        var bytes = System.Security.Cryptography.MD5.HashData(
            System.Text.Encoding.UTF8.GetBytes(prompt.Trim().ToLowerInvariant()));
        return BitConverter.ToInt32(bytes, 0);
    }
}

/// <summary>Shared regexes for intent matching.</summary>
internal static class Regexes
{
    /// <summary>Words that steer a sound request toward music composition.</summary>
    public static readonly System.Text.RegularExpressions.Regex MusicWords = new(
        @"\b(music|song|melody|tune|lullaby|anthem|compose)\b",
        System.Text.RegularExpressions.RegexOptions.Compiled);
}
