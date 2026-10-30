using System.Text.RegularExpressions;
using Cognify.Core;

namespace Cognify.Generation;

/// <summary>
/// The media generation module: lets the mind create, not just perceive.
/// It recognizes generation intents in text ("generate an image of...",
/// "create a video of...", "say hello", "make a sound of...") and produces
/// <see cref="MediaArtifact"/>s via the configured <see cref="IMediaBackend"/>.
/// All artifacts are kept in a bounded gallery and announced on the
/// <c>Mind.MediaGenerated</c> event.
/// </summary>
public sealed class MediaGenerationModule : CognitiveModuleBase
{
    private readonly GenerationOptions _options;
    private readonly object _gate = new();
    private readonly Queue<MediaArtifact> _gallery;
    private int _generated;

    /// <summary>Creates the module. Pass a custom backend for custom generation;
    /// by default every request is really rendered (PNG/GIF/AVI/WAV).</summary>
    public MediaGenerationModule(GenerationOptions options, IMediaBackend? backend = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Backend = backend ?? options.Backend as IMediaBackend
                  ?? new RenderingMediaBackend(options.MediaDirectory,
                      Math.Max(16000, options.SampleRate), options.EnableFileOutput);
        _gallery = new Queue<MediaArtifact>(Math.Max(1, options.GalleryCapacity));
    }

    /// <inheritdoc />
    public override string Name => "generation";

    /// <inheritdoc />
    public override int Order => 47; // action band: response set here is kept by thinking (order 50)

    /// <summary>The active media backend.</summary>
    public IMediaBackend Backend { get; }

    /// <summary>How many artifacts have been generated.</summary>
    public int GeneratedCount { get { lock (_gate) return _generated; } }

    /// <summary>The bounded gallery of recent artifacts, newest first.</summary>
    public IReadOnlyList<MediaArtifact> Gallery
    {
        get { lock (_gate) return _gallery.Reverse().ToList(); }
    }

    /// <summary>Generates an artifact programmatically (also used by the cycle).</summary>
    public MediaArtifact Generate(string prompt, MediaType type, double seconds = 0, string style = "")
    {
        var request = new GenerationRequest(prompt, type, seconds, style);
        var artifact = Backend.Generate(request);

        lock (_gate)
        {
            _generated++;
            _gallery.Enqueue(artifact);
            while (_gallery.Count > Math.Max(1, _options.GalleryCapacity))
                _gallery.Dequeue();
        }

        Host?.RaiseMediaGenerated(artifact);
        Host?.Stream?.Add("generation", $"created {artifact.Type}: {artifact.Describe()}");
        return artifact;
    }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        if (context.Stimulus.Kind is StimulusKind.Image or StimulusKind.Audio or StimulusKind.Video or StimulusKind.Feedback)
            return;

        var text = context.EffectiveContent;
        if (string.IsNullOrWhiteSpace(text)) return;

        var intent = MatchIntent(text);
        if (intent is null) return;

        var artifact = Generate(intent.Value.prompt, intent.Value.type, intent.Value.seconds,
            intent.Value.style);
        context.Bag["media.artifact"] = artifact;

        if (context.Response is null)
        {
            string extra = artifact.Data is not null
                ? $" The real {artifact.Extension.TrimStart('.').ToUpperInvariant()} bytes are attached."
                : "";
            context.Response = new ResponseDraft(
                $"Done — I generated {artifact.Type.ToString().ToLowerInvariant()} media for \"{artifact.Prompt}\". {artifact.Describe()}.{extra}",
                0.6, 870, "media-generation");
        }
    }

    private (string prompt, MediaType type, double seconds, string style)? MatchIntent(string text)
    {
        var lower = text.ToLowerInvariant();

        // "generate/create/make/draw an image/picture of X"
        var image = Regex.Match(lower, @"(?:generate|create|make|draw|paint)\s+(?:an?\s+)?(?:image|picture|photo)\s+(?:of\s+|showing\s+)?(.+)");
        if (image.Success)
            return (Clean(image.Groups[1].Value), MediaType.Image, 0, "");

        // "make a gif of X" / "animate X"
        var gif = Regex.Match(lower, @"(?:generate|create|make|animate)\s+(?:an?\s+)?gif\s+(?:of\s+|showing\s+)?(.+?)(?:\s+for\s+(\d+(?:\.\d+)?)\s*seconds?)?$");
        if (gif.Success)
        {
            double seconds = gif.Groups[2].Success
                ? double.Parse(gif.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture)
                : _options.DefaultVideoSeconds;
            return (Clean(gif.Groups[1].Value), MediaType.Gif, seconds, "");
        }

        // "generate/create/make a video/clip of X (for N seconds)"
        var video = Regex.Match(lower, @"(?:generate|create|make|render)\s+(?:an?\s+)?(?:video|clip|animation)\s+(?:of\s+|showing\s+)?(.+?)(?:\s+for\s+(\d+(?:\.\d+)?)\s*seconds?)?$");
        if (video.Success)
        {
            double seconds = video.Groups[2].Success
                ? double.Parse(video.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture)
                : _options.DefaultVideoSeconds;
            return (Clean(video.Groups[1].Value), MediaType.Video, seconds, "");
        }

        // "compose music about X" / "make a song about X" / "sing X"
        var music = Regex.Match(lower, @"(?:compose|make|create|play|sing)\s+(?:an?\s+)?(?:music|song|melody|tune|lullaby)\s*(?:about|of|for|that sounds)?\s*(.*)");
        if (music.Success)
            return (Clean(string.IsNullOrWhiteSpace(music.Groups[1].Value) ? lower : music.Groups[1].Value),
                MediaType.Audio, 0, "music");

        // "say X" / "speak X" / "read X aloud" — SPEECH GENERATION intent.
        // Deliberately strict: the imperative must START the message, so a
        // sentence that merely CONTAINS "say" ("do you know what memories
        // are... if so say yes and if not say no") stays a normal
        // conversational turn instead of hijacking the TTS path. Quoted
        // payloads ("say \"hello\"") still match anywhere.
        var speech = Regex.Match(lower, @"^(?:please\s+|yume\s*[,:]\s*)?(?:say|speak|read aloud|voice)\s+(?:the\s+words\s+|this\s+:?\s*)?[""']?(.+?)[""']?$");
        if (speech.Success && !lower.Contains(" if ") && !lower.StartsWith("if "))
            return (Clean(speech.Groups[1].Value), MediaType.Speech, 0, "");

        // Quoted speech anywhere: 'say "good morning"' mid-sentence.
        var quoted = Regex.Match(lower, @"(?:say|speak)\s+[""'](.+?)[""']");
        if (quoted.Success)
            return (Clean(quoted.Groups[1].Value), MediaType.Speech, 0, "");

        // "make a sound of X" / "generate audio of X" / "synthesize X"
        var audio = Regex.Match(lower, @"(?:make|generate|create|synthesize)\s+(?:an?\s+)?(?:sound|audio|noise|tone)\s*(?:of|like|for)?\s*(.+)");
        if (audio.Success)
            return (Clean(audio.Groups[1].Value), MediaType.Audio, 0, "");

        return null;
    }

    private static string Clean(string prompt) =>
        Regex.Replace(prompt.Trim().TrimEnd('.', '!', '?'), @"^(an?|the)\s+", "").Trim();
}
