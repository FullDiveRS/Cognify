namespace Cognify.Audio;

/// <summary>
/// What the auditory system extracted from one sound: its physical
/// characteristics plus, when it contains speech, a transcription.
/// </summary>
public sealed class AudioFeatures
{
    /// <summary>Natural-language summary of what was heard (non-speech sounds only).</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Recognized speech content; null when the clip is not speech.</summary>
    public string? Transcription { get; set; }

    /// <summary>Clip duration in seconds.</summary>
    public double DurationSeconds { get; set; }

    /// <summary>Perceived loudness (0..1).</summary>
    public double Loudness { get; set; } = 0.5;

    /// <summary>Estimated tempo in beats per minute (rhythmic sounds only; null otherwise).</summary>
    public double? TempoBpm { get; set; }

    /// <summary>True when the clip is judged to contain speech.</summary>
    public bool IsSpeech => Transcription is not null;

    /// <summary>Coarse sound category ("speech", "music", "noise", "tone", "unknown").</summary>
    public string Category { get; set; } = "unknown";

    /// <summary>One-line summary of what was heard.</summary>
    public string Summarize() =>
        IsSpeech
            ? $"Heard speech ({DurationSeconds:0.0}s): \"{Transcription}\""
            : $"{Description} ({Category}, {DurationSeconds:0.0}s, loudness {Loudness:0.00})";
}

/// <summary>
/// The input to an audio backend: a description or transcript plus any
/// metadata. Real backends decode the waveform itself; the built-in one
/// reasons over descriptive text and metadata.
/// </summary>
/// <param name="Content">Transcript of speech or a description of the sound.</param>
/// <param name="Metadata">Optional facts: duration, loudness, tempo, sample-rate...</param>
public sealed record AudioInput(string Content, IReadOnlyDictionary<string, string>? Metadata = null)
{
    /// <summary>Builds an audio input from a generic stimulus.</summary>
    public static AudioInput From(Core.Stimulus stimulus) =>
        new(stimulus.Content, stimulus.Metadata);
}

/// <summary>
/// Pluggable audio backend. Swap the built-in heuristic analyzer for a real
/// recognizer (whisper, cloud STT...) by implementing this interface and
/// assigning it via <c>WithAudio(a => a.Backend = myBackend)</c>.
/// </summary>
public interface IAudioBackend : Core.IModuleBackend
{
    /// <summary>Analyzes one audio input.</summary>
    AudioFeatures Analyze(AudioInput input);
}

/// <summary>
/// Deterministic, dependency-free audio backend. It "hears" by classifying
/// the input text: if it looks like a transcript (natural sentences) it is
/// treated as speech; sound-effect keywords ("bell", "thunder", "beep"...)
/// classify the clip as a non-speech sound; music keywords produce a tempo
/// estimate. Metadata (duration, loudness) always wins when present.
/// </summary>
public sealed class SimulatedAudioBackend : IAudioBackend
{
    private static readonly HashSet<string> SoundWords = new()
    {
        "bang", "beep", "bell", "birds", "boom", "buzz", "chime", "chatter", "clap",
        "crackle", "creak", "crash", "dog", "doorbell", "drum", "echo", "engine",
        "explosion", "footsteps", "gong", "hiss", "hum", "knock", "laughter", "purr",
        "rain", "ring", "roar", "rustle", "scream", "siren", "snap", "splash", "thud",
        "thunder", "tick", "whistle", "wind", "wine"
    };

    private static readonly HashSet<string> MusicWords = new()
    {
        "music", "song", "melody", "rhythm", "beat", "tune", "chords", "piano",
        "guitar", "violin", "drums", "orchestra", "singing", "humming", "lullaby", "jazz", "waltz"
    };

    /// <inheritdoc />
    public AudioFeatures Analyze(AudioInput input)
    {
        var metadata = input.Metadata ?? new Dictionary<string, string>();
        var text = (input.Content ?? string.Empty).Trim();
        var tokens = text.ToLowerInvariant()
            .Split(new[] { ' ', ',', '.', ';', ':', '!', '?', '"', '\n', '\t' },
                   StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToHashSet();

        var features = new AudioFeatures
        {
            DurationSeconds = ReadDouble(metadata, "duration", EstimateDuration(text)),
            Loudness = Math.Clamp(ReadDouble(metadata, "loudness", tokens.Contains("loud") ? 0.9 : tokens.Contains("quiet") ? 0.2 : 0.5), 0, 1)
        };

        // ---- classification -------------------------------------------
        bool hasSoundWord = tokens.Overlaps(SoundWords);
        bool hasMusicWord = tokens.Overlaps(MusicWords);
        bool looksLikeSpeech = LooksLikeSpeech(text);

        if (metadata.TryGetValue("kind", out var kind))
        {
            features.Category = kind.ToLowerInvariant();
        }
        else if (hasMusicWord)
        {
            features.Category = "music";
        }
        else if (hasSoundWord && !looksLikeSpeech)
        {
            features.Category = "noise";
        }
        else if (looksLikeSpeech)
        {
            features.Category = "speech";
        }
        else if (tokens.Contains("tone") || tokens.Contains("beep") || tokens.Contains("ping"))
        {
            features.Category = "tone";
        }
        else
        {
            features.Category = "unknown";
        }

        // ---- per-category extraction ------------------------------------
        switch (features.Category)
        {
            case "speech":
                features.Transcription = text.Length > 0 ? text : "(unintelligible)";
                features.Description = "A human voice speaking.";
                break;

            case "music":
                features.TempoBpm = ReadDouble(metadata, "tempo", EstimateTempo(tokens));
                features.Description = $"Music — {string.Join(", ", tokens.Where(MusicWords.Contains).Take(3))}.";
                break;

            case "noise":
                features.Description = $"A sound: {string.Join(", ", tokens.Where(SoundWords.Contains).Take(3))}.";
                break;

            case "tone":
                features.Description = "A steady tone.";
                break;

            default:
                features.Description = text.Length > 0 ? $"An unidentified sound: {Core.Mind.Truncate(text, 60)}." : "Silence, or something too faint to identify.";
                break;
        }

        return features;
    }

    private static readonly HashSet<string> FunctionWords = new()
    {
        "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "the", "a", "an",
        "is", "are", "was", "were", "do", "does", "did", "what", "how", "why", "when",
        "where", "who", "please", "tell", "can", "could", "would", "about", "this", "that",
        "and", "or", "but", "to", "of", "in", "on", "for", "with", "hello", "hi", "thanks"
    };

    private static bool LooksLikeSpeech(string text)
    {
        if (text.Length < 3) return false;
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length < 2) return false;
        // Speech-like: at least one function word (grammar glue) or a question mark.
        return words.Any(w => FunctionWords.Contains(w.TrimEnd(',', '.', '!', '?').ToLowerInvariant()))
               || text.Contains('?');
    }

    private static double EstimateDuration(string text) =>
        // ~2.5 words per second of speech, minimum half a second.
        Math.Max(0.5, text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length / 2.5);

    private static double EstimateTempo(HashSet<string> tokens) =>
        tokens.Contains("waltz") ? 120 : tokens.Contains("lullaby") ? 70 : tokens.Contains("drums") ? 140 : 100;

    private static double ReadDouble(IReadOnlyDictionary<string, string> metadata, string key, double fallback) =>
        metadata.TryGetValue(key, out var raw) &&
        double.TryParse(raw, System.Globalization.CultureInfo.InvariantCulture, out var value) ? value : fallback;
}
