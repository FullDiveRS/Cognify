using Cognify.Core;

namespace Cognify.Audio;

/// <summary>
/// The audio module: gives the mind ears. When a stimulus is an
/// <see cref="StimulusKind.Audio"/> (or a <see cref="StimulusKind.Video"/> with
/// sound), the configured <see cref="IAudioBackend"/> extracts features and —
/// for speech — a transcription. The transcription is then substituted into
/// the cycle so reasoning, memory and emotions operate on the spoken words.
/// </summary>
public sealed class AudioModule : CognitiveModuleBase
{
    private readonly AudioOptions _options;
    private readonly object _gate = new();
    private int _clipsHeard;

    private static readonly string[] QuestionWords =
        { "what", "who", "when", "where", "why", "how", "is", "are", "can", "do", "does", "could", "would", "please", "tell" };

    /// <summary>
    /// Creates the audio module. Pass a custom <paramref name="backend"/> for
    /// real speech recognition; when null, the backend is taken from
    /// <see cref="Core.AudioOptions.Backend"/> if one was assigned there,
    /// falling back to the built-in heuristic analyzer.
    /// </summary>
    public AudioModule(AudioOptions options, IAudioBackend? backend = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Backend = backend ?? options.Backend as IAudioBackend ?? new SimulatedAudioBackend();
    }

    /// <inheritdoc />
    public override string Name => "audio";

    /// <inheritdoc />
    public override int Order => 12; // perception band

    /// <summary>The active audio backend.</summary>
    public IAudioBackend Backend { get; }

    /// <summary>How many clips have been heard so far.</summary>
    public int ClipsHeard { get { lock (_gate) return _clipsHeard; } }

    /// <summary>The most recently heard audio features, if any.</summary>
    public AudioFeatures? LastHeard { get; private set; }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var kind = context.Stimulus.Kind;
        if (kind is not (StimulusKind.Audio or StimulusKind.Video)) return;

        var features = Backend.Analyze(AudioInput.From(context.Stimulus));
        lock (_gate)
        {
            _clipsHeard++;
            LastHeard = features;
        }

        context.Bag["audio.features"] = features;
        context.Bag["audio.category"] = features.Category;
        Host?.Stream?.Add("audio", $"heard: {features.Summarize()}");

        // Salient or startling sounds cause a small startle reaction.
        if (features.Loudness >= _options.SalienceThreshold)
            Host?.EmotionalState?.Feel(Emotions.EmotionType.Surprise,
                Math.Min(0.5, 0.1 + features.Loudness * 0.4));

        // Remember what was heard (the plain memory module only stores the raw
        // stimulus content, which for audio may be a file name or description).
        Host?.Memory?.Remember(
            $"[heard] {features.Summarize()}",
            Memory.MemoryKind.Episodic,
            importance: Math.Clamp(0.3 + features.Loudness * 0.3 + context.Stimulus.Intensity * 0.2, 0, 1),
            source: "audio");

        // Speech: surface the words so reasoning (which runs later) engages
        // with what was actually said.
        if (features.IsSpeech && _options.UseTranscription)
        {
            if (!string.IsNullOrWhiteSpace(features.Transcription))
            {
                context.Bag["audio.transcription"] = features.Transcription;
                context.EffectiveContent = features.Transcription!;
            }

            // If nothing has responded yet, acknowledge what was heard — but
            // only when no deeper module will answer the words themselves
            // (e.g. a spoken question that reasoning can handle).
            if (context.Response is null)
            {
                var transcript = features.Transcription ?? string.Empty;
                bool isQuestion = QuestionWords.Any(w => transcript.ToLowerInvariant().StartsWith(w, StringComparison.Ordinal))
                                  || transcript.Contains('?');
                bool deeperWillAnswer = Host?.HasModule<Reasoning.ReasoningModule>() == true;
                if (!(isQuestion && deeperWillAnswer))
                {
                    var reply = transcript.Length > 0
                        ? $"I heard you say: \"{Core.Mind.Truncate(transcript, 120)}\""
                        : "I heard a voice, but could not make out the words.";
                    context.Response = new ResponseDraft(reply, 0.55, 810, "audio-transcription");
                }
            }
        }
        else if (context.Response is null)
        {
            context.Response = new ResponseDraft(
                $"{features.Description} The {features.Category} was {features.DurationSeconds:0.0}s long.",
                0.4, 811, "audio-description");
        }
    }
}
