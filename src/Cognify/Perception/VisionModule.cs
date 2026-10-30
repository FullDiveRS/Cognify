using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Perception;

/// <summary>
/// The vision module: gives the mind eyes. When a stimulus is an
/// <see cref="StimulusKind.Image"/> or <see cref="StimulusKind.Video"/>, the
/// configured <see cref="IVisionBackend"/> turns it into a structured
/// <see cref="VisualScene"/> which is shared with the rest of the cognitive
/// cycle, remembered, and — for very rich or striking scenes — felt.
/// </summary>
public sealed class VisionModule : CognitiveModuleBase
{
    private readonly VisionOptions _options;
    private readonly object _gate = new();
    private int _scenesSeen;

    /// <summary>
    /// Creates the vision module. Pass a custom <paramref name="backend"/> for
    /// real image understanding; when null, the backend is taken from
    /// <see cref="Core.VisionOptions.Backend"/> if one was assigned there,
    /// falling back to the built-in heuristic analyzer.
    /// </summary>
    public VisionModule(VisionOptions options, IVisionBackend? backend = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Backend = backend
                  ?? options.Backend as IVisionBackend
                  ?? new SimulatedVisionBackend();
    }

    /// <inheritdoc />
    public override string Name => "vision";

    /// <inheritdoc />
    public override int Order => 11; // perception band, right after memory recall

    /// <summary>The active vision backend.</summary>
    public IVisionBackend Backend { get; }

    /// <summary>How many scenes have been perceived so far.</summary>
    public int ScenesSeen { get { lock (_gate) return _scenesSeen; } }

    /// <summary>The most recently perceived scene, if any.</summary>
    public VisualScene? LastScene { get; private set; }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        if (context.Stimulus.Kind is not (StimulusKind.Image or StimulusKind.Video)) return;

        var scene = Backend.Analyze(VisualInput.From(context.Stimulus));
        lock (_gate)
        {
            _scenesSeen++;
            LastScene = scene;
        }

        // Share the perception with every other module in this cycle.
        context.Bag["vision.scene"] = scene;
        context.Bag["vision.caption"] = scene.Caption;

        // Enrich the stimulus so text-based modules reason over what was seen.
        context.Bag["vision.summary"] = scene.Summarize();

        Host?.Stream?.Add("vision", $"saw: {scene.Summarize()}");

        // Remember what was seen — striking scenes stick.
        if (_options.RememberScenes)
        {
            var importance = Math.Clamp(0.35 + scene.Richness * 0.4 + context.Stimulus.Intensity * 0.2, 0, 1);
            Host?.Memory?.Remember(
                $"[saw] {scene.Summarize()}",
                Memory.MemoryKind.Episodic,
                importance: importance,
                source: "vision");
        }

        // Affective reaction: vivid and bright scenes lift the mood a little.
        if (scene.Richness > 0.5 || scene.Brightness > 0.8)
            Host?.EmotionalState?.Feel(EmotionType.Joy, 0.05 + scene.Richness * 0.1);
        else if (scene.Brightness < 0.2)
            Host?.EmotionalState?.Feel(EmotionType.Fear, 0.08);

        // If nothing else has responded yet, describe the scene.
        if (context.Response is null)
        {
            context.Response = new ResponseDraft(
                scene.Caption + " " + (scene.Objects.Count > 0
                    ? $"I make out {scene.Objects.Count} object(s): {string.Join(", ", scene.Objects.Take(4).Select(o => o.Label))}."
                    : "There is nothing distinctive I can name."),
                confidence: 0.45 + scene.Richness * 0.3,
                styleId: 800,
                styleName: "visual-description");
        }
    }
}
