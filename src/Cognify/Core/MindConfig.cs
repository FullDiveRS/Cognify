namespace Cognify.Core;

/// <summary>
/// How hard the mind thinks before it speaks — the same idea as thinking
/// budgets in modern language models. Higher levels generate more internal
/// thoughts (a larger "thinking token set") before committing to a reply.
/// </summary>
public enum ThinkingLevel
{
    /// <summary>A quick pause — one generated thought before answering.</summary>
    Low,

    /// <summary>The default: a brief think-through (two thoughts).</summary>
    Medium,

    /// <summary>A real deliberation (four thoughts).</summary>
    High,

    /// <summary>Deep deliberation (six thoughts).</summary>
    VeryHigh,

    /// <summary>Maximum deliberation (eight thoughts) — the full inner monologue.</summary>
    Max
}

/// <summary>
/// How far the mind is ALLOWED to think — a hard cap the model builder sets
/// at build time. The five presets match the user-facing ladder:
/// <list type="bullet">
/// <item><b>Only1</b> — the mind may only think at <see cref="ThinkingLevel.Low"/>.</item>
/// <item><b>Only2</b> — Low or Medium.</item>
/// <item><b>Only3</b> — Low, Medium or High.</item>
/// <item><b>Only4</b> — Low, Medium, High or Very High.</item>
/// <item><b>Only5</b> — everything up to Max (default).</item>
/// </list>
/// A level requested above the ceiling is silently clamped DOWN to the
/// ceiling's top level — a model capped at Only3 asked for Max still thinks,
/// just at High. The cap travels with the model into exported files
/// (<c>cognify.thinking.ceiling</c> GGUF metadata), so serving stacks
/// (llama.cpp / Unsloth) can read the minimum guarantee as well.
/// </summary>
public enum ThinkingCeiling
{
    /// <summary>Only 1: Low is the only allowed thinking level.</summary>
    Only1,
    /// <summary>Only 2: Low or Medium.</summary>
    Only2,
    /// <summary>Only 3: Low, Medium or High.</summary>
    Only3,
    /// <summary>Only 4: Low, Medium, High or Very High.</summary>
    Only4,
    /// <summary>Only 5: Low, Medium, High, Very High or Max — no cap (default).</summary>
    Only5
}

/// <summary>Helpers for <see cref="ThinkingCeiling"/>.</summary>
public static class ThinkingCeilingExtensions
{
    /// <summary>
    /// The highest <see cref="ThinkingLevel"/> the ceiling allows: Only1 → Low,
    /// Only2 → Medium, Only3 → High, Only4 → VeryHigh, Only5 → Max.
    /// </summary>
    public static ThinkingLevel MaxLevel(this ThinkingCeiling ceiling) => ceiling switch
    {
        ThinkingCeiling.Only1 => ThinkingLevel.Low,
        ThinkingCeiling.Only2 => ThinkingLevel.Medium,
        ThinkingCeiling.Only3 => ThinkingLevel.High,
        ThinkingCeiling.Only4 => ThinkingLevel.VeryHigh,
        _ => ThinkingLevel.Max
    };

    /// <summary>
    /// Clamps a requested level down to the ceiling. Levels at or below the
    /// ceiling pass through untouched; anything above it lands on the
    /// ceiling's top level.
    /// </summary>
    public static ThinkingLevel Clamp(this ThinkingCeiling ceiling, ThinkingLevel level) =>
        level > ceiling.MaxLevel() ? ceiling.MaxLevel() : level;
}

/// <summary>Options for the reasoning module.</summary>
public sealed class ReasoningOptions
{
    /// <summary>Maximum forward-chaining depth (number of inference passes).</summary>
    public int MaxInferenceDepth { get; set; } = 6;

    /// <summary>How many co-occurrence observations are required before inducing a new fact.</summary>
    public double InductionThreshold { get; set; } = 5;

    /// <summary>Enable unsupervised induction of new facts from repeated patterns.</summary>
    public bool AllowInduction { get; set; } = true;

    /// <summary>Enable abduction (finding the best explanation for an observation).</summary>
    public bool AllowAbduction { get; set; } = true;

    /// <summary>Fluent setter: <see cref="MaxInferenceDepth"/>.</summary>
    public ReasoningOptions Depth(int value) { MaxInferenceDepth = value; return this; }

    /// <summary>Fluent setter: <see cref="InductionThreshold"/>.</summary>
    public ReasoningOptions InduceAfter(int value) { InductionThreshold = value; return this; }
}

/// <summary>Options for the thinking module.</summary>
public sealed class ThinkingOptions
{
    /// <summary>Maximum steps recorded in a chain of thought.</summary>
    public int MaxThoughtSteps { get; set; } = 8;

    /// <summary>How many candidate response styles are considered in deliberation.</summary>
    public int DeliberationCandidates { get; set; } = 6;

    /// <summary>Minimum confidence for a conclusion to be stated assertively.</summary>
    public double ConfidenceFloor { get; set; } = 0.25;

    /// <summary>
    /// The mind's own thinking switch — can this model think before speaking
    /// at all? False short-circuits deliberation: the mind answers directly
    /// (from knowledge or a concise fallback) without the thought generator
    /// or the chain of thought. Mirrors the builder's
    /// <c>WithThinking(..., enabled: false)</c> but stays changeable at
    /// runtime through the options object.
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>How hard the mind thinks before speaking (token-set size).</summary>
    public ThinkingLevel Level { get; set; } = ThinkingLevel.Medium;

    /// <summary>
    /// The hard cap on thinking — the highest level this model may ever
    /// use (builder preset Only1..Only5). Requests above the ceiling are
    /// clamped down at runtime; see <see cref="ThinkingCeiling"/>.
    /// </summary>
    public ThinkingCeiling Ceiling { get; set; } = ThinkingCeiling.Only5;

    /// <summary>
    /// Flavor of the generated thoughts: "balanced", "focused", "playful",
    /// "poetic" or "plain". Unknown values fall back to "balanced".
    /// </summary>
    public string Tone { get; set; } = "balanced";

    /// <summary>
    /// Explicit thinking token budget overriding <see cref="Level"/> when
    /// greater than zero: roughly budget/50 generated thoughts (1..10).
    /// </summary>
    public int ThoughtBudget { get; set; } = 0;

    /// <summary>Fluent setter: <see cref="MaxThoughtSteps"/>.</summary>
    public ThinkingOptions Steps(int value) { MaxThoughtSteps = value; return this; }

    /// <summary>
    /// Fluent setter: <see cref="Ceiling"/> — caps how far this model may
    /// ever think (Only1 = Low only … Only5 = everything). The current
    /// <see cref="Level"/> is re-clamped immediately, and every later
    /// <see cref="At"/> request is clamped too.
    /// </summary>
    public ThinkingOptions UpTo(ThinkingCeiling value)
    {
        Ceiling = value;
        Level = Ceiling.Clamp(Level);
        MaxThoughtSteps = StepCeiling(Level);
        return this;
    }

    /// <summary>Fluent setter: <see cref="Level"/> — clamped to <see cref="Ceiling"/> (also retunes the step ceiling).</summary>
    public ThinkingOptions At(ThinkingLevel value)
    {
        Level = Ceiling.Clamp(value);
        MaxThoughtSteps = StepCeiling(Level);
        return this;
    }

    private int StepCeiling(ThinkingLevel level) => level switch
    {
        ThinkingLevel.Low => 6,
        ThinkingLevel.Medium => 8,
        ThinkingLevel.High => 12,
        ThinkingLevel.VeryHigh => 16,
        _ => 24
    };

    /// <summary>Fluent setter: <see cref="Enabled"/>.</summary>
    public ThinkingOptions Enable(bool value) { Enabled = value; return this; }

    /// <summary>Fluent setter: <see cref="Tone"/>.</summary>
    public ThinkingOptions WithTone(string value) { Tone = value ?? "balanced"; return this; }

    /// <summary>Fluent setter: <see cref="ThoughtBudget"/>.</summary>
    public ThinkingOptions Budget(int tokens) { ThoughtBudget = Math.Max(0, tokens); return this; }

    /// <summary>
    /// Generated-thought count for the current level/budget — the size of
    /// the mind's "thinking token set" (1..10) per cycle.
    /// </summary>
    public int ThoughtRounds
    {
        get
        {
            if (ThoughtBudget > 0) return Math.Clamp(ThoughtBudget / 50, 1, 10);
            return Ceiling.Clamp(Level) switch
            {
                ThinkingLevel.Low => 1,
                ThinkingLevel.Medium => 2,
                ThinkingLevel.High => 4,
                ThinkingLevel.VeryHigh => 6,
                _ => 8
            };
        }
    }
}

/// <summary>
/// Controls what the mind SHOWS in conversation. Everything here is
/// presentation, not cognition — the internals always run and stay available
/// programmatically (<see cref="Mind.Feeling"/>, <c>response.Chain</c>,
/// <c>response.Confidence</c>), but chat stays natural by default.
/// </summary>
public sealed class ExpressionOptions
{
    /// <summary>
    /// When true, feelings are ANNOUNCED in the mechanical introspection
    /// style — "I am picking up a quiet, neutral tone here (my current read:
    /// anticipation)". Default false: feelings color word choice naturally
    /// instead, and stay readable any time via <see cref="Mind.Feeling"/>.
    /// </summary>
    public bool ShowFeelingReads { get; set; } = false;

    /// <summary>
    /// When true (default), replies may end with a natural follow-up
    /// question ("Tell me more about them - I am curious who they are.").
    /// Set false for a mind that only answers and never asks.
    /// </summary>
    public bool AskFollowUpQuestions { get; set; } = true;

    /// <summary>
    /// When true, hosts may surface the response meta line (confidence,
    /// style, cycle). This is a convention flag for chat hosts — the data
    /// always lives on <see cref="MindResponse"/>.
    /// </summary>
    public bool ShowResponseMeta { get; set; } = false;

    /// <summary>
    /// When true, hosts may show the thought-generator trace (the mind's
    /// generated thoughts before speaking). Default false — thoughts stay
    /// on <c>response.Chain</c> / <see cref="ThinkingModule.LastThoughts"/>.
    /// </summary>
    public bool ShowThinking { get; set; } = false;

    /// <summary>Fluent setter: <see cref="ShowFeelingReads"/>.</summary>
    public ExpressionOptions Feelings(bool value) { ShowFeelingReads = value; return this; }

    /// <summary>Fluent setter: <see cref="AskFollowUpQuestions"/>.</summary>
    public ExpressionOptions FollowUps(bool value) { AskFollowUpQuestions = value; return this; }

    /// <summary>Fluent setter: <see cref="ShowResponseMeta"/>.</summary>
    public ExpressionOptions Meta(bool value) { ShowResponseMeta = value; return this; }

    /// <summary>Fluent setter: <see cref="ShowThinking"/>.</summary>
    public ExpressionOptions Thinking(bool value) { ShowThinking = value; return this; }
}

/// <summary>Options for the self-learning module.</summary>
public sealed class LearningOptions
{
    /// <summary>Dimension of the hashed feature vector fed to the micro-network.</summary>
    public int FeatureDim { get; set; } = 32;

    /// <summary>Hidden layer size of the micro-network.</summary>
    public int HiddenUnits { get; set; } = 20;

    /// <summary>Gradient-descent learning rate.</summary>
    public double LearningRate { get; set; } = 0.08;

    /// <summary>Base exploration rate for trying new response styles (0..1).</summary>
    public double BaseExploration { get; set; } = 0.15;

    /// <summary>How many recent experiences to keep for training.</summary>
    public int ExperienceCapacity { get; set; } = 200;

    /// <summary>Random seed for network initialization (deterministic by default).</summary>
    public int Seed { get; set; } = 42;

    /// <summary>Fluent setter: <see cref="LearningRate"/>.</summary>
    public LearningOptions Rate(double value) { LearningRate = value; return this; }

    /// <summary>Fluent setter: <see cref="BaseExploration"/>.</summary>
    public LearningOptions Exploration(double value) { BaseExploration = value; return this; }
}

/// <summary>Options for the emotion module.</summary>
public sealed class EmotionOptions
{
    /// <summary>Global multiplier applied to emotional reactions.</summary>
    public double IntensityScale { get; set; } = 1.0;

    /// <summary>Fraction of intensity lost per tick (0..1). Higher = emotions fade faster.</summary>
    public double DecayRate { get; set; } = 0.08;

    /// <summary>How fast mood follows emotion (0..1). Higher = mood follows faster.</summary>
    public double MoodInertia { get; set; } = 0.10;

    /// <summary>
    /// Fresh-appraisal surge (0..1) above which emotion can short-circuit
    /// deliberation — the dual-process "answer before thinking" gate
    /// (Zajonc's affect primacy). Blame/dismissal triggers bypass the gate
    /// (reflexive), everything else must cross this threshold.
    /// </summary>
    public double ImpulseThreshold { get; set; } = 0.55;

    /// <summary>Fluent setter: <see cref="IntensityScale"/>.</summary>
    public EmotionOptions Intensity(double value) { IntensityScale = value; return this; }

    /// <summary>Fluent setter: <see cref="DecayRate"/>.</summary>
    public EmotionOptions Decay(double value) { DecayRate = value; return this; }

    /// <summary>Fluent setter: <see cref="ImpulseThreshold"/>.</summary>
    public EmotionOptions Impulse(double threshold) { ImpulseThreshold = threshold; return this; }
}

/// <summary>Options for the memory module.</summary>
public sealed class MemoryOptions
{
    /// <summary>How many items the short-term buffer holds before the oldest is evicted.</summary>
    public int ShortTermCapacity { get; set; } = 12;

    /// <summary>Importance (0..1) above which a memory is consolidated into long-term store.</summary>
    public double LongTermThreshold { get; set; } = 0.60;

    /// <summary>Per-tick strength decay for long-term memories (higher = faster forgetting).</summary>
    public double DecayRate { get; set; } = 0.01;

    /// <summary>Strength gained each time a memory is recalled (rehearsal effect).</summary>
    public double RecallBoost { get; set; } = 0.05;

    /// <summary>Build associative links between co-occurring concepts.</summary>
    public bool UseAssociations { get; set; } = true;

    /// <summary>Fluent setter: <see cref="ShortTermCapacity"/>.</summary>
    public MemoryOptions Capacity(int value) { ShortTermCapacity = value; return this; }
}

/// <summary>A goal to give the mind at build time.</summary>
public sealed class GoalSpec
{
    /// <summary>Goal name (e.g. "learn-about-space").</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Human-readable description.</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Priority 0..1. Higher = pursued sooner.</summary>
    public double Priority { get; set; } = 0.5;

    /// <summary>Tags drive selection of planning strategies (curiosity / social / achievement / calm).</summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>Progress target (0..1+). Goal is achieved when progress reaches target.</summary>
    public double Target { get; set; } = 1.0;
}

/// <summary>Options for the goal module.</summary>
public sealed class GoalOptions
{
    /// <summary>Goals the mind starts with.</summary>
    public List<GoalSpec> InitialGoals { get; set; } = new();

    /// <summary>How fast drives grow per tick (0..1).</summary>
    public double DriveGrowth { get; set; } = 0.02;

    /// <summary>Drive level at which the mind autonomously spawns a goal.</summary>
    public double DriveSpawnThreshold { get; set; } = 0.75;

    /// <summary>Fluent helper: add an initial goal.</summary>
    public GoalOptions AddGoal(string name, double priority = 0.5, params string[] tags)
    {
        InitialGoals.Add(new GoalSpec { Name = name, Priority = priority, Tags = tags.ToList() });
        return this;
    }
}

/// <summary>Big Five personality trait values (0..1).</summary>
public sealed class PersonalityOptions
{
    /// <summary>Curiosity, imagination, preference for novelty.</summary>
    public double Openness { get; set; } = 0.5;

    /// <summary>Organization, deliberation, goal-directedness.</summary>
    public double Conscientiousness { get; set; } = 0.5;

    /// <summary>Sociability, enthusiasm, positive affect.</summary>
    public double Extraversion { get; set; } = 0.5;

    /// <summary>Warmth, trust, cooperation.</summary>
    public double Agreeableness { get; set; } = 0.5;

    /// <summary>Emotional sensitivity; high values amplify negative affect.</summary>
    public double Neuroticism { get; set; } = 0.5;

    /// <summary>Fluent setter for all five traits at once. Unset values stay 0.5.</summary>
    public PersonalityOptions Traits(double? openness = null, double? conscientiousness = null,
        double? extraversion = null, double? agreeableness = null, double? neuroticism = null)
    {
        if (openness.HasValue) Openness = openness.Value;
        if (conscientiousness.HasValue) Conscientiousness = conscientiousness.Value;
        if (extraversion.HasValue) Extraversion = extraversion.Value;
        if (agreeableness.HasValue) Agreeableness = agreeableness.Value;
        if (neuroticism.HasValue) Neuroticism = neuroticism.Value;
        return this;
    }
}

/// <summary>Options for persistence.</summary>
public sealed class PersistenceOptions
{
    /// <summary>Default SQLite file used by <see cref="Mind.SaveAsync"/> when no path is given.</summary>
    public string? SqlitePath { get; set; }

    /// <summary>If greater than 0, the mind auto-saves to <see cref="SqlitePath"/> every N cycles.</summary>
    public int AutoSaveEveryCycles { get; set; } = 0;
}

/// <summary>
/// Marker for pluggable per-module backends (vision, audio, media generation).
/// Core cannot reference the concrete backend interfaces (they live with their
/// modules), so option classes carry backends through this marker and the
/// module casts them. Assign e.g. an <c>IVisionBackend</c> to
/// <see cref="VisionOptions.Backend"/>.
/// </summary>
public interface IModuleBackend { }

/// <summary>Options for the vision module.</summary>
public sealed class VisionOptions
{
    /// <summary>Minimum confidence (0..1) for a detected object to be reported.</summary>
    public double DetectionThreshold { get; set; } = 0.25;

    /// <summary>Maximum number of objects reported per scene.</summary>
    public int MaxObjects { get; set; } = 8;

    /// <summary>Store what the mind sees as an episodic-style memory.</summary>
    public bool RememberScenes { get; set; } = true;

    /// <summary>
    /// Optional custom vision backend. Assign an <c>IVisionBackend</c>
    /// (Cognify.Perception) to replace the built-in heuristic analyzer.
    /// </summary>
    public IModuleBackend? Backend { get; set; }

    /// <summary>Fluent setter: <see cref="DetectionThreshold"/>.</summary>
    public VisionOptions Threshold(double value) { DetectionThreshold = value; return this; }
}

/// <summary>Options for the audio module.</summary>
public sealed class AudioOptions
{
    /// <summary>When true, a detected transcription replaces the raw content so downstream modules reason over the words.</summary>
    public bool UseTranscription { get; set; } = true;

    /// <summary>Loudness (0..1) above which the mind treats a sound as salient.</summary>
    public double SalienceThreshold { get; set; } = 0.45;

    /// <summary>
    /// Optional custom audio backend. Assign an <c>IAudioBackend</c>
    /// (Cognify.Audio) to replace the built-in heuristic analyzer.
    /// </summary>
    public IModuleBackend? Backend { get; set; }

    /// <summary>Fluent setter: <see cref="SalienceThreshold"/>.</summary>
    public AudioOptions Salience(double value) { SalienceThreshold = value; return this; }
}

/// <summary>Options for the embeddings module.</summary>
public sealed class EmbeddingOptions
{
    /// <summary>Dimensionality of the embedding vectors (feature-hash buckets).</summary>
    public int Dimensions { get; set; } = 64;

    /// <summary>How many semantic neighbours to surface per cycle.</summary>
    public int Neighbors { get; set; } = 3;

    /// <summary>Maximum number of embeddings retained in the semantic store.</summary>
    public int Capacity { get; set; } = 500;

    /// <summary>Similarity (0..1) above which two texts are considered near-duplicates.</summary>
    public double DuplicateThreshold { get; set; } = 0.92;

    /// <summary>Fluent setter: <see cref="Dimensions"/>.</summary>
    public EmbeddingOptions Dim(int value) { Dimensions = value; return this; }
}

/// <summary>Options for the multilingual module.</summary>
public sealed class MultilingualOptions
{
    /// <summary>BCP-47-ish code of the mind's preferred language (e.g. "en", "es", "zh").</summary>
    public string PreferredLanguage { get; set; } = "en";

    /// <summary>Automatically translate non-preferred input into the preferred language.</summary>
    public bool AutoTranslate { get; set; } = true;

    /// <summary>Additional phrase translations layered over the built-in phrasebook.</summary>
    public Dictionary<string, string> CustomPhrases { get; set; } = new();

    /// <summary>Fluent setter: <see cref="PreferredLanguage"/>.</summary>
    public MultilingualOptions Prefer(string language) { PreferredLanguage = language; return this; }
}

/// <summary>Options for the tool-use module.</summary>
public sealed class ToolOptions
{
    /// <summary>Register the built-in tools (clock, calculator, text-stats) at build time.</summary>
    public bool UseBuiltinTools { get; set; } = true;

    /// <summary>Automatically invoke a tool when the stimulus clearly requests one.</summary>
    public bool AutoInvoke { get; set; } = true;

    /// <summary>
    /// Registers the manual-only <c>web-search</c> tool at build time.
    /// Web search NEVER auto-fires inside the cognitive cycle (the tool is
    /// manual-only); it only runs when explicitly requested — the
    /// <c>/searchweb</c> command, a <c>use tool web-search</c> request, or
    /// <c>mind.SearchWeb(...)</c>.
    /// </summary>
    public bool AllowWebSearch { get; set; } = false;

    /// <summary>
    /// Optional custom web-search backend. Assign an
    /// <c>Cognify.Tools.IWebSearchBackend</c> to replace the default
    /// DuckDuckGo HTTP backend (e.g. Bing, SerpAPI, a local index).
    /// </summary>
    public object? WebSearchBackend { get; set; }

    /// <summary>Phrases that signal a tool request (matched case-insensitively).</summary>
    public List<string> InvokeTriggers { get; set; } = new()
    {
        "use tool", "call tool", "run tool", "invoke tool",
        "calculate", "what time", "what's the time", "current time",
        "count words", "how many words"
    };

    /// <summary>Fluent setter: <see cref="AutoInvoke"/>.</summary>
    public ToolOptions Auto(bool value) { AutoInvoke = value; return this; }

    /// <summary>Fluent setter: <see cref="AllowWebSearch"/>.</summary>
    public ToolOptions WebSearch(bool value) { AllowWebSearch = value; return this; }
}

/// <summary>Options for the code module.</summary>
public sealed class CodeOptions
{
    /// <summary>Default language for generated code ("csharp", "python", "javascript", "sql").</summary>
    public string PreferredLanguage { get; set; } = "csharp";

    /// <summary>Namespace used by generated C# code.</summary>
    public string Namespace { get; set; } = "Generated";

    /// <summary>Analyze code found in the stimulus (fenced blocks or code-looking text).</summary>
    public bool AnalyzeInput { get; set; } = true;

    /// <summary>Fluent setter: <see cref="PreferredLanguage"/>.</summary>
    public CodeOptions Language(string value) { PreferredLanguage = value; return this; }
}

/// <summary>Options for the media generation module.</summary>
public sealed class GenerationOptions
{
    /// <summary>Default clip length for generated video (seconds).</summary>
    public double DefaultVideoSeconds { get; set; } = 5;

    /// <summary>Frames per second for generated video.</summary>
    public int FramesPerSecond { get; set; } = 24;

    /// <summary>Sample rate for generated audio (Hz).</summary>
    public int SampleRate { get; set; } = 8000;

    /// <summary>Emit real, playable WAV bytes for speech/audio generation (deterministic tones).</summary>
    public bool EmitRealWav { get; set; } = true;

    /// <summary>How many generated artifacts to keep in the gallery.</summary>
    public int GalleryCapacity { get; set; } = 20;

    /// <summary>Directory where generated media files are saved (default "cognify_media").</summary>
    public string MediaDirectory { get; set; } = "cognify_media";

    /// <summary>Save every generated artifact to <see cref="MediaDirectory"/> as a real file.</summary>
    public bool EnableFileOutput { get; set; } = true;

    /// <summary>
    /// Optional custom media backend. Assign an <c>IMediaBackend</c>
    /// (Cognify.Generation) to replace the built-in rendering generator.
    /// </summary>
    public IModuleBackend? Backend { get; set; }

    /// <summary>Fluent setter: <see cref="DefaultVideoSeconds"/>.</summary>
    public GenerationOptions VideoSeconds(double value) { DefaultVideoSeconds = value; return this; }

    /// <summary>Fluent setter: <see cref="MediaDirectory"/>.</summary>
    public GenerationOptions MediaDir(string value) { MediaDirectory = value; return this; }
}

/// <summary>
/// Complete, serializable configuration for a <see cref="Mind"/>. Individual option
/// groups can be tuned via <c>MindBuilder.With*(options => ...)</c>.
/// </summary>
public sealed class MindConfig
{
    /// <summary>The mind's display name.</summary>
    public string Name { get; set; } = "Cognify";

    /// <summary>
    /// When true, capability-gated conveniences (<c>See</c>, <c>Hear</c>,
    /// <c>Teach</c>, <c>Reinforce</c>, <c>InvokeTool</c>, <c>GenerateMedia</c>,
    /// <c>DetectLanguage</c>, <c>SemanticSimilarity</c>) degrade gracefully
    /// instead of throwing <see cref="MindConfigurationException"/> when the
    /// matching module was not enabled. Default is false (strict mode).
    /// </summary>
    public bool LenientMode { get; set; } = false;

    /// <summary>Optional description of the mind's purpose.</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Reasoning options.</summary>
    public ReasoningOptions Reasoning { get; set; } = new();

    /// <summary>Thinking options.</summary>
    public ThinkingOptions Thinking { get; set; } = new();

    /// <summary>Expression options — what the mind shows in chat.</summary>
    public ExpressionOptions Expression { get; set; } = new();

    /// <summary>Self-learning options.</summary>
    public LearningOptions Learning { get; set; } = new();

    /// <summary>Emotion options.</summary>
    public EmotionOptions Emotions { get; set; } = new();

    /// <summary>Memory options.</summary>
    public MemoryOptions Memory { get; set; } = new();

    /// <summary>Goal/motivation options.</summary>
    public GoalOptions Goals { get; set; } = new();

    /// <summary>Personality options.</summary>
    public PersonalityOptions Personality { get; set; } = new();

    /// <summary>Gender identity options (how the mind describes itself + its pronouns).</summary>
    public GenderOptions Gender { get; set; } = new();

    /// <summary>Persistence options.</summary>
    public PersistenceOptions Persistence { get; set; } = new();

    /// <summary>Vision options.</summary>
    public VisionOptions Vision { get; set; } = new();

    /// <summary>Audio options.</summary>
    public AudioOptions Audio { get; set; } = new();

    /// <summary>Embedding options.</summary>
    public EmbeddingOptions Embeddings { get; set; } = new();

    /// <summary>Multilingual options.</summary>
    public MultilingualOptions Multilingual { get; set; } = new();

    /// <summary>Tool-use options.</summary>
    public ToolOptions Tools { get; set; } = new();

    /// <summary>Code options.</summary>
    public CodeOptions Code { get; set; } = new();

    /// <summary>Media generation options.</summary>
    public GenerationOptions Generation { get; set; } = new();
}
