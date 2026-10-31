using Cognify.Emotions;
using Cognify.Goals;
using Cognify.Learning;
using Cognify.Memory;
using Cognify.Persistence;
using Cognify.Personality;
using Cognify.Reasoning;
using Cognify.Thinking;

namespace Cognify.Core;

/// <summary>
/// Fluent builder for creating a <see cref="Mind"/>. Capabilities are opt-in:
/// only the modules you enable are attached.
///
/// <example>
/// <code>
/// var mind = MindBuilder.Create("Aria")
///     .Describe("A curious assistant")
///     .WithPersonality(p => p.Traits(openness: 0.9, extraversion: 0.7))
///     .WithEmotions(e => e.Intensity(0.9))
///     .WithReasoning()
///     .WithThinking()
///     .WithSelfLearning(l => l.Rate(0.1))
///     .WithMemory()
///     .WithGoals(g => g.AddGoal("learn", 0.8, "curiosity"))
///     .WithPersistence("aria.db")
///     .Build();
/// </code>
/// </example>
/// </summary>
public sealed class MindBuilder
{
    private readonly MindConfig _config = new();
    private readonly Dictionary<string, bool> _enabled = new();
    private readonly List<ICognitiveModule> _custom = new();
    private string _name = "Cognify";

    private MindBuilder() { }

    /// <summary>Starts building a mind with the given name.</summary>
    public static MindBuilder Create(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new MindConfigurationException("A mind needs a name.");
        return new MindBuilder { _name = name.Trim() };
    }

    /// <summary>Human-readable description of the mind's purpose.</summary>
    public MindBuilder Describe(string description)
    {
        _config.Description = description ?? string.Empty;
        return this;
    }

    // ------------------------------------------------------------------
    // Capability toggles
    // ------------------------------------------------------------------

    /// <summary>Enables the reasoning module (facts, rules, deduction / induction / abduction).</summary>
    public MindBuilder WithReasoning(Action<ReasoningOptions>? configure = null, bool enabled = true)
    {
        _enabled["reasoning"] = enabled;
        configure?.Invoke(_config.Reasoning);
        return this;
    }

    /// <summary>Enables the thinking module (chain of thought, deliberation, inner monologue).</summary>
    /// <param name="configure">Fine-tuning (level, tone, budget, ceiling).</param>
    /// <param name="enabled">The model's thinking switch — false means it cannot think before speaking at all.</param>
    /// <param name="ceiling">
    /// Hard cap on how far this model may ever think:
    /// <c>Only1</c> = Low only, <c>Only2</c> = up to Medium, <c>Only3</c> = up to High,
    /// <c>Only4</c> = up to Very High, <c>Only5</c> = everything (default).
    /// Requests above the cap are clamped down — e.g. a ceiling of Only3 asked
    /// for Max thinks at High. The cap is exported with the model
    /// (<c>cognify.thinking.ceiling</c>), so serving stacks can read it.
    /// </param>
    public MindBuilder WithThinking(Action<ThinkingOptions>? configure = null, bool enabled = true,
        ThinkingCeiling ceiling = ThinkingCeiling.Only5)
    {
        _enabled["thinking"] = enabled;
        _config.Thinking.UpTo(ceiling);
        configure?.Invoke(_config.Thinking);
        return this;
    }

    /// <summary>
    /// Configures what the mind SHOWS in conversation — the presentation
    /// layer, separate from cognition:
    /// <code>
    /// .WithExpression(e => e.Feelings(true)   // announce feelings mechanically
    ///                    .FollowUps(false)    // never end replies with questions
    ///                    .Meta(true)          // hosts may print confidence/style/cycle
    ///                    .Thinking(true))     // hosts may show the thought trace
    /// </code>
    /// Defaults keep chat natural: feelings color wording silently, the
    /// state stays on <see cref="Mind.Feeling"/>, and internals stay on the
    /// response object.
    /// </summary>
    public MindBuilder WithExpression(Action<ExpressionOptions>? configure)
    {
        configure?.Invoke(_config.Expression);
        return this;
    }

    /// <summary>Enables the self-learning module (micro neural net + reinforcement + induction).</summary>
    public MindBuilder WithSelfLearning(Action<LearningOptions>? configure = null, bool enabled = true)
    {
        _enabled["learning"] = enabled;
        configure?.Invoke(_config.Learning);
        return this;
    }

    /// <summary>Enables the emotion module (appraisal, mood, decay).</summary>
    public MindBuilder WithEmotions(Action<EmotionOptions>? configure = null, bool enabled = true)
    {
        _enabled["emotions"] = enabled;
        configure?.Invoke(_config.Emotions);
        return this;
    }

    /// <summary>Enables the memory module (short-term, long-term, episodic + associations).</summary>
    public MindBuilder WithMemory(Action<MemoryOptions>? configure = null, bool enabled = true)
    {
        _enabled["memory"] = enabled;
        configure?.Invoke(_config.Memory);
        return this;
    }

    /// <summary>Enables the goal module (drives, goals, planning).</summary>
    public MindBuilder WithGoals(Action<GoalOptions>? configure = null, bool enabled = true)
    {
        _enabled["goals"] = enabled;
        configure?.Invoke(_config.Goals);
        return this;
    }

    /// <summary>Enables the personality module (Big Five traits shaping behavior).</summary>
    public MindBuilder WithPersonality(Action<PersonalityOptions>? configure = null, bool enabled = true)
    {
        _enabled["personality"] = enabled;
        configure?.Invoke(_config.Personality);
        return this;
    }

    /// <summary>
    /// Sets the mind's gender identity: how it describes itself and the
    /// pronouns others use for it. Two presets exist —
    /// <c>.WithGender("girl", PronounSet.Female)</c> (she/her) and
    /// <c>.WithGender("boy", PronounSet.Male)</c> (he/him) — or leave the
    /// mind genderless entirely. they/them is NOT available as a mind's own
    /// pronoun set: it is reserved for describing other people whose gender
    /// the mind does not know. A mind without a gender never mentions one —
    /// behavior is unchanged.
    /// </summary>
    public MindBuilder WithGender(string identity, PronounSet? pronouns = null)
    {
        _config.Gender.Set(identity, pronouns);
        return this;
    }

    /// <summary>Sets the mind's gender identity via fluent configuration.</summary>
    public MindBuilder WithGender(Action<GenderOptions> configure)
    {
        configure?.Invoke(_config.Gender);
        return this;
    }

    /// <summary>Sets just the mind's pronoun set (no identity word).</summary>
    public MindBuilder WithPronouns(PronounSet pronouns)
    {
        _config.Gender.With(pronouns);
        return this;
    }

    /// <summary>
    /// Enables the vision module (scene analysis, object detection,
    /// captions). Assign <c>v.Backend = myIVisionBackend</c> for real image
    /// understanding.
    /// </summary>
    public MindBuilder WithVision(Action<VisionOptions>? configure = null, bool enabled = true)
    {
        _enabled["vision"] = enabled;
        configure?.Invoke(_config.Vision);
        return this;
    }

    /// <summary>
    /// Enables the audio module (sound classification, speech transcription).
    /// Assign <c>a.Backend = myIAudioBackend</c> for real speech recognition.
    /// </summary>
    public MindBuilder WithAudio(Action<AudioOptions>? configure = null, bool enabled = true)
    {
        _enabled["audio"] = enabled;
        configure?.Invoke(_config.Audio);
        return this;
    }

    /// <summary>
    /// Enables the embeddings module (semantic vectors, similarity search,
    /// semantic memory). Implements real hashed n-gram embeddings by default.
    /// </summary>
    public MindBuilder WithEmbeddings(Action<EmbeddingOptions>? configure = null, bool enabled = true)
    {
        _enabled["embeddings"] = enabled;
        configure?.Invoke(_config.Embeddings);
        return this;
    }

    /// <summary>
    /// Enables the multilingual module (language detection, phrasebook
    /// translation, auto-translation into the preferred language).
    /// </summary>
    public MindBuilder WithMultilingual(Action<MultilingualOptions>? configure = null, bool enabled = true)
    {
        _enabled["multilingual"] = enabled;
        configure?.Invoke(_config.Multilingual);
        return this;
    }

    /// <summary>
    /// Enables the tool-use module (tool registry, natural-language tool
    /// invocation, built-in clock/calculator/text-stats tools). Register
    /// your own tools on <c>mind.Tools.Registry</c> after building.
    /// </summary>
    public MindBuilder WithTools(Action<ToolOptions>? configure = null, bool enabled = true)
    {
        _enabled["tools"] = enabled;
        configure?.Invoke(_config.Tools);
        return this;
    }

    /// <summary>
    /// Enables the tool-use module AND registers the manual-only
    /// <c>web-search</c> tool (DuckDuckGo HTTP backend unless a custom
    /// <c>Tools.WebSearchBackend</c> was assigned). Web search never fires on
    /// its own inside the cognitive cycle — it runs only when explicitly
    /// requested via <see cref="Mind.SearchWeb"/> or the demo's
    /// <c>/searchweb</c> command.
    /// </summary>
    public MindBuilder WithWebSearch(Action<ToolOptions>? configure = null, bool enabled = true)
    {
        _enabled["tools"] = enabled;
        _config.Tools.AllowWebSearch = enabled;
        configure?.Invoke(_config.Tools);
        return this;
    }

    /// <summary>
    /// Enables the code module (code generation scaffolds + real static
    /// analysis of snippets).
    /// </summary>
    public MindBuilder WithCode(Action<CodeOptions>? configure = null, bool enabled = true)
    {
        _enabled["code"] = enabled;
        configure?.Invoke(_config.Code);
        return this;
    }

    /// <summary>
    /// Enables the media generation module (image/video specs + real WAV
    /// synthesis for speech/audio). Assign <c>g.Backend = myIMediaBackend</c>
    /// for a real generator.
    /// </summary>
    public MindBuilder WithMediaGeneration(Action<GenerationOptions>? configure = null, bool enabled = true)
    {
        _enabled["generation"] = enabled;
        configure?.Invoke(_config.Generation);
        return this;
    }

    /// <summary>Enables every core module with default options.</summary>
    public MindBuilder WithAllModules()
    {
        foreach (var key in new[] { "reasoning", "thinking", "learning", "emotions", "memory", "goals", "personality",
                                    "vision", "audio", "embeddings", "multilingual", "tools", "code", "generation" })
            _enabled[key] = true;
        return this;
    }

    /// <summary>
    /// Known capability names accepted by <see cref="Without"/> and <see cref="WithAllModules"/>.
    /// </summary>
    public static IReadOnlyCollection<string> KnownCapabilities { get; } = new[]
    {
        "reasoning", "thinking", "learning", "emotions", "memory", "goals", "personality",
        "vision", "audio", "embeddings", "multilingual", "tools", "code", "generation"
    };

    /// <summary>
    /// Turns capabilities <b>off</b>. Useful after <see cref="WithAllModules"/> to strip
    /// what you don't need:
    /// <code>
    /// var mind = MindBuilder.Create("ExampleAI")
    ///     .WithAllModules()
    ///     .Without("audio", "tools", "generation")
    ///     .Build();
    /// </code>
    /// Note the builder is opt-in by default — calling only
    /// <c>WithThinking().WithSelfLearning().WithMemory()</c> already excludes
    /// everything else, so <see cref="Without"/> is mainly for pruning a full set.
    /// </summary>
    public MindBuilder Without(params string[] capabilities)
    {
        if (capabilities is null || capabilities.Length == 0)
            throw new MindConfigurationException("Without() needs at least one capability name. " +
                $"Known: {string.Join(", ", KnownCapabilities)}");
        foreach (var raw in capabilities)
        {
            var key = raw?.Trim().ToLowerInvariant() ?? string.Empty;
            if (!KnownCapabilities.Contains(key))
                throw new MindConfigurationException(
                    $"Unknown capability '{raw}'. Known: {string.Join(", ", KnownCapabilities)}");
            _enabled[key] = false;
        }
        return this;
    }

    // ------------------------------------------------------------------
    // Extension + persistence
    // ------------------------------------------------------------------

    /// <summary>Registers a custom cognitive module ("and more" — extend the mind your way).</summary>
    public MindBuilder WithModule(ICognitiveModule module)
    {
        if (module is null) throw new ArgumentNullException(nameof(module));
        _custom.Add(module);
        return this;
    }

    /// <summary>Configures SQLite persistence for save/load and optional auto-save.</summary>
    /// <param name="path">SQLite database file path.</param>
    /// <param name="autoSaveEveryCycles">Auto-save interval in cycles; 0 disables auto-save.</param>
    public MindBuilder WithPersistence(string path, int autoSaveEveryCycles = 0)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new MindConfigurationException("Persistence path cannot be empty.");
        _config.Persistence.SqlitePath = path;
        _config.Persistence.AutoSaveEveryCycles = autoSaveEveryCycles;
        return this;
    }

    /// <summary>
    /// Enables <b>lenient mode</b>: when a capability was not enabled at build
    /// time, its convenience methods degrade gracefully instead of throwing
    /// <see cref="MindConfigurationException"/> — you don't <i>need</i> a
    /// module like vision to keep talking to the mind:
    /// <list type="bullet">
    /// <item><c>See</c> / <c>Hear</c> still run the full cognitive cycle on the
    /// description (the mind registers it as plain text, minus scene/audio
    /// analysis).</item>
    /// <item><c>Teach</c> keeps the sentence as a plain memory and returns 0.</item>
    /// <item><c>Reinforce</c> still stirs the mind's emotions, but trains nothing.</item>
    /// <item><c>InvokeTool</c> returns a failed <c>ToolResult</c>.</item>
    /// <item><c>GenerateMedia</c> / <c>DetectLanguage</c> return null and
    /// <c>SemanticSimilarity</c> returns 0.</item>
    /// </list>
    /// Strict mode (the default) keeps throwing so misconfigurations are loud.
    /// </summary>
    public MindBuilder Lenient(bool lenient = true)
    {
        _config.LenientMode = lenient;
        return this;
    }

    // ------------------------------------------------------------------
    // Build / restore
    // ------------------------------------------------------------------

    /// <summary>Builds the mind. Throws <see cref="MindConfigurationException"/> on invalid config.</summary>
    public Mind Build()
    {
        _config.Name = _name;
        Validate();

        var modules = new List<ICognitiveModule>();
        if (_enabled.GetValueOrDefault("personality")) modules.Add(new PersonalityModule(_config.Personality));
        if (_enabled.GetValueOrDefault("memory")) modules.Add(new MemoryModule(_config.Memory));
        if (_enabled.GetValueOrDefault("emotions")) modules.Add(new EmotionModule(_config.Emotions));
        if (_enabled.GetValueOrDefault("reasoning")) modules.Add(new ReasoningModule(_config.Reasoning));
        if (_enabled.GetValueOrDefault("goals")) modules.Add(new GoalModule(_config.Goals));
        if (_enabled.GetValueOrDefault("thinking")) modules.Add(new ThinkingModule(_config.Thinking));
        if (_enabled.GetValueOrDefault("learning"))
        {
            modules.Add(new SelfLearningModule(_config.Learning));
            // Self-improvement rides on the learning capability: it grades
            // spoken replies from implied user feedback and feeds the hybrid
            // engine, so Without("learning") removes both together.
            modules.Add(new Cognify.Learning.SelfImprovementModule());
        }

        // Capability extensions (perception + action).
        if (_enabled.GetValueOrDefault("vision"))
            modules.Add(new Cognify.Perception.VisionModule(_config.Vision,
                _config.Vision.Backend as Cognify.Perception.IVisionBackend));
        if (_enabled.GetValueOrDefault("audio"))
            modules.Add(new Cognify.Audio.AudioModule(_config.Audio,
                _config.Audio.Backend as Cognify.Audio.IAudioBackend));
        if (_enabled.GetValueOrDefault("embeddings"))
            modules.Add(new Cognify.Embeddings.EmbeddingModule(_config.Embeddings));
        if (_enabled.GetValueOrDefault("multilingual"))
            modules.Add(new Cognify.Multilingual.MultilingualModule(_config.Multilingual));
        if (_enabled.GetValueOrDefault("tools"))
            modules.Add(new Cognify.Tools.ToolModule(_config.Tools));
        if (_enabled.GetValueOrDefault("code"))
            modules.Add(new Cognify.Code.CodeModule(_config.Code));
        if (_enabled.GetValueOrDefault("generation"))
            modules.Add(new Cognify.Generation.MediaGenerationModule(_config.Generation));

        modules.AddRange(_custom);

        int seed = _config.Learning.Seed;
        var mind = new Mind(_config, modules, seed);

        // Register the manual-only web tools when enabled. They are
        // explicitly NOT auto-invocable — see WebSearchTool.ManualOnly.
        if (_config.Tools.AllowWebSearch && mind.Tools is { } tools)
        {
            if (tools.Registry.Get("web-search") is null)
                tools.Registry.Register(new Cognify.Tools.WebSearchTool(
                    _config.Tools.WebSearchBackend as Cognify.Tools.IWebSearchBackend));
            if (tools.Registry.Get("web-explore") is null)
                tools.Registry.Register(new Cognify.Tools.WebExploreTool(
                    _config.Tools.WebPageFetcher as Cognify.Tools.IWebPageFetcher,
                    _config.Tools.WebExploration));
        }

        if (_config.Persistence.SqlitePath is { } path)
        {
            mind.Store = new SqliteMindStore(path);
            // Database output at build time: the SQLite file (schema + initial
            // state) exists the moment the mind is built, so the AI model uses
            // SQLite from its very first cycle and Restore(path) works even
            // before the first explicit save.
            mind.Store.Save(MindSnapshot.Capture(mind));
        }

        return mind;
    }

    /// <summary>
    /// Restores a previously saved mind from a SQLite file. The mind is rebuilt
    /// with the full standard module set and its saved state is re-applied.
    /// </summary>
    public static Mind Restore(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new MindConfigurationException("Restore path cannot be empty.");
        if (!File.Exists(path))
            throw new PersistenceException($"No mind found at '{path}'.");

        var store = new SqliteMindStore(path);
        var snapshot = store.Load();

        var builder = Create(string.IsNullOrWhiteSpace(snapshot.Name) ? "Restored" : snapshot.Name)
            .Describe(snapshot.Description)
            .WithAllModules();

        var mind = builder.Build();
        snapshot.ApplyTo(mind);
        mind.Store = store;
        return mind;
    }

    private void Validate()
    {
        foreach (var (key, enabled) in _enabled.Where(kv => kv.Value))
        {
            _ = key switch
            {
                "reasoning" or "thinking" or "learning" or "emotions" or
                "memory" or "goals" or "personality" or
                "vision" or "audio" or "embeddings" or "multilingual" or
                "tools" or "code" or "generation" => true,
                _ => throw new MindConfigurationException($"Unknown capability '{key}'.")
            };
        }

        if (_config.Learning.FeatureDim < 8)
            throw new MindConfigurationException("Learning.FeatureDim must be at least 8.");
        if (_config.Learning.HiddenUnits < 2)
            throw new MindConfigurationException("Learning.HiddenUnits must be at least 2.");
        if (_config.Memory.ShortTermCapacity < 1)
            throw new MindConfigurationException("Memory.ShortTermCapacity must be at least 1.");
        if (_config.Embeddings.Dimensions < 16)
            throw new MindConfigurationException("Embeddings.Dimensions must be at least 16.");
        if (_config.Embeddings.Capacity < 1)
            throw new MindConfigurationException("Embeddings.Capacity must be at least 1.");
        if (_config.Generation.SampleRate < 1000)
            throw new MindConfigurationException("Generation.SampleRate must be at least 1000 Hz.");
        if (string.IsNullOrWhiteSpace(_config.Multilingual.PreferredLanguage))
            throw new MindConfigurationException("Multilingual.PreferredLanguage cannot be empty.");
    }
}
