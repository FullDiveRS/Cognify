using System.Diagnostics;
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
/// The central cognitive kernel. A <see cref="Mind"/> hosts a set of pluggable
/// cognitive modules and runs a structured cognitive cycle for every stimulus:
///
///   perceive -> remember -> feel -> reason -> pursue goals -> think -> learn -> respond
///
/// Modules are optional: build the mind with only the capabilities you want
/// via <see cref="MindBuilder"/>.
/// </summary>
public sealed class Mind : IDisposable
{
    private readonly object _gate = new();
    private readonly List<ICognitiveModule> _modules = new();
    private readonly Random _random;
    private readonly Stopwatch _clock = new();
    private int _cycles;

    internal Mind(MindConfig config, IEnumerable<ICognitiveModule> modules, int seed)
    {
        Config = config ?? throw new ArgumentNullException(nameof(config));
        Id = $"mind-{Guid.NewGuid():N}";
        _random = new Random(seed);
        _modules.AddRange(modules);
        foreach (var m in _modules) m.Attach(this);
        _modules.Sort((a, b) => a.Order.CompareTo(b.Order));
    }

    /// <summary>Unique identifier of this mind instance.</summary>
    public string Id { get; }

    /// <summary>The mind's display name.</summary>
    public string Name => Config.Name;

    /// <summary>The configuration used to build this mind.</summary>
    public MindConfig Config { get; }

    /// <summary>
    /// True when lenient mode is on: calls into capabilities that were not
    /// enabled (See, Hear, Teach, Reinforce, InvokeTool, GenerateMedia,
    /// DetectLanguage, SemanticSimilarity) degrade gracefully instead of
    /// throwing <see cref="MindConfigurationException"/>.
    /// </summary>
    public bool Lenient => Config.LenientMode;

    /// <summary>All attached modules, sorted by execution order.</summary>
    public IReadOnlyList<ICognitiveModule> Modules => _modules;

    /// <summary>Total cognitive cycles executed so far.</summary>
    public int CycleCount { get { lock (_gate) return _cycles; } }

    // ------------------------------------------------------------------
    // Typed accessors — null when the capability was not enabled at build time.
    // ------------------------------------------------------------------

    /// <summary>Emotional state of the mind (null if emotions are disabled).</summary>
    public EmotionalState? EmotionalState => Module<EmotionModule>()?.Current;

    /// <summary>
    /// Developer-facing snapshot of how the mind feels RIGHT NOW — the
    /// programmatic read of what chat no longer announces. Null when the
    /// emotion module is disabled.
    /// <code>
    /// Console.WriteLine(yume.Feeling?.CurrentStatus);
    /// // "feeling anticipation (intensity 0.15) - mood positive (+0.00)"
    /// </code>
    /// </summary>
    public FeelingSnapshot? Feeling => EmotionalState is { } state ? FeelingSnapshot.From(state) : null;

    /// <summary>Knowledge base (null if reasoning is disabled).</summary>
    public KnowledgeBase? Knowledge => Module<ReasoningModule>()?.Knowledge;

    /// <summary>Reasoning module (null if reasoning is disabled).</summary>
    public ReasoningModule? Reasoning => Module<ReasoningModule>();

    /// <summary>Memory system (null if memory is disabled).</summary>
    public MemoryModule? Memory => Module<MemoryModule>();

    /// <summary>Goal/motivation system (null if goals are disabled).</summary>
    public GoalModule? GoalSystem => Module<GoalModule>();

    /// <summary>Self-learning system (null if learning is disabled).</summary>
    public SelfLearningModule? Learning => Module<SelfLearningModule>();

    /// <summary>
    /// The mind's growing word vocabulary — words picked up from the user
    /// while talking (null when self-learning is disabled).
    /// </summary>
    public Lexicon? Lexicon => Learning?.Lexicon;

    /// <summary>Inner monologue stream (null if thinking is disabled).</summary>
    public ThoughtStream? Stream => Module<ThinkingModule>()?.Stream;

    /// <summary>
    /// The thinking module (null if thinking is disabled) — its
    /// <see cref="Thinking.ThinkingModule.LastThoughts"/> holds the thought
    /// generator's silent self-talk from the last cycle, produced BEFORE the
    /// mind spoke. Hosts show it only when
    /// <see cref="ExpressionOptions.ShowThinking"/> is on.
    /// </summary>
    public ThinkingModule? Thinking => Module<ThinkingModule>();

    /// <summary>Personality traits (null if personality is disabled).</summary>
    public BigFiveTraits? Personality => Module<PersonalityModule>()?.Traits;

    /// <summary>Vision system (null if vision is disabled).</summary>
    public Cognify.Perception.VisionModule? Vision => Module<Cognify.Perception.VisionModule>();

    /// <summary>Auditory system (null if audio is disabled).</summary>
    public Cognify.Audio.AudioModule? Hearing => Module<Cognify.Audio.AudioModule>();

    /// <summary>Semantic embeddings system (null if embeddings are disabled).</summary>
    public Cognify.Embeddings.EmbeddingModule? SemanticMemory => Module<Cognify.Embeddings.EmbeddingModule>();

    /// <summary>Multilingual system (null if multilingual is disabled).</summary>
    public Cognify.Multilingual.MultilingualModule? Languages => Module<Cognify.Multilingual.MultilingualModule>();

    /// <summary>Tool-use system (null if tools are disabled).</summary>
    public Cognify.Tools.ToolModule? Tools => Module<Cognify.Tools.ToolModule>();

    /// <summary>Code system (null if the code module is disabled).</summary>
    public Cognify.Code.CodeModule? CodeEngine => Module<Cognify.Code.CodeModule>();

    /// <summary>Media generation system (null if generation is disabled).</summary>
    public Cognify.Generation.MediaGenerationModule? MediaGeneration => Module<Cognify.Generation.MediaGenerationModule>();

    /// <summary>Active SQLite store (null if persistence was not configured).</summary>
    public SqliteMindStore? Store { get; internal set; }

    /// <summary>The last response produced by this mind.</summary>
    public MindResponse? LastResponse { get; private set; }

    /// <summary>
    /// The mind's trained vocabulary — sayings it learned through
    /// <see cref="Train"/> and may speak in its own voice later. Every
    /// trained saying widens what the mind can say, which is the main lever
    /// against repeating the same few sentences.
    /// </summary>
    public PhrasePool Phrases { get; } = new();

    /// <summary>
    /// The mind's n-gram language model — its statistical next-token predictor.
    /// It observes everything the mind reads and says (user turns, replies,
    /// taught facts, trained corpora, web results) and provides fluency
    /// scoring for deliberation plus literal next-token predictions.
    /// Always available; persists with the mind's snapshot.
    /// </summary>
    public Cognify.Learning.NextTokenModel LanguageModel { get; } = new();

    /// <summary>
    /// The mind's neural language model — a tiny decoder-only TRANSFORMER
    /// (the same architecture family as GPT: multi-head causal self-attention,
    /// GELU feed-forwards, learned positional embeddings, Adam-trained with
    /// full backpropagation — all pure C#, no external ML runtime).
    /// It observes the same text stream as <see cref="LanguageModel"/> and
    /// trains lazily: automatically once enough new language accumulates, or
    /// explicitly via <see cref="TrainDeepModel"/> (the demo's /trainbrain).
    /// Prediction blends both models through <see cref="Cognify.Learning.HybridNextToken"/>.
    /// Persists with the mind's snapshot.
    /// </summary>
    public Cognify.Learning.TransformerLM NeuralLanguageModel { get; internal set; } = new();

    /// <summary>Self-improvement system (null if learning is disabled).</summary>
    public Cognify.Learning.SelfImprovementModule? SelfImprovement =>
        Module<Cognify.Learning.SelfImprovementModule>();

    // ------------------------------------------------------------------
    // Gender identity
    // ------------------------------------------------------------------

    /// <summary>The mind's gender identity word ("" when none was configured).</summary>
    public string GenderIdentity => Config.Gender.Identity;

    /// <summary>The pronoun set others use for the mind (Unspecified when none was configured).</summary>
    public PronounSet Pronouns => Config.Gender.Pronouns;

    /// <summary>
    /// Sets the mind's gender identity at runtime — e.g. once the conversation
    /// revealed how the mind should see itself or be addressed. Applies with
    /// the next save, and shows up in <see cref="DescribeSelf"/> and the
    /// answer to "who are you?" immediately.
    /// </summary>
    public void SetGender(string identity, PronounSet? pronouns = null) =>
        Config.Gender.Set(identity, pronouns);

    /// <summary>Sets just the mind's pronoun set at runtime.</summary>
    public void SetPronouns(PronounSet pronouns) => Config.Gender.With(pronouns);

    /// <summary>
    /// The gender clause the mind uses when describing itself — "I am a girl,
    /// and I go by she/her pronouns." — or "" when no gender was configured,
    /// in which case self-description stays exactly what it used to be.
    /// </summary>
    public string GenderClause
    {
        get
        {
            var g = Config.Gender;
            string identityPart = "";
            if (!string.IsNullOrWhiteSpace(g.Identity))
            {
                var word = g.Identity.Trim();
                identityPart = StartsWithArticle(word) ? $"I am {word}" : $"I am {Article(word)} {word}";
            }
            string pronounPart = g.Pronouns.IsUnspecified
                ? ""
                : $"I go by {g.Pronouns.Subject}/{g.Pronouns.Object} pronouns";
            if (identityPart.Length > 0 && pronounPart.Length > 0) return $"{identityPart}, and {pronounPart}.";
            if (identityPart.Length > 0) return identityPart + ".";
            if (pronounPart.Length > 0) return pronounPart + ".";
            return "";
        }
    }

    /// <summary>
    /// A first-person self-introduction: name, purpose, gender (when the mind
    /// has one) and current mood. This is also what the mind answers to
    /// "who are you?" / "introduce yourself".
    /// </summary>
    public string DescribeSelf()
    {
        var name = Name;
        var desc = Config.Description;
        string self;
        if (string.IsNullOrWhiteSpace(desc))
            self = $"I am {name}, a mind built with Cognify";
        else if (desc.StartsWith(name, StringComparison.OrdinalIgnoreCase))
            self = desc.TrimEnd('.', '!');
        else
            self = $"I am {name} - {desc}";

        var parts = new List<string> { self + "." };
        var clause = GenderClause;
        if (clause.Length > 0) parts.Add(clause);
        // The numeric feeling readout is developer-facing introspection —
        // shown in chat only when Expression.ShowFeelingReads is on. The
        // live state stays on <see cref="Feeling"/> regardless.
        if (Config.Expression.ShowFeelingReads)
        {
            var feeling = EmotionalState?.Describe();
            if (feeling is not null) parts.Add($"Right now {feeling}.");
        }
        return string.Join(" ", parts);
    }

    private static bool StartsWithArticle(string word) =>
        word.StartsWith("a ", StringComparison.OrdinalIgnoreCase) ||
        word.StartsWith("an ", StringComparison.OrdinalIgnoreCase) ||
        word.StartsWith("the ", StringComparison.OrdinalIgnoreCase);

    private static string Article(string word) =>
        "aeiou".Contains(char.ToLowerInvariant(word[0])) ? "an" : "a";

    // ------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------

    /// <summary>Raised after every completed cognitive cycle.</summary>
    public event EventHandler<CycleEventArgs>? CycleCompleted;

    /// <summary>Raised when an emotion meaningfully changes.</summary>
    public event EventHandler<EmotionChangedEventArgs>? EmotionChanged;

    /// <summary>Raised for every step of the inner monologue.</summary>
    public event EventHandler<ThoughtProducedEventArgs>? ThoughtProduced;

    /// <summary>Raised when memories are recalled during a cycle.</summary>
    public event EventHandler<MemoryRecalledEventArgs>? MemoryRecalled;

    /// <summary>Raised when a goal changes.</summary>
    public event EventHandler<GoalUpdatedEventArgs>? GoalUpdated;

    /// <summary>Raised whenever the mind learns from feedback.</summary>
    public event EventHandler<LearnedEventArgs>? Learned;

    /// <summary>Raised when a module throws during a cycle (the cycle continues).</summary>
    public event EventHandler<ModuleErrorEventArgs>? ModuleError;

    /// <summary>Raised when a tool is invoked during a cycle.</summary>
    public event EventHandler<ToolInvokedEventArgs>? ToolInvoked;

    /// <summary>Raised when a media artifact is generated.</summary>
    public event EventHandler<MediaGeneratedEventArgs>? MediaGenerated;

    internal void RaiseEmotionChanged(EmotionType e, double intensity, double valence) =>
        EmotionChanged?.Invoke(this, new EmotionChangedEventArgs(e, intensity, valence));

    internal void RaiseThoughtProduced(ThoughtStep step) =>
        ThoughtProduced?.Invoke(this, new ThoughtProducedEventArgs(step));

    internal void RaiseMemoryRecalled(IReadOnlyList<MemoryRecord> memories) =>
        MemoryRecalled?.Invoke(this, new MemoryRecalledEventArgs(memories));

    internal void RaiseGoalUpdated(Goal goal, string change) =>
        GoalUpdated?.Invoke(this, new GoalUpdatedEventArgs(goal, change));

    internal void RaiseLearned(LearnedEventArgs args) => Learned?.Invoke(this, args);

    internal void RaiseCycleCompleted(MindResponse response) =>
        CycleCompleted?.Invoke(this, new CycleEventArgs(response));

    internal void RaiseModuleError(string module, Exception ex) =>
        ModuleError?.Invoke(this, new ModuleErrorEventArgs(module, ex));

    internal void RaiseToolInvoked(string toolName, IReadOnlyDictionary<string, string> arguments, Cognify.Tools.ToolResult result) =>
        ToolInvoked?.Invoke(this, new ToolInvokedEventArgs(toolName, arguments, result));

    internal void RaiseMediaGenerated(Cognify.Generation.MediaArtifact artifact) =>
        MediaGenerated?.Invoke(this, new MediaGeneratedEventArgs(artifact));

    internal Random Rng => _random;

    // ------------------------------------------------------------------
    // Module management
    // ------------------------------------------------------------------

    /// <summary>Returns the first attached module of the given type, or null.</summary>
    public T? Module<T>() where T : class, ICognitiveModule =>
        _modules.OfType<T>().FirstOrDefault();

    /// <summary>Returns true if a module of the given type is attached.</summary>
    public bool HasModule<T>() where T : class, ICognitiveModule =>
        _modules.OfType<T>().Any();

    /// <summary>Attaches an additional module at runtime. Returns true if added.</summary>
    public bool AttachModule(ICognitiveModule module)
    {
        if (module is null) throw new ArgumentNullException(nameof(module));
        lock (_gate)
        {
            if (_modules.Any(m => m.Name == module.Name)) return false;
            module.Attach(this);
            _modules.Add(module);
            _modules.Sort((a, b) => a.Order.CompareTo(b.Order));
            return true;
        }
    }

    /// <summary>Detaches a module by type. Returns true if removed.</summary>
    public bool DetachModule<T>() where T : class, ICognitiveModule
    {
        lock (_gate)
        {
            var m = _modules.OfType<T>().FirstOrDefault();
            return m is not null && _modules.Remove(m);
        }
    }

    // ------------------------------------------------------------------
    // The cognitive cycle
    // ------------------------------------------------------------------

    /// <summary>
    /// Processes a stimulus through the full cognitive cycle and produces a response.
    /// </summary>
    public MindResponse Process(Stimulus stimulus)
    {
        if (stimulus is null) throw new ArgumentNullException(nameof(stimulus));
        _clock.Restart();
        int cycle;
        CognitiveContext ctx;
        lock (_gate)
        {
            cycle = ++_cycles;
        }
        ctx = new CognitiveContext(this, cycle, stimulus);

        foreach (var module in _modules.ToArray())
        {
            try
            {
                module.Process(ctx);
            }
            catch (Exception ex)
            {
                ModuleError?.Invoke(this, new ModuleErrorEventArgs(module.Name, ex));
                Stream?.Add("error", $"module '{module.Name}' failed: {ex.Message}");
            }
        }

        ctx.Response ??= ComposeFallback(ctx);
        _clock.Stop();

        var response = new MindResponse(
            cycle, ctx.Response.Content, ctx.Response.Confidence,
            ctx.Response.StyleId, ctx.Response.StyleName,
            ctx.Chain, ctx.Reasoning, ctx.Emotions, ctx.Recalled,
            _clock.Elapsed, DateTimeOffset.UtcNow);

        // The language models learn from everything the mind reads AND says,
        // so their next-token predictions track the actual conversation.
        ObserveLanguage(stimulus.Content);
        ObserveLanguage(response.Content);

        lock (_gate) LastResponse = response;
        RaiseCycleCompleted(response);

        var saveEvery = Config.Persistence.AutoSaveEveryCycles;
        if (Store is not null && saveEvery > 0 && cycle % saveEvery == 0)
            _ = SaveAsync();

        return response;
    }

    /// <summary>Asynchronously processes a stimulus.</summary>
    public Task<MindResponse> ProcessAsync(Stimulus stimulus, CancellationToken ct = default) =>
        Task.Run(() => Process(stimulus), ct);

    /// <summary>
    /// Convenience: process a piece of text input.
    /// </summary>
    public MindResponse Process(string text, double intensity = 0.5) =>
        Process(Stimulus.Text(text, intensity));

    /// <summary>
    /// Teaches the mind facts expressed in plain English, e.g.
    /// "birds are animals", "dogs have fur", "cats can climb".
    /// Returns the number of facts learned.
    /// In lenient mode with reasoning disabled, the sentence is kept as a
    /// plain memory and 0 is returned instead of throwing.
    /// </summary>
    public int Teach(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence)) return 0;
        var facts = FactParser.Parse(sentence).ToList();
        var reasoning = Module<ReasoningModule>();
        if (reasoning is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Reasoning is not enabled; enable it with WithReasoning() to teach facts.");
            // Lenient: no knowledge base — keep the sentence as a memory.
            Memory?.Remember(sentence, MemoryKind.LongTerm, importance: 0.55, source: "taught");
            Stream?.Add("teach", "reasoning is not enabled; kept as a memory: " + sentence);
            return 0;
        }

        foreach (var f in facts) reasoning.Knowledge.Add(f);
        Memory?.Remember(sentence, MemoryKind.LongTerm, importance: 0.55, source: "taught");
        Stream?.Add("teach", $"learned {facts.Count} fact(s) from: {sentence}");
        EmotionalState?.Feel(EmotionType.Joy, 0.12); // learning feels good
        ObserveLanguage(sentence);
        return facts.Count;
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /// <summary>
    /// Trains the mind on a piece of text — the "train option" for growing a
    /// mind's knowledge AND its vocabulary. Each line can teach three kinds
    /// of things:
    ///
    ///   • facts ("birds are animals", "2 + 2 = 4") become knowledge,
    ///   • a counting sequence ("0, 1, 2, ..., 9") becomes the number line,
    ///   • plain conversational lines become SAYINGS — sentence shapes the
    ///     mind may speak in its own voice later. This is what "more words"
    ///     means concretely: the more sayings a mind owns, the more varied
    ///     its replies, and the less it repeats the same few sentences.
    ///
    /// Questions and conditional setups ("if i have no apples...") are
    /// requests, not knowledge or speech — they are skipped. Returns a
    /// <see cref="TrainingReport"/> of everything the pass added.
    /// </summary>
    public TrainingReport Train(string text)
    {
        var report = new TrainingReport();
        if (string.IsNullOrWhiteSpace(text)) return report;
        foreach (var line in text.Split('\n'))
            TrainLine(line, report);
        return report;
    }

    /// <summary>Bulk-trains the mind on a sequence of lines.</summary>
    public TrainingReport TrainAll(IEnumerable<string> lines)
    {
        var report = new TrainingReport();
        if (lines is null) return report;
        foreach (var line in lines)
            TrainLine(line, report);
        return report;
    }

    /// <summary>
    /// Trains the mind from a corpus file: one utterance per line, blank
    /// lines and '#' comment lines are ignored. Returns the training report.
    /// </summary>
    public TrainingReport TrainFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Training file not found: {path}", path);
        return TrainAll(File.ReadAllLines(path)
            .Where(l => !string.IsNullOrWhiteSpace(l) && !l.TrimStart().StartsWith("#")));
    }

    private void TrainLine(string? raw, TrainingReport report)
    {
        if (string.IsNullOrWhiteSpace(raw)) return;
        var line = raw.Trim();
        if (line.StartsWith("#")) return; // comment line in corpus files
        report.Lines++;

        // 1. Knowledge: facts (plain or arithmetic) into the knowledge base.
        int factsBefore = 0;
        var reasoning = Module<ReasoningModule>();
        if (reasoning is not null)
        {
            // "factsBefore" counts PARSED facts, not newly-added ones: a line
            // whose fact we already know is still knowledge-shaped — it must
            // never leak into the sayings vocabulary on re-training.
            var parsed = FactParser.Parse(line).ToList();
            foreach (var fact in parsed)
                if (reasoning.Knowledge.Add(fact))
                    report.FactsLearned++;
            factsBefore = parsed.Count;
            if (reasoning.Arithmetic.TeachSequence(line)) report.SequencesTaught++;
        }

        // 2. Vocabulary: a line that taught nothing and is not a request is
        //    conversational — it becomes a saying the mind can speak later.
        bool becameSaying = factsBefore == 0 && IsSayable(line) && Phrases.Add(line);
        if (becameSaying) report.SayingsAdded++;

        // 3. Sayings also become long-term memories (source: "trained"), so
        //    recall keeps working alongside the phrase pool.
        if (becameSaying)
            Memory?.Remember(line, MemoryKind.LongTerm, importance: 0.5, source: "trained");

        // Every training line also feeds the language models, so a trained
        // corpus directly improves the mind's language predictions.
        ObserveLanguage(line);

        Stream?.Add("train", $"trained on: {Truncate(line, 60)}");
        EmotionalState?.Feel(EmotionType.Joy, 0.08);
    }

    // Lines that assert nothing (questions, conditionals) never become
    // sayings — they are requests or setups, not things to say.
    private static readonly string[] QuestionStarters =
        { "what", "who", "when", "where", "why", "how", "is", "are", "do",
          "does", "did", "can", "could", "would", "will", "tell", "say", "name" };

    private static bool IsSayable(string line)
    {
        var t = line.Trim();
        if (t.Length < 3 || t.Length > 300 || t.Contains('?')) return false;
        var first = t.Split(' ', StringSplitOptions.RemoveEmptyEntries)[0]
                     .Trim(',', '.', '!', ':', ';').ToLowerInvariant();
        return first != "if" && !QuestionStarters.Contains(first);
    }

    /// <summary>
    /// Convenience: asks the mind a question in plain English (routed through
    /// the normal cognitive cycle so reasoning, memory and emotion all apply).
    /// </summary>
    public MindResponse Ask(string question) =>
        Process(new Stimulus(question, StimulusKind.Text, 0.6));

    /// <summary>
    /// Reinforces the last response with a reward in -1..1. Positive rewards
    /// train the network to produce more of the chosen response style.
    /// In lenient mode with self-learning disabled, the reward only stirs the
    /// mind's emotions instead of throwing.
    /// </summary>
    public void Reinforce(double reward)
    {
        var learning = Learning;
        if (learning is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Self-learning is not enabled; enable it with WithSelfLearning().");
        }
        else
        {
            learning.Reinforce(reward);
        }

        if (EmotionalState is not null)
        {
            EmotionalState.Feel(reward >= 0 ? EmotionType.Joy : EmotionType.Sadness,
                                Math.Min(1.0, Math.Abs(reward) * 0.6));
        }
    }

    // ------------------------------------------------------------------
    // Multimodal + agent conveniences
    // ------------------------------------------------------------------

    /// <summary>
    /// Shows the mind an image (a textual scene description unless a real
    /// vision backend is plugged in) and runs the full cognitive cycle on it.
    /// Throws <see cref="MindConfigurationException"/> when vision is disabled
    /// and the mind is not lenient; in lenient mode the description still goes
    /// through the full cognitive cycle as plain text.
    /// </summary>
    public MindResponse See(string description, IReadOnlyDictionary<string, string>? metadata = null, double intensity = 0.6)
    {
        if (!HasModule<Cognify.Perception.VisionModule>())
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Vision is not enabled; enable it with WithVision().");
            Stream?.Add("vision", "vision is not enabled; registering the scene as text");
        }
        return Process(Stimulus.Image(description, intensity, metadata));
    }

    /// <summary>
    /// Lets the mind hear a sound or speech clip and runs the full cognitive
    /// cycle on it. Throws <see cref="MindConfigurationException"/> when audio
    /// is disabled and the mind is not lenient; in lenient mode the content
    /// still goes through the full cognitive cycle as plain text.
    /// </summary>
    public MindResponse Hear(string content, IReadOnlyDictionary<string, string>? metadata = null, double intensity = 0.6)
    {
        if (!HasModule<Cognify.Audio.AudioModule>())
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Audio is not enabled; enable it with WithAudio().");
            Stream?.Add("audio", "audio is not enabled; registering the clip as text");
        }
        return Process(Stimulus.Audio(content, intensity, metadata));
    }

    /// <summary>
    /// Invokes a registered tool directly, outside the cognitive cycle.
    /// Throws <see cref="MindConfigurationException"/> when tools are disabled
    /// and the mind is not lenient; in lenient mode a failed result is
    /// returned instead.
    /// </summary>
    public Cognify.Tools.ToolResult InvokeTool(string toolName, IReadOnlyDictionary<string, string>? arguments = null)
    {
        var tools = Tools;
        if (tools is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Tool use is not enabled; enable it with WithTools().");
            return Cognify.Tools.ToolResult.Fail(
                $"Tool use is not enabled (lenient mode: '{toolName}' was not invoked).");
        }
        return tools.Invoke(toolName, arguments);
    }

    /// <summary>
    /// Generates an image / video / speech / sound artifact.
    /// Throws <see cref="MindConfigurationException"/> when generation is
    /// disabled and the mind is not lenient; in lenient mode null is returned.
    /// </summary>
    public Cognify.Generation.MediaArtifact? GenerateMedia(string prompt, Cognify.Generation.MediaType type,
        double seconds = 0, string style = "")
    {
        var generation = MediaGeneration;
        if (generation is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Media generation is not enabled; enable it with WithMediaGeneration().");
            return null; // lenient: nothing was generated
        }
        return generation.Generate(prompt, type, seconds, style);
    }

    /// <summary>
    /// Explicit web search: runs the manual-only <c>web-search</c> tool and,
    /// when it succeeds, LEARNS from the results — snippets become long-term
    /// memories (recallable later), clean "X is Y" sentences in them become
    /// knowledge-base facts (source "web"), and the text feeds the
    /// next-token language model. Words from the query are marked as spoken
    /// so the mind stops asking what they mean.
    ///
    /// This is deliberately the ONLY path to web search: the tool is
    /// manual-only and never auto-fires inside the cognitive cycle, so the
    /// mind goes online exactly when asked — never on its own.
    /// Throws <see cref="MindConfigurationException"/> when tools are
    /// disabled and the mind is not lenient; in lenient mode a failed result
    /// is returned instead.
    /// </summary>
    public Cognify.Tools.ToolResult SearchWeb(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return Cognify.Tools.ToolResult.Fail("No search query given.");
        var tools = Tools;
        if (tools is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Tool use is not enabled; enable it with WithTools() or WithWebSearch().");
            return Cognify.Tools.ToolResult.Fail(
                "Web search is not enabled (lenient mode: nothing was searched).");
        }

        // Restored minds may predate the web-search registration — make sure
        // the tool exists before invoking it.
        if (tools.Registry.Get("web-search") is null)
            tools.Registry.Register(new Cognify.Tools.WebSearchTool(
                Config.Tools.WebSearchBackend as Cognify.Tools.IWebSearchBackend));

        var result = tools.Invoke("web-search",
            new Dictionary<string, string> { ["query"] = query.Trim() }, query);
        if (result.Success)
            LearnFromWeb(query.Trim(), result);
        return result;
    }

    private void LearnFromWeb(string query, Cognify.Tools.ToolResult result)
    {
        var results = result.Data.TryGetValue("results", out var raw) &&
                      raw is IReadOnlyList<Cognify.Tools.WebSearchResult> list
            ? list
            : null;

        // 1. Recallable memories — she "remembers" what she read online.
        if (results is not null)
        {
            foreach (var r in results.Take(3))
                Memory?.Remember(
                    $"Web: {r.Title}. {r.Snippet} (source: {r.Url})",
                    MemoryKind.LongTerm, importance: 0.6, source: "web");
        }
        else
        {
            Memory?.Remember($"Web search on '{Truncate(query, 60)}': {Truncate(result.Output, 140)}",
                MemoryKind.LongTerm, importance: 0.55, source: "web");
        }

        // 2. Knowledge: clean factual sentences in the snippets become facts.
        if (results is not null && Reasoning is { } reasoning)
        {
            int added = 0;
            foreach (var r in results)
            {
                if (added >= 3) break;
                foreach (var fact in FactParser.Parse($"{r.Title}. {r.Snippet}"))
                {
                    var sourced = new Fact(fact.Subject, fact.Predicate, fact.Object,
                        Math.Min(fact.Confidence, 0.7), "web");
                    if (reasoning.Knowledge.Add(sourced))
                        added++;
                    if (added >= 3) break;
                }
            }
        }

        // 3. Language + vocabulary: the text feeds next-token prediction, and
        //    the query's words count as known now — she looked them up instead
        //    of asking what they mean.
        ObserveLanguage(result.Output);
        if (Lexicon is { } lex)
            foreach (var word in Emotions.MoodEngine.Tokenize(query))
                if (word.Length > 2)
                    lex.MarkSpoken(word);

        Stream?.Add("web", $"searched the web for: {Truncate(query, 60)} and learned from the results");
    }

    // ------------------------------------------------------------------
    // Seeing images, hearing audio, making media — the file-level APIs.
    // ------------------------------------------------------------------

    /// <summary>
    /// Trains the mind on a TEXT IMAGE: the built-in OCR reads the words out
    /// of a PNG/BMP/GIF/PPM image, and every readable line is trained exactly
    /// like a taught sentence — facts land in her knowledge base, clean
    /// sentences become sayings, and the vocabulary/lm pipelines see the
    /// words. Works best on clean, machine-rendered text (screenshots,
    /// rendered cards); handwriting comes back with low confidence.
    /// </summary>
    public Cognify.Imaging.ImageTrainingReport TrainImage(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Image file not found: {path}", path);
        var image = Cognify.Imaging.ImageIO.Load(path);
        var ocr = Cognify.Imaging.OcrEngine.Shared.Recognize(image);
        var report = Train(ocr.Text);
        Stream?.Add("vision", $"read an image: {Truncate(path, 50)} — {ocr.Describe()}");
        EmotionalState?.Feel(EmotionType.Joy, ocr.IsEmpty ? 0.0 : 0.1);
        return new Cognify.Imaging.ImageTrainingReport(path, image.Width, image.Height,
            ocr.Text, ocr.MeanConfidence, ocr.GlyphCount,
            report.Lines, report.FactsLearned, report.SayingsAdded);
    }

    /// <summary>Looks at an image and reports the text found in it, without training.</summary>
    public string LookAtImage(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Image file not found: {path}", path);
        var image = Cognify.Imaging.ImageIO.Load(path);
        var ocr = Cognify.Imaging.OcrEngine.Shared.Recognize(image);
        return ocr.IsEmpty
            ? $"I see a {image.Width}x{image.Height} image, but there is no text in it that I can read."
            : $"I see a {image.Width}x{image.Height} image and I read: \"{ocr.Text.Replace("\n", " / ")}\" ({ocr.Describe()}).";
    }

    /// <summary>
    /// Transcribes an audio file. Honest scope: she can transcribe speech
    /// she herself synthesized (her phrasebook), because her tiny on-device
    /// engine matches syllable fingerprints rather than running a giant
    /// speech model. Unmatched clips get a syllable count and a teach-me hint.
    /// </summary>
    public Cognify.Generation.TranscriptionResult TranscribeAudio(string path)
    {
        var backend = RequireRenderingBackend("transcribe audio");
        var clip = Cognify.Generation.WavCodec.Decode(File.ReadAllBytes(path));
        var result = backend.Transcribe(clip);
        Stream?.Add("audio", $"listened to {Truncate(path, 50)}: {result.Message}");
        return result;
    }

    /// <summary>
    /// Clones a voice from a recording: pitch, expressiveness, brightness
    /// and pace are measured from the sample and become her speaking voice.
    /// </summary>
    public Cognify.Generation.VoiceProfile CloneVoice(string path)
    {
        var backend = RequireRenderingBackend("clone a voice");
        var clip = Cognify.Generation.WavCodec.Decode(File.ReadAllBytes(path));
        var profile = backend.CloneVoiceFrom(clip);
        Memory?.Remember($"I cloned a voice from {Path.GetFileName(path)}: {profile.Describe()}",
            MemoryKind.LongTerm, importance: 0.6, source: "voice");
        Stream?.Add("audio", $"cloned a voice: {profile.Describe()}");
        EmotionalState?.Feel(EmotionType.Joy, 0.12);
        return profile;
    }

    /// <summary>Shows the voice profile currently used for speech.</summary>
    public string DescribeVoice() =>
        (MediaGeneration?.Backend as Cognify.Generation.RenderingMediaBackend)?.Voice.Describe()
        ?? "My voice profile is the built-in default (media generation disabled).";

    /// <summary>Converts an audio file to a new sample rate; returns a report.</summary>
    public string ConvertAudio(string path, int targetRateHz)
    {
        var backend = RequireRenderingBackend("convert audio");
        var clip = Cognify.Generation.WavCodec.Decode(File.ReadAllBytes(path));
        var converted = Cognify.Generation.Dsp.Resample(clip, targetRateHz);
        var bytes = Cognify.Generation.WavCodec.Encode(converted.Samples, converted.SampleRate);
        var saved = backend.Store.Save("converted", ".wav", bytes);
        return $"Converted {Path.GetFileName(path)} ({clip.SampleRate} Hz, {clip.Duration:0.0}s) " +
               $"to {converted.SampleRate} Hz — saved as {saved}.";
    }

    /// <summary>Splits an audio file into low / mid / high stems; returns a report.</summary>
    public string SeparateAudio(string path)
    {
        var backend = RequireRenderingBackend("separate audio");
        var clip = Cognify.Generation.WavCodec.Decode(File.ReadAllBytes(path));
        var stems = Cognify.Generation.Dsp.Separate(clip);
        var lines = new List<string> { $"Separated {Path.GetFileName(path)} into {stems.Count} stems:" };
        foreach (var stem in stems)
        {
            var bytes = Cognify.Generation.WavCodec.Encode(stem.Clip.Samples, stem.Clip.SampleRate);
            var saved = backend.Store.Save($"stem-{stem.Name}", ".wav", bytes);
            lines.Add($"  {stem.Name}: {stem.Clip.Duration:0.0}s at {stem.Clip.SampleRate} Hz — {saved}");
        }
        lines.Add("(a band split, not a neural separator — bass, body and sparkle, honestly labeled)");
        return string.Join("\n", lines);
    }

    /// <summary>
    /// Edits an audio file. Supported ops: <c>trim start end</c>,
    /// <c>gain factor</c>, <c>fade inSeconds outSeconds</c>, <c>normalize</c>,
    /// <c>reverse</c>, <c>concat otherFile</c>. Returns a report with the
    /// saved output path.
    /// </summary>
    public string EditAudio(string path, string operation, string arguments)
    {
        var backend = RequireRenderingBackend("edit audio");
        var clip = Cognify.Generation.WavCodec.Decode(File.ReadAllBytes(path));
        var parts = arguments.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Cognify.Generation.AudioClip edited = operation.ToLowerInvariant() switch
        {
            "trim" => Cognify.Generation.Dsp.Trim(clip,
                double.Parse(parts.ElementAtOrDefault(0) ?? "0", System.Globalization.CultureInfo.InvariantCulture),
                parts.Length > 1 ? double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture) : -1),
            "gain" => Cognify.Generation.Dsp.Gain(clip,
                double.Parse(parts.ElementAtOrDefault(0) ?? "1", System.Globalization.CultureInfo.InvariantCulture)),
            "fade" => Cognify.Generation.Dsp.Fade(clip,
                double.Parse(parts.ElementAtOrDefault(0) ?? "0.05", System.Globalization.CultureInfo.InvariantCulture),
                parts.Length > 1 ? double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture) : 0.05),
            "normalize" => Cognify.Generation.Dsp.Normalize(clip),
            "reverse" => Cognify.Generation.Dsp.Reverse(clip),
            "concat" => Cognify.Generation.Dsp.Concat(clip,
                Cognify.Generation.WavCodec.Decode(File.ReadAllBytes(
                    parts.ElementAtOrDefault(0) ?? throw new ArgumentException("concat needs a second file")))),
            _ => throw new ArgumentException(
                $"Unknown audio edit '{operation}'. Try: trim, gain, fade, normalize, reverse, concat."),
        };
        var bytes = Cognify.Generation.WavCodec.Encode(edited.Samples, edited.SampleRate);
        var saved = backend.Store.Save($"edited-{operation.ToLowerInvariant()}", ".wav", bytes);
        return $"Edited {Path.GetFileName(path)} with '{operation}': {edited.Duration:0.0}s at " +
               $"{edited.SampleRate} Hz — saved as {saved}.";
    }

    /// <summary>Lists the media files she has created (newest first).</summary>
    public IReadOnlyList<string> ListMedia()
    {
        if (MediaGeneration?.Backend is Cognify.Generation.RenderingMediaBackend rendering)
            return rendering.Store.ListFiles();
        return MediaGeneration?.Gallery.Select(a => a.Describe()).ToList()
               ?? [];
    }

    private Cognify.Generation.RenderingMediaBackend RequireRenderingBackend(string action)
    {
        if (MediaGeneration?.Backend is Cognify.Generation.RenderingMediaBackend rendering)
            return rendering;
        if (!Config.LenientMode)
            throw new MindConfigurationException(
                $"To {action} I need the built-in rendering media backend (WithMediaGeneration).");
        throw new MindConfigurationException($"Cannot {action} without the rendering media backend.");
    }

    /// <summary>
    /// Runs one conscious self-improvement pass: reviews the feedback ledger,
    /// retires sayings that consistently drew disapproval, and returns an
    /// explainable report. Throws <see cref="MindConfigurationException"/>
    /// when learning is disabled and the mind is not lenient; in lenient mode
    /// an empty report is returned.
    /// </summary>
    public Cognify.Learning.SelfImprovementReport SelfImprove()
    {
        var improvement = SelfImprovement;
        if (improvement is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Self-learning is not enabled; enable it with WithSelfLearning().");
            return new Cognify.Learning.SelfImprovementReport
            {
                Summary = "Self-improvement is not enabled (learning module missing).",
            };
        }
        return improvement.SelfImprove();
    }

    /// <summary>
    /// Feeds a piece of text to BOTH language models — the n-gram counts it
    /// immediately, the transformer buffers it for its next (lazy) training
    /// pass, which runs automatically once enough new language accumulates.
    /// </summary>
    private void ObserveLanguage(string text)
    {
        LanguageModel.Observe(text);
        NeuralLanguageModel.Observe(text);
        _ = NeuralLanguageModel.AutoTrain(); // no-op until ShouldAutoTrain
    }

    /// <summary>
    /// Next-token prediction: what would the mind say AFTER these words?
    /// Returns the top candidate tokens with normalized probabilities, best
    /// first. A HYBRID: the n-gram model contributes exact local statistics,
    /// the transformer contributes generalization to unseen contexts (and
    /// when she has not trained her transformer yet, the n-gram answers alone).
    /// </summary>
    public IReadOnlyList<Cognify.Learning.TokenPrediction> PredictNext(string context, int count = 5) =>
        Cognify.Learning.HybridNextToken.PredictNext(LanguageModel, NeuralLanguageModel, context, count);

    /// <summary>
    /// Fluency estimate for a candidate wording — how much it reads like the
    /// language the mind has absorbed, blending both language models. 0 means
    /// "no opinion"; closer to 0 from below reads more natural. Used by the
    /// thinking module to break close ties toward natural wordings.
    /// </summary>
    public double ScoreFluency(string text) =>
        Cognify.Learning.HybridNextToken.Fluency(LanguageModel, NeuralLanguageModel, text);

    /// <summary>
    /// Explicitly trains the mind's neural (transformer) language model on
    /// everything she has observed so far. The model also trains itself
    /// lazily during conversation; this is the deep pass — more epochs, full
    /// report. Never throws: with too little language it reports zero epochs.
    /// </summary>
    public Cognify.Learning.TransformerTrainingReport TrainDeepModel(int epochs = 12) =>
        NeuralLanguageModel.Train(epochs);

    /// <summary>
    /// Detects the language of a text. Throws when the multilingual module is
    /// not enabled and the mind is not lenient; in lenient mode null is returned.
    /// </summary>
    public Cognify.Multilingual.LanguageGuess? DetectLanguage(string text)
    {
        var languages = Languages;
        if (languages is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Multilingual is not enabled; enable it with WithMultilingual().");
            return null; // lenient: no detection without the module
        }
        return languages.Detect(text);
    }

    /// <summary>
    /// Cosine similarity (0..1) between two texts using the mind's embedding
    /// model. Throws when embeddings are disabled and the mind is not lenient;
    /// in lenient mode 0 is returned.
    /// </summary>
    public double SemanticSimilarity(string a, string b)
    {
        var embeddings = SemanticMemory;
        if (embeddings is null)
        {
            if (!Config.LenientMode)
                throw new MindConfigurationException("Embeddings are not enabled; enable them with WithEmbeddings().");
            return 0; // lenient: no similarity signal without the module
        }
        return embeddings.Similarity(a, b);
    }

    /// <summary>
    /// Advances background dynamics one tick: emotions decay, mood drifts,
    /// memories weaken, drives grow, goals may be spawned.
    /// Call periodically (e.g. once per second, or once per conversation turn).
    /// </summary>
    public void Tick()
    {
        lock (_gate)
        {
            foreach (var module in _modules) module.Tick();
        }
    }

    private ResponseDraft ComposeFallback(CognitiveContext ctx)
    {
        // Minimal composer used when the thinking module is disabled.
        // Memory content is the user's own words, so it is referenced
        // obliquely (never quoted back verbatim).
        if (ctx.Recalled.Count > 0)
        {
            var memoryTopic = ResponseComposer.TopicOf(ctx.Recalled[0].Content);
            return new ResponseDraft(
                $"That connects with what you told me earlier about {memoryTopic}.", 0.4, -1, "fallback");
        }
        return new ResponseDraft($"I registered '{Truncate(ctx.Stimulus.Content, 60)}'.", 0.3, -1, "fallback");
    }

    internal static string Truncate(string text, int max) =>
        text.Length <= max ? text : text[..max] + "...";

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /// <summary>
    /// Saves the mind's full state (config, emotions, memories, knowledge,
    /// goals, personality, learned network) to the configured SQLite store,
    /// or to the given path.
    /// </summary>
    public async Task SaveAsync(string? path = null, CancellationToken ct = default)
    {
        var store = path is null
            ? Store ?? throw new PersistenceException(
                "No persistence configured. Call WithPersistence(path) on the builder or pass a path.")
            : new SqliteMindStore(path);
        await store.SaveAsync(MindSnapshot.Capture(this), ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Restores mind state from a SQLite file into this mind (must have been
    /// built with the same set of core modules, e.g. via MindBuilder.Restore).
    /// </summary>
    public async Task LoadFromAsync(string path, CancellationToken ct = default)
    {
        var store = new SqliteMindStore(path);
        var snap = await store.LoadAsync(ct).ConfigureAwait(false);
        snap.ApplyTo(this);
    }

    /// <summary>Releases the SQLite store if one is attached.</summary>
    public void Dispose()
    {
        Store?.Dispose();
    }
}
