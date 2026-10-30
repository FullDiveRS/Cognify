using Cognify.Core;

namespace Cognify.Multilingual;

/// <summary>
/// The multilingual module: lets the mind understand which language it is
/// being addressed in. Every text stimulus is run through
/// <see cref="LanguageDetector"/>; the detected language is exposed to other
/// modules and, when <see cref="MultilingualOptions.AutoTranslate"/> is on
/// and a translation is known, the stimulus is translated into the mind's
/// preferred language before deeper processing.
/// </summary>
public sealed class MultilingualModule : CognitiveModuleBase
{
    private readonly MultilingualOptions _options;
    private readonly object _gate = new();
    private readonly Dictionary<string, int> _languageCounts = new();

    /// <summary>Creates the multilingual module with an optional custom phrasebook/translator.</summary>
    public MultilingualModule(MultilingualOptions options, Phrasebook? phrasebook = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Phrasebook = phrasebook ?? new Phrasebook();
        foreach (var (phrase, translation) in options.CustomPhrases)
            Phrasebook.AddPhrase(phrase, _options.PreferredLanguage, translation);
    }

    /// <inheritdoc />
    public override string Name => "multilingual";

    /// <inheritdoc />
    public override int Order => 13; // perception band, before embeddings

    /// <summary>The phrasebook used for translation.</summary>
    public Phrasebook Phrasebook { get; }

    /// <summary>
    /// An optional real translation backend; when set it is consulted before
    /// the built-in phrasebook.
    /// </summary>
    public ITranslationBackend? Translator { get; set; }

    /// <summary>The language the mind prefers to think in (code, e.g. "en").</summary>
    public string PreferredLanguage => _options.PreferredLanguage;

    /// <summary>Detects the language of a text (thin wrapper over <see cref="LanguageDetector.Detect"/>).</summary>
    public LanguageGuess Detect(string text) => LanguageDetector.Detect(text);

    /// <summary>Translates a text into the target language code.</summary>
    public string Translate(string text, string targetCode, string? fromCode = null)
    {
        var source = fromCode ?? LanguageDetector.Detect(text).Code;
        if (source == targetCode || source == "??") return text;
        return Translator?.Translate(text, source, targetCode)
               ?? Phrasebook.Translate(text, source, targetCode);
    }

    /// <summary>How many times each language has been seen (code → count).</summary>
    public IReadOnlyDictionary<string, int> LanguageCounts
    {
        get { lock (_gate) return new Dictionary<string, int>(_languageCounts); }
    }

    /// <summary>The language of the most recent text stimulus (code; "??"" when unknown).</summary>
    public string LastLanguage { get; private set; } = "??";

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var kind = context.Stimulus.Kind;
        if (kind is not (StimulusKind.Text or StimulusKind.Event or StimulusKind.Observation or StimulusKind.Internal))
            return; // audio transcription etc. is handled by their modules

        var text = context.Stimulus.Content;
        if (string.IsNullOrWhiteSpace(text)) return;

        var guess = LanguageDetector.Detect(text);
        lock (_gate)
        {
            var code = guess.Code;
            _languageCounts[code] = _languageCounts.GetValueOrDefault(code) + 1;
            LastLanguage = code;
        }

        context.Bag["language.detected"] = guess.Code;
        context.Bag["language.confidence"] = guess.Confidence;
        context.Bag["language.script"] = guess.Script;
        Host?.Stream?.Add("language",
            $"input is {guess.Language} ({guess.Script}, {guess.Confidence:P0} confidence)");

        // Translate into the preferred language when needed and possible.
        if (_options.AutoTranslate &&
            guess.Code != _options.PreferredLanguage &&
            guess.Code != "??")
        {
            var translated = Translate(text, _options.PreferredLanguage, guess.Code);
            if (!string.Equals(translated, text, StringComparison.Ordinal))
            {
                context.Bag["language.original"] = text;
                context.Bag["language.translated"] = translated;
                context.EffectiveContent = translated; // deeper modules reason over the translation
                Host?.Stream?.Add("language", $"translated to '{_options.PreferredLanguage}': \"{Core.Mind.Truncate(translated, 60)}\"");

                // Only short known phrases (greetings, thanks...) get a direct
                // translation response; longer input is answered after translation.
                if (context.Response is null && Phrasebook.Knows(text))
                {
                    context.Response = new ResponseDraft(
                        $"\"{text}\" means \"{translated}\" in {guess.Language} — got it!",
                        0.6 + guess.Confidence * 0.3, 820, "multilingual");
                }
            }
        }
    }
}
