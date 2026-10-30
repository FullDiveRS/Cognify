namespace Cognify.Multilingual;

/// <summary>
/// A pluggable translation backend. The built-in <see cref="Phrasebook"/>
/// covers common phrases; assign a real translator (an API, an LLM...) by
/// implementing this interface and setting <c>MultilingualModule.Translator</c>.
/// </summary>
public interface ITranslationBackend
{
    /// <summary>Translates text from the source language code to the target code.</summary>
    string Translate(string text, string fromCode, string toCode);
}

/// <summary>
/// The built-in phrasebook: a curated table of everyday phrases (greetings,
/// thanks, yes/no, question words) in the supported languages, plus support
/// for user-added phrases. Lookup is exact-match on normalized text, which is
/// enough for greetings and polite formulas — the bulk of real-world
/// cross-language chatter.
/// </summary>
public sealed class Phrasebook : ITranslationBackend
{
    // phrase-key → (language-code → translation)
    private readonly Dictionary<string, Dictionary<string, string>> _phrases;
    private readonly Dictionary<string, string> _custom = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates the phrasebook with the built-in phrase table.</summary>
    public Phrasebook()
    {
        _phrases = new Dictionary<string, Dictionary<string, string>>
        {
            // "hello"
            ["hello"] = new()
            {
                ["en"] = "hello", ["es"] = "hola", ["fr"] = "bonjour", ["de"] = "hallo",
                ["pt"] = "olá", ["it"] = "ciao", ["nl"] = "hallo", ["zh"] = "你好",
                ["ja"] = "こんにちは", ["ko"] = "안녕하세요", ["ru"] = "привет",
                ["uk"] = "привіт", ["ar"] = "مرحبا", ["hi"] = "नमस्ते"
            },
            // "thank you"
            ["thank you"] = new()
            {
                ["en"] = "thank you", ["es"] = "gracias", ["fr"] = "merci", ["de"] = "danke",
                ["pt"] = "obrigado", ["it"] = "grazie", ["nl"] = "dank je", ["zh"] = "谢谢",
                ["ja"] = "ありがとう", ["ko"] = "감사합니다", ["ru"] = "спасибо",
                ["uk"] = "дякую", ["ar"] = "شكرا", ["hi"] = "धन्यवाद"
            },
            // "yes"
            ["yes"] = new()
            {
                ["en"] = "yes", ["es"] = "sí", ["fr"] = "oui", ["de"] = "ja",
                ["pt"] = "sim", ["it"] = "sì", ["nl"] = "ja", ["zh"] = "是",
                ["ja"] = "はい", ["ko"] = "네", ["ru"] = "да", ["uk"] = "так",
                ["ar"] = "نعم", ["hi"] = "हाँ"
            },
            // "no"
            ["no"] = new()
            {
                ["en"] = "no", ["es"] = "no", ["fr"] = "non", ["de"] = "nein",
                ["pt"] = "não", ["it"] = "no", ["nl"] = "nee", ["zh"] = "不",
                ["ja"] = "いいえ", ["ko"] = "아니요", ["ru"] = "нет", ["uk"] = "ні",
                ["ar"] = "لا", ["hi"] = "नहीं"
            },
            // "goodbye"
            ["goodbye"] = new()
            {
                ["en"] = "goodbye", ["es"] = "adiós", ["fr"] = "au revoir", ["de"] = "auf wiedersehen",
                ["pt"] = "tchau", ["it"] = "arrivederci", ["nl"] = "tot ziens", ["zh"] = "再见",
                ["ja"] = "さようなら", ["ko"] = "안녕히 가세요", ["ru"] = "до свидания",
                ["uk"] = "до побачення", ["ar"] = "وداعا", ["hi"] = "अलविदा"
            },
            // "good morning"
            ["good morning"] = new()
            {
                ["en"] = "good morning", ["es"] = "buenos días", ["fr"] = "bonjour",
                ["de"] = "guten morgen", ["pt"] = "bom dia", ["it"] = "buongiorno",
                ["nl"] = "goedemorgen", ["zh"] = "早上好", ["ja"] = "おはよう",
                ["ko"] = "좋은 아침", ["ru"] = "доброе утро", ["uk"] = "доброго ранку",
                ["ar"] = "صباح الخير", ["hi"] = "सुप्रभात"
            },
            // "how are you"
            ["how are you"] = new()
            {
                ["en"] = "how are you", ["es"] = "cómo estás", ["fr"] = "comment ça va",
                ["de"] = "wie geht es dir", ["pt"] = "como vai", ["it"] = "come stai",
                ["nl"] = "hoe gaat het", ["zh"] = "你好吗", ["ja"] = "お元気ですか",
                ["ko"] = "어떻게 지내세요", ["ru"] = "как дела", ["uk"] = "як справи",
                ["ar"] = "كيف حالك", ["hi"] = "आप कैसे हैं"
            },
            // "please"
            ["please"] = new()
            {
                ["en"] = "please", ["es"] = "por favor", ["fr"] = "s'il vous plaît",
                ["de"] = "bitte", ["pt"] = "por favor", ["it"] = "per favore",
                ["nl"] = "alsjeblieft", ["zh"] = "请", ["ja"] = "お願いします",
                ["ko"] = "부탁합니다", ["ru"] = "пожалуйста", ["uk"] = "будь ласка",
                ["ar"] = "من فضلك", ["hi"] = "कृपया"
            },
            // "sorry"
            ["sorry"] = new()
            {
                ["en"] = "sorry", ["es"] = "lo siento", ["fr"] = "désolé",
                ["de"] = "entschuldigung", ["pt"] = "desculpa", ["it"] = "scusa",
                ["nl"] = "sorry", ["zh"] = "对不起", ["ja"] = "ごめんなさい",
                ["ko"] = "죄송합니다", ["ru"] = "извини", ["uk"] = "вибачте",
                ["ar"] = "آسف", ["hi"] = "क्षमा करें"
            },
            // "i love you"
            ["i love you"] = new()
            {
                ["en"] = "i love you", ["es"] = "te quiero", ["fr"] = "je t'aime",
                ["de"] = "ich liebe dich", ["pt"] = "eu te amo", ["it"] = "ti amo",
                ["nl"] = "ik hou van je", ["zh"] = "我爱你", ["ja"] = "愛してる",
                ["ko"] = "사랑해", ["ru"] = "я тебя люблю", ["uk"] = "я тебе кохаю",
                ["ar"] = "أحبك", ["hi"] = "मैं तुमसे प्यार करता हूँ"
            }
        };
    }

    /// <summary>Phrases known to the phrasebook (normalized keys).</summary>
    public IReadOnlyCollection<string> KnownPhrases => _phrases.Keys.Concat(_custom.Keys).ToList();

    /// <summary>
    /// Adds or overrides a phrase translation. <paramref name="phrase"/> is the
    /// normalized English key (e.g. "thank you"); <paramref name="languageCode"/>
    /// is the target language; <paramref name="translation"/> the text to use.
    /// </summary>
    public void AddPhrase(string phrase, string languageCode, string translation)
    {
        if (string.IsNullOrWhiteSpace(phrase) || string.IsNullOrWhiteSpace(translation))
            throw new ArgumentException("Phrase and translation must be non-empty.");
        _custom[Normalize(phrase)] = translation;
        _custom[$"{Normalize(phrase)}::{languageCode.ToLowerInvariant()}"] = translation;
    }

    /// <summary>Translates a known phrase between any two supported languages.</summary>
    public string Translate(string text, string fromCode, string toCode)
    {
        if (string.IsNullOrWhiteSpace(text)) return text;
        if (fromCode == toCode) return text;

        var key = Normalize(text);

        // Custom overrides win.
        if (_custom.TryGetValue($"{key}::{toCode.ToLowerInvariant()}", out var custom))
            return custom;

        // Exact phrase lookup (English-keyed phrases).
        if (_phrases.TryGetValue(key, out var table) && table.TryGetValue(toCode.ToLowerInvariant(), out var translation))
            return translation;

        // Reverse lookup: the text is a known translation in the source
        // language (e.g. "hola" when translating Spanish → French).
        var reverseKey = FindPhraseKey(text, fromCode);
        if (reverseKey is not null &&
            _phrases.TryGetValue(reverseKey, out var reverseTable) &&
            reverseTable.TryGetValue(toCode.ToLowerInvariant(), out var reverseTranslation))
            return reverseTranslation;

        // Word-by-word fallback for simple sentences of known words.
        var words = Normalize(text).Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length is > 1 and <= 4 &&
            words.All(w => _phrases.ContainsKey(w)))
        {
            var translated = words.Select(w =>
                _phrases[w].TryGetValue(toCode.ToLowerInvariant(), out var t) ? t : w);
            return string.Join(' ', translated);
        }

        // Unknown: return original so the caller knows it's untranslated.
        return text;
    }

    /// <summary>
    /// Finds the canonical phrase key whose translation in
    /// <paramref name="fromCode"/> equals the given text (case-insensitive).
    /// </summary>
    private string? FindPhraseKey(string text, string fromCode)
    {
        var normalized = Normalize(text);
        foreach (var (key, table) in _phrases)
        {
            if (table.TryGetValue(fromCode.ToLowerInvariant(), out var translation) &&
                Normalize(translation) == normalized)
                return key;
        }
        return null;
    }

    /// <summary>True when the phrasebook knows this exact phrase (as a key or as a known translation).</summary>
    public bool Knows(string text)
    {
        var key = Normalize(text);
        if (_phrases.ContainsKey(key) || _custom.ContainsKey(key)) return true;
        foreach (var table in _phrases.Values)
        {
            foreach (var translation in table.Values)
                if (Normalize(translation) == key) return true;
        }
        return false;
    }

    private static string Normalize(string text) =>
        string.Join(' ', text.ToLowerInvariant()
            .Split(new[] { ' ', ',', '.', ';', ':', '!', '?', '"', '\n', '\t' },
                   StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
}
