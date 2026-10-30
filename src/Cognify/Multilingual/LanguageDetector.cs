using System.Text;

namespace Cognify.Multilingual;

/// <summary>
/// Languages the built-in detector can distinguish. Codes follow ISO 639-1.
/// </summary>
public enum Language
{
    /// <summary>Unknown / undetectable.</summary>
    Unknown = 0,
    English, Spanish, French, German, Portuguese, Italian, Dutch,
    Chinese, Japanese, Korean, Russian, Ukrainian, Arabic, Hindi
}

/// <summary>
/// A detected language with a confidence score (0..1).
/// </summary>
/// <param name="Language">The detected language.</param>
/// <param name="Code">ISO 639-1 style code (e.g. "en").</param>
/// <param name="Confidence">How sure the detector is (0..1).</param>
/// <param name="Script">The dominant writing script ("latin", "cyrillic", "han"...).</param>
public sealed record LanguageGuess(Language Language, string Code, double Confidence, string Script);

/// <summary>
/// Real, dependency-free language detection in two stages:
/// <list type="number">
/// <item><b>Script analysis</b> — Unicode block counting identifies Han
/// (Chinese), Kana (Japanese), Hangul (Korean), Cyrillic, Arabic and
/// Devanagari writing outright; Han with Kana present means Japanese.</item>
/// <item><b>Stopword profiling</b> — for Latin-script text, function words
/// ("the", "el", "le", "der"...) are profiled; the language whose stopword
/// set captures the most frequent tokens wins.</item>
/// </list>
/// </summary>
public static class LanguageDetector
{
    private static readonly Dictionary<string, Language> CodeToLanguage = new()
    {
        ["en"] = Language.English, ["es"] = Language.Spanish, ["fr"] = Language.French,
        ["de"] = Language.German, ["pt"] = Language.Portuguese, ["it"] = Language.Italian,
        ["nl"] = Language.Dutch, ["zh"] = Language.Chinese, ["ja"] = Language.Japanese,
        ["ko"] = Language.Korean, ["ru"] = Language.Russian, ["uk"] = Language.Ukrainian,
        ["ar"] = Language.Arabic, ["hi"] = Language.Hindi
    };

    private static readonly Dictionary<Language, string> LanguageToCode =
        CodeToLanguage.ToDictionary(kv => kv.Value, kv => kv.Key);

    private static readonly Dictionary<Language, string[]> Stopwords = new()
    {
        [Language.English] = new[] { "the", "is", "are", "and", "you", "this", "that", "what", "how", "with", "have", "not", "can", "it" },
        [Language.Spanish] = new[] { "el", "la", "los", "las", "es", "y", "que", "de", "un", "una", "como", "está", "para", "por", "con" },
        [Language.French] = new[] { "le", "la", "les", "est", "et", "que", "de", "un", "une", "comment", "pour", "avec", "pas", "je", "tu" },
        [Language.German] = new[] { "der", "die", "das", "ist", "und", "nicht", "ein", "eine", "wie", "mit", "was", "sind", "ich", "du" },
        [Language.Portuguese] = new[] { "o", "a", "os", "as", "é", "e", "que", "de", "um", "uma", "como", "para", "com", "não" },
        [Language.Italian] = new[] { "il", "lo", "la", "le", "è", "e", "che", "di", "un", "una", "come", "per", "con", "non" },
        [Language.Dutch] = new[] { "de", "het", "een", "is", "en", "niet", "van", "wat", "hoe", "met", "zijn", "dat" }
    };

    // Signature words: language-distinctive content words (greetings, politeness
    // formulas...) that identify a language even when no stopword appears —
    // e.g. the single word "gracias".
    private static readonly Dictionary<Language, string[]> SignatureWords = new()
    {
        [Language.English] = new[] { "hello", "thanks", "welcome", "goodbye", "please" },
        [Language.Spanish] = new[] { "gracias", "hola", "adiós", "buenos", "tardes", "noches", "amigo", "mucho", "gusto" },
        [Language.French] = new[] { "bonjour", "merci", "bonsoir", "salut", "au", "revoir", "s'il", "plaît", "ça", "va" },
        [Language.German] = new[] { "hallo", "danke", "tschüss", "wiedersehen", "bitte", "guten", "morgen", "gerne" },
        [Language.Portuguese] = new[] { "olá", "obrigado", "obrigada", "tchau", "bom", "dia", "boa", "noite" },
        [Language.Italian] = new[] { "ciao", "grazie", "arrivederci", "prego", "buongiorno", "buonasera", "amore" },
        [Language.Dutch] = new[] { "hallo", "dankje", "bedankt", "alsjeblieft", "tot", "ziens", "goedemorgen" },
        [Language.Russian] = new[] { "спасибо", "привет", "пожалуйста", "здравствуйте", "до", "свидания", "доброе", "утро" },
        [Language.Ukrainian] = new[] { "дякую", "привіт", "будь", "ласка", "доброго", "ранку", "до", "побачення" },
        [Language.Chinese] = new[] { "你好", "谢谢", "再见", "请", "早上好" },
        [Language.Japanese] = new[] { "ありがとう", "こんにちは", "さようなら", "おはよう" },
        [Language.Korean] = new[] { "감사합니다", "안녕하세요", "안녕히", "부탁합니다" }
    };

    /// <summary>Detects the language of a piece of text. Never throws.</summary>
    public static LanguageGuess Detect(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new LanguageGuess(Language.Unknown, "??", 0, "none");

        var script = DetectScript(text, out int hanCount, out int kanaCount, out int hangulCount,
                                  out int cyrillicCount, out int arabicCount, out int devanagariCount,
                                  out int latinCount);

        // ---- non-Latin scripts are unambiguous -------------------------
        if (hangulCount > 0) return new LanguageGuess(Language.Korean, "ko", 0.95, "hangul");
        if (kanaCount > 0) return new LanguageGuess(Language.Japanese, "ja", 0.95, "han+kana");

        if (hanCount > 0) return new LanguageGuess(Language.Chinese, "zh", 0.93, "han");
        if (arabicCount > 0) return new LanguageGuess(Language.Arabic, "ar", 0.93, "arabic");
        if (devanagariCount > 0) return new LanguageGuess(Language.Hindi, "hi", 0.93, "devanagari");

        if (cyrillicCount > 0)
        {
            // Ukrainian vs Russian: Ukrainian-only letters і/ї/є/ґ vs Russian-only ы/э/ъ.
            bool hasUk = text.Any(c => "іїєґІЇЄҐ".Contains(c));
            bool hasRu = text.Any(c => "ыэъЫЭЪ".Contains(c));
            if (hasUk && !hasRu) return new LanguageGuess(Language.Ukrainian, "uk", 0.9, "cyrillic");
            if (hasRu) return new LanguageGuess(Language.Russian, "ru", 0.9, "cyrillic");
            return new LanguageGuess(Language.Russian, "ru", 0.7, "cyrillic");
        }

        if (latinCount == 0)
            return new LanguageGuess(Language.Unknown, "??", 0, script);

        // ---- Latin-script: stopword + signature profiling ---------------
        var words = text.ToLowerInvariant()
            .Split(new[] { ' ', ',', '.', ';', ':', '!', '?', '"', '\n', '\t' },
                   StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(w => w.Length > 0)
            .ToList();

        if (words.Count == 0)
            return new LanguageGuess(Language.Unknown, "??", 0.1, "latin");

        var scores = new Dictionary<Language, double>();
        foreach (var (language, stopwordList) in Stopwords)
        {
            double hits = words.Count(w => stopwordList.Contains(w));
            if (SignatureWords.TryGetValue(language, out var signatures))
                hits += words.Count(w => signatures.Contains(w)) * 1.5; // signature words are stronger signals
            scores[language] = hits / words.Count;
        }

        var best = scores.OrderByDescending(kv => kv.Value).First();
        if (best.Value <= 0)
            return new LanguageGuess(Language.Unknown, "??", 0.15, "latin");

        double confidence = Math.Clamp(0.4 + best.Value * 1.2, 0, 0.95);
        return new LanguageGuess(best.Key, LanguageToCode[best.Key], confidence, "latin");
    }

    /// <summary>Returns true when the code is one of the supported language codes.</summary>
    public static bool IsSupported(string code) =>
        code is not null && CodeToLanguage.ContainsKey(code.ToLowerInvariant());

    /// <summary>Maps a language code ("en") to its enum value (Unknown when unsupported).</summary>
    public static Language FromCode(string code) =>
        code is not null && CodeToLanguage.TryGetValue(code.ToLowerInvariant(), out var language)
            ? language : Language.Unknown;

    /// <summary>Maps a language enum value to its code ("??"" when unknown).</summary>
    public static string ToCode(Language language) =>
        LanguageToCode.TryGetValue(language, out var code) ? code : "??";

    private static string DetectScript(string text, out int han, out int kana, out int hangul,
        out int cyrillic, out int arabic, out int devanagari, out int latin)
    {
        han = kana = hangul = cyrillic = arabic = devanagari = latin = 0;
        foreach (char c in text)
        {
            int cp = c;
            if (c >= 0xD800 && c <= 0xDBFF) continue; // high surrogate: skip (sufficient for detection)
            if (cp is >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF) han++;
            else if (cp is >= 0x3040 and <= 0x30FF) kana++;
            else if (cp is >= 0xAC00 and <= 0xD7AF or >= 0x1100 and <= 0x11FF) hangul++;
            else if (cp is >= 0x0400 and <= 0x04FF) cyrillic++;
            else if (cp is >= 0x0600 and <= 0x06FF or >= 0x0750 and <= 0x077F) arabic++;
            else if (cp is >= 0x0900 and <= 0x097F) devanagari++;
            else if (cp is (>= 0x0041 and <= 0x007A) or (>= 0x00C0 and <= 0x024F)) latin++;
        }

        if (han > 0) return "han";
        if (hangul > 0) return "hangul";
        if (kana > 0) return "kana";
        if (arabic > 0) return "arabic";
        if (devanagari > 0) return "devanagari";
        if (cyrillic > 0) return "cyrillic";
        if (latin > 0) return "latin";
        return "none";
    }
}
