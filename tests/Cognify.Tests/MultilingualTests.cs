using Cognify.Core;
using Cognify.Multilingual;

namespace Cognify.Tests;

/// <summary>Tests for language detection, the phrasebook and the module.</summary>
public class MultilingualTests
{
    // ---- Detection: scripts ---------------------------------------------

    [Theory]
    [InlineData("你好世界", Language.Chinese, "zh")]
    [InlineData("こんにちは世界", Language.Japanese, "ja")]
    [InlineData("안녕하세요", Language.Korean, "ko")]
    [InlineData("привет мир", Language.Russian, "ru")]
    [InlineData("مرحبا بالعالم", Language.Arabic, "ar")]
    [InlineData("नमस्ते दुनिया", Language.Hindi, "hi")]
    public void Detect_IdentifiesScripts(string text, Language expected, string code)
    {
        var guess = LanguageDetector.Detect(text);
        Assert.Equal(expected, guess.Language);
        Assert.Equal(code, guess.Code);
        Assert.True(guess.Confidence > 0.5);
    }

    [Fact]
    public void Detect_JapaneseWithKanji_IsJapaneseNotChinese()
    {
        var guess = LanguageDetector.Detect("こんにちは、元気ですか。世界。");
        Assert.Equal(Language.Japanese, guess.Language);
    }

    [Fact]
    public void Detect_UkrainianVersusRussian()
    {
        // Ukrainian-specific letter 'i' (Cyrillic і) vs Russian-only ы/э/ъ.
        Assert.Equal(Language.Ukrainian, LanguageDetector.Detect("дякую, мені це подобається").Language);
        Assert.Equal(Language.Russian, LanguageDetector.Detect("спасибо, это хорошо").Language);
    }

    // ---- Detection: Latin stopword profiles ------------------------------

    [Theory]
    [InlineData("what is the weather like today", Language.English, "en")]
    [InlineData("el perro está en la casa", Language.Spanish, "es")]
    [InlineData("le chat est sur la table", Language.French, "fr")]
    [InlineData("der hund ist in dem haus", Language.German, "de")]
    [InlineData("il gatto è sulla sedia", Language.Italian, "it")]
    public void Detect_IdentifiesLatinLanguages(string text, Language expected, string code)
    {
        var guess = LanguageDetector.Detect(text);
        Assert.Equal(expected, guess.Language);
        Assert.Equal(code, guess.Code);
    }

    [Fact]
    public void Detect_EmptyText_IsUnknown()
    {
        var guess = LanguageDetector.Detect("   ");
        Assert.Equal(Language.Unknown, guess.Language);
        Assert.Equal("??", guess.Code);
    }

    [Fact]
    public void Detect_UnknownLatinWords_IsLowConfidence()
    {
        var guess = LanguageDetector.Detect("xyzzy plugh quux");
        Assert.True(guess.Confidence < 0.4);
    }

    [Fact]
    public void CodeMapping_RoundTrips()
    {
        Assert.Equal(Language.Spanish, LanguageDetector.FromCode("es"));
        Assert.Equal("es", LanguageDetector.ToCode(Language.Spanish));
        Assert.Equal(Language.Unknown, LanguageDetector.FromCode("xx"));
        Assert.False(LanguageDetector.IsSupported("xx"));
    }

    // ---- Phrasebook -------------------------------------------------------

    [Fact]
    public void Phrasebook_TranslatesKnownPhrases()
    {
        var book = new Phrasebook();
        Assert.Equal("hola", book.Translate("hello", "en", "es"));
        Assert.Equal("merci", book.Translate("Thank you!", "en", "fr"));
        Assert.Equal("danke", book.Translate("thank you", "en", "de"));
    }

    [Fact]
    public void Phrasebook_TranslatesBetweenAnyTwoLanguages()
    {
        var book = new Phrasebook();
        Assert.Equal("bonjour", book.Translate("hola", "es", "fr"));
        Assert.Equal("ありがとう", book.Translate("thank you", "en", "ja"));
    }

    [Fact]
    public void Phrasebook_TranslatesWordByWord_ForSimpleSentences()
    {
        var book = new Phrasebook();
        var translated = book.Translate("yes please", "en", "es");
        Assert.Contains("sí", translated);
        Assert.Contains("por favor", translated);
    }

    [Fact]
    public void Phrasebook_UnknownPhrase_ReturnsOriginal()
    {
        var book = new Phrasebook();
        var original = "completely unknown quantum entanglement phrase";
        Assert.Equal(original, book.Translate(original, "en", "es"));
    }

    [Fact]
    public void Phrasebook_SameLanguage_ReturnsOriginal()
    {
        var book = new Phrasebook();
        Assert.Equal("hello", book.Translate("hello", "en", "en"));
    }

    [Fact]
    public void Phrasebook_CustomPhrases_OverrideBuiltins()
    {
        var book = new Phrasebook();
        book.AddPhrase("hello", "es", "¡qué pasa!");
        Assert.Equal("¡qué pasa!", book.Translate("hello", "en", "es"));
    }

    // ---- Module ------------------------------------------------------------

    [Fact]
    public void Module_DetectsLanguage_AndRecordsCounts()
    {
        using var mind = MindBuilder.Create("Polyglot").WithMultilingual().Build();
        mind.Ask("hello");
        mind.Ask("what is the weather");

        var languages = mind.Languages!;
        Assert.Equal("en", languages.LastLanguage);
        Assert.True(languages.LanguageCounts["en"] >= 2);
    }

    [Fact]
    public void Module_AutoTranslatesKnownPhrase()
    {
        using var mind = MindBuilder.Create("Polyglot")
            .WithMultilingual(m => m.Prefer("en"))
            .Build();

        var response = mind.Ask("gracias");
        Assert.Contains("gracias", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("es", mind.Languages!.LastLanguage);
    }

    [Fact]
    public void Module_ForeignQuestion_TranslatesEffectiveContent()
    {
        // A Spanish question about a taught fact: after translation the
        // reasoning module should still be able to answer it.
        using var mind = MindBuilder.Create("Polyglot")
            .WithMultilingual(m => m.Prefer("en"))
            .WithReasoning()
            .WithThinking()
            .Build();

        mind.Teach("birds are animals");
        var response = mind.Ask("el pájaro es un animal?");

        // The translation of an unknown sentence stays original, so reasoning
        // sees the raw text; but the detection must have happened.
        Assert.Equal("es", mind.Languages!.LastLanguage);
        Assert.NotNull(response);
    }

    [Fact]
    public void Module_ScriptLanguages_AreDetected()
    {
        using var mind = MindBuilder.Create("Polyglot").WithMultilingual().Build();
        mind.Ask("你好");
        Assert.Equal("zh", mind.Languages!.LastLanguage);
    }

    [Fact]
    public void Module_IgnoresImageStimuli()
    {
        using var mind = MindBuilder.Create("Polyglot").WithMultilingual().Build();
        mind.Process(Stimulus.Image("a red apple"));
        Assert.Equal("??", mind.Languages!.LastLanguage);
    }
}
