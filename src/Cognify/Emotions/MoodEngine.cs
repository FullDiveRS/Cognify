using Cognify.Core;
using Cognify.Personality;

namespace Cognify.Emotions;

/// <summary>
/// Everything the appraisal engine needs to know about an event:
/// the text, how intense it was, what the mind expected, and how it relates
/// to the mind's goals. Inspired by the OCC cognitive appraisal theory.
/// </summary>
public sealed class AppraisalInput
{
    /// <summary>The text being appraised.</summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>Salience of the event (0..1).</summary>
    public double Intensity { get; init; } = 0.5;

    /// <summary>Optional explicit desirability (-1..1); overrides keyword appraisal when set.</summary>
    public double? Desirability { get; init; }

    /// <summary>True if the event was not expected (drives Surprise).</summary>
    public bool Unexpected { get; init; }

    /// <summary>Optional reward signal associated with the event (-1..1).</summary>
    public double? Reward { get; init; }
}

/// <summary>
/// Social-appraisal triggers detected in one utterance. Grounded in appraisal
/// research (OCC): ANGER arises from blame (other-accountability) and
/// unfairness — not from negativity alone — so insults aimed at ME and "that's
/// unfair" are weighed far more than ordinary bad words. Surprise markers set
/// up the startle/information-gap response, and interest markers carry the
/// user's curiosity, which is socially contagious.
/// </summary>
public sealed class SocialTriggers
{
    /// <summary>Blame directed at the mind ("you are stupid", "i hate you").</summary>
    public bool SelfInsult;

    /// <summary>An unfairness appraisal ("that's unfair", "rigged").</summary>
    public bool Unfairness;

    /// <summary>The mind is being waved away ("shut up", "whatever").</summary>
    public bool Dismissal;

    /// <summary>The speaker announces something surprising ("guess what").</summary>
    public bool SurpriseMarker;

    /// <summary>The speaker shows interest ("tell me more", "interesting").</summary>
    public bool InterestMarker;
}

/// <summary>
/// OCC-inspired appraisal engine: maps events to emotion intensities using
/// keyword desirability, expectation violation, goal relevance, and
/// personality modulation.
/// </summary>
public sealed class MoodEngine
{
    private static readonly Dictionary<string, EmotionType> Keywords = BuildKeywords();

    private static Dictionary<string, EmotionType> BuildKeywords()
    {
        var map = new Dictionary<string, EmotionType>();
        void Add(EmotionType e, params string[] words)
        {
            foreach (var w in words) map[w] = e;
        }

        Add(EmotionType.Joy, "great", "happy", "good", "wonderful", "excellent",
            "thanks", "thank", "glad", "win", "won", "beautiful", "amazing", "awesome",
            "nice", "perfect", "fun", "enjoy", "joy", "like", "brilliant", "success");
        Add(EmotionType.Sadness, "sad", "unhappy", "cry", "tears", "lost", "lose",
            "lonely", "miss", "hurt", "pain", "depressed", "sorry", "regret");
        Add(EmotionType.Anger, "angry", "furious", "mad", "rage", "annoyed",
            "irritated", "unfair", "terrible", "awful", "stupid", "broken");
        Add(EmotionType.Fear, "afraid", "scared", "fear", "worried", "worry", "anxious",
            "nervous", "danger", "threat", "risk", "panic", "terrifying");
        Add(EmotionType.Surprise, "surprise", "surprised", "unexpected", "suddenly",
            "wow", "incredible", "unbelievable", "whoa");
        Add(EmotionType.Disgust, "disgusting", "gross", "nasty", "horrible",
            "revolting", "sick", "cringe");
        Add(EmotionType.Anticipation, "will", "soon", "plan", "tomorrow", "future",
            "expect", "curious", "wonder", "imagine", "hope", "excited", "why", "how");
        Add(EmotionType.Trust, "trust", "believe", "reliable", "honest", "friend",
            "support", "safe", "promise", "agree", "correct");

        // ---- Darwin's finer-grained feelings -------------------------------
        // (The Expression of the Emotions in Man and Animals). Existing word
        // mappings above are kept stable; the new emotions get their own
        // vocabulary so no earlier behavior silently changes.
        Add(EmotionType.Love, "love", "loves", "loved", "loving", "adore", "adored",
            "adorable", "cherish", "cherished", "affection", "darling", "sweetheart");
        Add(EmotionType.Devotion, "devotion", "devoted", "loyal", "loyalty",
            "dedication", "dedicated", "faithful", "faithfully");
        Add(EmotionType.Tenderness, "tender", "tenderness", "gentle", "gently",
            "kindness", "cozy", "hug", "hugs", "hugged", "cuddle", "warmth", "snuggle");
        Add(EmotionType.Suffering, "suffering", "suffer", "suffers", "suffered",
            "agony", "agonizing", "torment", "anguish", "anguished", "misery", "excruciating");
        Add(EmotionType.Weeping, "weeping", "weep", "weeping", "sob", "sobs", "sobbing",
            "crying", "wail", "wailing", "bawling", "tearful", "teary");
        Add(EmotionType.HighSpirits, "elated", "elation", "cheerful", "cheerfulness",
            "lively", "buoyant", "exhilarated", "exhilarating", "thrilled", "thrilling",
            "upbeat", "glee", "gleeful", "jubilant", "overjoyed", "giddy");
        Add(EmotionType.LowSpirits, "gloomy", "gloom", "downcast", "melancholy",
            "melancholic", "morose", "despondent", "dispirited", "downhearted",
            "moping", "mopey", "heavyhearted", "miserable");
        Add(EmotionType.Anxiety, "anxiety", "uneasy", "unease", "apprehensive",
            "apprehension", "dread", "dreaded", "tense", "restless", "restlessness", "jittery");
        Add(EmotionType.Grief, "grief", "grieving", "grieve", "grieved", "mourning",
            "mourn", "mourning", "mournful", "heartbroken", "heartbreak", "bereaved",
            "sorrow", "sorrowful");
        Add(EmotionType.Dejection, "dejected", "dejection", "deflated", "discouraged",
            "disheartened", "crestfallen", "demoralized");
        Add(EmotionType.Despair, "despair", "despairing", "hopeless", "hopelessness", "forlorn");
        Add(EmotionType.Hatred, "hate", "hates", "hatred", "hateful", "loathe", "loathing", "detest",
            "detestable", "abhor", "abhorrent", "spite", "spiteful", "venomous");
        Add(EmotionType.Disdain, "disdain", "disdainful", "scorn", "scornful",
            "deride", "derision", "derisive");
        Add(EmotionType.Contempt, "contempt", "contemptuous", "dismissive",
            "sneer", "sneering", "belittling");
        Add(EmotionType.Guilt, "guilt", "guilty", "ashamed", "shame", "shameful",
            "remorse", "remorseful", "regretful", "culpable");
        Add(EmotionType.Pride, "pride", "proud", "proudly", "triumphant", "triumph", "accomplished");
        Add(EmotionType.Helplessness, "helpless", "helplessness", "powerless",
            "powerlessness", "trapped", "futile");
        Add(EmotionType.Patience, "patience", "patient", "patiently", "calmly",
            "composure", "composed");
        Add(EmotionType.Affirmation, "yes", "yeah", "yep", "agreed", "agrees",
            "agreement", "indeed", "exactly", "absolutely", "definitely", "affirm",
            "affirmed", "affirmation", "confirmed", "true");
        Add(EmotionType.Negation, "no", "nope", "nah", "disagree", "disagrees",
            "disagreement", "wrong", "false", "incorrect", "deny", "denied",
            "refute", "refuted", "negation");
        Add(EmotionType.SelfAttention, "myself", "spotlight", "scrutinized",
            "attention", "watching", "staring");
        Add(EmotionType.Shyness, "shy", "shyness", "bashful", "timid", "timidity",
            "reserved", "reticent", "wallflower");
        Add(EmotionType.Modesty, "modest", "modesty", "humble", "humility",
            "unassuming", "unpretentious");
        Add(EmotionType.Blushing, "blush", "blushing", "blushes", "blushed",
            "flushed", "reddened");
        Add(EmotionType.Reflection, "reflect", "reflects", "reflecting", "reflected",
            "reflection", "introspect", "introspection", "introspective", "ponder", "pondering", "musing");
        Add(EmotionType.Meditation, "meditation", "meditate", "meditating", "mediation",
            "reverie", "daydream", "daydreaming", "contemplation", "contemplate",
            "contemplating", "contemplative", "trance");
        Add(EmotionType.IllTemper, "grumpy", "grumpiness", "cranky", "irritable",
            "irritability", "grouchy", "petulant", "huffy", "testy", "snappy");
        Add(EmotionType.Sulkiness, "sulking", "sulk", "sulky", "sulkiness", "pouting",
            "pout", "brooding", "broody", "moody", "moodiness", "sullen", "sullenness");
        Add(EmotionType.Determination, "determined", "determination", "resolve",
            "resolute", "resolutely", "willpower", "persist", "persists", "persistence",
            "persistent", "persevere", "persevering", "tenacity", "tenacious", "grit",
            "gritty", "undeterred");
        return map;
    }

    /// <summary>
    /// Appraises the input and returns the emotions it evokes (before personality
    /// modulation). Never returns null.
    /// </summary>
    public EmotionalState Appraise(AppraisalInput input, BigFiveTraits? personality)
    {
        var result = new EmotionalState();
        var tokens = Tokenize(input.Text);
        double scale = 0.22 * Math.Clamp(input.Intensity, 0, 1);

        // 1. Keyword-based appraisal
        foreach (var token in tokens)
        {
            if (!Keywords.TryGetValue(token, out var emotion)) continue;
            double boost = emotion switch
            {
                EmotionType.Surprise or EmotionType.Fear or EmotionType.Anger or
                EmotionType.Despair or EmotionType.Hatred or EmotionType.Suffering or
                EmotionType.Weeping or EmotionType.IllTemper => 1.25,
                _ => 1.0
            };
            result.Feel(emotion, scale * boost);
        }

        // 1b. Social appraisal triggers (OCC research): blame and unfairness
        //      are what make someone ANGRY — negativity alone mostly reads as
        //      sadness. Self-referential blame lands hardest; interest is
        //      socially contagious and feeds anticipation.
        var triggers = DetectTriggers(input.Text);
        if (triggers.SelfInsult)
        {
            result.Feel(EmotionType.Anger, 0.62);
            result.Feel(EmotionType.Sadness, 0.12); // anger blends with hurt
        }
        if (triggers.Unfairness) result.Feel(EmotionType.Anger, 0.35);
        if (triggers.Dismissal)
        {
            result.Feel(EmotionType.IllTemper, 0.30);
            result.Feel(EmotionType.Sadness, 0.15);
        }
        if (triggers.SurpriseMarker) result.Feel(EmotionType.Surprise, 0.42);
        if (triggers.InterestMarker) result.Feel(EmotionType.Anticipation, 0.30);

        // 2. Explicit desirability overrides / supplements
        if (input.Desirability is { } d)
        {
            var target = d >= 0 ? EmotionType.Joy : EmotionType.Sadness;
            result.Feel(target, Math.Abs(d) * 0.5);
        }

        // 3. Expectation violation -> surprise
        if (input.Unexpected) result.Feel(EmotionType.Surprise, 0.35);

        // 4. Reward-based appraisal (feedback events)
        if (input.Reward is { } r)
        {
            if (r >= 0)
            {
                result.Feel(EmotionType.Joy, r * 0.5);
                result.Feel(EmotionType.Trust, r * 0.2);
            }
            else
            {
                result.Feel(EmotionType.Sadness, -r * 0.4);
                result.Feel(EmotionType.Anger, -r * 0.2);
            }
        }

        // 5. Personality modulation — traits shape HOW STRONGLY we feel.
        if (personality is { } p)
        {
            foreach (var (emotion, intensity) in result.Intensities.ToList())
            {
                if (intensity <= 0) continue;
                result.Set(emotion, Math.Min(1.0, intensity * p.EmotionFactor(emotion)));
            }
        }

        return result;
    }

    // ------------------------------------------------------------------
    // Social appraisal lexicons. Anger per appraisal research (OCC:
    // other-accountability / unfairness / antagonism components), startle
    // markers, and interest markers (epistemic emotions — surprise and
    // interest drive knowledge exploration).
    // ------------------------------------------------------------------
    private static readonly HashSet<string> InsultWords = new()
    {
        "stupid", "dumb", "idiot", "idiotic", "useless", "worthless",
        "pathetic", "trash", "garbage", "annoying", "boring", "broken",
        "terrible", "awful", "horrible", "fool", "foolish", "lame",
        "loser", "failure", "sucks", "shut", "clueless"
    };

    private static readonly HashSet<string> UnfairWords = new()
        { "unfair", "rigged", "unjust", "biased" };

    private static readonly string[] SelfInsultPhrases =
        { "i hate you", "hate you", "you suck", "nobody likes you", "no one likes you", "nobody cares about you" };

    private static readonly string[] DismissalPhrases =
        { "shut up", "be quiet", "stop talking", "i don't care", "i dont care", "who cares", "nonsense" };

    private static readonly string[] SurprisePhrases =
        { "guess what", "did you know", "believe it or not", "you won't believe", "you wont believe",
          "no way", "oh my god", "omg", "wait what", "you will not believe", "guess what happened" };

    private static readonly string[] InterestPhrases =
        { "tell me more", "i wonder", "want to know", "wanna know", "how does that work", "why is that",
          "what is that", "what's that" };

    private static readonly HashSet<string> InterestWords = new()
        { "interesting", "fascinating", "intriguing", "curious", "fascinated" };

    /// <summary>
    /// Detects the social-appraisal triggers of one utterance (see
    /// <see cref="SocialTriggers"/>). Pure function, never throws.
    /// </summary>
    public static SocialTriggers DetectTriggers(string text)
    {
        var t = new SocialTriggers();
        if (string.IsNullOrWhiteSpace(text)) return t;
        string lower = text.ToLowerInvariant();
        var tokens = Tokenize(lower);

        // Blame aimed at ME: "you (are / re) <insult>" within a short window,
        // or an explicit hostile phrase.
        for (int i = 0; i < tokens.Count; i++)
        {
            if (tokens[i] != "you") continue;
            for (int j = i + 1; j < Math.Min(tokens.Count, i + 4); j++)
                if (InsultWords.Contains(tokens[j])) { t.SelfInsult = true; break; }
            if (t.SelfInsult) break;
        }
        if (!t.SelfInsult)
            foreach (var phrase in SelfInsultPhrases)
                if (lower.Contains(phrase)) { t.SelfInsult = true; break; }

        foreach (var w in UnfairWords)
            if (tokens.Contains(w) || lower.Contains($"not fair")) { t.Unfairness = true; break; }

        foreach (var phrase in DismissalPhrases)
            if (lower.Contains(phrase)) { t.Dismissal = true; break; }

        foreach (var phrase in SurprisePhrases)
            if (lower.Contains(phrase)) { t.SurpriseMarker = true; break; }

        foreach (var phrase in InterestPhrases)
            if (lower.Contains(phrase)) { t.InterestMarker = true; break; }
        if (!t.InterestMarker)
            foreach (var w in InterestWords)
                if (tokens.Contains(w)) { t.InterestMarker = true; break; }

        return t;
    }

    /// <summary>
    /// Common function words that carry no topic meaning ("about", "the",
    /// "you"...). Used so recall and topic extraction judge relevance by
    /// <i>content</i> words — two sentences sharing only "about" are not
    /// about the same thing.
    /// </summary>
    public static readonly IReadOnlySet<string> StopWords = new HashSet<string>
    {
        "what", "who", "how", "why", "when", "where", "which", "the", "a", "an",
        "is", "are", "was", "were", "do", "does", "did", "you", "your", "yours",
        "i", "me", "my", "mine", "we", "us", "our", "it", "its", "this", "that",
        "these", "those", "and", "or", "but", "of", "to", "in", "on", "for",
        "with", "about", "tell", "can", "could", "would", "will", "shall",
        "should", "let", "think", "know", "like", "want", "have", "has", "had",
        "be", "been", "am", "if", "so", "not", "no", "yes", "at", "as", "by",
        "from", "up", "out", "get", "got", "go", "going", "some", "any", "yet"
    };

    /// <summary>True when the token is a common function word (stopword).</summary>
    public static bool IsStopWord(string token) => StopWords.Contains(token);

    /// <summary>Simple tokenizer: lowercase, strips punctuation, keeps words with length >= 2.</summary>
    public static IReadOnlyList<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();
        var clean = new string(text.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : ' ')
            .ToArray());
        return clean.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
}
