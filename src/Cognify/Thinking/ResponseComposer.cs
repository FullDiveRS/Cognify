using System.Text.RegularExpressions;
using Cognify.Core;
using Cognify.Emotions;
using Cognify.Personality;
using Cognify.Reasoning;

namespace Cognify.Thinking;

/// <summary>A candidate response produced during deliberation.</summary>
public sealed class Candidate
{
    internal Candidate(string content, string styleName, int styleId, double prior, double confidence, string? body = null)
    {
        Content = content;
        StyleName = styleName;
        StyleId = styleId;
        Prior = prior;
        Confidence = confidence;
        Body = body ?? content;
    }

    /// <summary>The response text.</summary>
    public string Content { get; }

    /// <summary>The core body text before style decoration (opener/question wrappers).</summary>
    internal string Body { get; }

    /// <summary>Name of the response style.</summary>
    public string StyleName { get; }

    /// <summary>Style id (used for reward attribution / Q-table).</summary>
    public int StyleId { get; }

    /// <summary>Static prior score (personality + emotion fit).</summary>
    internal double Prior { get; }

    /// <summary>Baseline confidence if chosen.</summary>
    internal double Confidence { get; }
}

/// <summary>
/// Builds candidate responses in different styles from everything the cycle
/// gathered: direct answers, inferred facts, recalled memories, emotional
/// state and goals. Personality decides which styles feel natural.
/// </summary>
internal sealed class ResponseComposer
{
    private readonly BigFiveTraits? _personality;
    private readonly Random _rng;

    // Templates spoken recently (as TemplateKeys) — the same sentence shape
    // must not come out twice even when a number inside it changed ("learned
    // 1 new thing" vs "learned 3 new things" is the same sentence).
    private readonly HashSet<string> _avoid = new(StringComparer.Ordinal);

    /// <summary>
    /// Normalizes a sentence to its template: case, punctuation around
    /// whitespace and every number collapse to "#" — so "I just learned 3
    /// new thing(s)" and "I just learned 1 new thing(s)" are the SAME line
    /// for repetition purposes.
    /// </summary>
    internal static string TemplateKey(string text) =>
        Regex.Replace(
            Regex.Replace(text.Trim().ToLowerInvariant(), @"\d+(?:\.\d+)?", "#"),
            @"\s+", " ");

    // Style ids double as Q-table keys — keep stable across versions.
    internal const int StyleInformative = 0;
    internal const int StyleEmpathetic = 1;
    internal const int StyleCurious = 2;
    internal const int StyleConcise = 3;
    internal const int StyleReflective = 4;
    internal const int StyleEncouraging = 5;
    internal const int StyleTrained = 6;
    internal const int StyleImpulse = 7;

    internal ResponseComposer(BigFiveTraits? personality, Random rng)
    {
        _personality = personality;
        _rng = rng;
    }

    /// <summary>Registers templates that must not come out of this cycle again.</summary>
    internal void SetAvoid(params string?[] texts)
    {
        foreach (var t in texts)
            if (!string.IsNullOrWhiteSpace(t)) _avoid.Add(TemplateKey(t));
    }

    internal IReadOnlyList<Candidate> ComposeCandidates(
        CognitiveContext ctx, string? directAnswer)
    {
        var emotions = ctx.Emotions;
        var valence = emotions?.Valence ?? 0;
        var arousal = emotions?.Arousal ?? 0;
        var dominant = emotions?.Dominant ?? EmotionType.Joy;
        var topic = TopicOf(ctx.EffectiveContent);
        var learned = ctx.Reasoning?.FactsLearned ?? 0;
        var candidates = new List<Candidate>();

        // 0. Informative ---------------------------------------------------
        // Body selection priority: direct knowledge > just-learned facts >
        // fresh inferences > recalled context. Recall is referenced OBLIQUELY
        // (never quoted verbatim) — memory content is usually the user's own
        // words, and quoting it back made the mind parrot the user.
        string body;
        if (directAnswer is not null)
        {
            body = directAnswer;
            if (learned > 0)
            {
                // The turn both answered AND taught — acknowledge the new
                // knowledge briefly instead of burying the answer.
                body += " " + Pick(
                    $"I also picked up {learned} new fact(s) from this.",
                    $"That also taught me {learned} new thing(s), by the way.",
                    $"Filed {learned} new fact(s) away while we were at it.");
            }
        }
        else if (learned > 0)
        {
            body = Pick(
                $"Got it - I just learned {learned} new thing(s) from that. They are part of my knowledge now.",
                $"Thank you for teaching me - that added {learned} new fact(s) to what I know.",
                $"That clicked into place - {learned} new piece(s) of knowledge just connected for me.",
                $"I am building quite the picture now - {learned} new fact(s) went onto the shelf.",
                $"New knowledge absorbed - {learned} new fact(s) I did not have a moment ago.",
                $"That lesson landed - my knowledge base grew by {learned} new fact(s) just now.");
        }
        else if (ctx.Reasoning is { } r && r.Inferred.Count > 0)
        {
            body = $"Here is what I can work out: {ReasoningModule.Polish(r.Inferred[0].ToString())}. " +
                   (r.Inferred.Count > 1 ? $"That builds on {r.Inferred.Count - 1} other inference(s) I just made." : "");
        }
        else if (ctx.Recalled.Count > 0)
        {
            var memoryTopic = TopicOf(ctx.Recalled[0].Content);
            body = Pick(
                $"It connects with what you told me earlier about {memoryTopic}, though I cannot conclude anything new from it yet.",
                $"I remember our earlier exchange about {memoryTopic} - build on that a little and I can go further.",
                $"Nothing new to conclude yet; it echoes something you said about {memoryTopic} before.",
                $"That stirs a memory about {memoryTopic} - give me one more detail and I might connect it.");
        }
        else
        {
            // Nothing known: say so honestly, and when the CURRENT feeling is
            // an epistemic one (surprise, anticipation) express that
            // confusion/curiosity NATURALLY — the feeling shapes the words
            // instead of being announced in brackets.
            body = dominant is EmotionType.Surprise or EmotionType.Anticipation
                ? Pick($"Hm - {topic} is beyond what I know so far; teach me and I will connect the dots.",
                       $"I honestly do not know {topic} yet - which makes me curious. Teach me a little and I will reason with it.",
                       $"That is new to me - {topic} has a hole-shaped gap in my knowledge. Care to fill it?")
                : Pick(
                    $"I do not have solid knowledge about {topic} yet - teach me and I will reason with it.",
                    $"Nothing in my knowledge base covers {topic} yet - tell me more about it?",
                    $"That is beyond what I know so far; teach me about {topic} and I will connect the dots.",
                    $"My knowledge has a {topic}-shaped hole in it - care to fill it?");
        }
        candidates.Add(new Candidate(body, "informative", StyleInformative,
            prior: 0.55, confidence: directAnswer is not null ? 0.85 : 0.5, body));

        // 1. Empathetic ------------------------------------------------------
        // Feelings are FELT, not announced: by default the opener just
        // sounds like the mood (ShowFeelingReads = false). The mechanical
        // "(my current read: ...)" introspection appears only when the
        // developer turns ShowFeelingReads on — the raw state stays readable
        // any time via mind.Feeling.CurrentStatus.
        if (emotions is not null)
        {
            var expr = ctx.Mind.Config.Expression;
            // Epistemic emotions (surprise, anticipation) get openers that
            // EXPRESS the confusion/curiosity naturally instead of naming it.
            string feeling = dominant is EmotionType.Surprise or EmotionType.Anticipation
                ? Pick("Hm - that one caught me off guard",
                       "Oh - I did not see that coming",
                       "Wait - that genuinely surprises me",
                       "Now I am curious about this one",
                       "That makes me want to know more")
                : valence < -0.25
                    ? Pick("I can sense this carries something heavy",
                           "That sounds like it weighs on you",
                           "There is real weight in what you shared")
                    : valence > 0.25
                        ? Pick("There is a genuinely bright note in this",
                               "This one lights something up in me",
                               "I can feel the good energy in this")
                        : Pick("I feel steady and even about this",
                               "Nothing dramatic stirs in me right now",
                               "I am settled and calm about this");
            string tail = dominant is EmotionType.Fear or EmotionType.Anger
                or EmotionType.Despair or EmotionType.Grief or EmotionType.Suffering
                or EmotionType.Weeping or EmotionType.Anxiety or EmotionType.Helplessness
                or EmotionType.Hatred or EmotionType.IllTemper or EmotionType.LowSpirits
                ? " Let me know if you want me to think through it calmly with you."
                : valence < -0.25 ? " I am here either way." : "";
            string content = expr.ShowFeelingReads
                ? $"{feeling} (my current read: {EmotionalState.Humanize(dominant)}). " + body.Trim() + tail
                : $"{feeling}. " + body.Trim() + tail;
            candidates.Add(new Candidate(
                content,
                "empathetic", StyleEmpathetic,
                prior: 0.4 + Math.Abs(valence) * 0.5 + (_personality?.Agreeableness ?? 0.5) * 0.1,
                confidence: 0.55, body));
        }

        // 2. Curious ---------------------------------------------------------
        // Epistemic emotions drive questions (information-gap theory):
        // surprise and interest create a knowledge gap, and the natural
        // human move is to ASK about it — "what is it?", "tell me more".
        // When the user talks about a person whose gender the mind does not
        // know, the question refers to that person with they/them — the ONLY
        // place they/them is used: describing unknown people, never the mind
        // itself.
        string question;
        // Fresh-appraisal flag from the emotion module (this turn's surprise
        // or interest spike) — NOT the standing blend, which would latch.
        bool epistemicSurge = ctx.Bag.ContainsKey("emotions.epistemic");
        // Follow-up questions are presentation, not cognition: the developer
        // turns them off with WithExpression(e => e.FollowUps(false)) and the
        // mind only answers. The new-word weave is NOT a follow-up — it is
        // her own natural reaction to learning a word — and stays available.
        bool wantQuestion = ctx.Mind.Config.Expression.AskFollowUpQuestions;
        if (MentionsUnknownPerson(ctx.EffectiveContent))
        {
            question = Pick(
                "Who are they to you? I do not know them yet - what are they like?",
                "What are they like? I have not met them, so all I know is what you tell me.",
                "Tell me more about them - who are they, exactly?");
        }
        else if (epistemicSurge)
        {
            question = Pick(
                $"What is {topic}, exactly? You have me curious now.",
                $"Oh? Tell me more about {topic} - I am genuinely interested.",
                $"Now I want to know more about {topic} - what is it?",
                $"Wait - what is the whole story with {topic}? Tell me everything.");
        }
        else if (ctx.Recalled.Count > 0)
        {
            question = Pick($"How does that compare with what you had in mind about {topic}?",
                   $"Is that close to what you were thinking about {topic}?");
        }
        else
        {
            question = Pick($"What should I know about {topic} that I do not yet?",
                   $"Tell me something about {topic} I might be missing?");
        }

        // Growing vocabulary: when the mind has picked up a word it has never
        // woven into a reply before, the curious candidate asks about it.
        // Each new word spawns a brand-new sentence, so the mind's spoken
        // variety grows WITH the conversation. The word is only consumed if
        // this candidate actually wins (thinking marks it spoken).
        string wordWeave = "";
        var lexicon = ctx.Mind.Learning?.Lexicon;
        var freshWord = lexicon?.FreshWordWithAge();
        if (freshWord is { } candidateWord)
        {
            // Words heard long ago but never spoken get honest phrasing —
            // calling a word from ten cycles ago "brand-new" sounds glitchy.
            bool isOld = ctx.Cycle - candidateWord.FirstCycle > 8;
            wordWeave = isOld
                ? Pick(
                    $" By the way - I have been holding onto the word \"{candidateWord.Word}\" since earlier. What does it mean in your world?",
                    $" Also - \"{candidateWord.Word}\" is still a word I have not placed yet. How do you use it?")
                : Pick(
                    $" By the way - I just picked up the word \"{candidateWord.Word}\" from you. What does it mean in your world?",
                    $" Also - \"{candidateWord.Word}\" is a brand-new word for me. How do you use it?",
                    $" And my vocabulary just grew by one: \"{candidateWord.Word}\". What does it mean to you?");
            ctx.Bag["lexicon.weave"] = candidateWord.Word;
        }

        candidates.Add(new Candidate(
            wantQuestion ? $"{body} {question}{wordWeave}" : $"{body}{wordWeave}",
            "curious", StyleCurious,
            prior: 0.4 + (_personality?.Openness ?? 0.5) * 0.45 + (epistemicSurge ? 0.38 : 0),
            confidence: 0.45, body));

        // 3. Concise ----------------------------------------------------------
        string shortBody = directAnswer is not null
            ? FirstSentence(directAnswer)
            : learned > 0
                ? Pick($"Noted - {learned} new fact(s) stored.", $"Learned {learned} new thing(s). Thanks!",
                       $"{learned} new fact(s) filed away.", $"+{learned} new fact(s) in my knowledge base. Thanks!")
                : ctx.Recalled.Count > 0
                    ? Pick($"Still fuzzy on {topic}.", $"No answer on {topic} yet - I only recall our chat about it.")
                    : Pick($"Not sure yet about {topic}.", $"No answer on {topic} yet.");
        candidates.Add(new Candidate(shortBody, "concise", StyleConcise,
            prior: 0.45 + (1 - arousal) * 0.2, confidence: 0.4, shortBody));

        // 4. Reflective ---------------------------------------------------------
        if (ctx.Recalled.Count > 0)
        {
            var memory = ctx.Recalled[0];
            // The lead-in stays complementary: `body` already references the
            // recall obliquely, so the opener only adds a "familiar" cue.
            candidates.Add(new Candidate(
                Pick("That rings a bell.", "Familiar territory.",
                     "I have heard this before.", "This feels like familiar ground.") + " " + body,
                "reflective", StyleReflective,
                prior: 0.45 + memory.Strength * 0.3 + (_personality?.Openness ?? 0.5) * 0.15,
                confidence: 0.5, body));
        }

        // 5. Encouraging ------------------------------------------------------------
        // The private goal ("make the user smile" and friends) is deliberately
        // NOT spoken about — it biases deliberation in the thinking module
        // instead. What comes out is just warmth.
        candidates.Add(new Candidate(
            $"{Pick("Good of you to bring this up!", "I am glad you asked.", "Happy you brought this to me.", "I appreciate you sharing this.", "Always good to hear from you.")} " +
            $"{body}",
            "encouraging", StyleEncouraging,
            prior: 0.4 + (_personality?.Extraversion ?? 0.5) * 0.4 + Math.Max(0, valence) * 0.2,
            confidence: 0.45, body));

        // 6. Trained -----------------------------------------------------------
        // Sayings the mind was TRAINED to say are its own spoken vocabulary.
        // Each trained line is one more sentence shape available, so a
        // well-trained mind simply has more to draw from here — this is the
        // structural fix for "she keeps repeating the same sentence". Only
        // fresh sayings (templates not spoken recently) are offered, and
        // when something relevant to the conversation exists it leads.
        var pool = ctx.Mind.Phrases;
        if (pool.Count > 0)
        {
            var fresh = pool.PickRelevant(ctx.EffectiveContent, _rng, 4)
                .Where(s => !_avoid.Contains(TemplateKey(s)))
                .ToList();
            if (fresh.Count > 0)
            {
                // Smalltalk boost: when there is no answer, no new knowledge,
                // no inference and no recall, speaking from the mind's own
                // trained voice is the most natural thing to do — this is
                // what a trained vocabulary is FOR. Anti-repetition steering
                // keeps it from winning twice in a row.
                bool nothingToSay = directAnswer is null && learned == 0 &&
                                    (ctx.Reasoning?.Inferred.Count ?? 0) == 0 &&
                                    ctx.Recalled.Count == 0;
                string saying = fresh[0];
                candidates.Add(new Candidate(
                    saying, "trained", StyleTrained,
                    prior: 0.42 + (_personality?.Openness ?? 0.5) * 0.2 + (nothingToSay ? 0.25 : 0),
                    confidence: 0.5, saying));
            }
        }

        return candidates;
    }

    /// <summary>
    /// Picks one of the phrasing variants at random, skipping any whose
    /// TEMPLATE was spoken recently (previous reply, last confirmation, the
    /// last few bodies) so two consecutive turns never speak the same line —
    /// even when only a number inside the line changed.
    /// </summary>
    private string Pick(params string[] options)
    {
        var fresh = options.Where(o => !_avoid.Contains(TemplateKey(o))).ToArray();
        var pool = fresh.Length > 0 ? fresh : options;
        return pool[_rng.Next(pool.Length)];
    }

    /// <summary>
    /// Extracts a conversational topic word from text. Questions put the topic
    /// last ("what is your favorite food?" -&gt; "food"), so the last content
    /// token wins; dangling auxiliaries, greeting words, time fillers and
    /// hex-ish tokens (media artifact ids like aa9546f4...) are skipped, and
    /// when the tail token is thin ("many", "plus") the last substantial
    /// content word is preferred.
    /// </summary>
    internal static string TopicOf(string text)
    {
        var content = MoodEngine.Tokenize(text)
            .Where(t => !MoodEngine.IsStopWord(t) && t.Length > 2
                        && !Greetings.Contains(t) && !TimeFillers.Contains(t)
                        && !t.Any(char.IsDigit)) // "media-aa9546..." is not a topic
            .ToList();
        while (content.Count > 1 && TrailingFillers.Contains(content[^1]))
            content.RemoveAt(content.Count - 1);
        for (int i = content.Count - 1; i >= 0; i--)
            if (content[i].Length >= 4)
                return content[i];
        return content.Count > 0 ? content[^1] : "that";
    }

    private static readonly HashSet<string> Greetings = new()
        { "hello", "hey", "greetings", "please", "okay", "ok", "thanks", "thank" };

    // Time/degree words never name a subject — "what is the answer to ...
    // how many do i currently have?" must not become "about currently".
    private static readonly HashSet<string> TimeFillers = new()
    {
        "currently", "now", "then", "today", "recently", "still", "already",
        "again", "always", "never", "maybe", "perhaps", "really", "actually",
        "just", "very", "many", "much", "also", "too", "quite", "pretty"
    };

    private static readonly HashSet<string> TrailingFillers = new()
        { "have", "be", "do", "did", "will", "may", "might", "must", "get", "got", "way", "one" };

    // Words that introduce a PERSON whose gender the mind cannot know. When
    // one appears, the mind refers to that person with they/them — the only
    // context where those pronouns are used (never for the mind itself).
    private static readonly HashSet<string> UnknownPersonWords = new()
    {
        "someone", "somebody", "person", "stranger", "friend", "buddy", "pal",
        "neighbor", "neighbour", "visitor", "guest", "colleague", "coworker",
        "classmate", "cousin", "sibling", "they", "them", "their"
    };

    /// <summary>True when the text talks about a person of unknown gender.</summary>
    internal static bool MentionsUnknownPerson(string text) =>
        MoodEngine.Tokenize(text).Any(t => UnknownPersonWords.Contains(t));

    private static string FirstSentence(string text)
    {
        int cut = text.IndexOfAny(new[] { '.', '!', '?' });
        return cut > 0 ? text[..(cut + 1)] : text;
    }
}
