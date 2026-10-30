using System.Text.RegularExpressions;

namespace Cognify.Reasoning;

/// <summary>
/// A single piece of symbolic knowledge: subject - predicate - object with a
/// confidence, e.g. (bird, is-a, animal, 0.95).
/// </summary>
public sealed record Fact(
    string Subject,
    string Predicate,
    string Object,
    double Confidence = 1.0,
    string? Source = null)
{
    /// <summary>Renders the fact as "subject predicate object".</summary>
    public override string ToString() => $"{Subject} {Predicate} {Object}";
}

/// <summary>
/// Parses simple English sentences into facts:
/// "birds are animals" -> (bird, is-a, animal); "dogs have fur" -> (dog, has, fur).
/// </summary>
public static class FactParser
{
    private static readonly string[] Verbs =
        { "has", "can", "eats", "lives", "likes", "loves", "hates", "needs", "wants", "makes", "uses", "knows" };

    private static readonly Dictionary<string, string> VerbNormalization = new()
    {
        ["are"] = "is-a", ["is"] = "is", ["has"] = "has", ["have"] = "has",
        ["can"] = "can", ["eats"] = "eats", ["eat"] = "eats",
        ["lives"] = "lives-in", ["likes"] = "likes", ["loves"] = "likes",
        ["hates"] = "hates", ["needs"] = "needs", ["wants"] = "needs",
        ["makes"] = "makes", ["uses"] = "uses", ["knows"] = "knows",
        // Dictionary-style knowledge: "joy means a bright feeling of gladness"
        // becomes (joy, means, bright feeling of gladness) — recallable via
        // "what does joy mean?" instead of degrading into a saying.
        ["means"] = "means"
    };

    /// <summary>
    /// Parses text into zero or more facts. Multi-sentence input
    /// ("Birds are animals. Animals are living things.") yields one fact per
    /// sentence instead of one garbled fact from the whole string.
    /// </summary>
    public static IEnumerable<Fact> Parse(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence)) yield break;
        foreach (var part in Regex.Split(sentence.Trim(), @"(?<=[.!?;])\s+"))
        {
            if (string.IsNullOrWhiteSpace(part)) continue;
            foreach (var fact in ParseSingle(part.Trim()))
                yield return fact;
        }
    }

    private static IEnumerable<Fact> ParseSingle(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence)) yield break;

        // Questions are requests, not knowledge. "Hello Yume, what is 2 + 2?"
        // used to parse as the fact (hello-yume-what, is, 2-2) and the mind
        // claimed to have "learned" it — questions never become facts.
        var trimmed = sentence.Trim();
        var firstWord = trimmed.Split(' ', StringSplitOptions.RemoveEmptyEntries)[0]
                               .Trim(',', '.', '!', '?', ':', ';')
                               .ToLowerInvariant();
        if (trimmed.Contains('?') || QuestionStarters.Contains(firstWord)) yield break;

        // Conditionals assert nothing: "if i have no apples and i buy an
        // apple" is a hypothetical setup, not knowledge about the world.
        if (firstWord == "if") yield break;

        // Taught arithmetic ("2 + 2 is 4", "two plus two is four") is real
        // knowledge, but of a structured kind: it becomes a ("2+2", equals, 4)
        // fact instead of a garbled ("2-2", is, 4) one. If the sentence is
        // arithmetic-shaped but was NOT accepted as a teaching (wrong result,
        // unsupported form), it must never fall through to the generic parser
        // either — "2 + 2 is 5" is a mistake, not a fact about "2-2".
        var arithmetic = ArithmeticEngine.TryTeach(trimmed);
        if (arithmetic is not null)
        {
            yield return arithmetic;
            yield break;
        }
        if (ArithmeticEngine.LooksLikeArithmeticStatement(trimmed)) yield break;

        var words = TokenizeWords(sentence);
        if (words.Length < 3) yield break;


        // Pattern: "X is/are a/an Y" or "X is Y" or "X VERB Y"
        for (int i = 1; i < words.Length - 1; i++)
        {
            var raw = words[i];
            if (!VerbNormalization.TryGetValue(raw, out var predicate)) continue;

            // Existentials have no subject: "there is a 10 based number
            // system" announces a thing, it asserts nothing about a thing.
            if (i >= 1 && words[i - 1] == "there" && raw is "is" or "are") continue;

            // "0 can be nothing or can be a holder" — "can be" is the copula
            // of possibility, not a capability claim ("birds can fly" is).
            if (raw == "can" && i + 1 < words.Length && words[i + 1] == "be") continue;

            // Unlabeled questions ("yume what is 2 + 2" — no '?', no leading
            // question word) would otherwise become facts like (yume-what, is,
            // 2-2). Any question word in the subject or object means: not knowledge.
            if (words[..i].Concat(words[(i + 1)..]).Any(IsQuestionWord)) yield break;

            // Strip leading articles from the subject phrase ("a dog" -> "dog").
            var subjectWords = words[..i].ToList();
            while (subjectWords.Count > 0 && subjectWords[0] is "a" or "an" or "the")
                subjectWords.RemoveAt(0);

            // Negations assert nothing: "birds are not animals" is a denial,
            // not a fact that birds are "not animal". Storing it would poison
            // the knowledge base (and make the mind agree with its own
            // contradiction). Denials are felt (surprise), not stored.
            if (subjectWords.Any(w => w is "not" or "never" || w == "n't")) yield break;

            var subject = Singularize(subjectWords.ToArray());

            var objWords = words[(i + 1)..].ToList();

            // Strip articles
            objWords.RemoveAll(w => w is "a" or "an" or "the");
            if (objWords.Count == 0) continue;
            if (objWords.Any(w => w is "not" or "never" or "no" or "nobody" or "nothing" or "nt"))
                yield break;

            var obj = Singularize(objWords.ToArray());
            if (subject.Length == 0 || obj.Length == 0) continue;

            yield return new Fact(subject, predicate, obj, Confidence: 0.9, Source: "parsed");
            yield break; // one fact per sentence keeps things clean
        }
    }

    private static readonly HashSet<string> QuestionStarters = new()
    {
        "what", "who", "when", "where", "why", "how", "which", "whose",
        "is", "are", "am", "do", "does", "did", "can", "could",
        "will", "would", "should", "may"
    };

    private static bool IsQuestionWord(string word) =>
        word is "what" or "who" or "when" or "where" or "why" or "how" or "which" or "whose";

    private static string[] TokenizeWords(string sentence) =>
        new string(sentence.ToLowerInvariant()
                .Select(c => char.IsLetterOrDigit(c) || c == '\'' ? c : ' ')
                .ToArray())
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);

    private static string Singularize(string[] words)
    {
        if (words.Length == 0) return string.Empty;
        var last = words[^1];
        // "-ness" words (happiness, kindness) are already singular: stripping
        // the trailing 's' used to mangle them into "happines"/"kindnes".
        if (last.Length > 3 && last.EndsWith('s') &&
            !last.EndsWith("ss") && !last.EndsWith("ness"))
            last = last[..^1];
        // Multi-word phrases stay natural language: "before affect" — joined
        // with spaces, NOT the old slug form ("before-affect") the mind used
        // to speak back verbatim when recalling facts.
        return string.Join(' ', words[..^1].Append(last));
    }
}
