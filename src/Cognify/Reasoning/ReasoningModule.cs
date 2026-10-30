using System.Text.RegularExpressions;
using Cognify.Core;

namespace Cognify.Reasoning;

/// <summary>
/// The reasoning module gives the mind symbolic knowledge and inference.
/// It parses taught facts, answers questions, runs forward chaining over
/// rules, induces new knowledge from repeated patterns, and can abduce
/// explanations for observations.
/// </summary>
public sealed class ReasoningModule : CognitiveModuleBase
{
    private readonly ReasoningOptions _options;
    private readonly Dictionary<(string A, string B), int> _coOccurrence = new();

    internal ReasoningModule(ReasoningOptions options) => _options = options;

    /// <inheritdoc />
    public override string Name => "reasoning";

    /// <inheritdoc />
    public override int Order => 30;

    /// <summary>The mind's knowledge base.</summary>
    public KnowledgeBase Knowledge { get; } = new();

    /// <summary>The inference engine bound to the knowledge base.</summary>
    public InferenceEngine Engine { get; private set; } = null!;

    /// <summary>
    /// Arithmetic reasoner over the same knowledge base: taught facts are
    /// looked up, unknown ones are derived from related taught facts, and
    /// word problems are evaluated step by step. The calculator tool is only
    /// consulted (by the tool module) when this engine comes up empty.
    /// </summary>
    public ArithmeticEngine Arithmetic { get; private set; } = null!;

    /// <inheritdoc />
    public override void Attach(Mind mind)
    {
        base.Attach(mind);
        Engine = new InferenceEngine(Knowledge, _options);
        Arithmetic = new ArithmeticEngine(Knowledge);
    }

    /// <summary>Adds a fact directly to the knowledge base.</summary>
    public void Learn(Fact fact) => Knowledge.Add(fact);

    /// <summary>Adds a logic rule directly (e.g. IF ?s is-a bird THEN ?s can fly).</summary>
    public void Learn(LogicRule rule) => Knowledge.AddRule(rule);

    /// <summary>
    /// Answers a question using knowledge + hierarchical expansion.
    /// Arithmetic questions ("what is 2 + 2?") are answered from taught
    /// arithmetic first. Returns null when the mind does not know.
    /// </summary>
    public string? AnswerQuestion(string question)
    {
        var q = question.TrimEnd('?', '!', '.').Trim();
        var lower = q.ToLowerInvariant();
        var tokens = Emotions.MoodEngine.Tokenize(lower).ToArray();

        // Identity: "introduce yourself" / "who are you" — answer from the
        // mind's own name, description, gender and current mood instead of
        // treating it as an unknown topic.
        if (lower.Contains("introduce yourself") ||
            lower.Contains("who are you") ||
            lower.Contains("tell me about yourself") ||
            lower is "who am i talking to" or "what are you")
        {
            return Host?.DescribeSelf() ?? "I am a mind built with Cognify.";
        }

        if (tokens.Length == 0) return null;

        // Preambles must not hide the question: "now what is after?" is a
        // question about "after" — the filler words up front are dropped
        // before the question patterns below try to match.
        tokens = SkipLeadingFillers(tokens);
        if (tokens.Length == 0) return null;

        // Arithmetic and word problems come before knowledge lookup: "what is
        // 2 + 2?" must never be treated as a question about the topic "2".
        var math = Arithmetic.TryAnswerQuestion(question) ?? Arithmetic.TryWordProblem(question);
        if (math is not null) return math.Reply;

        // "what is the meaning of X" — a dictionary-style lookup over the
        // means facts the mind was taught ("joy means a bright feeling...").
        if (tokens.Length >= 6 && tokens[0] == "what" && tokens[1] == "is" &&
            tokens[2] == "the" && tokens[3] == "meaning" && tokens[4] == "of")
        {
            return DescribeEntity(SingularizeRange(tokens[5..]));
        }

        // "define X" — same lookup, dictionary phrasing.
        if (tokens[0] == "define" && tokens.Length >= 2)
        {
            return DescribeEntity(SingularizeRange(tokens[1..]));
        }

        // "what is X" / "what are X" / "who is X"
        if (tokens[0] is "what" or "who" && tokens.Length >= 3 &&
            (tokens[1] == "is" || tokens[1] == "are"))
        {
            var entity = SingularizeRange(tokens[2..].Where(t => t != "a" && t != "an" && t != "the"));
            return DescribeEntity(entity);
        }

        // "what does X have/can X do" style
        if (tokens[0] == "what" && tokens.Length >= 3 && tokens[1] == "does")
        {
            var entity = SingularizeRange(new[] { tokens[2] });
            return DescribeEntity(entity);
        }

        // "is X Y" / "are X Y" -> yes/no check
        if (tokens[0] is "is" or "are" && tokens.Length >= 3)
        {
            var entity = tokens[1];
            var claim = SingularizeRange(tokens[2..].Where(t => t != "a" && t != "an" && t != "the"));
            if (CheckClaim(entity, claim) is { } verdict) return verdict;
        }

        // Trailing claim after a preamble: "if a bird is an animal and animals
        // are living things, is a bird living?" — the real question rides
        // after the LAST is/are, so check that clause too.
        int lastCopula = Array.LastIndexOf(tokens, "is");
        if (lastCopula < 0) lastCopula = Array.LastIndexOf(tokens, "are");
        if (lastCopula > 0)
        {
            var rest = tokens[(lastCopula + 1)..]
                .Where(t => t != "a" && t != "an" && t != "the").ToArray();
            if (rest.Length >= 2)
            {
                var trailingEntity = SingularizeRange(rest[..^1]);
                var trailingClaim = SingularizeRange(rest[^1..]);
                if (CheckClaim(trailingEntity, trailingClaim) is { } trailingVerdict)
                    return trailingVerdict;
            }
        }

        return null;
    }

    private static readonly HashSet<string> LeadingFillers = new()
    {
        "now", "then", "also", "so", "well", "okay", "ok", "and", "plus",
        "today", "please", "anyway", "just", "still", "simply"
    };

    private static string[] SkipLeadingFillers(string[] tokens)
    {
        int i = 0;
        while (i < tokens.Length && LeadingFillers.Contains(tokens[i])) i++;
        return i == 0 ? tokens : tokens[i..];
    }

    private string? DescribeEntity(string entity)
    {
        var direct = Knowledge.Query(entity, null, null);
        var lines = new List<string>();

        foreach (var fact in direct.Where(f => f.Predicate != "related-to"))
            lines.Add(Polish($"{fact.Subject} {PredicatePhrase(fact.Predicate)} {fact.Object}."));

        // Ancestor chain, skipping is-a relations already stated directly.
        var statedParents = direct.Where(f => f.Predicate == "is-a").Select(f => f.Object).ToHashSet();
        foreach (var ancestor in Engine.AncestorsOf(entity))
            if (!statedParents.Contains(ancestor.Object))
                lines.Add($"{entity} is a kind of {ancestor.Object}.");

        if (lines.Count == 0) return null;
        return string.Join(" ", lines.Distinct());
    }

    private static string PredicatePhrase(string predicate) => predicate switch
    {
        "is-a" => "is a",
        "is" => "is",
        "equals" => "equals",
        "has" => "has",
        "can" => "can",
        "eats" => "eats",
        "lives-in" => "lives in",
        "likes" => "likes",
        "means" => "means",
        _ => predicate.Replace('-', ' ')
    };

    /// <summary>
    /// Makes stored knowledge readable when spoken: the standalone pronoun
    /// gets its capital ("if i buy" -> "if I buy") and the sentence starts
    /// with a capital letter. Stored facts stay lowercase for matching; this
    /// is a display-only polish.
    /// </summary>
    public static string Polish(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return text;
        var polished = Regex.Replace(text, @"\bi\b", "I");
        return char.ToUpperInvariant(polished[0]) + polished[1..];
    }

    private string? CheckClaim(string entity, string claim)
    {
        var aboutEntity = Knowledge.Query(entity, null, null);

        // Letter-and-digit skeleton match: "living thing" vs "living-thing"
        // vs old persisted slugs all compare equal.
        static string Skeleton(string s) =>
            new(s.ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());
        string claimKey = Skeleton(claim);

        bool direct = aboutEntity.Any(f => Skeleton(f.Object) == claimKey);

        // Prefix match lets "is a bird living?" hit the taught "living thing"
        // concept ("living" vs "living thing") without exact equality.
        bool PrefixHit(string obj) =>
            Skeleton(obj) == claimKey ||
            (claimKey.Length >= 4 && Skeleton(obj).StartsWith(claimKey, StringComparison.Ordinal));

        Fact? chainHit = Engine.AncestorsOf(entity).FirstOrDefault(f =>
            PrefixHit(f.Object) || Knowledge.Query(f.Object, null, claim).Count > 0);
        bool predicateMatch = aboutEntity.Any(f => f.Predicate == claim);

        if (aboutEntity.Count == 0) return null; // unknown entity: let other modules respond
        if (direct || chainHit is not null)
        {
            string basis = chainHit is { } hit && !direct
                ? $" - {entity} is a kind of {hit.Object.Replace('-', ' ')}"
                : "";
            return Polish($"Yes - {entity} relates to {claim.Replace('-', ' ')} based on what I know{basis}.");
        }
        if (predicateMatch) return Polish($"Yes - {entity} {claim.Replace('-', ' ')} according to my knowledge.");
        return Polish($"I don't have evidence that {entity} {claim.Replace('-', ' ')}.");
    }

    private static string SingularizeRange(IEnumerable<string> words)
    {
        var list = words.ToList();
        if (list.Count == 0) return string.Empty;
        var last = list[^1];
        // "-ness" words (happiness, kindness) are already singular — parity
        // with FactParser.Singularize.
        if (last.Length > 3 && last.EndsWith('s') &&
            !last.EndsWith("ss") && !last.EndsWith("ness"))
            last = last[..^1];
        // Natural phrases, not slugs: "living thing", never "living-thing"
        // (parity with FactParser.Singularize).
        return string.Join(' ', list.Take(list.Count - 1).Append(last));
    }

    /// <summary>
    /// True when the statement NEGATES something the mind knows
    /// ("birds are not animals" vs. taught "birds are animals"). A violated
    /// expectation is the textbook trigger of surprise — the emotion module
    /// uses this to startle, and the answer (if any) still comes from
    /// reasoning, not from the emotion.
    /// </summary>
    public bool KnownContradiction(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;

        // Token-based scan (contraction-aware): "birds aren't animals" and
        // "birds are not animals" both become ... is are not animals ...
        foreach (var part in Regex.Split(text.Trim(), @"(?<=[.!?;])\s+"))
        {
            var words = part.Trim().TrimEnd('.', '!', '?').ToLowerInvariant()
                .Replace("n't", " not")
                .Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length < 4) continue;

            for (int i = 0; i < words.Length - 2; i++)
            {
                if (words[i] is not ("is" or "are")) continue;
                if (words[i + 1] != "not") continue;

                var subjectWords = words[..i]
                    .Where(w => !ContradictionFillers.Contains(w)).ToArray();
                var claimWords = words[(i + 2)..]
                    .Where(w => !ContradictionFillers.Contains(w)).ToArray();
                if (subjectWords.Length == 0 || claimWords.Length == 0) break;

                var entity = SingularizeRange(subjectWords);
                var claim = SingularizeRange(claimWords);

                static string Skeleton(string s) =>
                    new(s.ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());
                string claimKey = Skeleton(claim);

                bool contradicted = Knowledge.Query(entity, null, null).Any(f =>
                    f.Predicate is "is-a" or "is" &&
                    (Skeleton(f.Object) == claimKey ||
                     (claimKey.Length >= 4 && Skeleton(f.Object).StartsWith(claimKey, StringComparison.Ordinal)) ||
                     (Skeleton(f.Object).Length >= 4 && claimKey.StartsWith(Skeleton(f.Object), StringComparison.Ordinal))));
                if (contradicted) return true;
                break; // only the first "X is not Y" clause per sentence
            }
        }
        return false;
    }

    private static readonly HashSet<string> ContradictionFillers = new()
        { "a", "an", "the", "well", "actually", "really", "just", "very" };

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        var text = context.EffectiveContent;

        // 0. Learn facts stated in plain sentences — parity with CognifyPy's
        //    cognitive cycle. "birds are animals" said in normal conversation
        //    becomes knowledge here, not only via the explicit Teach() API.
        int learnedNow = 0;
        foreach (var fact in FactParser.Parse(text))
            if (Knowledge.Add(fact)) learnedNow++;

        // 0b. A taught counting sequence ("the system is 0, 1, 2, ..., 9")
        //     becomes the mind's number line — counting along it later is
        //     reasoning from teaching, not calculation, so no calculator is
        //     ever needed for sums that stay on the taught line.
        Arithmetic.TeachSequence(text);

        // 1. Direct question answering. Arithmetic questions are answered
        //    from taught facts (or derived from them); the tool module reads
        //    the Bag flags and only falls back to the calculator when the
        //    mind could not work the answer out on its own.
        var mathAnswer = Arithmetic.TryAnswerQuestion(text) ?? Arithmetic.TryWordProblem(text);
        var answer = mathAnswer?.Reply ?? AnswerQuestion(text);
        if (answer is not null)
        {
            var result = new ReasoningResult();
            result.Answer = answer;
            result.FactsLearned = learnedNow;
            if (mathAnswer is not null)
            {
                context.Bag["reasoning.mathAnswered"] = "1";
                result.AddExplanation($"Arithmetic {mathAnswer.Expression} = {mathAnswer.Result} " +
                                      (mathAnswer.DirectlyTaught ? "(a fact you taught me)" : "(reasoned from taught arithmetic)"));
            }
            else
            {
                result.AddExplanation($"Direct knowledge lookup for: {text}");
            }
            if (learnedNow > 0)
                result.AddExplanation($"Learned {learnedNow} new fact(s) from the input.");
            context.Reasoning = result;
            return;
        }

        // 1b. It looked like an arithmetic question but the mind could not
        //     answer it from taught knowledge — let the calculator help (only
        //     if tools are attached), and be transparent about it.
        if (mathAnswer is null &&
            (Arithmetic.LooksLikeArithmeticQuestion(text) || Arithmetic.LooksLikeWordProblem(text)))
        {
            context.Bag["reasoning.mathAttempted"] = "1";
        }

        // 2. For statements, run forward chaining to expand knowledge.
        var result2 = new ReasoningResult();
        result2.FactsLearned = learnedNow;
        if (learnedNow > 0)
            result2.AddExplanation($"Learned {learnedNow} new fact(s) from the input.");
        var derived = Engine.ForwardChain();
        foreach (var fact in derived.Take(5))
            result2.AddExplanation($"Inferred: {fact} (confidence {fact.Confidence:0.00})");
        foreach (var fact in derived) result2.AddInferred(fact);

        // 3. Track concept co-occurrence for induction.
        if (_options.AllowInduction)
        {
            var tokens = Emotions.MoodEngine.Tokenize(text)
                .Where(t => t.Length > 3)
                .Distinct()
                .Take(6)
                .ToList();
            for (int i = 0; i < tokens.Count; i++)
                for (int j = i + 1; j < tokens.Count; j++)
                {
                    var key = (tokens[i], tokens[j]);
                    _coOccurrence[key] = _coOccurrence.GetValueOrDefault(key) + 1;
                }

            var induced = Engine.Induce(_coOccurrence, _options.InductionThreshold);
            foreach (var fact in induced)
                result2.AddExplanation($"Induced new relation from repeated observation: {fact}");
        }

        context.Reasoning = result2;
        context.Bag["reasoning.newFacts"] = derived.Count;
    }
}
