using System.Globalization;
using System.Text.RegularExpressions;

namespace Cognify.Reasoning;

/// <summary>
/// The outcome of an arithmetic reasoning pass: what was asked, what the mind
/// concluded, and HOW it got there (a fact the user taught, or reasoning built
/// on top of taught facts).
/// </summary>
/// <param name="Expression">Normalized expression, e.g. "2+2".</param>
/// <param name="Result">The computed result as text, e.g. "4".</param>
/// <param name="Reply">The full spoken answer (includes a short explanation).</param>
/// <param name="DirectlyTaught">True when the exact fact was taught by the user.</param>
/// <param name="Confidence">How confident the mind is in this answer.</param>
public sealed record ArithmeticResult(
    string Expression,
    string Result,
    string Reply,
    bool DirectlyTaught,
    double Confidence)
{
    /// <summary>Human-readable derivation steps (empty when directly taught).</summary>
    public IReadOnlyList<string> Steps { get; init; } = Array.Empty<string>();
}

/// <summary>
/// Reasons about arithmetic the way a child does: from what it was TAUGHT.
///
/// <list type="number">
/// <item>Teaching: "2 + 2 = 4", "two plus two is four", "2 + 2 makes 4" become
/// knowledge-base facts ("2+2" equals "4"). A fact is only stored when the
/// stated result is actually correct.</item>
/// <item>Teaching a number line: "the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
/// becomes the mind's counting ladder. Counting along it ("3 + 4" — count on
/// from 3) is reasoning from teaching, not calculation.</item>
/// <item>Questions: "what is 2 + 2?" is looked up first. If never taught, the
/// engine DERIVES the answer from related taught facts — e.g. with
/// "2 + 1 = 3" and "3 + 1 = 4" it counts on from 2: 3, 4.</item>
/// <item>Word problems: "if I have 1 apple and buy another one, how many
/// apples do I have?" extracts the quantities and transfers, then evaluates
/// each step from taught knowledge. Natural quantities work too: "no
/// apples" is 0, "an apple"/"a apple"/"one apple" is 1.</item>
/// </list>
///
/// Only when the engine cannot work an expression out does the caller fall
/// back to the calculator tool — the mind never NEEDS the calculator for
/// anything it was taught (or can derive from what it was taught).
/// </summary>
public sealed class ArithmeticEngine
{
    private readonly KnowledgeBase _kb;

    /// <summary>Creates an arithmetic reasoner over the given knowledge base.</summary>
    public ArithmeticEngine(KnowledgeBase kb) => _kb = kb;

    private const int MaxSteps = 32;

    // ====================================================================
    // Teaching: statements like "2 + 2 = 4" become knowledge-base facts.
    // ====================================================================

    private static readonly Regex TeachPattern = new(
        @"^(?:the answer to\s+)?(?<a>-?\d+(?:\.\d+)?)\s*(?<op>[+\-*/])\s*(?<b>-?\d+(?:\.\d+)?)\s*" +
        @"(?:=|is|equals|makes|gives|will be)\s*(?<r>-?\d+(?:\.\d+)?)$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// Parses a taught arithmetic statement ("2 + 2 = 4", "two plus two is
    /// four", "2 + 2 makes 4") into a knowledge fact. Returns null when the
    /// sentence is not a correct arithmetic teaching statement — the generic
    /// fact parser then gets its chance.
    /// </summary>
    public static Fact? TryTeach(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence)) return null;

        var text = Normalize(sentence.Trim().TrimEnd('.', '!', '?', ','));
        var m = TeachPattern.Match(text);
        if (!m.Success) return null;

        double a = ParseNum(m.Groups["a"].Value);
        double b = ParseNum(m.Groups["b"].Value);
        double r = ParseNum(m.Groups["r"].Value);
        char op = m.Groups["op"].Value[0];

        // Sanity: the taught result must actually be correct — a wrong "fact"
        // would poison every later derivation.
        double expected = op switch
        {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => b == 0 ? double.NaN : a / b,
            _ => double.NaN
        };
        if (double.IsNaN(expected) || Math.Abs(expected - r) > 1e-9) return null;

        return new Fact(Key(a, op, b), "equals", Num(r), Confidence: 0.95, Source: "taught-arithmetic");
    }

    // ====================================================================
    // Questions: "what is 2 + 2?", "how much is two plus two?", "2+2?"
    // ====================================================================

    private static readonly Regex Expression = new(
        @"(?<a>-?\d+(?:\.\d+)?)\s*(?<op>[+\-*/])\s*(?<b>-?\d+(?:\.\d+)?)",
        RegexOptions.Compiled);

    private static readonly Regex Questionish = new(
        @"\b(what|whats|how much|how many|calculate|compute|solve|tell me)\b|\?",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>
    /// True when the text contains an arithmetic expression shaped like a
    /// question (or a bare expression). Used to decide whether the calculator
    /// may step in as a fallback when reasoning comes up empty.
    /// </summary>
    public bool LooksLikeArithmeticQuestion(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        return Expression.IsMatch(Normalize(text)) && Questionish.IsMatch(text);
    }

    /// <summary>
    /// Answers an arithmetic question from taught knowledge only:
    /// exact taught fact first, then derivation from related taught facts.
    /// Returns null when the mind cannot work it out — it will say so
    /// instead of pretending (the calculator is the caller's fallback).
    /// </summary>
    public ArithmeticResult? TryAnswerQuestion(string input)
    {
        if (string.IsNullOrWhiteSpace(input)) return null;

        var normalized = Normalize(input);
        if (!Questionish.IsMatch(input) && !IsBareExpression(normalized)) return null;

        Match? best = null;
        foreach (Match m in Expression.Matches(normalized))
            if (best is null || m.Length > best.Length) best = m;
        if (best is null) return null;

        double a = ParseNum(best.Groups["a"].Value);
        double b = ParseNum(best.Groups["b"].Value);
        char op = best.Groups["op"].Value[0];
        string expr = Key(a, op, b);

        // 1. The fact was taught directly.
        if (Lookup(expr) is { } taught)
            return new ArithmeticResult(expr, taught,
                $"{Display(expr)} is {taught} - you taught me that one.",
                DirectlyTaught: true, Confidence: 0.95);

        // 2. Derive it from related taught facts.
        return op switch
        {
            '+' => DeriveAdd(expr, a, b),
            '-' => DeriveSub(expr, a, b),
            '*' => DeriveMul(expr, a, b),
            '/' => DeriveDiv(expr, a, b),
            _ => null
        };
    }

    private static bool IsBareExpression(string normalized) =>
        Regex.IsMatch(normalized.TrimEnd('?', ' ', '.'),
            @"^-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+$");

    /// <summary>
    /// True when a statement (already past the question guard) carries an
    /// arithmetic expression, e.g. "2 + 2 is 5". Such sentences are either
    /// accepted teaching (handled above) or mistakes — never generic facts.
    /// </summary>
    public static bool LooksLikeArithmeticStatement(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence)) return false;
        var normalized = Normalize(sentence);
        return Expression.IsMatch(normalized);
    }

    // ====================================================================
    // Word problems: "if I have 1 apple and buy another one, how many
    // apples do I have?" — quantities + transfer verbs, evaluated from
    // taught knowledge step by step.
    // ====================================================================

    // "how may" is a common typo for "how many" — a real user types it, and
    // treating it as a non-question made the mind ignore the whole problem.
    private static readonly Regex HowMany = new(@"\bhow (?:many|much|may)\b",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // Quantities may be natural language: "no apples" (0), "an apple" (1).
    private static readonly Regex Initial = new(
        @"\b(?:i have|we have|i(?:'ve)? got|you have|there are|there is|i started with|i start with|she has|he has)\s+" +
        @"(?<n>-?\d+(?:\.\d+)?|no\b|zero\b|an?\b)(?:\s+(?<unit>[a-z]+))?",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex Gain = new(
        @"\b(?:buy|bought|get|gets|got|receive|receives|received|find|finds|found|gain|gains|gained|" +
        @"add|adds|added|make|makes|made|bake|bakes|baked|pick|picks|picked)\s+" +
        @"(?:another(?:\s+one)?|one more|(?<n>\d+)(?:\s+more)?|" +
        @"an?\s+(?!lot\b|few\b|couple\b|bit\b|number\b|handful\b)[a-z]+|one\b)",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly Regex Lose = new(
        @"\b(?:eat|eats|ate|lose|loses|lost|give away|gives away|gave away|sell|sells|sold|" +
        @"break|breaks|broke|drop|drops|dropped)\s+" +
        @"(?:(?<n>\d+(?:\.\d+)?)|an?\s+(?!lot\b|few\b|couple\b|bit\b)[a-z]+|one\b)",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private static readonly HashSet<string> NonUnits = new()
        { "of", "more", "less", "them", "it", "to", "and", "idea", "clue", "doubt", "chance" };

    /// <summary>"no" -&gt; 0, "a"/"an"/"one" -&gt; 1, digits parse directly.</summary>
    private static double ParseQuantity(string raw) => raw switch
    {
        "no" or "zero" => 0,
        "a" or "an" => 1,
        _ => ParseNum(raw)
    };

    /// <summary>True when the text has the "how many ...?" word-problem shape.</summary>
    public bool LooksLikeWordProblem(string text) =>
        !string.IsNullOrWhiteSpace(text) && HowMany.IsMatch(text);

    // Extracted problem setup: starting quantity, whether it was stated
    // explicitly, the unit noun (may be null) and the transfer events in
    // reading order.
    private readonly record struct WordProblemPlan(
        double Start, bool ExplicitStart, string? Unit,
        List<(int Index, char Op, double Amount, string Label)> Events);

    /// <summary>
    /// Extracts a word problem's setup: the starting quantity and every
    /// gain/loss event, in the order they appear. Returns null when the text
    /// has no "how many" question or no starting quantity at all.
    /// </summary>
    private WordProblemPlan? PlanWordProblem(string input)
    {
        if (string.IsNullOrWhiteSpace(input) || !HowMany.IsMatch(input)) return null;

        var text = Normalize(input);

        // Several candidate starts may appear ("there is a 10 based number
        // system ... if i have no apples"). Prefer explicit quantities over
        // bare articles, then unit-bearing matches, then the later one — the
        // real setup usually sits closest to the "how many" question.
        Match? init = null;
        int initScore = -1;
        foreach (Match m in Initial.Matches(text))
        {
            int score = StartScore(m);
            if (score > initScore || (score == initScore && m.Index > init!.Index))
            {
                init = m;
                initScore = score;
            }
        }
        if (init is null) return null;

        double start = ParseQuantity(init.Groups["n"].Value);
        bool explicitStart = char.IsDigit(init.Groups["n"].Value[0])
                             || init.Groups["n"].Value is "no" or "zero";
        string? unit = init.Groups["unit"].Success ? init.Groups["unit"].Value : null;
        if (unit is not null && (NonUnits.Contains(unit) || unit.Length < 3)) unit = null;

        var events = new List<(int Index, char Op, double Amount, string Label)>();
        foreach (Match m in Gain.Matches(text))
        {
            double n = m.Groups["n"].Success ? ParseNum(m.Groups["n"].Value) : 1;
            events.Add((m.Index, '+', n, m.Value.Trim()));
        }
        foreach (Match m in Lose.Matches(text))
            events.Add((m.Index, '-', m.Groups["n"].Success ? ParseNum(m.Groups["n"].Value) : 1, m.Value.Trim()));
        events.Sort((x, y) => x.Index.CompareTo(y.Index));

        return new WordProblemPlan(start, explicitStart, unit, events);
    }

    /// <summary>"I have 3 apples, how many do I have?" — answered from the
    /// stated quantity itself (only when it was explicit).</summary>
    private ArithmeticResult StatedQuantityResult(double start, string? unit)
    {
        string unit0 = unit is null ? "" : $" {Plural(unit, Num(start))}";
        return new ArithmeticResult(
            Num(start), Num(start),
            $"You start with {Num(start)}{unit0}, so you still have {Num(start)}{unit0} - that is what you told me.",
            DirectlyTaught: false, Confidence: 0.7);
    }

    /// <summary>
    /// Solves a small addition/subtraction word problem by extracting the
    /// starting quantity and each gain/loss (in the order they appear), then
    /// evaluating every step from taught knowledge. Returns null when anything
    /// is missing or untaught — the mind stays honest about what it cannot do.
    /// </summary>
    public ArithmeticResult? TryWordProblem(string input)
    {
        if (PlanWordProblem(input) is not { } plan) return null;
        var (start, explicitStart, unit, events) = plan;

        // No transfers at all: "I have 3 apples, how many apples do I have?"
        // is answered from the stated quantity — but only when it was stated
        // explicitly ("a dog" alone is too weak to answer from).
        if (events.Count == 0)
            return explicitStart ? StatedQuantityResult(start, unit) : null;

        var steps = new List<string>();
        double cur = start;
        foreach (var (_, op, amount, label) in events)
        {
            string expr = Key(cur, op, amount);
            string? value = Lookup(expr);
            if (value is null)
            {
                var derived = op == '+' ? DeriveAdd(expr, cur, amount) : DeriveSub(expr, cur, amount);
                if (derived is null) return null; // honest: cannot work it out
                value = derived.Result;
                steps.AddRange(derived.Steps);
            }

            string unitPhrase = unit is null ? "" : $" {Plural(unit, value)}";
            steps.Add($"{label}: {Num(cur)} {op} {Num(amount)} = {value}{unitPhrase}");
            cur = ParseNum(value);
        }

        string startUnit = unit is null ? "" : $" {Plural(unit, Num(start))}";
        string resultUnit = unit is null ? "" : $" {Plural(unit, Num(cur))}";
        string reply = $"You start with {Num(start)}{startUnit}, and all of that works out to " +
                       $"{Num(cur)}{resultUnit} - {steps[^1]}.";
        return new ArithmeticResult(
            Expression: string.Join(" ", events.Select(e => $"{e.Op}{Num(e.Amount)}")),
            Result: Num(cur), Reply: reply, DirectlyTaught: false, Confidence: 0.75);
    }

    /// <summary>
    /// Solves a word problem the calculator's way: same extraction as
    /// <see cref="TryWordProblem"/>, but every step is plain arithmetic
    /// instead of taught knowledge. This is the transparent fallback the tool
    /// module uses when taught reasoning could not work the problem out —
    /// the mind says openly that it leaned on the calculator.
    /// </summary>
    public ArithmeticResult? TryWordProblemMechanically(string input)
    {
        if (PlanWordProblem(input) is not { } plan) return null;
        var (start, explicitStart, unit, events) = plan;

        if (events.Count == 0)
            return explicitStart ? StatedQuantityResult(start, unit) : null;

        var steps = new List<string>();
        double cur = start;
        foreach (var (_, op, amount, label) in events)
        {
            double next = op == '+' ? cur + amount : cur - amount;
            string unitPhrase = unit is null ? "" : $" {Plural(unit, Num(next))}";
            steps.Add($"{label}: {Num(cur)} {op} {Num(amount)} = {Num(next)}{unitPhrase}");
            cur = next;
        }

        string startUnit = unit is null ? "" : $" {Plural(unit, Num(start))}";
        string resultUnit = unit is null ? "" : $" {Plural(unit, Num(cur))}";
        string reply = $"You start with {Num(start)}{startUnit}, and all of that works out to " +
                       $"{Num(cur)}{resultUnit} - {steps[^1]}.";
        return new ArithmeticResult(
            Expression: string.Join(" ", events.Select(e => $"{e.Op}{Num(e.Amount)}")),
            Result: Num(cur), Reply: reply, DirectlyTaught: false, Confidence: 0.6);
    }

    /// <summary>
    /// Ranks an Initial match: explicit quantities (digits, "no") beat bare
    /// articles ("a"), a following unit noun beats none. Ties go to the later
    /// match — the setup nearest the question.
    /// </summary>
    private int StartScore(Match m)
    {
        string n = m.Groups["n"].Value;
        int score = char.IsDigit(n[0]) || n is "no" or "zero" ? 2 : 0;
        if (m.Groups["unit"].Success && !NonUnits.Contains(m.Groups["unit"].Value)
            && m.Groups["unit"].Value.Length >= 3)
            score += 1;
        return score;
    }

    // ====================================================================
    // The counting ladder — a taught number line ("0, 1, 2, 3, 4, 5, 6, 7,
    // 8, 9"). Counting along it is reasoning from what was taught.
    // ====================================================================

    private (double Min, double Max)? _ladder;

    /// <summary>
    /// True when the input taught a counting sequence (at least five numbers
    /// running upward by one, e.g. "the system is 0, 1, 2, 3, 4, 5, 6, 7, 8,
    /// 9"). The sequence becomes the mind's number line: it can now count on
    /// and back along it without ever needing a calculator.
    /// </summary>
    public bool TeachSequence(string input)
    {
        if (string.IsNullOrWhiteSpace(input)) return false;

        bool taught = false;
        foreach (var part in Regex.Split(input.Trim(), @"(?<=[.!?;])\s+"))
        {
            var tokens = Regex.Matches(Normalize(part), @"-?\d+(?:\.\d+)?")
                              .Select(m => ParseNum(m.Value))
                              .ToList();
            int i = 0;
            while (i < tokens.Count)
            {
                int j = i;
                while (j + 1 < tokens.Count && Math.Abs(tokens[j + 1] - tokens[j] - 1) < 1e-9) j++;
                if (j - i + 1 >= 5 && IsCountable(tokens[i]) && IsCountable(tokens[j]))
                {
                    RememberLadder(tokens[i], tokens[j]);
                    taught = true;
                }
                i = j + 1;
            }
        }
        return taught;
    }

    private void RememberLadder(double min, double max)
    {
        // Overlapping or touching ranges extend the line; a disjoint run is
        // a new number line and replaces the old one.
        if (_ladder is not { } cur || min > cur.Max + 1 || max < cur.Min - 1)
            _ladder = (min, max);
        else
            _ladder = (Math.Min(cur.Min, min), Math.Max(cur.Max, max));
    }

    // ====================================================================
    // Derivation — reasoning on top of taught facts
    // ====================================================================

    /// <summary>
    /// a + b by climbing taught facts: prefer the largest taught jump
    /// ("cur + k = m"), fall back to counting on one at a time ("cur + 1").
    /// When no path leads up from a, a taught "b + a" fact (commutativity)
    /// still counts.
    /// </summary>
    private ArithmeticResult? DeriveAdd(string expr, double a, double b)
    {
        var steps = new List<string>();
        if (IsCountable(a) && IsCountable(b) && b >= 0)
        {
            if (b == 0)
                return new ArithmeticResult(expr, Num(a),
                    $"{Display(expr)} is {Num(a)} - adding zero changes nothing.",
                    DirectlyTaught: false, Confidence: 0.7);

            // "no apples and i buy an apple": starting from NOTHING and adding
            // b leaves b. Zero is not an unknown quantity here — the word
            // problem itself stated it ("no apples" = none), so this is an
            // identity the mind can reason out without ever being taught it.
            if (a == 0)
            {
                steps.Add($"{Num(0)} + {Num(b)} = {Num(b)} (starting from nothing)");
                return new ArithmeticResult(expr, Num(b),
                    $"{Display(expr)} is {Num(b)} - starting from nothing and adding {Num(b)} gives {Num(b)}.",
                    DirectlyTaught: false, Confidence: 0.8) { Steps = steps };
            }

            double cur = a, remaining = b;
            while (remaining > 0 && steps.Count < MaxSteps)
            {
                if (FindJump(cur, '+', remaining) is not (double k, double landed))
                    break;
                steps.Add($"{Num(cur)} + {Num(k)} = {Num(landed)}" + (k == 1 ? " (counting on)" : ""));
                cur = landed;
                remaining -= k;
            }
            if (remaining == 0 && steps.Count > 0)
                return new ArithmeticResult(expr, Num(cur),
                    $"{Display(expr)} is {Num(cur)} - I worked it out from what you taught me: {string.Join(", then ", steps)}.",
                    DirectlyTaught: false, Confidence: 0.8) { Steps = steps };
        }

        // Commutativity: maybe the mirror fact "b + a" was the taught one.
        string flipped = Key(b, '+', a);
        if (Lookup(flipped) is { } mirror)
            return new ArithmeticResult(expr, mirror,
                $"{Display(expr)} is {mirror} - same as {Display(flipped)}, which you taught me.",
                DirectlyTaught: false, Confidence: 0.78);

        return null;
    }

    /// <summary>
    /// a - b by stepping down taught facts: either a taught subtraction
    /// ("cur - k = m") or the inverse of a taught addition ("p + k = cur"
    /// means cur - k = p, and cur - p = k as well).
    /// </summary>
    private ArithmeticResult? DeriveSub(string expr, double a, double b)
    {
        if (!IsCountable(a) || !IsCountable(b) || a < 0 || b < 0) return null;

        var steps = new List<string>();
        double cur = a, remaining = b;
        while (remaining > 0 && steps.Count < MaxSteps)
        {
            if (FindJump(cur, '-', remaining) is not (double k, double landed))
                return null;
            steps.Add($"{Num(cur)} - {Num(k)} = {Num(landed)}");
            cur = landed;
            remaining -= k;
        }

        if (b == 0)
            return new ArithmeticResult(expr, Num(a),
                $"{Display(expr)} is {Num(a)} - taking zero away changes nothing.",
                DirectlyTaught: false, Confidence: 0.7);

        return steps.Count > 0
            ? new ArithmeticResult(expr, Num(cur),
                $"{Display(expr)} is {Num(cur)} - I worked backwards from what you taught me: {string.Join(", then ", steps)}.",
                DirectlyTaught: false, Confidence: 0.8) { Steps = steps }
            : null;
    }

    /// <summary>a * b (small whole b) as repeated addition of a.</summary>
    private ArithmeticResult? DeriveMul(string expr, double a, double b)
    {
        if (!IsCountable(a) || !IsCountable(b) || b < 0 || b > 12 || Math.Abs(a) > 1000) return null;
        if (b == 0)
            return new ArithmeticResult(expr, "0",
                $"{Display(expr)} is 0 - {Num(a)} added together zero times is still 0.",
                DirectlyTaught: false, Confidence: 0.7);

        var steps = new List<string>();
        double cur = a;
        for (int i = 2; i <= (int)b; i++)
        {
            var add = DeriveAdd(Key(cur, '+', a), cur, a);
            if (add is null) return null;
            steps.Add($"{Num(cur)} + {Num(a)} = {add.Result}");
            cur = ParseNum(add.Result);
        }
        return new ArithmeticResult(expr, Num(cur),
            $"{Display(expr)} is {Num(cur)} - that is {Num(a)} added together {Num(b)} times: {string.Join(", ", steps)}.",
            DirectlyTaught: false, Confidence: 0.75) { Steps = steps };
    }

    /// <summary>a / b (small whole, exact) as repeated subtraction.</summary>
    private ArithmeticResult? DeriveDiv(string expr, double a, double b)
    {
        if (!IsCountable(a) || !IsCountable(b) || b <= 0 || a < 0 || Math.Abs(a) > 400) return null;

        double cur = a;
        int times = 0;
        while (cur > 0 && times <= MaxSteps)
        {
            if (cur < b) return null; // not exact — honest failure
            var sub = DeriveSub(Key(cur, '-', b), cur, b);
            if (sub is null) return null;
            cur = ParseNum(sub.Result);
            times++;
        }
        if (cur != 0) return null;
        return new ArithmeticResult(expr, Num(times),
            $"{Display(expr)} is {Num(times)} - I took {Num(b)} away from {Num(a)} {Num(times)} time(s) until nothing was left.",
            DirectlyTaught: false, Confidence: 0.7);
    }

    /// <summary>
    /// Finds the largest taught step of at most <paramref name="max"/> from
    /// <paramref name="cur"/>: a taught "cur + k" or "cur - k" fact, the
    /// inverse of a taught addition ("p + j = cur" gives both cur - j = p
    /// and cur - p = j), or — at k = 1 — one step along the taught number
    /// line (counting on/off by one is reasoning from the taught sequence).
    /// </summary>
    private (double K, double Landed)? FindJump(double cur, char op, double max)
    {
        for (double k = Math.Floor(max); k >= 1; k--)
        {
            if (Lookup(Key(cur, op, k)) is { } m)
                return (k, ParseNum(m));
            if (op == '-' && LookupInverse(cur, k) is { } p)
                return (k, ParseNum(p));
            if (k == 1 && _ladder is { } line && IsCountable(cur)
                && cur >= line.Min - 1e-9 && cur <= line.Max + 1e-9)
            {
                double step = op == '+' ? cur + 1 : cur - 1;
                if (step >= line.Min - 1e-9 && step <= line.Max + 1e-9)
                    return (1, step);
            }
        }
        return null;
    }

    // ====================================================================
    // Knowledge access + normalization helpers
    // ====================================================================

    private string? Lookup(string expressionKey)
    {
        var facts = _kb.Query(expressionKey, "equals", null);
        return facts.Count > 0 ? facts[0].Object : null;
    }

    /// <summary>Finds a taught fact "p + j = cur"; returns p when j == k, or j when p == k.</summary>
    private string? LookupInverse(double cur, double k)
    {
        foreach (var fact in _kb.Query(null, "equals", Num(cur)))
        {
            var m = Regex.Match(fact.Subject, @"^(?<p>-?\d+(?:\.\d+)?)\+(?<j>-?\d+(?:\.\d+)?)$");
            if (!m.Success) continue;
            double p = ParseNum(m.Groups["p"].Value);
            double j = ParseNum(m.Groups["j"].Value);
            if (Math.Abs(j - k) < 1e-9) return Num(p);
            if (Math.Abs(p - k) < 1e-9) return Num(j);
        }
        return null;
    }

    // ---- number words: "two plus twenty-one" -> "2 + 21" ----
    private static readonly Dictionary<string, int> NumberWords = new()
    {
        ["zero"] = 0, ["one"] = 1, ["two"] = 2, ["three"] = 3, ["four"] = 4,
        ["five"] = 5, ["six"] = 6, ["seven"] = 7, ["eight"] = 8, ["nine"] = 9,
        ["ten"] = 10, ["eleven"] = 11, ["twelve"] = 12, ["thirteen"] = 13,
        ["fourteen"] = 14, ["fifteen"] = 15, ["sixteen"] = 16, ["seventeen"] = 17,
        ["eighteen"] = 18, ["nineteen"] = 19, ["twenty"] = 20, ["thirty"] = 30,
        ["forty"] = 40, ["fifty"] = 50, ["sixty"] = 60, ["seventy"] = 70,
        ["eighty"] = 80, ["ninety"] = 90
    };

    private static readonly Regex NumberRun = new(
        @"\b(?<run>" + string.Join("|", NumberWords.Keys.OrderByDescending(k => k.Length)) +
        @")(?:[\s-](?<run>" + string.Join("|", NumberWords.Keys.OrderByDescending(k => k.Length)) + @"))*\b",
        RegexOptions.Compiled);

    private static readonly Regex MultiwordOps = new(
        @"\b(?:multiplied by|divided by)\b", RegexOptions.Compiled);

    private static readonly Regex OpWords = new(
        @"\b(?<op>plus|minus|less|times|over)\b", RegexOptions.Compiled);

    private static readonly Regex AndBetweenNumbers = new(
        @"(?<=\d)\s+and\s+(?=\d)", RegexOptions.Compiled);

    private static readonly Regex XBetweenNumbers = new(
        @"(?<=\d)\s+x\s+(?=\d)", RegexOptions.Compiled);

    /// <summary>
    /// Lowercases, converts number words to digits ("two" -> "2", "twenty
    /// one" -> "21"), operator words to symbols ("plus" -> "+", "and" between
    /// two numbers -> "+") and unifies unicode operators. "and" between words
    /// ("cats and dogs") is never touched.
    /// </summary>
    internal static string Normalize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;

        var lower = text.ToLowerInvariant()
                        .Replace('×', '*').Replace('÷', '/')
                        .Replace('−', '-').Replace('–', '-');

        lower = MultiwordOps.Replace(lower, m =>
            m.Value.StartsWith("multiplied", StringComparison.Ordinal) ? " * " : " / ");

        lower = NumberRun.Replace(lower, m =>
        {
            long total = 0, current = 0;
            foreach (var w in Regex.Split(m.Groups["run"].Value, @"[\s-]+"))
            {
                if (!NumberWords.TryGetValue(w, out int v)) continue;
                if (v < 10 && current >= 20) current += v;          // twenty-one
                else { total += current; current = v; }
            }
            return $" {total + current} ";
        });

        lower = AndBetweenNumbers.Replace(lower, " + ");
        lower = XBetweenNumbers.Replace(lower, " * ");
        lower = OpWords.Replace(lower, m => m.Groups["op"].Value switch
        {
            "plus" => " + ",
            "minus" or "less" => " - ",
            "times" => " * ",
            "over" => " / ",
            _ => " "
        });

        return Regex.Replace(lower, @"\s+", " ").Trim();
    }

    internal static string Display(string expr) =>
        Regex.Replace(expr, @"([+\-*/])", m => m.Value switch
        {
            "+" => " + ",
            "-" => " - ",
            "*" => " * ",
            "/" => " / ",
            _ => " "
        }).Replace("  ", " ").Trim();

    internal static string Num(double value) =>
        Math.Abs(value - Math.Round(value)) < 1e-9
            ? Math.Round(value).ToString(CultureInfo.InvariantCulture)
            : value.ToString("0.######", CultureInfo.InvariantCulture);

    private static double ParseNum(string s) =>
        double.Parse(s, CultureInfo.InvariantCulture);

    private static bool IsCountable(double v) =>
        Math.Abs(v - Math.Round(v)) < 1e-9 && Math.Abs(v) <= 1000;

    private static string Key(double a, char op, double b) => $"{Num(a)}{op}{Num(b)}";

    private static string Plural(string unit, string amount) =>
        amount == "1" && unit.EndsWith('s') && !unit.EndsWith("ss") ? unit[..^1]
        : amount != "1" && !unit.EndsWith('s') ? unit + "s"
        : unit;
}
