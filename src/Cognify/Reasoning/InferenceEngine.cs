using Cognify.Core;

namespace Cognify.Reasoning;

/// <summary>
/// Result of a reasoning pass: which facts were inferred, explanations of how,
/// and (for direct questions) a composed answer.
/// </summary>
public sealed class ReasoningResult
{
    /// <summary>Facts newly derived during this pass.</summary>
    public IReadOnlyList<Fact> Inferred => _inferred;

    /// <summary>Human-readable explanations of the inference steps.</summary>
    public IReadOnlyList<string> Explanations => _explanations;

    /// <summary>True when this pass produced anything useful.</summary>
    public bool Success => _inferred.Count > 0 || _explanations.Count > 0 || Answer is not null;

    /// <summary>For direct questions ("what is a bird?"), the composed answer.</summary>
    public string? Answer { get; internal set; }

    /// <summary>
    /// Number of facts parsed out of the current input and added to the
    /// knowledge base this pass (i.e. the user just taught the mind something).
    /// </summary>
    public int FactsLearned { get; internal set; }

    private readonly List<Fact> _inferred = new();
    private readonly List<string> _explanations = new();

    internal void AddInferred(Fact fact) => _inferred.Add(fact);
    internal void AddExplanation(string text) => _explanations.Add(text);
}

/// <summary>
/// A compact symbolic inference engine over the <see cref="KnowledgeBase"/>:
/// <list type="bullet">
/// <item><b>Deduction</b> — forward chaining of variable-carrying rules (modus ponens).</item>
/// <item><b>Induction</b> — generalizes repeated co-occurring pairs into new related-to facts.</item>
/// <item><b>Abduction</b> — finds the most plausible rule + antecedents that explain an observation.</item>
/// </list>
/// </summary>
public sealed class InferenceEngine
{
    private readonly KnowledgeBase _kb;
    private readonly ReasoningOptions _options;

    /// <summary>Creates an inference engine over the given knowledge base.</summary>
    public InferenceEngine(KnowledgeBase kb, ReasoningOptions options)
    {
        _kb = kb;
        _options = options;
    }

    // ------------------------------------------------------------------
    // Deduction — forward chaining
    // ------------------------------------------------------------------

    /// <summary>
    /// Runs forward chaining: repeatedly applies every rule until no new facts
    /// emerge (bounded by MaxInferenceDepth). Returns derived facts.
    /// </summary>
    public IReadOnlyList<Fact> ForwardChain(int? maxDepth = null)
    {
        int depth = Math.Clamp(maxDepth ?? _options.MaxInferenceDepth, 1, 32);
        var derived = new List<Fact>();
        var queue = new Queue<Fact>(_kb.AllFacts());

        for (int pass = 0; pass < depth && queue.Count > 0; pass++)
        {
            var frontier = queue.ToList();
            queue.Clear();

            foreach (var rule in _kb.AllRules())
            {
                foreach (var (bindings, confidence) in MatchAntecedents(rule, frontier))
                {
                    var conclusion = Instantiate(rule.Then, bindings);
                    var fact = new Fact(conclusion.Subject, conclusion.Predicate, conclusion.Object,
                        Math.Min(rule.Confidence, confidence),
                        $"rule:{rule.GetHashCode()}");
                    if (_kb.Add(fact))
                    {
                        derived.Add(fact);
                        queue.Enqueue(fact);
                    }
                }
            }
        }
        return derived;
    }

    private IEnumerable<(Dictionary<string, string> Bindings, double Confidence)> MatchAntecedents(
        LogicRule rule, IReadOnlyList<Fact> candidates)
    {
        // Recursive binding across antecedent patterns.
        var results = new List<(Dictionary<string, string>, double)>();
        MatchFrom(rule.When, 0, new Dictionary<string, string>(), 1.0, candidates, results);
        return results;
    }

    private void MatchFrom(
        IReadOnlyList<FactPattern> patterns, int index,
        Dictionary<string, string> bindings, double confidence,
        IReadOnlyList<Fact> candidates,
        List<(Dictionary<string, string>, double)> results)
    {
        if (index == patterns.Count)
        {
            results.Add((new Dictionary<string, string>(bindings), confidence));
            return;
        }

        var pattern = patterns[index];
        var pool = _kb.Query(
            pattern.SubjectIsVar ? null : pattern.Subject,
            pattern.Predicate,
            pattern.ObjectIsVar ? null : pattern.Object);

        foreach (var fact in pool.Count > 0 ? pool : candidates.Where(f =>
                     (pattern.SubjectIsVar || f.Subject == pattern.Subject) &&
                     (pattern.ObjectIsVar || f.Object == pattern.Object)))
        {
            if (fact.Predicate != pattern.Predicate) continue;
            if (!pattern.SubjectIsVar && fact.Subject != pattern.Subject) continue;
            if (!pattern.ObjectIsVar && fact.Object != pattern.Object) continue;

            var next = new Dictionary<string, string>(bindings);
            bool consistent = true;

            if (pattern.SubjectIsVar)
            {
                if (next.TryGetValue(pattern.Subject, out var boundS) && boundS != fact.Subject)
                    consistent = false;
                else next[pattern.Subject] = fact.Subject;
            }
            if (consistent && pattern.ObjectIsVar)
            {
                if (next.TryGetValue(pattern.Object, out var boundO) && boundO != fact.Object)
                    consistent = false;
                else next[pattern.Object] = fact.Object;
            }
            if (!consistent) continue;

            MatchFrom(patterns, index + 1, next, confidence * fact.Confidence, candidates, results);
        }
    }

    private static FactPattern Instantiate(FactPattern pattern, Dictionary<string, string> bindings) =>
        new(
            pattern.SubjectIsVar && bindings.TryGetValue(pattern.Subject, out var s) ? s : pattern.Subject,
            pattern.Predicate,
            pattern.ObjectIsVar && bindings.TryGetValue(pattern.Object, out var o) ? o : pattern.Object);

    // ------------------------------------------------------------------
    // Deduction — syllogistic expansion of is-a hierarchies
    // ------------------------------------------------------------------

    /// <summary>
    /// Expands the is-a hierarchy upwards from an entity:
    /// bird is-a animal, animal is-a living-thing => returns the full ancestor chain.
    /// </summary>
    public IReadOnlyList<Fact> AncestorsOf(string entity)
    {
        var chain = new List<Fact>();
        var seen = new HashSet<string> { entity };
        var frontier = new List<string> { entity };

        while (frontier.Count > 0)
        {
            var next = new List<string>();
            foreach (var node in frontier)
            {
                foreach (var fact in _kb.Query(node, "is-a", null))
                {
                    if (seen.Add(fact.Object))
                    {
                        chain.Add(fact);
                        next.Add(fact.Object);
                    }
                }
            }
            frontier = next;
        }
        return chain;
    }

    // ------------------------------------------------------------------
    // Induction — generalize repeated patterns
    // ------------------------------------------------------------------

    /// <summary>
    /// Induces new facts from frequently co-occurring concept pairs.
    /// Concepts observed together >= threshold times get a "related-to" fact
    /// with confidence proportional to frequency.
    /// </summary>
    public IReadOnlyList<Fact> Induce(IReadOnlyDictionary<(string A, string B), int> coOccurrence, double threshold)
    {
        var induced = new List<Fact>();
        foreach (var ((a, b), count) in coOccurrence.OrderByDescending(kv => kv.Value))
        {
            if (count < threshold) break;
            if (a == b) continue;
            var fact = new Fact(a, "related-to", b,
                Confidence: Math.Min(0.75, 0.4 + count * 0.05),
                Source: "induced");
            if (_kb.Add(fact)) induced.Add(fact);
        }
        return induced;
    }

    // ------------------------------------------------------------------
    // Abduction — best explanation for an observation
    // ------------------------------------------------------------------

    /// <summary>
    /// Tries to explain an observed fact by finding a rule whose consequent
    /// matches and whose antecedents are (or can be) satisfied.
    /// Returns explanations in human-readable form.
    /// </summary>
    public IReadOnlyList<string> Abduce(Fact observation, int maxExplanations = 3)
    {
        var explanations = new List<string>();
        foreach (var rule in _kb.AllRules())
        {
            var then = rule.Then;
            if (then.Predicate != observation.Predicate) continue;
            if (!then.SubjectIsVar && then.Subject != observation.Subject) continue;
            if (!then.ObjectIsVar && then.Object != observation.Object) continue;

            // Try to satisfy the antecedents against known facts.
            var bindings = new Dictionary<string, string>();
            if (then.SubjectIsVar) bindings[then.Subject] = observation.Subject;
            if (then.ObjectIsVar) bindings[then.Object] = observation.Object;

            if (rule.When.All(pattern => AntecedentSatisfied(pattern, bindings)))
            {
                explanations.Add(
                    $"{observation} is plausibly explained by rule [{rule}] " +
                    $"because {string.Join("; ", rule.When.Select(p => Instantiate(p, bindings).ToString()))}.");
                if (explanations.Count >= maxExplanations) break;
            }
        }
        return explanations;
    }

    private bool AntecedentSatisfied(FactPattern pattern, Dictionary<string, string> bindings)
    {
        var subject = pattern.SubjectIsVar && bindings.TryGetValue(pattern.Subject, out var s) ? s : pattern.Subject;
        var obj = pattern.ObjectIsVar && bindings.TryGetValue(pattern.Object, out var o) ? o : pattern.Object;
        return _kb.Query(subject, pattern.Predicate, obj).Count > 0;
    }
}
