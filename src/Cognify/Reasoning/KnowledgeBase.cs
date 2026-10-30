using System.Text.Json.Serialization;

namespace Cognify.Reasoning;

/// <summary>
/// A pattern inside a rule antecedent or consequent. Variables are written
/// "?x" and bind across the rule, enabling real forward chaining.
/// </summary>
public sealed record FactPattern(string Subject, string Predicate, string Object)
{
    /// <summary>True if the subject is a variable.</summary>
    [JsonIgnore]
    public bool SubjectIsVar => Subject.StartsWith('?');

    /// <summary>True if the object is a variable.</summary>
    [JsonIgnore]
    public bool ObjectIsVar => Object.StartsWith('?');

    /// <summary>Renders the pattern.</summary>
    public override string ToString() => $"{Subject} {Predicate} {Object}";
}

/// <summary>
/// A logic rule: IF all antecedent patterns match THEN derive the consequent.
/// Example: IF ?s is-a bird AND ?s has ?w THEN bird-of ?s ?w ... simpler form:
/// IF ?s is-a bird THEN ?s can fly.
/// </summary>
public sealed record LogicRule(
    IReadOnlyList<FactPattern> When,
    FactPattern Then,
    double Confidence = 0.8,
    string? Source = null)
{
    /// <summary>Renders the rule in readable form.</summary>
    public override string ToString() =>
        $"IF {string.Join(" AND ", When)} THEN {Then}";
}

/// <summary>
/// An indexed store of facts and rules. Fast wildcard queries via subject/predicate
/// indexes; thread-safe.
/// </summary>
public sealed class KnowledgeBase
{
    private readonly object _gate = new();
    private readonly HashSet<Fact> _facts = new();
    private readonly Dictionary<string, List<Fact>> _bySubject = new();
    private readonly Dictionary<string, List<Fact>> _byObject = new();
    private readonly List<LogicRule> _rules = new();

    /// <summary>Raised whenever a new fact is added.</summary>
    public event EventHandler<Fact>? FactAdded;

    /// <summary>Number of stored facts.</summary>
    public int FactCount { get { lock (_gate) return _facts.Count; } }

    /// <summary>Number of stored rules.</summary>
    public int RuleCount { get { lock (_gate) return _rules.Count; } }

    /// <summary>Adds a fact. Returns false if it already existed.</summary>
    public bool Add(Fact fact)
    {
        lock (_gate)
        {
            if (!_facts.Add(fact)) return false;

            if (!_bySubject.TryGetValue(fact.Subject, out var subList))
            {
                subList = new List<Fact>();
                _bySubject[fact.Subject] = subList;
            }
            subList.Add(fact);

            if (!_byObject.TryGetValue(fact.Object, out var objList))
            {
                objList = new List<Fact>();
                _byObject[fact.Object] = objList;
            }
            objList.Add(fact);
        }
        FactAdded?.Invoke(this, fact);
        return true;
    }

    /// <summary>Adds a rule to the knowledge base.</summary>
    public void AddRule(LogicRule rule)
    {
        lock (_gate) _rules.Add(rule);
    }

    /// <summary>Removes all facts matching subject/predicate/object (null = wildcard).</summary>
    public int Remove(string? subject, string? predicate, string? obj)
    {
        lock (_gate)
        {
            var doomed = _facts
                .Where(f => (subject is null || f.Subject == subject) &&
                            (predicate is null || f.Predicate == predicate) &&
                            (obj is null || f.Object == obj))
                .ToList();
            foreach (var f in doomed)
            {
                _facts.Remove(f);
                _bySubject[f.Subject].Remove(f);
                _byObject[f.Object].Remove(f);
            }
            return doomed.Count;
        }
    }

    /// <summary>Queries facts; pass null for any field to use a wildcard.</summary>
    public IReadOnlyList<Fact> Query(string? subject, string? predicate, string? obj)
    {
        lock (_gate)
        {
            IEnumerable<Fact> source = _facts;
            if (subject is not null && _bySubject.TryGetValue(subject, out var subList))
                source = subList;
            else if (obj is not null && _byObject.TryGetValue(obj, out var objList))
                source = objList;

            return source
                .Where(f => (subject is null || f.Subject == subject) &&
                            (predicate is null || f.Predicate == predicate) &&
                            (obj is null || f.Object == obj))
                .OrderByDescending(f => f.Confidence)
                .ToList();
        }
    }

    /// <summary>Returns all facts (snapshot).</summary>
    public IReadOnlyList<Fact> AllFacts()
    {
        lock (_gate) return _facts.OrderBy(f => f.Subject).ThenBy(f => f.Predicate).ToList();
    }

    /// <summary>Returns all rules (snapshot).</summary>
    public IReadOnlyList<LogicRule> AllRules()
    {
        lock (_gate) return _rules.ToList();
    }

    /// <summary>True if the exact fact (ignoring confidence/source) exists.</summary>
    public bool Contains(Fact fact)
    {
        lock (_gate) return _facts.Contains(fact);
    }
}
