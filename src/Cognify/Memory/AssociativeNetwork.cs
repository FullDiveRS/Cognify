namespace Cognify.Memory;

/// <summary>
/// A weighted concept graph. Concepts that occur together get linked; recall
/// activates seed concepts and spreads activation through the network, so
/// related-but-unmentioned memories can surface too.
/// </summary>
public sealed class AssociativeNetwork
{
    private readonly object _gate = new();
    private readonly Dictionary<string, Dictionary<string, double>> _edges = new();

    /// <summary>Number of concepts in the network.</summary>
    public int NodeCount { get { lock (_gate) return _edges.Count; } }

    /// <summary>Strengthens the bidirectional link between two concepts.</summary>
    public void Link(string a, string b, double weight = 0.2)
    {
        if (a == b || string.IsNullOrWhiteSpace(a) || string.IsNullOrWhiteSpace(b)) return;
        lock (_gate)
        {
            AddEdge(a, b, weight);
            AddEdge(b, a, weight);
        }
    }

    private void AddEdge(string from, string to, double weight)
    {
        if (!_edges.TryGetValue(from, out var inner))
        {
            inner = new Dictionary<string, double>();
            _edges[from] = inner;
        }
        inner[to] = Math.Min(1.0, inner.GetValueOrDefault(to) + weight);
    }

    /// <summary>
    /// Spreads activation from seed concepts and returns the most activated
    /// related concepts (excluding the seeds).
    /// </summary>
    public IReadOnlyList<(string Concept, double Activation)> Activate(
        IEnumerable<string> seeds, int depth = 2, int max = 8, double decay = 0.5)
    {
        lock (_gate)
        {
            var activation = new Dictionary<string, double>();
            var seedSet = new HashSet<string>(seeds);

            foreach (var seed in seedSet)
                activation[seed] = 1.0;

            var frontier = seedSet.ToList();
            for (int level = 0; level < depth && frontier.Count > 0; level++)
            {
                var next = new List<string>();
                foreach (var node in frontier)
                {
                    if (!_edges.TryGetValue(node, out var inner)) continue;
                    double baseLevel = activation.GetValueOrDefault(node);
                    foreach (var (neighbor, weight) in inner)
                    {
                        double gain = baseLevel * weight * decay;
                        if (gain < 0.02) continue;
                        if (!seedSet.Contains(neighbor) &&
                            (!activation.TryGetValue(neighbor, out var existing) || gain > existing))
                        {
                            activation[neighbor] = gain;
                            next.Add(neighbor);
                        }
                    }
                }
                frontier = next;
            }

            return activation
                .Where(kv => !seedSet.Contains(kv.Key))
                .OrderByDescending(kv => kv.Value)
                .Take(max)
                .Select(kv => (kv.Key, kv.Value))
                .ToList();
        }
    }

    /// <summary>Returns all edges as (a, b, weight) triples for persistence.</summary>
    public IReadOnlyList<(string A, string B, double Weight)> AllEdges()
    {
        lock (_gate)
        {
            var list = new List<(string, string, double)>();
            foreach (var (a, inner) in _edges)
                foreach (var (b, w) in inner)
                    if (string.CompareOrdinal(a, b) < 0)
                        list.Add((a, b, w));
            return list;
        }
    }

    /// <summary>Restores an edge from persistence.</summary>
    public void RestoreEdge(string a, string b, double weight)
    {
        lock (_gate)
        {
            AddEdge(a, b, 0); // ensure nodes exist
            AddEdge(b, a, 0);
            _edges[a][b] = Math.Clamp(weight, 0, 1);
            _edges[b][a] = Math.Clamp(weight, 0, 1);
        }
    }
}
