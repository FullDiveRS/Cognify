using System.Text.Json;
using Cognify.Emotions;
using Cognify.Goals;
using Cognify.Learning;
using Cognify.Memory;
using Cognify.Reasoning;

namespace Cognify.Persistence;

/// <summary>
/// A serializable snapshot of a mind's entire state: config, emotions, mood,
/// memories, knowledge (facts + rules), goals, drives, personality-affecting
/// config and the learned neural network. Captured by
/// <see cref="SqliteMindStore"/> and re-applied to a mind on restore.
/// </summary>
public sealed class MindSnapshot
{
    // Profile ---------------------------------------------------------
    /// <summary>Mind name.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Mind description.</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Executed cycles at capture time.</summary>
    public int Cycles { get; set; }

    /// <summary>When the snapshot was taken.</summary>
    public DateTimeOffset SavedAt { get; set; }

    /// <summary>Serialized <see cref="Core.MindConfig"/>.</summary>
    public string ConfigJson { get; set; } = "{}";

    // Affect ------------------------------------------------------------
    /// <summary>Emotion intensities by name.</summary>
    public Dictionary<string, double> Emotions { get; set; } = new();

    /// <summary>Mood valence (-1..1).</summary>
    public double MoodValence { get; set; }

    /// <summary>Mood arousal (0..1).</summary>
    public double MoodArousal { get; set; }

    // Memory --------------------------------------------------------------
    /// <summary>Stored memories.</summary>
    public List<MemoryDto> Memories { get; set; } = new();

    /// <summary>Associative edges (a, b, weight).</summary>
    public List<string[]> Associations { get; set; } = new();

    // Knowledge -------------------------------------------------------------
    /// <summary>Facts (subject, predicate, object, confidence, source).</summary>
    public List<string[]> Facts { get; set; } = new();

    /// <summary>Serialized rules.</summary>
    public List<RuleDto> Rules { get; set; } = new();

    // Goals ------------------------------------------------------------------
    /// <summary>Goals.</summary>
    public List<GoalDto> Goals { get; set; } = new();

    /// <summary>Drive levels (name, level, growth).</summary>
    public List<string[]> Drives { get; set; } = new();

    // Learning -------------------------------------------------------------------
    /// <summary>Serialized neural network.</summary>
    public string? NetworkJson { get; set; }

    /// <summary>Serialized style-conditioned network (the neural half of the hybrid engine).</summary>
    public string? StyleNetworkJson { get; set; }

    /// <summary>Q-table entries (styleId, q, count).</summary>
    public List<string[]> QTable { get; set; } = new();

    /// <summary>Learning statistics.</summary>
    public LearningStatsDto Stats { get; set; } = new();

    /// <summary>Words the mind picked up from conversation (word, count, firstCycle, lastCycle).</summary>
    public List<string[]> LexiconEntries { get; set; } = new();

    /// <summary>Words already woven into a spoken reply (never asked about again).</summary>
    public List<string> LexiconSpoken { get; set; } = new();

    /// <summary>Serialized n-gram language model (next-token prediction).</summary>
    public string? LanguageModelJson { get; set; }

    /// <summary>Serialized neural language model (transformer weights, vocabulary, buffer).</summary>
    public string? NeuralLanguageModelJson { get; set; }

    /// <summary>Serialized self-improvement ledger (template scores, feedback counts).</summary>
    public string? SelfImprovementJson { get; set; }

    // Semantic memory ---------------------------------------------------
    /// <summary>Semantic (embedding) memory entries captured from the store.</summary>
    public List<SemanticDto> SemanticEntries { get; set; } = new();

    // Training -------------------------------------------------------------
    /// <summary>Trained sayings (the mind's spoken vocabulary).</summary>
    public List<string> Sayings { get; set; } = new();

    // DTOs ---------------------------------------------------------------
    /// <summary>Serializable memory record.</summary>
    public sealed class MemoryDto
    {
        public string Kind { get; set; } = "ShortTerm";
        public string Content { get; set; } = string.Empty;
        public double Importance { get; set; }
        public double Strength { get; set; }
        public DateTimeOffset CreatedAt { get; set; }
        public DateTimeOffset? LastRecalledAt { get; set; }
        public int RecallCount { get; set; }
        public string? EmotionType { get; set; }
        public double EmotionIntensity { get; set; }
        public string? Source { get; set; }
    }

    /// <summary>Serializable logic rule.</summary>
    public sealed class RuleDto
    {
        public List<string[]> When { get; set; } = new();
        public string[] Then { get; set; } = Array.Empty<string>();
        public double Confidence { get; set; }
        public string? Source { get; set; }
    }

    /// <summary>Serializable goal.</summary>
    public sealed class GoalDto
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public double Priority { get; set; }
        public double Progress { get; set; }
        public double Target { get; set; } = 1.0;
        public int Status { get; set; }
        public List<string> Tags { get; set; } = new();
        public DateTimeOffset? Deadline { get; set; }
    }

    /// <summary>Serializable learning statistics.</summary>
    public sealed class LearningStatsDto
    {
        public int TotalExperiences { get; set; }
        public int TrainingSteps { get; set; }
        public double AverageReward { get; set; }
        public double LastLoss { get; set; }
        public double ExplorationRate { get; set; }

        /// <summary>Symbolic/neural blend weight (0 = all symbolic, 1 = all neural).</summary>
        public double NeuralWeight { get; set; }
    }

    /// <summary>Serializable semantic (embedding) memory entry.</summary>
    public sealed class SemanticDto
    {
        public string Text { get; set; } = string.Empty;
        public List<double> Vector { get; set; } = new();
        public string Tag { get; set; } = string.Empty;
        public DateTimeOffset CreatedAt { get; set; }
    }

    // Capture / apply -----------------------------------------------------

    /// <summary>Captures the full state of a mind.</summary>
    public static MindSnapshot Capture(Core.Mind mind)
    {
        var snap = new MindSnapshot
        {
            Name = mind.Name,
            Description = mind.Config.Description,
            Cycles = mind.CycleCount,
            SavedAt = DateTimeOffset.UtcNow,
            ConfigJson = JsonSerializer.Serialize(mind.Config)
        };

        if (mind.EmotionalState is { } emo)
        {
            foreach (var (emotion, intensity) in emo.Snapshot())
                snap.Emotions[emotion.ToString()] = intensity;
            snap.MoodValence = emo.MoodValence;
            snap.MoodArousal = emo.MoodArousal;
        }

        if (mind.Memory is { } mem)
        {
            var seenIds = new HashSet<Guid>();
            foreach (var m in mem.RecentShortTerm().Concat(mem.LongTermMemories()).Concat(mem.Episodes()))
            {
                if (!seenIds.Add(m.Id)) continue; // a record may live in short-term + long-term
                snap.Memories.Add(new MemoryDto
                {
                    Kind = m.Kind.ToString(),
                    Content = m.Content,
                    Importance = m.Importance,
                    Strength = m.Strength,
                    CreatedAt = m.CreatedAt,
                    LastRecalledAt = m.LastRecalledAt,
                    RecallCount = m.RecallCount,
                    EmotionType = m.EmotionType?.ToString(),
                    EmotionIntensity = m.EmotionIntensity,
                    Source = m.Source
                });
            }
            snap.Associations = mem.Associations.AllEdges()
                .Select(e => new[] { e.A, e.B, e.Weight.ToString(System.Globalization.CultureInfo.InvariantCulture) })
                .ToList();
        }

        if (mind.Knowledge is { } kb)
        {
            snap.Facts = kb.AllFacts()
                .Select(f => new[] { f.Subject, f.Predicate, f.Object,
                    f.Confidence.ToString(System.Globalization.CultureInfo.InvariantCulture), f.Source ?? "" })
                .ToList();
            snap.Rules = kb.AllRules().Select(r => new RuleDto
            {
                When = r.When.Select(p => new[] { p.Subject, p.Predicate, p.Object }).ToList(),
                Then = new[] { r.Then.Subject, r.Then.Predicate, r.Then.Object },
                Confidence = r.Confidence,
                Source = r.Source
            }).ToList();
        }

        if (mind.GoalSystem is { } goals)
        {
            snap.Goals = goals.AllGoals().Select(g => new GoalDto
            {
                Name = g.Name,
                Description = g.Description,
                Priority = g.Priority,
                Progress = g.Progress,
                Target = g.Target,
                Status = (int)g.Status,
                Tags = g.Tags.ToList(),
                Deadline = g.Deadline
            }).ToList();
            snap.Drives = goals.Drives.Values
                .Select(d => new[] { d.Name, d.Level.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    d.Growth.ToString(System.Globalization.CultureInfo.InvariantCulture) })
                .ToList();
        }

        if (mind.Learning is { } learning)
        {
            snap.NetworkJson = learning.Net.Serialize();
            snap.StyleNetworkJson = learning.StyleNet.Serialize();
            snap.QTable = learning.SnapshotQTable()
                .Select(kv => new[] { kv.Key.ToString(),
                    kv.Value.Q.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    kv.Value.Count.ToString() })
                .ToList();
            snap.Stats = new LearningStatsDto
            {
                TotalExperiences = learning.Stats.TotalExperiences,
                TrainingSteps = learning.Stats.TrainingSteps,
                AverageReward = learning.Stats.AverageReward,
                LastLoss = learning.Stats.LastLoss,
                ExplorationRate = learning.Stats.ExplorationRate,
                NeuralWeight = learning.NeuralWeight
            };
            snap.LexiconEntries = learning.Lexicon.Snapshot();
            snap.LexiconSpoken = learning.Lexicon.SnapshotSpoken();
        }

        if (mind.LanguageModel is { } lm && lm.TotalTokens > 0)
            snap.LanguageModelJson = lm.Serialize();
        if (mind.NeuralLanguageModel is { } nlm && nlm.IsTrained)
            snap.NeuralLanguageModelJson = nlm.Serialize();
        if (mind.SelfImprovement is { } si)
            snap.SelfImprovementJson = si.Serialize();

        if (mind.SemanticMemory is { } semantic)
        {
            foreach (var entry in semantic.Store.All())
            {
                snap.SemanticEntries.Add(new SemanticDto
                {
                    Text = entry.Text,
                    Vector = entry.Vector.ToList(),
                    Tag = entry.Tag,
                    CreatedAt = entry.CreatedAt
                });
            }
        }

        snap.Sayings = mind.Phrases.All().ToList();

        return snap;
    }

    /// <summary>Applies this snapshot to a (newly built) mind.</summary>
    public void ApplyTo(Core.Mind mind)
    {
        // The saved config is not applied wholesale (the mind may be rebuilt
        // with different module wiring), but personality traits and gender
        // identity are part of "who the mind is" — restore them from the
        // saved config.
        TryRestorePersonality(mind);
        TryRestoreGender(mind);

        if (mind.EmotionalState is { } emo)
        {
            foreach (var (name, intensity) in Emotions)
                if (Enum.TryParse<EmotionType>(name, out var emotion))
                    emo.Set(emotion, intensity);
            emo.MoodValence = MoodValence;
            emo.MoodArousal = MoodArousal;
        }

        if (mind.Memory is { } mem)
        {
            foreach (var m in Memories)
            {
                EmotionType? emotion = Enum.TryParse<EmotionType>(m.EmotionType, out var parsed) ? parsed : null;
                var kind = Enum.TryParse<MemoryKind>(m.Kind, out var kindParsed) ? kindParsed : MemoryKind.LongTerm;
                var record = mem.Remember(m.Content, kind, m.Importance, emotion, m.EmotionIntensity, m.Source);
                record.Restore(m.Strength, m.RecallCount, m.LastRecalledAt, m.Source);
            }

            foreach (var edge in Associations)
                if (edge.Length == 3 &&
                    double.TryParse(edge[2], System.Globalization.CultureInfo.InvariantCulture, out var w))
                    mem.Associations.RestoreEdge(edge[0], edge[1], w);
        }

        if (mind.Knowledge is { } kb)
        {
            foreach (var f in Facts)
                if (f.Length >= 4 &&
                    double.TryParse(f[3], System.Globalization.CultureInfo.InvariantCulture, out var conf))
                    kb.Add(new Fact(f[0], f[1], f[2], conf, f.Length > 4 && f[4].Length > 0 ? f[4] : null));
            foreach (var r in Rules)
            {
                if (r.Then.Length != 3) continue;
                kb.AddRule(new LogicRule(
                    r.When.Where(p => p.Length == 3)
                        .Select(p => new FactPattern(p[0], p[1], p[2])).ToList(),
                    new FactPattern(r.Then[0], r.Then[1], r.Then[2]),
                    r.Confidence, r.Source));
            }
        }

        if (mind.GoalSystem is { } goals)
        {
            foreach (var g in Goals)
            {
                var goal = goals.AddGoal(g.Name, g.Description, g.Priority, g.Tags, g.Target, g.Deadline);
                goal.Restore(g.Progress, (GoalStatus)Math.Clamp(g.Status, 0, 2));
            }
            foreach (var d in Drives)
                if (d.Length == 3 && goals.Drives.TryGetValue(d[0], out var drive) &&
                    double.TryParse(d[1], System.Globalization.CultureInfo.InvariantCulture, out var level))
                    drive.SetLevel(level);
        }

        if (mind.Learning is { } learning && NetworkJson is not null)
        {
            var qTable = new Dictionary<int, (double, int)>();
            foreach (var row in QTable)
                if (row.Length == 3 && int.TryParse(row[0], out var id) &&
                    double.TryParse(row[1], System.Globalization.CultureInfo.InvariantCulture, out var q) &&
                    int.TryParse(row[2], out var count))
                    qTable[id] = (q, count);

            // Snapshots from before the hybrid engine have no neural weight
            // (stored as 0) — keep the module's default blend in that case.
            double? weight = Stats.NeuralWeight > 0 ? Stats.NeuralWeight : null;

            learning.RestoreState(NetworkJson, qTable, new LearningStats
            {
                TotalExperiences = Stats.TotalExperiences,
                TrainingSteps = Stats.TrainingSteps,
                AverageReward = Stats.AverageReward,
                LastLoss = Stats.LastLoss,
                ExplorationRate = Stats.ExplorationRate
            }, StyleNetworkJson, weight, LexiconEntries, LexiconSpoken);
        }

        if (mind.SemanticMemory is { } semantic && SemanticEntries.Count > 0)
        {
            foreach (var entry in SemanticEntries)
            {
                if (string.IsNullOrWhiteSpace(entry.Text) || entry.Vector.Count == 0) continue;
                semantic.Store.Add(entry.Text, entry.Vector.ToArray(), entry.Tag);
            }
        }

        if (mind.LanguageModel is { } restoredLm && !string.IsNullOrWhiteSpace(LanguageModelJson))
        {
            var lm = Cognify.Learning.NextTokenModel.Deserialize(LanguageModelJson);
            if (lm.TotalTokens > 0)
                restoredLm.Merge(lm); // keep anything observed before the load too
        }

        // Neural (transformer) language model — absent in older snapshots,
        // in which case the mind simply starts with a fresh untrained one
        // and re-learns from everything she observes from now on. The
        // snapshot carries the model's own architecture: rebuild the model
        // with THAT config so a mind trained on a bigger transformer stays
        // on it instead of silently shrinking to current defaults.
        if (!string.IsNullOrWhiteSpace(NeuralLanguageModelJson))
        {
            var neuralCfg = Cognify.Learning.TransformerLM.ConfigFromJson(NeuralLanguageModelJson);
            if (neuralCfg is not null &&
                (neuralCfg.Dim != mind.NeuralLanguageModel.Config.Dim ||
                 neuralCfg.Heads != mind.NeuralLanguageModel.Config.Heads ||
                 neuralCfg.Layers != mind.NeuralLanguageModel.Config.Layers ||
                 neuralCfg.ContextLen != mind.NeuralLanguageModel.Config.ContextLen))
            {
                mind.NeuralLanguageModel = new Cognify.Learning.TransformerLM(neuralCfg);
            }
            mind.NeuralLanguageModel.TryRestore(NeuralLanguageModelJson);
        }

        if (mind.SelfImprovement is { } restoredSi && !string.IsNullOrWhiteSpace(SelfImprovementJson))
            restoredSi.Restore(SelfImprovementJson);

        if (Sayings.Count > 0)
            mind.Phrases.Restore(Sayings);
    }

    private void TryRestorePersonality(Core.Mind mind)
    {
        if (mind.Personality is null) return;
        try
        {
            var saved = System.Text.Json.JsonSerializer.Deserialize<Core.MindConfig>(ConfigJson);
            if (saved?.Personality is { } p)
            {
                mind.Personality.Openness = Math.Clamp(p.Openness, 0, 1);
                mind.Personality.Conscientiousness = Math.Clamp(p.Conscientiousness, 0, 1);
                mind.Personality.Extraversion = Math.Clamp(p.Extraversion, 0, 1);
                mind.Personality.Agreeableness = Math.Clamp(p.Agreeableness, 0, 1);
                mind.Personality.Neuroticism = Math.Clamp(p.Neuroticism, 0, 1);
            }
        }
        catch (System.Text.Json.JsonException)
        {
            // Corrupt config payload: keep the freshly built traits.
        }
    }

    private void TryRestoreGender(Core.Mind mind)
    {
        try
        {
            var saved = System.Text.Json.JsonSerializer.Deserialize<Core.MindConfig>(ConfigJson);
            if (saved?.Gender is { } g && !g.IsUnset)
            {
                SanitizeGender(g);
                if (!g.IsUnset) mind.Config.Gender = g;
            }
        }
        catch (System.Text.Json.JsonException)
        {
            // Corrupt config payload: keep the freshly built gender.
        }
    }

    /// <summary>
    /// Older snapshots may carry pronoun sets the gender system no longer
    /// offers (they/them as the mind's own pronouns, custom sets). The mind
    /// is restored with what it can keep: she/her, he/him, or none at all.
    /// </summary>
    private static void SanitizeGender(Core.GenderOptions g)
    {
        var p = g.Pronouns;
        // JSON restore yields value-equal copies (not the preset singletons),
        // so compare by label, not by reference.
        bool known = p.IsUnspecified || p.IsUnknownPerson
            || p.Label == Core.PronounSet.Female.Label
            || p.Label == Core.PronounSet.Male.Label;
        if (!known || p.IsUnknownPerson)
            g.Pronouns = Core.PronounSet.Unspecified; // custom sets are no longer offered
        var identity = g.Identity.Trim().ToLowerInvariant();
        if (Core.PronounSet.IsReservedLabel(identity))
            g.Identity = string.Empty;
    }
}
