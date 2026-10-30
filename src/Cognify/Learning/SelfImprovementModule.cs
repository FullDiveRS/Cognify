using System.Text.Json;
using Cognify.Core;
using Cognify.Thinking;

namespace Cognify.Learning;

/// <summary>What one self-improvement review pass changed.</summary>
public sealed record SelfImprovementReport
{
    /// <summary>Conversation turns the tracker has watched.</summary>
    public int CyclesObserved { get; init; }

    /// <summary>Turns that read as approval of the previous reply.</summary>
    public int PositiveSignals { get; init; }

    /// <summary>Turns that read as disapproval of the previous reply.</summary>
    public int NegativeSignals { get; init; }

    /// <summary>How many reply wordings are currently tracked.</summary>
    public int TemplatesTracked { get; init; }

    /// <summary>Templates demoted (spoken replies that drew disapproval).</summary>
    public int TemplatesDemoted { get; init; }

    /// <summary>Templates promoted (spoken replies that drew approval).</summary>
    public int TemplatesPromoted { get; init; }

    /// <summary>Trained sayings pruned during reviews (repeatedly disliked).</summary>
    public int SayingsPruned { get; init; }

    /// <summary>Human-readable one-line summary.</summary>
    public string Summary { get; init; } = "";

    /// <summary><see cref="Object.ToString"/> that shows the summary line.</summary>
    public override string ToString() => Summary;
}

/// <summary>
/// The self-improvement module — the mind studying its own conversations and
/// getting better from them, with no teacher required.
///
/// Three concrete mechanisms (all deterministic and testable):
///
/// 1. IMPLIED FEEDBACK: the user's next message often grades the previous
///    reply ("good job!", "no, that's wrong", "stop saying that"). The
///    module classifies those cues and turns them into real reward for the
///    hybrid learning engine — reinforcing exactly the style and wording
///    that was spoken, one turn late (feedback always arrives late).
/// 2. WORDING SCORES: every reply template keeps an approval score (EMA).
///    Disliked wordings are demoted — deliberation steers away from them —
///    and approved ones are promoted. <see cref="SelfImprove"/> prunes
///    trained sayings whose score stays negative: she literally stops
///    saying the things that never land.
/// 3. SELF-REVIEW: <see cref="SelfImprove"/> runs a conscious pass over the
///    ledger (also exposed as the demo's <c>/improve</c> command) and
///    reports what changed, so self-improvement is explainable, not magic.
/// </summary>
public sealed class SelfImprovementModule : CognitiveModuleBase
{
    private readonly object _gate = new();

    // Template key -> approval score in -1..1 (EMA of implied feedback).
    private readonly Dictionary<string, double> _templateScores = new(StringComparer.Ordinal);

    private int _cyclesObserved;
    private int _positives;
    private int _negatives;
    private int _autoRewards;
    private int _prunedSayings;

    // The previous turn's spoken reply, kept so the NEXT turn's feedback can
    // be credited to it.
    private string? _prevReply;
    private string? _prevTemplate;
    private int _prevStyle;
    private double[]? _prevFeatures;
    private double _prevValence;
    private double _prevArousal;
    private int _prevCycle = -1;

    /// <summary>Creates the module.</summary>
    public SelfImprovementModule() { }

    /// <inheritdoc />
    public override string Name => "self-improvement";

    /// <inheritdoc />
    public override int Order => 65; // after thinking (50) + learning (60)

    /// <summary>How many conversation turns have been watched.</summary>
    public int CyclesObserved { get { lock (_gate) return _cyclesObserved; } }

    /// <summary>Approval signals detected so far.</summary>
    public int PositiveSignals { get { lock (_gate) return _positives; } }

    /// <summary>Disapproval signals detected so far.</summary>
    public int NegativeSignals { get { lock (_gate) return _negatives; } }

    /// <summary>
    /// Deliberation reads these: penalty in 0..1 for a reply template that
    /// historically drew disapproval (0 = neutral / unknown).
    /// </summary>
    public double TemplatePenalty(string templateKey)
    {
        lock (_gate)
        {
            var s = _templateScores.TryGetValue(templateKey, out var v) ? v : 0;
            return s < 0 ? Math.Min(1, -s) : 0;
        }
    }

    /// <summary>
    /// Deliberation reads these: boost in 0..1 for a reply template that
    /// historically drew approval (0 = neutral / unknown).
    /// </summary>
    public double TemplateBoost(string templateKey)
    {
        lock (_gate)
        {
            var s = _templateScores.TryGetValue(templateKey, out var v) ? v : 0;
            return s > 0 ? Math.Min(1, s) : 0;
        }
    }

    // Cues that grade the previous reply. Deliberately conservative — a
    // false "dislike" would suppress good wordings.
    public static readonly string[] PositiveCues =
    {
        "good job", "well done", "nice work", "that's right", "thats right",
        "that is right", "correct", "exactly", "perfect", "great answer",
        "good answer", "smart", "you got it", "that helped", "thanks",
        "thank you", "i like that", "love that", "amazing", "awesome",
        "you're smart", "you are smart", "clever", "proud of you",
    };

    public static readonly string[] NegativeCues =
    {
        "wrong", "incorrect", "not right", "that's not", "thats not",
        "that is not", "you don't understand", "you do not understand",
        "you're not listening", "you are not listening", "stop saying",
        "stop talking like", "bad answer", "nonsense", "that made no sense",
        "you're repeating", "you are repeating", "already said that",
        "stupid", "dumb", "idiot", "shut up", "annoying", "boring",
        "you misheard", "you misunderstood", "no, that", "no that",
    };

    /// <summary>+1 when the text clearly approves the previous reply, -1 when
    /// it clearly disapproves, 0 when it carries no feedback cue.</summary>
    public static int ClassifyFeedback(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return 0;
        var lower = text.ToLowerInvariant();
        // A question is a question, not a grade — "are you sure that's right?"
        // must not count as approval even though it contains "right".
        if (lower.Contains('?')) return 0;
        foreach (var cue in NegativeCues)
            if (lower.Contains(cue)) return -1;
        foreach (var cue in PositiveCues)
            if (lower.Contains(cue)) return 1;
        return 0;
    }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        if (context.Stimulus.Kind == StimulusKind.Feedback)
            return; // explicit feedback already flows through Reinforce()

        // Backfill: when thinking is disabled the previous cycle's reply was
        // still being composed while this module ran — by NOW it is committed
        // and available as Host.LastResponse. Grade it on the next turn.
        if (Host?.LastResponse is { } last &&
            last.CycleNumber == context.Cycle - 1)
        {
            lock (_gate)
            {
                if (_prevCycle == context.Cycle - 1 && _prevReply is null)
                {
                    _prevReply = last.Content;
                    _prevTemplate = ResponseComposer.TemplateKey(last.Content);
                    _prevStyle = last.StyleId;
                }
            }
        }

        int cue = ClassifyFeedback(context.EffectiveContent);
        if (cue != 0)
            ApplyImpliedFeedback(cue, context);

        // Remember this turn so the next one can grade it. During module
        // execution, context.Response holds THIS cycle's reply when the
        // thinking module is on (it runs earlier in the cycle); when it is
        // off, the fallback is composed after the modules and gets picked up
        // by the backfill above on the next cycle.
        lock (_gate)
        {
            _prevReply = context.Response?.Content;
            _prevTemplate = _prevReply is null ? null : ResponseComposer.TemplateKey(_prevReply);
            _prevStyle = context.Response?.StyleId ?? 0;
            _prevFeatures = context.Features;
            _prevValence = context.Emotions?.Valence ?? 0;
            _prevArousal = context.Emotions?.Arousal ?? 0.5;
            _prevCycle = context.Cycle;
            _cyclesObserved++;
        }
    }

    private void ApplyImpliedFeedback(int cue, CognitiveContext context)
    {
        string? template;
        int style;
        double[]? features;
        double valence, arousal;
        lock (_gate)
        {
            // Only grade a reply from the immediately previous cycle — older
            // feedback is too noisy to attribute.
            if (_prevTemplate is null || _prevCycle != context.Cycle - 1)
                return;
            template = _prevTemplate;
            style = _prevStyle;
            features = _prevFeatures;
            valence = _prevValence;
            arousal = _prevArousal;
        }

        double reward = cue > 0 ? 0.65 : -0.75;
        lock (_gate)
        {
            double current = _templateScores.TryGetValue(template, out var v) ? v : 0;
            // EMA: a wording's reputation moves a little each time it is graded.
            current += 0.35 * (reward - current);
            _templateScores[template] = Math.Clamp(current, -1, 1);
            if (cue > 0) _positives++; else _negatives++;
            _autoRewards++;
        }

        // Close the reinforcement loop on the actual spoken style.
        var learner = Host?.Learning;
        if (learner is not null && features is not null)
        {
            learner.ObserveAction(features, style, valence, arousal);
            learner.Reinforce(reward);
        }

        Host?.Stream?.Add("self-improvement",
            $"implied {(cue > 0 ? "approval" : "disapproval")} of my last reply — " +
            $"wording {(cue > 0 ? "promoted" : "demoted")}, style #{style} reinforced ({reward:+0.00;-0.00})");
    }

    // ------------------------------------------------------------------
    // The conscious review pass (also the /improve command).
    // ------------------------------------------------------------------

    /// <summary>
    /// Grades a specific spoken reply with an explicit reward in -1..1.
    /// The normal path is implied feedback from the next user message, but
    /// hosts that KNOW the verdict (a UI thumbs-up button, an evaluation
    /// harness) can grade replies directly. Moves the wording's reputation
    /// exactly like implied feedback does.
    /// </summary>
    public void RecordFeedback(string replyText, double reward)
    {
        if (string.IsNullOrWhiteSpace(replyText)) return;
        reward = Math.Clamp(reward, -1, 1);
        var template = ResponseComposer.TemplateKey(replyText);
        lock (_gate)
        {
            double current = _templateScores.TryGetValue(template, out var v) ? v : 0;
            current += 0.35 * (reward - current);
            _templateScores[template] = Math.Clamp(current, -1, 1);
            if (reward >= 0) _positives++; else _negatives++;
        }
    }

    /// <summary>
    /// Runs one self-improvement review: prunes trained sayings whose
    /// wording score is clearly negative and summarizes everything the
    /// module has learned from feedback so far.
    /// </summary>
    public SelfImprovementReport SelfImprove()
    {
        int pruned = 0;
        int demoted = 0;
        int promoted = 0;
        var phrases = Host?.Phrases;
        if (phrases is not null)
        {
            foreach (var saying in phrases.All())
            {
                var key = ResponseComposer.TemplateKey(saying);
                lock (_gate)
                    if (_templateScores.TryGetValue(key, out var s) && s <= -0.45)
                    {
                        if (phrases.Remove(saying))
                        {
                            pruned++;
                            _prunedSayings++;
                            Host?.Stream?.Add("self-improvement",
                                $"retired a saying that never landed: \"{Mind.Truncate(saying, 50)}\"");
                        }
                    }
            }
        }

        lock (_gate)
        {
            foreach (var (key, score) in _templateScores)
            {
                if (score <= -0.45) demoted++;
                else if (score >= 0.45) promoted++;
            }
        }

        int cycles, positives, negatives, tracked;
        lock (_gate)
        {
            cycles = _cyclesObserved;
            positives = _positives;
            negatives = _negatives;
            tracked = _templateScores.Count;
        }
        var summary =
            $"Reviewed {cycles} turn(s) — {positives} approval(s), " +
            $"{negatives} disapproval(s); {tracked} wording(s) tracked " +
            $"({promoted} promoted, {demoted} demoted), " +
            $"{pruned} saying(s) retired this pass.";
        var report = new SelfImprovementReport
        {
            CyclesObserved = cycles,
            PositiveSignals = positives,
            NegativeSignals = negatives,
            TemplatesTracked = tracked,
            TemplatesDemoted = demoted,
            TemplatesPromoted = promoted,
            SayingsPruned = pruned,
            Summary = summary,
        };
        Host?.Stream?.Add("self-improvement", summary);
        return report;
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    private sealed class Dto
    {
        public List<string[]>? Templates { get; set; } // [key, score]
        public int Cycles { get; set; }
        public int Positives { get; set; }
        public int Negatives { get; set; }
        public int PrunedSayings { get; set; }
    }

    /// <summary>Serializes the improvement ledger (for persistence).</summary>
    public string Serialize()
    {
        lock (_gate)
        {
            var dto = new Dto
            {
                Cycles = _cyclesObserved,
                Positives = _positives,
                Negatives = _negatives,
                PrunedSayings = _prunedSayings,
                Templates = _templateScores
                    .OrderByDescending(kv => Math.Abs(kv.Value))
                    .Take(300)
                    .Select(kv => new[] { kv.Key, kv.Value.ToString("R", System.Globalization.CultureInfo.InvariantCulture) })
                    .ToList(),
            };
            return JsonSerializer.Serialize(dto);
        }
    }

    /// <summary>Restores the improvement ledger from a snapshot.</summary>
    public void Restore(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return;
        try
        {
            var dto = JsonSerializer.Deserialize<Dto>(json);
            if (dto is null) return;
            lock (_gate)
            {
                _templateScores.Clear();
                if (dto.Templates is not null)
                    foreach (var row in dto.Templates)
                        if (row.Length == 2 &&
                            double.TryParse(row[1], System.Globalization.CultureInfo.InvariantCulture, out var s))
                            _templateScores[row[0]] = Math.Clamp(s, -1, 1);
                _cyclesObserved = dto.Cycles;
                _positives = dto.Positives;
                _negatives = dto.Negatives;
                _prunedSayings = dto.PrunedSayings;
                _prevReply = null;
                _prevTemplate = null;
                _prevFeatures = null;
                _prevCycle = -1;
            }
        }
        catch (JsonException)
        {
            // Corrupt payload: start the ledger fresh.
        }
    }
}
