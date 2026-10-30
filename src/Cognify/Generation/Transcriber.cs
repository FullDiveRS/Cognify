namespace Cognify.Generation;

/// <summary>The result of a transcription attempt.</summary>
public sealed record TranscriptionResult
{
    /// <summary>Creates the result.</summary>
    public TranscriptionResult(string text, double confidence, int syllableCount,
        bool matched, string message)
    {
        Text = text;
        Confidence = confidence;
        SyllableCount = syllableCount;
        Matched = matched;
        Message = message;
    }

    /// <summary>The best matching phrase, or "" when nothing matched.</summary>
    public string Text { get; }

    /// <summary>Match confidence 0..1 (0 when unmatched).</summary>
    public double Confidence { get; }

    /// <summary>How many syllable-sized energy bursts were heard.</summary>
    public int SyllableCount { get; }

    /// <summary>True when the clip matched a phrase she can speak.</summary>
    public bool Matched { get; }

    /// <summary>Human-readable explanation of the outcome.</summary>
    public string Message { get; }
}

/// <summary>
/// A small speech transcriber with an honest scope: it recognizes speech
/// that the mind herself synthesized (or a close re-recording of it), by
/// comparing the clip's energy-envelope "fingerprints" (syllable burst
/// pattern plus pitch) against a phrasebook built every time she speaks.
/// Foreign audio gets a syllable count and a truthful "teach me" reply
/// rather than confident nonsense.
/// </summary>
public sealed class Transcriber
{
    private const double BurstThresholdFactor = 0.45;

    private readonly object _gate = new();
    private readonly List<(string Phrase, EnvelopeFingerprint Print)> _phrasebook = new();

    /// <summary>How many phrases the transcriber currently knows.</summary>
    public int KnownPhrases { get { lock (_gate) return _phrasebook.Count; } }

    /// <summary>Registers a phrase with the fingerprint of its own synthesis.</summary>
    public void RegisterTemplate(string phrase, AudioClip clip)
    {
        if (string.IsNullOrWhiteSpace(phrase) || clip.Samples.Length == 0) return;
        lock (_gate)
        {
            _phrasebook.RemoveAll(p => p.Phrase.Equals(phrase.Trim(), StringComparison.OrdinalIgnoreCase));
            _phrasebook.Add((phrase.Trim(), Fingerprint(clip)));
        }
    }

    /// <summary>The phrases currently recognizable.</summary>
    public IReadOnlyList<string> Phrases
    {
        get { lock (_gate) return _phrasebook.Select(p => p.Phrase).ToList(); }
    }

    /// <summary>Transcribes a clip against the learned phrasebook.</summary>
    public TranscriptionResult Transcribe(AudioClip clip)
    {
        var fingerprint = Fingerprint(clip);
        if (fingerprint.BurstCount == 0)
            return new TranscriptionResult("", 0, 0, false,
                "I hear silence — there is no speech in that clip.");

        lock (_gate)
        {
            string best = "";
            double bestScore = 0;
            foreach (var (phrase, print) in _phrasebook)
            {
                double score = Similarity(fingerprint, print);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = phrase;
                }
            }
            if (_phrasebook.Count == 0 || bestScore < 0.62)
            {
                return new TranscriptionResult("", bestScore, fingerprint.BurstCount, false,
                    $"I hear about {fingerprint.BurstCount} syllable(s), but I can only transcribe " +
                    "speech I synthesized myself — teach me the phrase (say it with the speech skill) and I will remember it.");
            }
            return new TranscriptionResult(best, bestScore, fingerprint.BurstCount, true,
                $"I hear \"{best}\" ({bestScore:P0} match).");
        }
    }

    // ------------------------------------------------------------------
    // Envelope fingerprints
    // ------------------------------------------------------------------

    /// <summary>One clip's fingerprint: syllable burst count, normalized burst
    /// durations, and the mean pitch of the voiced part.</summary>
    internal sealed record EnvelopeFingerprint(int BurstCount, double[] BurstPattern, double MeanPitch)
    {
        /// <summary>Round for stable comparisons.</summary>
        public static double[] Round(double[] values) =>
            values.Select(v => Math.Round(v, 2)).ToArray();
    }

    /// <summary>Computes the envelope fingerprint of a clip.</summary>
    internal static EnvelopeFingerprint Fingerprint(AudioClip clip)
    {
        // 10 ms RMS envelope, smoothed.
        int window = Math.Max(1, clip.SampleRate / 100);
        int frames = clip.Samples.Length / window;
        if (frames < 4)
            return new EnvelopeFingerprint(0, [], 0);

        var envelope = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            for (int i = 0; i < window; i++)
            {
                var sample = clip.Samples[f * window + i];
                sum += sample * sample;
            }
            envelope[f] = Math.Sqrt(sum / window);
        }
        var smoothed = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            double sum = envelope[f];
            int count = 1;
            if (f > 0) { sum += envelope[f - 1]; count++; }
            if (f + 1 < frames) { sum += envelope[f + 1]; count++; }
            smoothed[f] = sum / count;
        }

        double peak = smoothed.Max();
        if (peak < 0.01)
            return new EnvelopeFingerprint(0, [], 0);

        // Syllable bursts: contiguous runs above threshold.
        double threshold = peak * BurstThresholdFactor;
        var bursts = new List<int>();
        bool inBurst = false;
        foreach (var value in smoothed)
        {
            if (value >= threshold)
            {
                if (!inBurst) bursts.Add(0);
                bursts[^1]++;
                inBurst = true;
            }
            else
            {
                inBurst = false;
            }
        }
        bursts.RemoveAll(b => b < 2); // drop clicks

        double total = Math.Max(1, bursts.Sum());
        var pattern = EnvelopeFingerprint.Round(bursts.Select(b => b / total).ToArray());
        return new EnvelopeFingerprint(bursts.Count, pattern, MeanPitch(clip));
    }

    private static double MeanPitch(AudioClip clip)
    {
        // Coarse mean pitch via average zero-crossing rate of loud frames.
        int window = Math.Max(1, clip.SampleRate / 100);
        var rates = new List<double>();
        for (int start = 0; start + window <= clip.Samples.Length; start += window)
        {
            double energy = 0;
            int crossings = 0;
            for (int i = start; i < start + window; i++)
            {
                energy += clip.Samples[i] * clip.Samples[i];
                if (i > start && (clip.Samples[i] >= 0) != (clip.Samples[i - 1] >= 0)) crossings++;
            }
            if (Math.Sqrt(energy / window) > 0.02)
                rates.Add(crossings / (window / (double)clip.SampleRate));
        }
        rates.Sort();
        return rates.Count == 0 ? 0 : rates[rates.Count / 2];
    }

    private static double Similarity(EnvelopeFingerprint a, EnvelopeFingerprint b)
    {
        if (a.BurstCount != b.BurstCount) return 0;
        if (a.BurstPattern.Length == 0 || b.BurstPattern.Length == 0) return 0;
        // Pattern distance (L1 over normalized burst lengths).
        double distance = 0;
        for (int i = 0; i < a.BurstPattern.Length; i++)
            distance += Math.Abs(a.BurstPattern[i] - b.BurstPattern[i]);
        double patternScore = 1 - distance / 2;

        // Pitch closeness: within 20% counts as a full match.
        double pitchScore = 1;
        if (a.MeanPitch > 0 && b.MeanPitch > 0)
        {
            double ratio = Math.Max(a.MeanPitch, b.MeanPitch) / Math.Max(1, Math.Min(a.MeanPitch, b.MeanPitch));
            pitchScore = Math.Clamp(1 - (ratio - 1) / 0.2, 0, 1);
        }
        return patternScore * 0.8 + pitchScore * 0.2;
    }
}
