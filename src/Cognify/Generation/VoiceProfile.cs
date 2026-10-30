namespace Cognify.Generation;

/// <summary>
/// A voice identity: base pitch, expressiveness range, brightness and pace.
/// <see cref="VoiceCloneEngine"/> estimates one from a real recording; the
/// speech synthesizer then speaks with that voice.
/// </summary>
public sealed record VoiceProfile
{
    /// <summary>Median fundamental frequency in Hz (typical speech 85–255).</summary>
    public double BasePitchHz { get; init; } = 165;

    /// <summary>How far the pitch swings when speaking, in semitones.</summary>
    public double PitchRangeSemitones { get; init; } = 4;

    /// <summary>Spectral brightness, 0 (dark/soft) .. 1 (bright/hissy).</summary>
    public double Brightness { get; init; } = 0.4;

    /// <summary>Speaking pace in words per minute (typical 120–190).</summary>
    public double WordsPerMinute { get; init; } = 150;

    /// <summary>Where the profile came from ("default", a file path, ...).</summary>
    public string Source { get; init; } = "default";

    /// <summary>One-line description.</summary>
    public string Describe() =>
        $"voice profile ({Source}): {BasePitchHz:0} Hz base pitch, ±{PitchRangeSemitones:0.0} semitones, " +
        $"brightness {Brightness:P0}, {WordsPerMinute:0} wpm";
}

/// <summary>
/// Builds a <see cref="VoiceProfile"/> from a recording by measuring it:
/// median pitch via autocorrelation over voiced frames, spectral brightness
/// via zero-crossing rate, and pace via syllable-sized energy peaks. It is
/// a statistical "clone" (timbre parameters, not a neural voice print) —
/// the honest scope for a dependency-free engine.
/// </summary>
public static class VoiceCloneEngine
{
    private const int FrameSize = 1024;   // ~46 ms at 22.05 kHz
    private const int Hop = 512;

    /// <summary>Analyzes a WAV recording and derives a voice profile.</summary>
    public static VoiceProfile CloneFrom(AudioClip clip)
    {
        if (clip.Duration < 0.5)
            throw new ArgumentException("voice sample too short — need at least 0.5 s of speech");

        var pitches = new List<double>();
        var zcrs = new List<double>();
        var energies = new List<double>();
        int window = Math.Min(FrameSize, clip.Samples.Length);
        double zcrScale = clip.SampleRate / (double)window / 2; // normalize ZCR to Hz-ish

        for (int start = 0; start + window <= clip.Samples.Length; start += Hop)
        {
            var frame = clip.Samples.AsSpan(start, window);
            double energy = 0;
            int crossings = 0;
            for (int i = 0; i < window; i++)
            {
                energy += frame[i] * frame[i];
                if (i > 0 && (frame[i] >= 0) != (frame[i - 1] >= 0)) crossings++;
            }
            energy = Math.Sqrt(energy / window);
            energies.Add(energy);
            zcrs.Add(crossings * zcrScale);
            if (energy < 0.02) continue; // unvoiced/silence — skip pitch

            double? pitch = EstimatePitch(frame, clip.SampleRate);
            if (pitch is not null) pitches.Add(pitch.Value);
        }

        if (pitches.Count == 0)
            throw new ArgumentException("no voiced frames found — is this a speech recording?");

        pitches.Sort();
        double median = pitches[pitches.Count / 2];
        double low = pitches[(int)(pitches.Count * 0.1)];
        double high = pitches[(int)(pitches.Count * 0.9)];
        double range = 12 * Math.Log(Math.Max(high, 1) / Math.Max(low, 1)) / Math.Log(2);
        double brightness = Math.Clamp(Median(zcrs) / 3000.0, 0, 1);

        // Syllable-ish energy peaks per second → words per minute (≈1.5 syllables/word).
        var syllables = CountEnergyPeaks(energies, Hop / (double)clip.SampleRate);
        double wpm = Math.Clamp(syllables / clip.Duration * 60 / 1.5, 60, 260);

        return new VoiceProfile
        {
            BasePitchHz = Math.Clamp(median, 60, 400),
            PitchRangeSemitones = Math.Clamp(range, 0.5, 12),
            Brightness = brightness,
            WordsPerMinute = wpm,
            Source = "cloned"
        };
    }

    /// <summary>Autocorrelation pitch estimate for one frame (60–400 Hz).</summary>
    private static double? EstimatePitch(ReadOnlySpan<double> frame, int sampleRate)
    {
        int minLag = sampleRate / 400;
        int maxLag = Math.Min(sampleRate / 60, frame.Length / 2);
        double best = 0;
        int bestLag = -1;
        double energy0 = 0;
        for (int i = 0; i < frame.Length; i++) energy0 += frame[i] * frame[i];
        if (energy0 < 1e-9) return null;

        for (int lag = minLag; lag <= maxLag; lag++)
        {
            double correlation = 0;
            for (int i = 0; i + lag < frame.Length; i++)
                correlation += frame[i] * frame[i + lag];
            correlation /= frame.Length - lag;
            if (correlation > best)
            {
                best = correlation;
                bestLag = lag;
            }
        }
        if (bestLag <= 0 || best / energy0 < 0.25) return null;
        return sampleRate / (double)bestLag;
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0;
        var sorted = values.OrderBy(v => v).ToList();
        return sorted[sorted.Count / 2];
    }

    private static int CountEnergyPeaks(List<double> energies, double hopSeconds)
    {
        if (energies.Count < 3) return 0;
        // Smooth with a 3-frame average, then count local maxima above a floor.
        var smoothed = new double[energies.Count];
        for (int i = 0; i < energies.Count; i++)
        {
            double sum = energies[i];
            int count = 1;
            if (i > 0) { sum += energies[i - 1]; count++; }
            if (i + 1 < energies.Count) { sum += energies[i + 1]; count++; }
            smoothed[i] = sum / count;
        }
        double floor = smoothed.Average() * 1.15;
        int peaks = 0;
        for (int i = 1; i < smoothed.Length - 1; i++)
            if (smoothed[i] > smoothed[i - 1] && smoothed[i] >= smoothed[i + 1] && smoothed[i] > floor)
                peaks++;
        // hopSeconds converts frames → seconds; report per second internally.
        return peaks;
    }
}
