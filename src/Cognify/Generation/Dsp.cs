namespace Cognify.Generation;

/// <summary>The kind of filter for <see cref="Dsp.Filter"/>.</summary>
public enum FilterKind
{
    /// <summary>Everything below the cutoff (bass stem).</summary>
    LowPass,
    /// <summary>Everything above the cutoff (treble stem).</summary>
    HighPass,
    /// <summary>The band between two cutoffs (midrange stem).</summary>
    BandPass
}

/// <summary>
/// The audio editing / analysis toolkit behind the mind's "edit", "convert"
/// and "separate" abilities: trimming, gain, fades, normalization, reverse,
/// concatenation, sample-rate conversion, biquad filtering and a three-band
/// stem separator. All operations are pure functions on <see cref="AudioClip"/>.
/// </summary>
public static class Dsp
{
    /// <summary>Keeps [start, end) seconds; negative end means "to the end".</summary>
    public static AudioClip Trim(AudioClip clip, double start, double end)
    {
        int first = Math.Clamp((int)(start * clip.SampleRate), 0, clip.Samples.Length);
        int last = end < 0 ? clip.Samples.Length
            : Math.Clamp((int)(end * clip.SampleRate), first, clip.Samples.Length);
        var trimmed = clip.Samples[first..last];
        return new AudioClip(trimmed, clip.SampleRate);
    }

    /// <summary>Multiplies every sample by a gain factor (1 = unchanged).</summary>
    public static AudioClip Gain(AudioClip clip, double factor)
    {
        var output = new double[clip.Samples.Length];
        for (int i = 0; i < output.Length; i++) output[i] = Math.Clamp(clip.Samples[i] * factor, -1, 1);
        return new AudioClip(output, clip.SampleRate);
    }

    /// <summary>Applies linear fade-in/out (seconds; 0 skips that side).</summary>
    public static AudioClip Fade(AudioClip clip, double fadeIn, double fadeOut)
    {
        var output = (double[])clip.Samples.Clone();
        int inSamples = (int)(fadeIn * clip.SampleRate);
        int outSamples = (int)(fadeOut * clip.SampleRate);
        for (int i = 0; i < inSamples && i < output.Length; i++)
            output[i] *= i / (double)inSamples;
        for (int i = 0; i < outSamples && i < output.Length; i++)
            output[output.Length - 1 - i] *= i / (double)outSamples;
        return new AudioClip(output, clip.SampleRate);
    }

    /// <summary>Scales the clip so its peak hits the target (default 0.95).</summary>
    public static AudioClip Normalize(AudioClip clip, double peak = 0.95)
    {
        double max = 0;
        foreach (var sample in clip.Samples) max = Math.Max(max, Math.Abs(sample));
        return max < 1e-6 ? clip : Gain(clip, peak / max);
    }

    /// <summary>Plays the clip backwards.</summary>
    public static AudioClip Reverse(AudioClip clip)
    {
        var output = new double[clip.Samples.Length];
        for (int i = 0; i < output.Length; i++) output[i] = clip.Samples[clip.Samples.Length - 1 - i];
        return new AudioClip(output, clip.SampleRate);
    }

    /// <summary>Joins two clips (resampling the second if rates differ).</summary>
    public static AudioClip Concat(AudioClip first, AudioClip second)
    {
        var aligned = second.SampleRate == first.SampleRate
            ? second : Resample(second, first.SampleRate);
        var output = new double[first.Samples.Length + aligned.Samples.Length];
        first.Samples.CopyTo(output, 0);
        aligned.Samples.CopyTo(output, first.Samples.Length);
        return new AudioClip(output, first.SampleRate);
    }

    /// <summary>Linear-interpolation resampling to a new sample rate.</summary>
    public static AudioClip Resample(AudioClip clip, int targetRate)
    {
        if (targetRate == clip.SampleRate || clip.Samples.Length == 0)
            return clip;
        int outputLength = (int)((long)clip.Samples.Length * targetRate / clip.SampleRate);
        var output = new double[Math.Max(1, outputLength)];
        double step = clip.Samples.Length / (double)output.Length;
        for (int i = 0; i < output.Length; i++)
        {
            double position = i * step;
            int index = (int)position;
            double fraction = position - index;
            double a = clip.Samples[index];
            double b = index + 1 < clip.Samples.Length ? clip.Samples[index + 1] : a;
            output[i] = a + (b - a) * fraction;
        }
        return new AudioClip(output, targetRate);
    }

    /// <summary>Applies a biquad filter (Butterworth-ish, RBJ cookbook).</summary>
    public static AudioClip Filter(AudioClip clip, FilterKind kind, double cutoffHz, double q = 0.707)
    {
        double w0 = 2 * Math.PI * cutoffHz / clip.SampleRate;
        double cosW0 = Math.Cos(w0), sinW0 = Math.Sin(w0);
        double alpha = sinW0 / (2 * q);

        double b0, b1, b2, a0, a1, a2;
        switch (kind)
        {
            case FilterKind.LowPass:
                b0 = (1 - cosW0) / 2; b1 = 1 - cosW0; b2 = (1 - cosW0) / 2;
                a0 = 1 + alpha; a1 = -2 * cosW0; a2 = 1 - alpha;
                break;
            case FilterKind.HighPass:
                b0 = (1 + cosW0) / 2; b1 = -(1 + cosW0); b2 = (1 + cosW0) / 2;
                a0 = 1 + alpha; a1 = -2 * cosW0; a2 = 1 - alpha;
                break;
            default: // band pass (constant peak gain)
                b0 = alpha; b1 = 0; b2 = -alpha;
                a0 = 1 + alpha; a1 = -2 * cosW0; a2 = 1 - alpha;
                break;
        }
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;

        var output = new double[clip.Samples.Length];
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double x0 = clip.Samples[i];
            double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            output[i] = Math.Clamp(y0, -1, 1);
            x2 = x1; x1 = x0; y2 = y1; y1 = y0;
        }
        return new AudioClip(output, clip.SampleRate);
    }

    /// <summary>One result of the three-band separator.</summary>
    public sealed record Stem(string Name, AudioClip Clip);

    /// <summary>
    /// Splits a clip into low / mid / high stems (bass, vocals-and-body,
    /// sparkle). A real source separator needs trained models; this band
    /// split is the honest, dependency-free approximation and is useful for
    /// FX and karaoke-style experiments.
    /// </summary>
    public static IReadOnlyList<Stem> Separate(AudioClip clip)
    {
        var low = Filter(clip, FilterKind.LowPass, 300);
        var high = Filter(clip, FilterKind.HighPass, 3000);
        var mid = Filter(Filter(clip, FilterKind.HighPass, 300), FilterKind.LowPass, 3000);
        return new[] { new Stem("low", low), new Stem("mid", mid), new Stem("high", high) };
    }

    /// <summary>RMS loudness of a clip (0..1).</summary>
    public static double Loudness(AudioClip clip)
    {
        double sum = 0;
        foreach (var sample in clip.Samples) sum += sample * sample;
        return Math.Sqrt(sum / Math.Max(1, clip.Samples.Length));
    }
}
