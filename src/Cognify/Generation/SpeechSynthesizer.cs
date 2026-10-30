using System.Security.Cryptography;
using System.Text;

namespace Cognify.Generation;

/// <summary>
/// A tiny formant (source–filter) text-to-speech engine: letter-to-sound
/// rules turn text into phoneme segments, a glottal pulse/noise source is
/// filtered through three cascaded formant resonators, and pitch/intonation
/// come from the <see cref="VoiceProfile"/>. The result is robotic but
/// word-shaped, real speech — no external engine, no keys.
/// </summary>
public sealed class SpeechSynthesizer
{
    private readonly int _sampleRate;

    /// <summary>Creates a synthesizer (16 kHz default — speech band).</summary>
    public SpeechSynthesizer(int sampleRate = 16000)
    {
        if (sampleRate < 8000) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        _sampleRate = sampleRate;
    }

    /// <summary>Current sample rate.</summary>
    public int SampleRate => _sampleRate;

    // Vowel formant targets (F1, F2, F3).
    private static readonly Dictionary<char, (double F1, double F2, double F3)> Vowels = new()
    {
        ['a'] = (700, 1220, 2600),
        ['e'] = (500, 1900, 2550),
        ['i'] = (330, 2300, 3000),
        ['o'] = (500, 900, 2400),
        ['u'] = (350, 800, 2300),
    };

    /// <summary>One synthesized segment.</summary>
    private sealed record Phoneme(string Kind, double F1, double F2, double F3,
        double DurationMs, bool Voiced, double Amplitude);

    /// <summary>Synthesizes speech for the text (deterministic per text+profile).</summary>
    public AudioClip Synthesize(string text, VoiceProfile? profile = null)
    {
        profile ??= new VoiceProfile();
        var phonemes = TextToPhonemes(text);
        if (phonemes.Count == 0)
            return new AudioClip([], _sampleRate);

        var random = new Random(StableSeed(text + profile.Source));
        bool question = text.TrimEnd().EndsWith('?');
        int totalSamples = (int)(phonemes.Sum(p => p.DurationMs) / 1000.0 * _sampleRate) + 1;
        var output = new double[totalSamples];
        double pitchHz = profile.BasePitchHz;
        int cursor = 0;
        double previousF1 = Vowels['a'].F1, previousF2 = Vowels['a'].F2, previousF3 = Vowels['a'].F3;
        double phase = 0;

        foreach (var phoneme in phonemes)
        {
            int length = Math.Max(1, (int)(phoneme.DurationMs / 1000.0 * _sampleRate));
            length = Math.Min(length, Math.Max(1, totalSamples - cursor));
            // 20 ms formant glide from the previous segment into this one.
            int glide = Math.Min(length, (int)(0.02 * _sampleRate));
            for (int i = 0; i < length; i++)
            {
                double t = i / (double)_sampleRate;
                double progress = glide > 0 && i < glide ? i / (double)glide : 1.0;
                double f1 = previousF1 + (phoneme.F1 - previousF1) * progress;
                double f2 = previousF2 + (phoneme.F2 - previousF2) * progress;
                double f3 = previousF3 + (phoneme.F3 - previousF3) * progress;

                // Source: glottal pulses (voiced) or shaped noise (unvoiced).
                double source;
                if (phoneme.Voiced)
                {
                    phase += pitchHz / _sampleRate;
                    if (phase >= 1) phase -= 1;
                    source = phase < 0.4 ? 2 * phase / 0.4 - 1 : 1 - 2 * (phase - 0.4) / 0.6;
                    // Pitch declination across the sentence + micro-jitter.
                    pitchHz = profile.BasePitchHz * PitchContour(cursor / (double)totalSamples, question)
                              * (1 + (random.NextDouble() - 0.5) * 0.04);
                }
                else
                {
                    source = (random.NextDouble() * 2 - 1) * (0.4 + profile.Brightness * 0.6);
                }

                // Three cascaded formant resonators.
                double value = source * 0.35;
                value = Resonate(value, f1, 80);
                value = Resonate(value, f2, 90) * 0.7;
                value = Resonate(value, f3, 120) * 0.4;

                // Segment envelope (soft attack/decay), then mix in.
                double envelope = Math.Sin(Math.PI * Math.Clamp(i / (double)length, 0, 1));
                envelope = 0.35 + 0.65 * envelope;
                output[cursor + i] += value * envelope * phoneme.Amplitude;
            }
            previousF1 = phoneme.F1;
            previousF2 = phoneme.F2;
            previousF3 = phoneme.F3;
            cursor += length;
        }

        // Master normalize + fade so nothing clips or clicks.
        var clip = Dsp.Normalize(new AudioClip(output, _sampleRate), 0.85);
        return Dsp.Fade(clip, 0.01, 0.03);
    }

    private double _r1, _r2; // resonator state (per-instance; single-threaded use)

    private double Resonate(double input, double frequency, double bandwidth)
    {
        double r = Math.Exp(-Math.PI * bandwidth / _sampleRate);
        double theta = 2 * Math.PI * frequency / _sampleRate;
        double a1 = 2 * r * Math.Cos(theta);
        double a2 = -(r * r);
        double y = input + a1 * _r1 + a2 * _r2;
        _r2 = _r1;
        _r1 = y;
        return y * (1 - r); // rough gain normalization
    }

    private static double PitchContour(double progress, bool question)
    {
        // Start slightly high, drift down; questions rise again at the end.
        double contour = 1.04 - 0.06 * progress;
        if (question && progress > 0.7) contour *= 1.0 + 0.12 * (progress - 0.7) / 0.3;
        return contour;
    }

    // ------------------------------------------------------------------
    // Letter-to-sound
    // ------------------------------------------------------------------

    /// <summary>Converts text into phoneme segments.</summary>
    private List<Phoneme> TextToPhonemes(string text)
    {
        var phonemes = new List<Phoneme>();
        foreach (var rawWord in text.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var word = new string(rawWord.Where(char.IsAsciiLetter).ToArray());
            if (word.Length == 0)
            {
                phonemes.Add(Silence(60));
                continue;
            }
            AppendWord(phonemes, word);
            phonemes.Add(Silence(rawWord.EndsWith('.') || rawWord.EndsWith('!')
                         ? 220 : rawWord.EndsWith(',') ? 140 : 70));
        }
        return phonemes;
    }

    private void AppendWord(List<Phoneme> phonemes, string word)
    {
        int i = 0;
        while (i < word.Length)
        {
            var two = i + 1 < word.Length ? word.Substring(i, 2) : "";
            switch (two)
            {
                case "oo": phonemes.Add(Vowel('u', 130)); i += 2; continue;
                case "ee" or "ea": phonemes.Add(Vowel('i', 120)); i += 2; continue;
                case "ou" or "ow": phonemes.Add(Vowel('o', 140)); i += 2; continue;
                case "th": phonemes.Add(Fricative("th", 80, 0.25)); i += 2; continue;
                case "sh": phonemes.Add(Fricative("sh", 85, 0.3)); i += 2; continue;
                case "ch": phonemes.Add(Plosive('k', false)); phonemes.Add(Fricative("sh", 70, 0.3)); i += 2; continue;
                case "ph": phonemes.Add(Fricative("f", 80, 0.3)); i += 2; continue;
                case "ck": phonemes.Add(Plosive('k', false)); i += 2; continue;
                case "qu": phonemes.Add(Plosive('k', false)); phonemes.Add(Glide('u')); i += 2; continue;
                case "ai" or "ay": phonemes.Add(Vowel('e', 130)); i += 2; continue;
            }

            char c = word[i];
            if (char.IsAsciiLetter(c))
            {
                if (Vowels.ContainsKey(c) || c is 'y' or 'w')
                {
                    // 'y'/'w' between consonants act as vowels; elsewhere glides.
                    bool vowelUse = c is 'y' && i > 0 && i + 1 < word.Length &&
                                    char.IsAsciiLetter(word[i + 1]) && !Vowels.ContainsKey(word[i + 1]);
                    if (Vowels.ContainsKey(c)) phonemes.Add(Vowel(c, 120));
                    else if (vowelUse) phonemes.Add(Vowel('i', 110));
                    else phonemes.Add(Glide(c));
                }
                else
                {
                    phonemes.Add(Consonant(c));
                }
            }
            i++;
        }
    }

    private static Phoneme Vowel(char c, double ms) =>
        new("vowel", Vowels[c].F1, Vowels[c].F2, Vowels[c].F3, ms, true, 0.9);

    private static Phoneme Glide(char c) => c switch
    {
        'w' => new Phoneme("glide", 300, 610, 2200, 70, true, 0.7),
        'y' => new Phoneme("glide", 270, 2300, 3000, 60, true, 0.7),
        'r' => new Phoneme("glide", 400, 1100, 1600, 75, true, 0.7),
        'l' => new Phoneme("glide", 360, 1300, 2700, 75, true, 0.7),
        _ => new Phoneme("glide", 400, 1200, 2400, 60, true, 0.7)
    };

    private static Phoneme Consonant(char c) => c switch
    {
        'm' => new Phoneme("nasal", 250, 1100, 2200, 80, true, 0.7),
        'n' => new Phoneme("nasal", 280, 1700, 2600, 75, true, 0.7),
        'b' or 'd' or 'g' => Plosive(c, true),
        'p' or 't' or 'k' => Plosive(c, false),
        'f' => Fricative("f", 80, 0.3),
        'v' => new Phoneme("fricative", 300, 1200, 2400, 70, true, 0.4),
        's' => Fricative("s", 95, 0.3),
        'z' => new Phoneme("fricative", 300, 1400, 2500, 70, true, 0.4),
        'h' => new Phoneme("fricative", 500, 1500, 2500, 55, false, 0.25),
        'j' => new Phoneme("affricate", 300, 1800, 2600, 90, true, 0.5),
        'x' => Plosive('k', false),
        _ => new Phoneme("fricative", 400, 1400, 2400, 60, false, 0.3)
    };

    private static Phoneme Plosive(char c, bool voiced) => new("plosive",
        350, 1700, 2500, voiced ? 55 : 45, voiced, 0.6);

    private static Phoneme Fricative(string kind, double ms, double amplitude) => kind switch
    {
        "s" => new Phoneme("fricative", 400, 4500, 6000, ms, false, amplitude),
        "sh" => new Phoneme("fricative", 400, 2200, 3200, ms, false, amplitude),
        "f" or "th" => new Phoneme("fricative", 400, 1600, 2400, ms, false, amplitude),
        _ => new Phoneme("fricative", 400, 1800, 2600, ms, false, amplitude)
    };

    private static Phoneme Silence(double ms) => new("silence", 400, 1200, 2400, ms, false, 0);

    private static int StableSeed(string text)
    {
        var hash = MD5.HashData(Encoding.UTF8.GetBytes(text.Trim().ToLowerInvariant()));
        return BitConverter.ToInt32(hash, 0);
    }
}
