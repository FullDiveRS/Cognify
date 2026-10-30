using System.Security.Cryptography;
using System.Text;

namespace Cognify.Generation;

/// <summary>
/// Turns a text prompt into a short composed piece: mood parsing picks a
/// scale and tempo, a deterministic random walk writes the melody, chords
/// and a bass line, and everything is synthesized with envelopes into real
/// WAV samples. Deterministic per prompt.
/// </summary>
public sealed class MusicSynthesizer
{
    private static readonly Dictionary<string, int> NoteIndex = new()
    {
        ["c"] = 0, ["c#"] = 1, ["db"] = 1, ["d"] = 2, ["d#"] = 3, ["eb"] = 3,
        ["e"] = 4, ["f"] = 5, ["f#"] = 6, ["gb"] = 6, ["g"] = 7, ["g#"] = 8,
        ["ab"] = 8, ["a"] = 9, ["a#"] = 10, ["bb"] = 10, ["b"] = 11,
    };

    private static readonly Dictionary<string, int[]> Scales = new()
    {
        ["major"] = [0, 2, 4, 5, 7, 9, 11, 12],
        ["minor"] = [0, 2, 3, 5, 7, 8, 10, 12],
        ["pentatonic"] = [0, 2, 4, 7, 9, 12, 14, 16],
    };

    private sealed record MoodConfig(string Scale, int Bpm, string Wave, int Octave,
        string Name);

    private static readonly Dictionary<string, MoodConfig> Moods = new()
    {
        ["happy"] = new("major", 132, "triangle", 4, "happy"),
        ["sad"] = new("minor", 66, "sine", 3, "sad"),
        ["calm"] = new("pentatonic", 84, "sine", 4, "calm"),
        ["energetic"] = new("major", 160, "square", 4, "energetic"),
        ["dark"] = new("minor", 100, "triangle", 3, "dark"),
    };

    private static readonly Dictionary<string, string[]> MoodWords = new()
    {
        ["happy"] = ["happy", "joyful", "cheerful", "bright", "sunny", "fun", "playful"],
        ["sad"] = ["sad", "melancholy", "gloomy", "blue", "somber", "rain"],
        ["calm"] = ["calm", "peaceful", "gentle", "soft", "serene", "quiet", "zen"],
        ["energetic"] = ["energetic", "fast", "exciting", "intense", "dance", "rock", "hyper"],
        ["dark"] = ["dark", "scary", "eerie", "mysterious", "spooky", "night"],
    };

    private readonly int _sampleRate;

    /// <summary>Creates the synthesizer (22.05 kHz default).</summary>
    public MusicSynthesizer(int sampleRate = 22050) => _sampleRate = sampleRate;

    /// <summary>The parsed mood of a prompt ("happy" by default).</summary>
    public static string ParseMood(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        foreach (var (mood, words) in MoodWords)
            if (words.Any(lower.Contains))
                return mood;
        return "happy";
    }

    /// <summary>A short description of the piece a prompt will produce.</summary>
    public static string Describe(string prompt)
    {
        var mood = ParseMood(prompt);
        var config = Moods[mood];
        return $"a {config.Name} {config.Scale}-scale piece at {config.Bpm} bpm";
    }

    /// <summary>Composes and synthesizes the piece (deterministic per prompt).</summary>
    public AudioClip Compose(string prompt)
    {
        var mood = ParseMood(prompt);
        var config = Moods[mood];
        var random = new Random(StableSeed(prompt));

        // Explicit notes in the prompt ("A4 C5 E5") override the melody walk.
        var explicitNotes = System.Text.RegularExpressions.Regex
            .Matches(prompt, @"\b([A-G][#b]?\d)\b")
            .Select(m => NoteFrequency(m.Groups[1].Value))
            .ToList();

        int beatMs = (int)(60_000.0 / config.Bpm);
        var melody = new List<(double Freq, double Ms)>();
        if (explicitNotes.Count > 0)
        {
            foreach (var freq in explicitNotes)
                melody.Add((freq, 300));
        }
        else
        {
            var scale = Scales[config.Scale];
            int degree = 0;
            for (int i = 0; i < 8; i++)
            {
                int step = random.Next(5) switch { 0 => -2, 1 => -1, 2 => 0, 3 => 1, _ => 1 };
                degree = Math.Clamp(degree + step, 0, scale.Length - 1);
                int midi = 12 * (config.Octave + 1) + scale[degree];
                melody.Add((440.0 * Math.Pow(2, (midi - 69) / 12.0),
                    beatMs * (random.Next(3) == 2 ? 1.0 : 0.5)));
            }
        }

        // Chord pad: triads on scale degrees 0, 3, 4, 0 held under the melody.
        var scale2 = Scales[config.Scale];
        int[] progression = [0, 3, 4, 0];
        var chords = new List<double[]>();
        foreach (var rootDegree in progression)
        {
            int rootMidi = 12 * (config.Octave - 1 + 1) + scale2[rootDegree];
            double root = 440.0 * Math.Pow(2, (rootMidi - 69) / 12.0);
            double third = root * Math.Pow(2, (config.Scale == "minor" ? 3 : 4) / 12.0);
            double fifth = root * Math.Pow(2, 7 / 12.0);
            chords.Add([root, third, fifth]);
        }

        int melodySamples = (int)(melody.Sum(n => n.Ms) / 1000.0 * _sampleRate);
        var output = new double[Math.Max(1, melodySamples)];

        // Melody voice.
        int cursor = 0;
        foreach (var (freq, ms) in melody)
        {
            int length = (int)(ms / 1000.0 * _sampleRate);
            AppendTone(output, cursor, Math.Min(length, output.Length - cursor),
                freq, config.Wave, 0.32, 0.01, ms / 1000.0 * 0.3);
            cursor += length;
        }

        // Chord pad (whole progression, looped under the melody).
        int chordLength = output.Length / chords.Count;
        for (int c = 0; c < chords.Count; c++)
        {
            int start = c * chordLength;
            int length = Math.Min(chordLength, output.Length - start);
            foreach (var freq in chords[c])
                AppendTone(output, start, length, freq, "sine", 0.06, 0.05, 0.1);
        }

        // Bass: root note an octave down, one per chord.
        for (int c = 0; c < chords.Count; c++)
        {
            int start = c * chordLength;
            int length = Math.Min(chordLength, output.Length - start);
            AppendTone(output, start, length, chords[c][0] / 2, "triangle", 0.14, 0.01, 0.05);
        }

        var clip = Dsp.Normalize(new AudioClip(output, _sampleRate), 0.9);
        return Dsp.Fade(clip, 0.02, 0.08);
    }

    private void AppendTone(double[] output, int start, int length, double freq,
        string wave, double volume, double attack, double release)
    {
        if (length <= 0) return;
        int attackSamples = Math.Max(1, (int)(attack * _sampleRate));
        int releaseSamples = Math.Max(1, (int)(release * _sampleRate));
        for (int i = 0; i < length; i++)
        {
            double t = i / (double)_sampleRate;
            double phase = 2 * Math.PI * freq * t;
            double value = Waveform(wave, phase) + 0.25 * Waveform(wave, phase * 2);
            double envelope = 1.0;
            if (i < attackSamples) envelope = i / (double)attackSamples;
            else if (i > length - releaseSamples)
                envelope = Math.Max(0, (length - i) / (double)releaseSamples);
            output[start + i] += value * envelope * volume;
        }
    }

    private static double Waveform(string kind, double phase) => kind switch
    {
        "square" => Math.Sin(phase) >= 0 ? 0.6 : -0.6,
        "triangle" => 2.0 / Math.PI * Math.Asin(Math.Sin(phase)),
        _ => Math.Sin(phase)
    };

    /// <summary>Converts a note name ("A4") to frequency in Hz.</summary>
    public static double NoteFrequency(string note)
    {
        var match = System.Text.RegularExpressions.Regex.Match(
            note.Trim().ToLowerInvariant(), @"^([a-g][#b]?)(\d)$");
        if (!match.Success)
            throw new ArgumentException($"invalid note '{note}'");
        int semitones = NoteIndex[match.Groups[1].Value] + 12 * (int.Parse(match.Groups[2].Value) + 1) - 69;
        return 440.0 * Math.Pow(2, semitones / 12.0);
    }

    private static int StableSeed(string text)
    {
        var hash = MD5.HashData(Encoding.UTF8.GetBytes(text.Trim().ToLowerInvariant()));
        return BitConverter.ToInt32(hash, 0);
    }
}
