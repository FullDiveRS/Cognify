using Cognify.Audio;
using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Tests;

/// <summary>Tests for the simulated audio backend and audio module.</summary>
public class AudioTests
{
    [Fact]
    public void Backend_ClassifiesSpeech()
    {
        var backend = new SimulatedAudioBackend();
        var features = backend.Analyze(new AudioInput("hello there friend, how are you doing today"));

        Assert.True(features.IsSpeech);
        Assert.Equal("speech", features.Category);
        Assert.Contains("hello", features.Transcription, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Backend_ClassifiesNoise()
    {
        var backend = new SimulatedAudioBackend();
        var features = backend.Analyze(new AudioInput("thunder crash"));

        Assert.False(features.IsSpeech);
        Assert.Equal("noise", features.Category);
        Assert.Contains("thunder", features.Description, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Backend_ClassifiesMusic_WithTempo()
    {
        var backend = new SimulatedAudioBackend();
        var features = backend.Analyze(new AudioInput("jazz music with drums"));

        Assert.Equal("music", features.Category);
        Assert.NotNull(features.TempoBpm);
        Assert.True(features.TempoBpm > 0);
    }

    [Fact]
    public void Backend_MetadataDurationWins()
    {
        var backend = new SimulatedAudioBackend();
        var metadata = new Dictionary<string, string> { ["duration"] = "42.5", ["loudness"] = "0.9" };
        var features = backend.Analyze(new AudioInput("beep", metadata));

        Assert.Equal(42.5, features.DurationSeconds, precision: 2);
        Assert.Equal(0.9, features.Loudness, precision: 2);
    }

    [Fact]
    public void Module_Speech_StimulusIsTranscribed()
    {
        using var mind = MindBuilder.Create("Listener").WithAudio().Build();
        var response = mind.Hear("please tell me about dogs");

        Assert.Equal(1, mind.Hearing!.ClipsHeard);
        Assert.Contains("please tell me about dogs", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_Speech_FeedsReasoningOverWords()
    {
        // The transcription becomes the effective content, so the reasoning
        // module can answer a question that arrived by "voice".
        using var mind = MindBuilder.Create("Listener")
            .WithAudio()
            .WithReasoning()
            .WithThinking()
            .Build();

        mind.Teach("birds are animals");
        var response = mind.Hear("what is a bird?");

        Assert.Contains("animal", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_NonSpeech_DescribesSound()
    {
        using var mind = MindBuilder.Create("Listener").WithAudio().Build();
        var response = mind.Hear("thunder crash boom");

        Assert.Contains("sound", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("noise", mind.Hearing!.LastHeard!.Category);
    }

    [Fact]
    public void Module_LoudSound_Startles()
    {
        using var mind = MindBuilder.Create("Listener")
            .WithAudio(a => a.Salience(0.4))
            .WithEmotions()
            .Build();

        var metadata = new Dictionary<string, string> { ["loudness"] = "0.95" };
        mind.Process(Stimulus.Audio("explosion bang", metadata: metadata));

        Assert.True(mind.EmotionalState!.Intensities.GetValueOrDefault(EmotionType.Surprise) > 0,
            "a very loud sound should surprise the mind");
    }

    [Fact]
    public void Module_RemembersClips()
    {
        using var mind = MindBuilder.Create("Listener").WithAudio().WithMemory().Build();
        mind.Hear("a doorbell rings somewhere");
        Assert.Contains(mind.Memory!.Episodes(), m => m.Source == "audio");
    }

    [Fact]
    public void Hear_Throws_WhenAudioDisabled()
    {
        using var mind = MindBuilder.Create("Deaf").Build();
        Assert.Throws<MindConfigurationException>(() => mind.Hear("hello"));
    }

    [Fact]
    public void CustomBackend_IsUsed()
    {
        var stub = new StubAudioBackend();
        using var mind = MindBuilder.Create("Listener")
            .WithAudio(a => a.Backend = stub)
            .Build();

        mind.Hear("anything");
        Assert.Equal(1, stub.Calls);
        Assert.Same(stub.Features, mind.Hearing!.LastHeard);
    }

    private sealed class StubAudioBackend : IAudioBackend
    {
        public int Calls;
        public readonly AudioFeatures Features = new() { Description = "stub", Category = "noise" };
        public AudioFeatures Analyze(AudioInput input) { Calls++; return Features; }
    }
}
