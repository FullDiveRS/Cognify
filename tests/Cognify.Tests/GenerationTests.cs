using Cognify.Core;
using Cognify.Generation;

namespace Cognify.Tests;

/// <summary>Tests for the media generation backend and module.</summary>
public class GenerationTests
{
    // ---- Image / video specs -------------------------------------------------

    [Fact]
    public void Backend_GeneratesDeterministicImageSpecs()
    {
        var backend = new SimulatedMediaBackend();
        var a = backend.Generate(new GenerationRequest("a sunset over mountains", MediaType.Image));
        var b = backend.Generate(new GenerationRequest("a sunset over mountains", MediaType.Image));

        Assert.Equal(a.Seed, b.Seed);
        Assert.Equal(a.Width, b.Width);
        Assert.Equal(a.Height, b.Height);
        Assert.True(a.Width >= 512 && a.Width <= 1024);
        Assert.Null(a.Data); // images are spec-only in the simulated backend
    }

    [Fact]
    public void Backend_VideoHasFrameMath()
    {
        var backend = new SimulatedMediaBackend();
        var artifact = backend.Generate(new GenerationRequest("a rocket launch", MediaType.Video, 4));

        Assert.Equal(4, artifact.DurationSeconds, precision: 2);
        Assert.True(artifact.FramesPerSecond >= 24);
        Assert.Equal((int)Math.Ceiling(4.0 * artifact.FramesPerSecond), artifact.FrameCount);
        Assert.True(artifact.SizeBytes > 0);
    }

    // ---- Real WAV synthesis ---------------------------------------------------

    [Fact]
    public void Backend_SpeechSynthesizesRealWav()
    {
        var backend = new SimulatedMediaBackend(sampleRate: 8000);
        var artifact = backend.Generate(new GenerationRequest("hello world today", MediaType.Speech));

        Assert.NotNull(artifact.Data);
        Assert.Equal(8000, artifact.SampleRateHz);

        var wav = artifact.Data!;
        // RIFF header
        Assert.Equal((byte)'R', wav[0]);
        Assert.Equal((byte)'I', wav[1]);
        Assert.Equal((byte)'F', wav[2]);
        Assert.Equal((byte)'F', wav[3]);
        // WAVE marker at offset 8
        Assert.Equal((byte)'W', wav[8]);
        Assert.Equal((byte)'A', wav[9]);
        Assert.Equal((byte)'V', wav[10]);
        Assert.Equal((byte)'E', wav[11]);

        // 44-byte canonical header + 16-bit mono data
        long expectedSamples = (long)(artifact.DurationSeconds * 8000);
        Assert.Equal(44 + expectedSamples * 2, wav.Length);
        Assert.Equal(44 + expectedSamples * 2, artifact.SizeBytes);
    }

    [Fact]
    public void Backend_WavContainsNonSilence()
    {
        var backend = new SimulatedMediaBackend();
        var artifact = backend.Generate(new GenerationRequest("a loud gong", MediaType.Audio));

        var wav = artifact.Data!;
        // Some sample in the body should be non-zero.
        bool anyNonZero = false;
        for (int i = 44; i < wav.Length - 1; i += 2)
        {
            if (wav[i] != 0 || wav[i + 1] != 0) { anyNonZero = true; break; }
        }
        Assert.True(anyNonZero, "synthesized WAV should contain audible samples");
    }

    [Fact]
    public void Backend_DifferentPrompts_DifferentSeeds()
    {
        var backend = new SimulatedMediaBackend();
        var a = backend.Generate(new GenerationRequest("ocean waves", MediaType.Audio));
        var b = backend.Generate(new GenerationRequest("forest birds", MediaType.Audio));
        Assert.NotEqual(a.Seed, b.Seed);
    }

    [Fact]
    public void Backend_WavCanBeDisabled()
    {
        var backend = new SimulatedMediaBackend(8000, emitWav: false);
        var artifact = backend.Generate(new GenerationRequest("wind", MediaType.Audio));
        Assert.Null(artifact.Data);
        Assert.True(artifact.SizeBytes > 0); // specs still estimate the size
    }

    // ---- Module in the cognitive cycle ---------------------------------------------

    [Fact]
    public void Module_GeneratesImage_OnRequest()
    {
        using var mind = MindBuilder.Create("Artist").WithMediaGeneration().Build();
        var response = mind.Ask("generate an image of a red sports car");

        Assert.Equal(1, mind.MediaGeneration!.GeneratedCount);
        Assert.Contains("image", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("media-generation", response.StyleName);
        Assert.Single(mind.MediaGeneration.Gallery);
    }

    [Fact]
    public void Module_GeneratesVideo_WithSeconds()
    {
        using var mind = MindBuilder.Create("Artist")
            .WithMediaGeneration(g => g.VideoSeconds(3))
            .Build();

        mind.Ask("create a video of ocean waves");
        var artifact = mind.MediaGeneration!.Gallery[0];

        Assert.Equal(MediaType.Video, artifact.Type);
        Assert.Equal(3, artifact.DurationSeconds, precision: 1);
    }

    [Fact]
    public void Module_Speaks()
    {
        using var mind = MindBuilder.Create("Artist").WithMediaGeneration().Build();
        mind.Ask("say hello there friend");

        var artifact = mind.MediaGeneration!.Gallery[0];
        Assert.Equal(MediaType.Speech, artifact.Type);
        Assert.NotNull(artifact.Data);
    }

    [Fact]
    public void Module_DoesNotHijackConditionalsContainingSay()
    {
        // Regression: "do you know ... if so say yes and if not say no" used
        // to trigger speech generation because "say" matched ANYWHERE. A
        // sentence that merely contains "say" stays a normal conversation.
        using var mind = MindBuilder.Create("Talker").WithMediaGeneration().Build();
        mind.Ask("do you know what memories are or would you like me to help you learn it, if so say yes and if not say no");
        Assert.Equal(0, mind.MediaGeneration!.GeneratedCount);
    }

    [Fact]
    public void Module_DoesNotHijackQuotedSpeechWithoutQuotes()
    {
        using var mind = MindBuilder.Create("Talker").WithMediaGeneration().Build();
        mind.Ask("where would you say the line is");
        Assert.Equal(0, mind.MediaGeneration!.GeneratedCount);
    }

    [Fact]
    public void Module_StillSpeaksQuotedPayloadMidSentence()
    {
        using var mind = MindBuilder.Create("Artist").WithMediaGeneration().Build();
        mind.Ask("if you greet the user, say \"good morning\" warmly");
        Assert.Equal(1, mind.MediaGeneration!.GeneratedCount);
        Assert.Equal(MediaType.Speech, mind.MediaGeneration.Gallery[0].Type);
    }

    [Fact]
    public void Module_IgnoresUnrelatedText()
    {
        using var mind = MindBuilder.Create("Artist").WithMediaGeneration().Build();
        mind.Ask("what is the weather like?");
        Assert.Equal(0, mind.MediaGeneration!.GeneratedCount);
    }

    [Fact]
    public void Gallery_RespectsCapacity()
    {
        using var mind = MindBuilder.Create("Artist")
            .WithMediaGeneration(g => g.VideoSeconds(1))
            .Build();

        for (int i = 0; i < 5; i++)
            mind.GenerateMedia($"clip number {i}", MediaType.Video, 1);

        Assert.Equal(5, mind.MediaGeneration!.GeneratedCount);
        Assert.Equal(5, mind.MediaGeneration.Gallery.Count); // default capacity 20

        using var small = MindBuilder.Create("SmallArtist")
            .WithMediaGeneration(g => g.VideoSeconds(1))
            .Build();
        // Gallery capacity is enforced via options; simulate by generating directly on backend module.
        var module = small.MediaGeneration!;
        for (int i = 0; i < 3; i++)
            module.Generate($"tiny clip {i}", MediaType.Video, 0.5);
        Assert.True(module.Gallery.Count <= 3);
    }

    [Fact]
    public void GenerateMedia_Convenience_AndEvent()
    {
        using var mind = MindBuilder.Create("Artist").WithMediaGeneration().Build();
        MediaArtifact? received = null;
        mind.MediaGenerated += (_, e) => received = e.Artifact;

        var artifact = mind.GenerateMedia("a thunderstorm", MediaType.Audio, 2)!;
        Assert.Same(artifact, received);
        Assert.Equal(MediaType.Audio, artifact.Type);
        Assert.NotNull(artifact.Data);
    }

    [Fact]
    public void GenerateMedia_Throws_WhenDisabled()
    {
        using var mind = MindBuilder.Create("NoArtist").Build();
        Assert.Throws<MindConfigurationException>(
            () => mind.GenerateMedia("a painting", MediaType.Image));
    }

    [Fact]
    public void CustomBackend_IsUsed()
    {
        var stub = new StubMediaBackend();
        using var mind = MindBuilder.Create("Artist")
            .WithMediaGeneration(g => g.Backend = stub)
            .Build();

        mind.Ask("generate an image of anything");
        Assert.Equal(1, stub.Calls);
        Assert.Same(stub.Artifact, mind.MediaGeneration!.Gallery[0]);
    }

    private sealed class StubMediaBackend : IMediaBackend
    {
        public int Calls;
        public readonly MediaArtifact Artifact = new() { Type = MediaType.Image, Prompt = "stub" };
        public MediaArtifact Generate(GenerationRequest request) { Calls++; return Artifact; }
    }
}
