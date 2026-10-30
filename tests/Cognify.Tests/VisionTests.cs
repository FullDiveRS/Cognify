using Cognify.Core;
using Cognify.Emotions;
using Cognify.Perception;

namespace Cognify.Tests;

/// <summary>Tests for the simulated vision backend and vision module.</summary>
public class VisionTests
{
    [Fact]
    public void Backend_DetectsKnownObjects()
    {
        var backend = new SimulatedVisionBackend();
        var scene = backend.Analyze(new VisualInput("a red apple on a wooden table"));

        Assert.Contains(scene.Objects, o => o.Label == "apple");
        Assert.Contains(scene.Objects, o => o.Label == "table");
        Assert.Equal("red", Assert.Single(scene.Colors));
        Assert.Equal("indoor", scene.SceneType);
    }

    [Fact]
    public void Backend_SpatialHints_PlaceObjects()
    {
        var backend = new SimulatedVisionBackend();
        var scene = backend.Analyze(new VisualInput("a cat on the left, a dog on the right"));

        var cat = Assert.Single(scene.Objects, o => o.Label == "cat");
        var dog = Assert.Single(scene.Objects, o => o.Label == "dog");
        Assert.True(cat.X < 0.3, $"cat should be left-placed, was {cat.X}");
        Assert.True(dog.X > 0.7, $"dog should be right-placed, was {dog.X}");
    }

    [Fact]
    public void Backend_BrightnessFromWords()
    {
        var backend = new SimulatedVisionBackend();
        Assert.True(backend.Analyze(new VisualInput("a dark night street")).Brightness < 0.3);
        Assert.True(backend.Analyze(new VisualInput("a bright sunny beach")).Brightness > 0.7);
    }

    [Fact]
    public void Backend_MetadataOverridesHeuristics()
    {
        var backend = new SimulatedVisionBackend();
        var metadata = new Dictionary<string, string>
        {
            ["width"] = "1920", ["height"] = "1080", ["brightness"] = "0.1"
        };
        var scene = backend.Analyze(new VisualInput("a bright sunny day", metadata));

        Assert.Equal(1920, scene.Width);
        Assert.Equal(1080, scene.Height);
        Assert.Equal(0.1, scene.Brightness, precision: 2);
    }

    [Fact]
    public void Backend_DarkScene_IsAppraisedAsLowBrightness()
    {
        var backend = new SimulatedVisionBackend();
        var scene = backend.Analyze(new VisualInput("a shadow, dark corner with no objects"));
        Assert.True(scene.Brightness < 0.3);
        Assert.Empty(scene.Objects);
        Assert.Contains("dark", scene.Caption, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_ProcessesImageStimulus_AndResponds()
    {
        using var mind = MindBuilder.Create("Seer").WithVision().WithMemory().Build();
        var response = mind.See("a red apple on a table");

        Assert.NotNull(mind.Vision);
        Assert.Equal(1, mind.Vision.ScenesSeen);
        Assert.NotNull(mind.Vision.LastScene);
        Assert.Contains("apple", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.True(response.Confidence > 0.4);
    }

    [Fact]
    public void Module_IgnoresTextStimulus()
    {
        using var mind = MindBuilder.Create("Seer").WithVision().Build();
        mind.Ask("just a plain question");
        Assert.Equal(0, mind.Vision!.ScenesSeen);
    }

    [Fact]
    public void Module_RemembersScenes()
    {
        using var mind = MindBuilder.Create("Seer").WithVision().WithMemory().Build();
        mind.See("a dog chasing a ball in the park");
        Assert.Contains(mind.Memory!.Episodes(), m => m.Source == "vision");
    }

    [Fact]
    public void Module_StrikingScene_Joy()
    {
        using var mind = MindBuilder.Create("Seer").WithVision().WithEmotions().Build();
        mind.See("a bright sunny garden with flowers, trees and a colorful bird");
        Assert.True(mind.EmotionalState!.Intensities.GetValueOrDefault(EmotionType.Joy) > 0,
            "a vivid bright scene should lift joy");
    }

    [Fact]
    public void CustomBackend_IsUsed()
    {
        var stub = new StubVisionBackend();
        using var mind = MindBuilder.Create("Seer")
            .WithVision(v => v.Backend = stub)
            .Build();

        mind.See("anything at all");
        Assert.Equal(1, stub.Calls);
        Assert.Same(stub.Scene, mind.Vision!.LastScene);
    }

    private sealed class StubVisionBackend : IVisionBackend
    {
        public int Calls;
        public readonly VisualScene Scene = new() { Caption = "stub caption" };
        public VisualScene Analyze(VisualInput input) { Calls++; return Scene; }
    }

    [Fact]
    public void See_Throws_WhenVisionDisabled()
    {
        using var mind = MindBuilder.Create("Blind").Build();
        Assert.Throws<MindConfigurationException>(() => mind.See("a cat"));
    }

    [Fact]
    public void VideoStimulus_AlsoTriggersVision()
    {
        using var mind = MindBuilder.Create("Seer").WithVision().Build();
        mind.Process(Stimulus.Video("a car driving down a road"));
        Assert.Equal(1, mind.Vision!.ScenesSeen);
    }
}
