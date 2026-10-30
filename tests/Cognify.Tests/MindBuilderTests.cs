using Cognify.Core;

namespace Cognify.Tests;

/// <summary>Tests for the fluent MindBuilder and the Mind kernel.</summary>
public class MindBuilderTests
{
    [Fact]
    public void Create_WithName_BuildsMind()
    {
        using var mind = MindBuilder.Create("TestMind").Build();
        Assert.Equal("TestMind", mind.Name);
        Assert.StartsWith("mind-", mind.Id);
        Assert.Equal(0, mind.CycleCount);
    }

    [Fact]
    public void Create_WithBlankName_Throws()
    {
        Assert.Throws<MindConfigurationException>(() => MindBuilder.Create("  ").Build());
    }

    [Fact]
    public void DefaultMind_HasNoModules()
    {
        using var mind = MindBuilder.Create("Minimal").Build();
        Assert.Empty(mind.Modules);
    }

    [Fact]
    public void WithAllModules_AttachesAllFourteen()
    {
        using var mind = MindBuilder.Create("Full").WithAllModules().Build();
        Assert.Equal(15, mind.Modules.Count); // the 14 capabilities + the self-improvement module riding on learning
        Assert.True(mind.HasModule<Cognify.Emotions.EmotionModule>());
        Assert.True(mind.HasModule<Cognify.Reasoning.ReasoningModule>());
        Assert.True(mind.HasModule<Cognify.Thinking.ThinkingModule>());
        Assert.True(mind.HasModule<Cognify.Learning.SelfLearningModule>());
        Assert.True(mind.HasModule<Cognify.Memory.MemoryModule>());
        Assert.True(mind.HasModule<Cognify.Goals.GoalModule>());
        Assert.True(mind.HasModule<Cognify.Personality.PersonalityModule>());
        Assert.True(mind.HasModule<Cognify.Perception.VisionModule>());
        Assert.True(mind.HasModule<Cognify.Audio.AudioModule>());
        Assert.True(mind.HasModule<Cognify.Embeddings.EmbeddingModule>());
        Assert.True(mind.HasModule<Cognify.Multilingual.MultilingualModule>());
        Assert.True(mind.HasModule<Cognify.Tools.ToolModule>());
        Assert.True(mind.HasModule<Cognify.Code.CodeModule>());
        Assert.True(mind.HasModule<Cognify.Generation.MediaGenerationModule>());
    }

    [Fact]
    public void Modules_RunInOrder()
    {
        using var mind = MindBuilder.Create("Ordered").WithAllModules().Build();
        var orders = mind.Modules.Select(m => m.Order).ToList();
        Assert.Equal(orders.OrderBy(o => o), orders);
    }

    [Fact]
    public void TypedAccessors_WorkWhenEnabled_AreNullWhenDisabled()
    {
        using var full = MindBuilder.Create("Full").WithAllModules().Build();
        Assert.NotNull(full.EmotionalState);
        Assert.NotNull(full.Knowledge);
        Assert.NotNull(full.Memory);
        Assert.NotNull(full.GoalSystem);
        Assert.NotNull(full.Learning);
        Assert.NotNull(full.Stream);
        Assert.NotNull(full.Personality);

        using var bare = MindBuilder.Create("Bare").Build();
        Assert.Null(bare.EmotionalState);
        Assert.Null(bare.Knowledge);
    }

    [Fact]
    public void Process_BareMind_ProducesFallbackResponse()
    {
        using var mind = MindBuilder.Create("Bare").Build();
        var response = mind.Process("hello world");
        Assert.False(string.IsNullOrWhiteSpace(response.Content));
        Assert.Equal(1, mind.CycleCount);
        Assert.Equal(-1, response.StyleId); // fallback style
    }

    [Fact]
    public void CycleCompleted_EventFires()
    {
        using var mind = MindBuilder.Create("Events").Build();
        MindResponse? captured = null;
        mind.CycleCompleted += (_, e) => captured = e.Response;
        mind.Process("trigger");
        Assert.NotNull(captured);
        Assert.Equal(1, captured!.CycleNumber);
    }

    [Fact]
    public void ModuleError_DoesNotStopCycle()
    {
        using var mind = MindBuilder.Create("Errors").Build();
        mind.AttachModule(new ExplodingModule());
        string? failed = null;
        mind.ModuleError += (_, e) => failed = e.ModuleName;
        var response = mind.Process("still works");
        Assert.Equal("exploder", failed);
        Assert.False(string.IsNullOrWhiteSpace(response.Content));
    }

    private sealed class ExplodingModule : CognitiveModuleBase
    {
        public override string Name => "exploder";
        public override int Order => 1;
        public override void Process(CognitiveContext context) =>
            throw new InvalidOperationException("boom");
    }

    [Fact]
    public void CustomModule_CanRespond()
    {
        using var mind = MindBuilder.Create("Custom")
            .WithModule(new EchoModule())
            .Build();
        var response = mind.Process("anything");
        Assert.Equal("echo:anything", response.Content);
        Assert.Equal(99, response.StyleId);
    }

    private sealed class EchoModule : CognitiveModuleBase
    {
        public override string Name => "echo";
        public override int Order => 99;
        public override void Process(CognitiveContext context) =>
            context.Response = new ResponseDraft($"echo:{context.Stimulus.Content}", 1.0, 99, "echo");
    }

    [Fact]
    public async Task ProcessAsync_Works()
    {
        using var mind = MindBuilder.Create("Async").Build();
        var response = await mind.ProcessAsync(Stimulus.Text("hi"));
        Assert.False(string.IsNullOrWhiteSpace(response.Content));
    }
}
