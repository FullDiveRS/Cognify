using Cognify.Core;
using Cognify.Emotions;
using Cognify.Generation;
using Xunit;

namespace Cognify.Tests;

/// <summary>
/// Lenient mode: convenience methods for capabilities that were not enabled
/// degrade gracefully instead of throwing MindConfigurationException.
/// Strict mode (default) keeps throwing so misconfigurations stay loud.
/// </summary>
public class LenientModeTests
{
    private static Mind LenientMind() =>
        MindBuilder.Create("Soft")
            .Describe("lenient test mind")
            .WithThinking()
            .WithMemory()
            .WithEmotions()
            .Lenient()
            .Build();

    [Fact]
    public void Lenient_Flag_IsExposed()
    {
        using var lenient = LenientMind();
        using var strict = MindBuilder.Create("Strict").WithThinking().Build();
        Assert.True(lenient.Lenient);
        Assert.False(strict.Lenient);
    }

    // ------------------------------------------------------------ See / Hear

    [Fact]
    public void See_Throws_WhenNotLenient()
    {
        using var mind = MindBuilder.Create("Strict").WithThinking().Build();
        Assert.Throws<MindConfigurationException>(() => mind.See("a cat"));
    }

    [Fact]
    public void Lenient_See_StillRunsFullCycle_WithoutVision()
    {
        using var mind = LenientMind();
        var response = mind.See("a red apple on a wooden table");
        Assert.NotNull(response);
        Assert.False(string.IsNullOrWhiteSpace(response.Content));
        Assert.Equal(1, mind.CycleCount); // the stimulus really went through the cycle
    }

    [Fact]
    public void Lenient_Hear_StillRunsFullCycle_WithoutAudio()
    {
        using var mind = LenientMind();
        var response = mind.Hear("a thunderstorm rolling in");
        Assert.NotNull(response);
        Assert.False(string.IsNullOrWhiteSpace(response.Content));
        Assert.Equal(1, mind.CycleCount);
    }

    [Fact]
    public void Lenient_See_NotesDegradation_InThoughtStream()
    {
        using var mind = LenientMind();
        mind.See("a small garden");
        Assert.Contains(mind.Stream!.Recent(50), s => s.Phase == "vision");
    }

    // ------------------------------------------------------- Teach / Reinforce

    [Fact]
    public void Lenient_Teach_ReturnsZero_ButKeepsMemory()
    {
        using var mind = LenientMind();
        int learned = mind.Teach("birds are animals");
        Assert.Equal(0, learned);                      // no knowledge base
        Assert.True(mind.Memory!.StrengthOf("birds are animals") > 0); // kept as memory
    }

    [Fact]
    public void Lenient_Reinforce_DoesNotThrow_AndStillFeels()
    {
        using var mind = LenientMind();
        mind.Reinforce(0.8); // no learning module — must not throw

        // The reward still lands emotionally even though nothing was trained.
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Joy] > 0);
    }

    // ----------------------------------------------------------------- Tools

    [Fact]
    public void Lenient_InvokeTool_ReturnsFailedResult()
    {
        using var mind = LenientMind();
        var result = mind.InvokeTool("clock");
        Assert.False(result.Success);
        Assert.Contains("not enabled", result.Output);
    }

    // -------------------------------------------------- Generation / language

    [Fact]
    public void Lenient_GenerateMedia_ReturnsNull()
    {
        using var mind = LenientMind();
        Assert.Null(mind.GenerateMedia("a painting", MediaType.Image));
    }

    [Fact]
    public void Lenient_DetectLanguage_ReturnsNull()
    {
        using var mind = LenientMind();
        Assert.Null(mind.DetectLanguage("bonjour le monde"));
    }

    [Fact]
    public void Lenient_SemanticSimilarity_ReturnsZero()
    {
        using var mind = LenientMind();
        Assert.Equal(0, mind.SemanticSimilarity("hello", "hello"));
    }

    // ------------------------------------------------- Strict mode still loud

    [Fact]
    public void Strict_Mode_StillThrows_ForEveryGatedCall()
    {
        using var mind = MindBuilder.Create("Strict").WithThinking().Build();
        Assert.Throws<MindConfigurationException>(() => mind.Teach("birds are animals"));
        Assert.Throws<MindConfigurationException>(() => mind.Reinforce(0.5));
        Assert.Throws<MindConfigurationException>(() => mind.InvokeTool("clock"));
        Assert.Throws<MindConfigurationException>(() => mind.GenerateMedia("x", MediaType.Image));
        Assert.Throws<MindConfigurationException>(() => mind.DetectLanguage("hola"));
        Assert.Throws<MindConfigurationException>(() => mind.SemanticSimilarity("a", "b"));
    }

    // ------------------------------------------ Lenient + persistence roundtrip

    [Fact]
    public async Task Lenient_Mind_SavesAndRestores()
    {
        var path = Path.Combine(Path.GetTempPath(), $"cognify-lenient-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilder.Create("Softy").WithThinking().WithMemory().Lenient()
                       .WithPersistence(path).Build())
            {
                mind.See("a quiet lake"); // would throw in strict mode without vision
                await mind.SaveAsync();
            }

            using var restored = MindBuilder.Restore(path);
            Assert.Equal("Softy", restored.Name);
            // What the lenient mind "saw" landed in the SQLite output.
            Assert.Contains(restored.Memory!.Recall("lake", 5), m => m.Content.Contains("lake"));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
