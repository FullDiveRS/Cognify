using Cognify.Core;
using Cognify.Emotions;
using Cognify.Memory;

namespace Cognify.Tests;

/// <summary>End-to-end persistence tests: save a mind to SQLite, restore it, verify state.</summary>
public class PersistenceTests : IDisposable
{
    private readonly string _path = System.IO.Path.Combine(
        System.IO.Path.GetTempPath(), $"cognify-test-{Guid.NewGuid():N}.db");

    public void Dispose()
    {
        if (File.Exists(_path)) File.Delete(_path);
    }

    [Fact]
    public async Task Save_CreatesFile()
    {
        using (var mind = BuildMind())
            await mind.SaveAsync(_path);
        Assert.True(File.Exists(_path));
        Assert.True(new FileInfo(_path).Length > 0);
    }

    [Fact]
    public async Task Restore_KeepsKnowledge()
    {
        using (var mind = BuildMind())
        {
            mind.Teach("birds are animals");
            mind.Teach("animals are living things");
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal(2, restored.Knowledge!.FactCount);
        var answer = restored.Ask("what is a bird?");
        Assert.Contains("animal", answer.Content);
    }

    [Fact]
    public async Task Restore_KeepsEmotions()
    {
        using (var mind = BuildMind())
        {
            mind.EmotionalState!.Feel(EmotionType.Joy, 0.8);
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.True(restored.EmotionalState!.Intensities[EmotionType.Joy] > 0.7);
    }

    [Fact]
    public async Task Restore_KeepsMemories()
    {
        using (var mind = BuildMind())
        {
            mind.Memory!.Remember("the secret code is 1234", MemoryKind.LongTerm, importance: 0.9);
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal(1, restored.Memory!.Counts.LongTerm);
        var recalled = restored.Memory.Recall("secret code", max: 1);
        Assert.NotEmpty(recalled);
        Assert.Contains("1234", recalled[0].Content);
    }

    [Fact]
    public async Task Restore_KeepsGoals()
    {
        using (var mind = BuildMind())
        {
            mind.GoalSystem!.AddGoal("restore-test", "verify goals", 0.7, new[] { "curiosity" });
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Contains(restored.GoalSystem!.AllGoals(), g => g.Name == "restore-test");
    }

    [Fact]
    public async Task Restore_KeepsLearningStats()
    {
        using (var mind = BuildMind())
        {
            mind.Process("something to learn from");
            mind.Reinforce(0.7);
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal(1, restored.Learning!.Stats.TotalExperiences);
        Assert.True(restored.Learning.Stats.TrainingSteps > 0);
    }

    [Fact]
    public async Task Restore_NetworkWeightsMatch()
    {
        using (var mind = BuildMind())
        {
            mind.Process("train");
            mind.Reinforce(0.9);
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        var original = MindBuilder.Restore(_path); // load twice, compare predictions
        var input = new double[restored.Learning!.Net.InputDim];
        input[0] = 1.0;
        var a = restored.Learning.Net.Predict(input);
        var b = original.Learning!.Net.Predict(input);
        Assert.Equal(a, b);
        original.Dispose();
    }

    [Fact]
    public async Task Restore_KeepsPersonality()
    {
        using (var mind = MindBuilder.Create("Unique")
                   .WithAllModules()
                   .WithPersonality(p => p.Traits(openness: 0.85))
                   .WithPersistence(_path)
                   .Build())
        {
            await mind.SaveAsync();
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal(0.85, restored.Personality!.Openness, precision: 3);
    }

    [Fact]
    public void Restore_MissingFile_Throws()
    {
        Assert.Throws<PersistenceException>(() =>
            MindBuilder.Restore(System.IO.Path.Combine(
                System.IO.Path.GetTempPath(), $"no-such-{Guid.NewGuid():N}.db")));
    }

    [Fact]
    public async Task Save_ThenReload_WorksRepeatedly()
    {
        var taught = new List<string>();
        for (int i = 0; i < 3; i++)
        {
            taught.Add($"iteration fact {i} is testable");
            using (var mind = BuildMind())
            {
                foreach (var fact in taught) mind.Teach(fact);
                await mind.SaveAsync(_path);
            }
            using var restored = MindBuilder.Restore(_path);
            Assert.Equal(taught.Count, restored.Knowledge!.FactCount);
        }
    }

    [Fact]
    public async Task LoadFrom_AppliesToExistingMind()
    {
        using (var mind = BuildMind())
        {
            mind.Teach("whales are mammals");
            await mind.SaveAsync(_path);
        }

        using var other = BuildMind();
        await other.LoadFromAsync(_path);
        Assert.Equal(1, other.Knowledge!.FactCount);
    }

    [Fact]
    public async Task SaveRestore_KeepsSemanticMemory()
    {
        using (var mind = BuildMind())
        {
            mind.Ask("cats love to eat fish every day");
            mind.Ask("the mountain was covered in snow");
            Assert.Equal(2, mind.SemanticMemory!.Store.Count);
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal(2, restored.SemanticMemory!.Store.Count);

        // Vectors must survive: the neighbor search should still rank the
        // stored cat sentence above the unrelated mountain one.
        var hits = restored.SemanticMemory.FindSimilar("cats and their fish", topK: 2);
        Assert.NotEmpty(hits);
        Assert.Contains("cats", hits[0].Entry.Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task SaveRestore_KeepsToolAndVisionModules()
    {
        using (var mind = BuildMind())
        {
            mind.InvokeTool("calculator", new Dictionary<string, string> { ["expression"] = "3 * 4" });
            mind.See("a red apple on a table");
            await mind.SaveAsync(_path);
        }

        // Restore rebuilds the full module set: action/perception modules
        // must be present and functional afterwards.
        using var restored = MindBuilder.Restore(_path);
        Assert.True(restored.HasModule<Cognify.Tools.ToolModule>());
        Assert.True(restored.HasModule<Cognify.Perception.VisionModule>());

        var calc = restored.InvokeTool("calculator",
            new Dictionary<string, string> { ["expression"] = "20 + 22" });
        Assert.True(calc.Success);
        Assert.Equal(42, (double)calc.Data["value"], precision: 6);
    }

    [Fact]
    public void Build_WritesDatabaseOutput_Immediately()
    {
        // WithPersistence + Build(): the SQLite file must exist right away,
        // seeded with schema and the mind's initial state — before any
        // explicit Save call.
        Assert.False(File.Exists(_path));
        using (var mind = MindBuilder.Create("DbOutput")
                   .WithThinking()
                   .WithMemory()
                   .WithPersistence(_path)
                   .Build())
        {
            Assert.True(File.Exists(_path));
            Assert.True(new FileInfo(_path).Length > 0);
        }

        // The build-time output is a valid mind database: restorable as-is.
        using var restored = MindBuilder.Restore(_path);
        Assert.Equal("DbOutput", restored.Name);
    }

    [Fact]
    public async Task Build_DatabaseOutput_RoundTrips_State()
    {
        using (var mind = MindBuilder.Create("Grower")
                   .WithThinking()
                   .WithMemory()
                   .WithPersistence(_path)
                   .Build())
        {
            // The DB already existed at Build(); cycles keep using the same file.
            mind.Ask("the giraffe walked slowly across the savanna");
            await mind.SaveAsync();
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal("Grower", restored.Name);
        Assert.Contains(restored.Memory!.Recall("giraffe", 5),
            m => m.Content.Contains("giraffe"));
    }

    private static Mind BuildMind() =>
        MindBuilder.Create("Persisted")
            .WithAllModules()
            .Build();
}
