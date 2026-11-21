using System.Text.Json;
using System.Text.RegularExpressions;
using Cognify.Core;
using Cognify.Export;
using Cognify.Reasoning;

namespace Cognify.Tests;

/// <summary>
/// REASONING TOKEN BUDGETS AND LEVELS, LINED UP WITH THINKING — the
/// reasoning module now shares the thinking ladder: the same five levels,
/// the same ceiling, the same default budgets (Low 50 … Max 400), a global
/// one-size budget override, a per-level ladder, and a level→depth ladder
/// for forward chaining (Medium keeps the historical 6 passes). Its
/// explanation trace appears ALONGSIDE thinking in the chain of thought —
/// including when thinking is off — and the exported GGUF carries thinking
/// + reasoning levels (llama.cpp effort names) with both budget ladders in
/// the manifest. Mirrors CognifyPy's TestReasoningBudgets.
/// </summary>
public class ReasoningBudgetTests
{
    private static Mind BuildMind(Action<ReasoningOptions>? configure = null,
        bool thinking = true) =>
        MindBuilder.Create("Reasoner")
            .WithReasoning(configure)
            .WithMemory()
            .WithThinking(enabled: thinking)
            .Build();

    // ---- ladder API ------------------------------------------------------

    [Fact]
    public void Default_budgets_match_thinking_level_for_level()
    {
        foreach (ThinkingLevel level in Enum.GetValues<ThinkingLevel>())
            Assert.Equal(ThinkingOptions.DefaultBudget(level),
                         ReasoningOptions.DefaultBudget(level.FromThinking()));
        var r = new ReasoningOptions();
        Assert.Equal(ReasoningLevel.Medium, r.Level);
        Assert.Equal(100, r.TokensFor(ReasoningLevel.Medium));
        Assert.Equal(ReasoningCeiling.Only5, r.Ceiling);
    }

    [Fact]
    public void Reasoning_level_ladder_mirrors_thinking_rung_for_rung()
    {
        // ReasoningLevel is its own enum with the SAME rungs in the SAME
        // order — the cross-ladder maps must round-trip 1:1.
        var thinking = Enum.GetValues<ThinkingLevel>();
        var reasoning = Enum.GetValues<ReasoningLevel>();
        Assert.Equal(thinking.Length, reasoning.Length);
        for (int i = 0; i < thinking.Length; i++)
        {
            Assert.Equal((int)thinking.GetValue(i)!, (int)reasoning.GetValue(i)!);
            Assert.Equal(thinking.GetValue(i),
                ((ReasoningLevel)reasoning.GetValue(i)!).ToThinking());
        }
        Assert.Equal(ReasoningCeiling.Only4,
            ThinkingCeiling.Only4.FromThinking());
    }

    [Fact]
    public void Global_budget_overrides_every_level()
    {
        var r = new ReasoningOptions().Budget(120);
        foreach (ReasoningLevel level in Enum.GetValues<ReasoningLevel>())
            Assert.Equal(120, r.TokensFor(level));
        // Budget(0) clears the global — the ladder takes over again.
        Assert.Equal(400, r.Budget(0).TokensFor(ReasoningLevel.Max));
    }

    [Fact]
    public void Per_level_budget_writes_and_clears_with_ceiling_clamp()
    {
        var r = new ReasoningOptions().UpTo(ReasoningCeiling.Only2);
        r.Budget(ReasoningLevel.Max, 600);
        Assert.Equal(600, r.TokensFor(ReasoningLevel.Max));      // clamped onto Medium
        Assert.Equal(600, r.TokensFor(ReasoningLevel.Medium));
        r.Budget(ReasoningLevel.Medium, 0);                      // zero clears the entry
        Assert.Equal(100, r.TokensFor(ReasoningLevel.Medium));   // default ladder again
    }

    [Fact]
    public void Level_sizes_depth_ladder_and_Depth_pins_it()
    {
        var r = new ReasoningOptions();
        Assert.Equal(6, r.EffectiveDepth);                      // Medium keeps history
        r.At(ReasoningLevel.Low);
        Assert.Equal(2, r.EffectiveDepth);
        r.At(ReasoningLevel.Max);
        Assert.Equal(12, r.MaxInferenceDepth);
        r.Depth(4);                                             // explicit pin wins…
        Assert.Equal(4, r.EffectiveDepth);
        Assert.True(r.DepthIsCustom);
        r.At(ReasoningLevel.Low);                                // …and stops the ladder
        Assert.Equal(4, r.EffectiveDepth);
    }

    [Fact]
    public void Ceiling_clamps_the_level()
    {
        var r = new ReasoningOptions().UpTo(ReasoningCeiling.Only2).At(ReasoningLevel.Max);
        Assert.Equal(ReasoningLevel.Medium, r.Level);
    }

    // ---- cycle -----------------------------------------------------------

    [Fact]
    public void Reasoning_trace_appears_alongside_thinking_in_the_chain()
    {
        using var mind = BuildMind();
        mind.Teach("birds are animals");
        var response = mind.Process("animals are living things");  // learns in-cycle

        // Reasoning spent its budget and produced a trace line…
        Assert.NotNull(response.Reasoning);
        Assert.Contains(response.Reasoning!.Explanations, e => e.StartsWith("Learned"));
        Assert.Equal(100, response.Reasoning!.BudgetLimit);
        Assert.True(response.Reasoning!.BudgetUsed > 0);

        // …and thinking surfaced it ALONGSIDE its own phases.
        Assert.Contains(response.Chain.Steps, s => s.Text.StartsWith("Reasoning:"));
        Assert.Contains(response.Chain.Steps, s => s.Phase == ThoughtPhase.Observe);
        Assert.Contains(response.Chain.Steps, s => s.Phase == ThoughtPhase.Hypothesize);
    }

    [Fact]
    public void Reasoning_trace_survives_thinking_disabled_at_build_time()
    {
        // No thinking module attached at all: there is no chain of thought,
        // but reasoning still ran and its budgeted trace rides on the
        // response for hosts to show.
        using var mind = BuildMind(thinking: false);
        mind.Teach("birds are animals");
        var response = mind.Process("animals are living things");
        Assert.Null(response.Chain);
        Assert.NotNull(response.Reasoning);
        Assert.Contains(response.Reasoning!.Explanations, e => e.StartsWith("Learned"));
        Assert.Equal(100, response.Reasoning!.BudgetLimit);
    }

    [Fact]
    public void Reasoning_trace_appears_in_the_chain_when_thinking_switches_off_at_runtime()
    {
        // Module attached but the switch flipped off mid-session: the honest
        // one-step answer path still carries the reasoning trace.
        using var mind = BuildMind();
        mind.Config.Thinking.Enabled = false;   // same options object the module reads
        mind.Teach("birds are animals");
        var response = mind.Process("animals are living things");
        Assert.NotNull(response.Chain);
        Assert.Contains(response.Chain!.Steps, s => s.Text.StartsWith("Reasoning:"));
    }

    [Fact]
    public void Tiny_budget_trims_the_trace_and_records_the_spend()
    {
        using var mind = BuildMind(r => r.Budget(1));   // one token — nothing fits
        mind.Teach("birds are animals");
        var response = mind.Process("animals are living things");

        Assert.NotNull(response.Reasoning);
        Assert.Empty(response.Reasoning!.Explanations);   // trimmed, not dropped facts
        Assert.Equal(1, response.Reasoning!.BudgetLimit);
        Assert.Equal(0, response.Reasoning!.BudgetUsed);
        // The knowledge base is untouched — only the visible trace is trimmed.
        Assert.True(mind.Reasoning!.Knowledge.Query("animal", null, null).Count > 0);
    }

    // ---- builder ---------------------------------------------------------

    [Fact]
    public void WithCognitiveLevel_sets_both_modules_when_enabled()
    {
        using var mind = MindBuilder.Create("Leveled")
            .WithReasoning().WithMemory().WithThinking()
            .WithCognitiveLevel(ThinkingLevel.VeryHigh)
            .Build();
        Assert.Equal(ThinkingLevel.VeryHigh, mind.Config.Thinking.Level);
        Assert.Equal(ReasoningLevel.VeryHigh, mind.Config.Reasoning.Level);
        Assert.Equal(10, mind.Config.Reasoning.MaxInferenceDepth);   // VeryHigh depth
        Assert.Equal(16, mind.Config.Thinking.MaxThoughtSteps);      // VeryHigh steps
    }

    [Fact]
    public void WithCognitiveLevel_clamps_and_skips_disabled_modules()
    {
        using var mind = MindBuilder.Create("Clamped")
            // Reasoning configured at High but built DISABLED — the shared
            // level call must leave its options untouched.
            .WithReasoning(r => r.At(ReasoningLevel.High), enabled: false)
            .WithThinking()
            .WithCognitiveLevel(ThinkingLevel.Max, ThinkingCeiling.Only2)
            .Build();
        Assert.Equal(ThinkingLevel.Medium, mind.Config.Thinking.Level);  // clamped
        Assert.Equal(ReasoningLevel.High, mind.Config.Reasoning.Level);   // skipped
    }

    // ---- export (llama.cpp) ----------------------------------------------

    [Fact]
    public void Exported_gguf_carries_levels_and_budget_ladders_for_both_modules()
    {
        var dir = Path.Combine(Path.GetTempPath(), "cognify-reasoning-levels", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using var mind = MindBuilder.Create("ExportLevels")
                .WithReasoning()
                .WithMemory()
                .WithThinking()
                .Build();
            string corpus = Path.Combine(dir, "levels.txt");
            File.WriteAllLines(corpus, new[]
            {
                "ExportLevels is quiet.", "ExportLevels likes code.", "The sky is blue."
            });
            mind.TrainFile(corpus);
            mind.TrainDeepModel(1);

            string path = Path.Combine(dir, "levels.gguf");
            MindExporter.ExportGgufForLlamaCpp(mind, path);
            var gguf = new GgufReader(path);

            // KV metadata — llama.cpp-aligned effort names + token budgets.
            Assert.True(gguf.Metadata["cognify.reasoning.enabled"].AsBool());
            Assert.Equal("medium", gguf.Metadata["cognify.reasoning.level"].AsString());
            Assert.Equal(100u, gguf.Metadata["cognify.reasoning.budget"].AsUInt32());
            Assert.Equal(100u, gguf.Metadata["cognify.thinking.budget"].AsUInt32());

            // Manifest — riding in the chat template, readable from /props:
            // level ladders + per-level budgets for BOTH modules.
            var template = gguf.Metadata["tokenizer.chat_template"].AsString();
            var m = Regex.Match(template, @"cognify_manifest = '(.*?)' %\}");
            Assert.True(m.Success);
            using var doc = JsonDocument.Parse(m.Groups[1].Value.Replace("\\u0027", "'"));
            var thinking = doc.RootElement.GetProperty("thinking");
            var reasoning = doc.RootElement.GetProperty("reasoning");
            Assert.Equal(gguf.Metadata["cognify.thinking.level"].AsString(),
                         thinking.GetProperty("level").GetString());
            Assert.Equal("medium", reasoning.GetProperty("level").GetString());
            Assert.Equal(100, reasoning.GetProperty("budget").GetInt32());
            Assert.Equal("low", reasoning.GetProperty("levels")[0].GetString());
            Assert.Equal("xhigh", reasoning.GetProperty("levels")[3].GetString()); // VeryHigh → xhigh
            Assert.Equal("max", reasoning.GetProperty("levels")[4].GetString());
            Assert.Equal(50, reasoning.GetProperty("budgets").GetProperty("low").GetInt32());
            Assert.Equal(400, thinking.GetProperty("budgets").GetProperty("max").GetInt32());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Exported_gguf_carries_a_vocabulary_sized_sampling_profile()
    {
        var dir = Path.Combine(Path.GetTempPath(), "cognify-sampling", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using var mind = MindBuilder.Create("SamplingMeta")
                .WithReasoning()
                .WithMemory()
                .WithThinking()
                .Build();
            string corpus = Path.Combine(dir, "sampling.txt");
            File.WriteAllLines(corpus, new[]
            {
                "SamplingMeta is quiet.", "SamplingMeta likes code.", "The sky is blue."
            });
            mind.TrainFile(corpus);
            mind.TrainDeepModel(1);

            string path = Path.Combine(dir, "sampling.gguf");
            MindExporter.ExportGgufForLlamaCpp(mind, path);
            var gguf = new GgufReader(path);

            // Tiny vocabulary (greeting-scale) → the tightest tier: a low
            // temperature, a strong repetition penalty and a hard reply cap —
            // the exact settings that stop the "what you is what you" loop.
            Assert.Equal(0.7f, gguf.Metadata["cognify.sampling.temperature"].AsFloat32());
            Assert.Equal(0.9f, gguf.Metadata["cognify.sampling.top_p"].AsFloat32());
            Assert.Equal(40u, gguf.Metadata["cognify.sampling.top_k"].AsUInt32());
            Assert.Equal(1.2f, gguf.Metadata["cognify.sampling.repeat_penalty"].AsFloat32());
            Assert.Equal(128u, gguf.Metadata["cognify.sampling.max_tokens"].AsUInt32());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
