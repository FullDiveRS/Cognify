using Cognify.Core;
using Cognify.Reasoning;

namespace Cognify.Tests;

/// <summary>Tests for facts, rules and the inference engine.</summary>
public class ReasoningTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Logician")
            .WithReasoning(r => r.Depth(6).InduceAfter(50))
            .WithThinking()
            .Build();

    [Fact]
    public void FactParser_ParsesIsAre()
    {
        var facts = FactParser.Parse("birds are animals").ToList();
        var fact = Assert.Single(facts);
        Assert.Equal("bird", fact.Subject);
        Assert.Equal("is-a", fact.Predicate);
        Assert.Equal("animal", fact.Object);
    }

    [Fact]
    public void FactParser_ParsesHas()
    {
        var fact = Assert.Single(FactParser.Parse("a dog has fur"));
        Assert.Equal("dog", fact.Subject);
        Assert.Equal("has", fact.Predicate);
        Assert.Equal("fur", fact.Object);
    }

    [Fact]
    public void FactParser_ParsesCan()
    {
        var fact = Assert.Single(FactParser.Parse("cats can climb"));
        Assert.Equal("cat", fact.Subject);
        Assert.Equal("can", fact.Predicate);
    }

    [Fact]
    public void KnowledgeBase_AddQueryRoundtrip()
    {
        var kb = new KnowledgeBase();
        kb.Add(new Fact("bird", "is-a", "animal"));
        kb.Add(new Fact("dog", "is-a", "animal"));
        Assert.Single(kb.Query("bird", null, null));
        Assert.Equal(2, kb.Query(null, "is-a", "animal").Count);
        Assert.Single(kb.Query("dog", "is-a", "animal"));
        Assert.Empty(kb.Query("fish", null, null));
    }

    [Fact]
    public void KnowledgeBase_NoDuplicateFacts()
    {
        var kb = new KnowledgeBase();
        Assert.True(kb.Add(new Fact("a", "is", "b")));
        Assert.False(kb.Add(new Fact("a", "is", "b")));
        Assert.Equal(1, kb.FactCount);
    }

    [Fact]
    public void Teach_StoresFacts()
    {
        using var mind = BuildMind();
        int learned = mind.Teach("birds are animals");
        Assert.Equal(1, learned);
        Assert.Equal(1, mind.Knowledge!.FactCount);
    }

    [Fact]
    public void Teach_WithoutReasoning_Throws()
    {
        using var mind = MindBuilder.Create("NoReason").Build();
        Assert.Throws<MindConfigurationException>(() => mind.Teach("birds are animals"));
    }

    [Fact]
    public void Ask_WhatIs_AnswersFromKnowledge()
    {
        using var mind = BuildMind();
        mind.Teach("birds are animals");
        var response = mind.Ask("what is a bird?");
        Assert.Contains("animal", response.Content);
        Assert.NotNull(response.Reasoning?.Answer);
    }

    [Fact]
    public void Ask_HierarchicalDeduction()
    {
        using var mind = BuildMind();
        mind.Teach("birds are animals");
        mind.Teach("animals are living things");
        var response = mind.Ask("what is a bird?");
        // Facts are spoken in natural words now — "living thing", not the old
        // slug "living-thing".
        Assert.Contains("living thing", response.Content); // transitive is-a expansion
        Assert.DoesNotContain("living-thing", response.Content);
    }

    [Fact]
    public void Ask_UnknownEntity_GracefulResponse()
    {
        using var mind = BuildMind();
        var response = mind.Ask("what is a blorptastic?");
        Assert.NotNull(response);
        Assert.False(string.IsNullOrWhiteSpace(response.Content));
    }

    [Fact]
    public void ForwardChain_DerivesNewFacts()
    {
        using var mind = BuildMind();
        var reasoning = mind.Module<ReasoningModule>()!;
        reasoning.Learn(new Fact("tweety", "is-a", "bird"));
        reasoning.Learn(new LogicRule(
            new[] { new FactPattern("?s", "is-a", "bird") },
            new FactPattern("?s", "can", "fly")));
        var derived = reasoning.Engine.ForwardChain();
        Assert.Contains(derived, f => f.Subject == "tweety" && f.Predicate == "can" && f.Object == "fly");
    }

    [Fact]
    public void ForwardChain_TransitiveIsA()
    {
        var kb = new KnowledgeBase();
        var engine = new InferenceEngine(kb, new ReasoningOptions { MaxInferenceDepth = 6 });
        kb.Add(new Fact("a", "is-a", "b"));
        kb.Add(new Fact("b", "is-a", "c"));
        kb.AddRule(new LogicRule(
            new[] { new FactPattern("?x", "is-a", "?y"), new FactPattern("?y", "is-a", "?z") },
            new FactPattern("?x", "is-a", "?z")));
        var derived = engine.ForwardChain();
        Assert.Contains(derived, f => f.Subject == "a" && f.Object == "c");
    }

    [Fact]
    public void AncestorsOf_ReturnsChain()
    {
        var kb = new KnowledgeBase();
        var engine = new InferenceEngine(kb, new ReasoningOptions());
        kb.Add(new Fact("bird", "is-a", "animal"));
        kb.Add(new Fact("animal", "is-a", "living-thing"));
        var chain = engine.AncestorsOf("bird");
        Assert.Equal(2, chain.Count);
        Assert.Equal("animal", chain[0].Object);
        Assert.Equal("living-thing", chain[1].Object);
    }

    [Fact]
    public void Abduce_ExplainsObservation()
    {
        var kb = new KnowledgeBase();
        var engine = new InferenceEngine(kb, new ReasoningOptions { AllowAbduction = true });
        kb.AddRule(new LogicRule(
            new[] { new FactPattern("?s", "is-a", "bird") },
            new FactPattern("?s", "can", "fly")));
        kb.Add(new Fact("tweety", "is-a", "bird"));
        var explanations = engine.Abduce(new Fact("tweety", "can", "fly"));
        Assert.NotEmpty(explanations);
        Assert.Contains("bird", explanations[0]);
    }

    [Fact]
    public void Induce_CreatesRelatedToFact()
    {
        var kb = new KnowledgeBase();
        var engine = new InferenceEngine(kb, new ReasoningOptions());
        var coOccurrence = new Dictionary<(string, string), int> { [("coffee", "morning")] = 6 };
        var induced = engine.Induce(coOccurrence, threshold: 5);
        var fact = Assert.Single(induced);
        Assert.Equal("coffee", fact.Subject);
        Assert.Equal("related-to", fact.Predicate);
        Assert.Equal("morning", fact.Object);
        Assert.Equal("induced", fact.Source);
    }
}
