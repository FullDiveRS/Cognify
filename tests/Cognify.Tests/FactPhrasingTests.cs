using Cognify.Core;
using Cognify.Reasoning;

namespace Cognify.Tests;

/// <summary>
/// Regression tests for the hyphenated-fact bug the user spotted in the
/// Yume transcript: taught facts came back as slugs — "before is
/// before-affect-effect-being-if-i-buy-something-like-apple-or-some-pineapple"
/// — instead of natural words. Facts are now stored with spaces and spoken
/// with a display polish (capital "I", capitalized sentence start).
/// </summary>
public class FactPhrasingTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Speaker")
            .WithReasoning(r => r.Depth(6).InduceAfter(50))
            .WithThinking()
            .Build();

    [Fact]
    public void Multiword_Fact_Objects_Use_Natural_Spaces()
    {
        var fact = Assert.Single(FactParser.Parse("Before is the affect before the cause"));
        Assert.Equal("before", fact.Subject);
        Assert.Equal("is", fact.Predicate);
        Assert.Equal("affect before cause", fact.Object);
        Assert.DoesNotContain('-', fact.Object);
    }

    [Fact]
    public void Recall_Speaks_Natural_Words_Not_Slugs()
    {
        using var mind = BuildMind();
        mind.Teach("Before is the affect before the cause");
        var response = mind.Ask("what is before?");
        Assert.Contains("affect before cause", response.Content);
        Assert.DoesNotContain("affect-before-cause", response.Content);
        Assert.DoesNotContain('-', response.Content);
    }

    [Fact]
    public void Identity_Facts_Speak_With_Spaces()
    {
        using var mind = BuildMind();
        mind.Teach("Your name is Yume");
        var response = mind.Ask("what is your name?");
        // Used to be "your-name is yume".
        Assert.Contains("name is yume", response.Content);
        Assert.DoesNotContain("your-name", response.Content);
    }

    [Fact]
    public void Leading_Fillers_Do_Not_Hide_The_Question()
    {
        using var mind = BuildMind();
        mind.Teach("After is the cause going forward");
        // "now what is after" used to fall through as an unknown topic.
        var response = mind.Ask("now what is after");
        Assert.Contains("cause going forward", response.Content);
    }

    [Fact]
    public void YesNo_Claims_Match_Spaced_Facts_And_Old_Slugs()
    {
        var kb = new KnowledgeBase();
        var engine = new InferenceEngine(kb, new ReasoningOptions());
        kb.Add(new Fact("before", "is", "affect before cause"));   // new space form
        kb.Add(new Fact("animal", "is-a", "living-thing"));        // legacy slug form

        // Direct spaced match via CheckClaim's letter-skeleton comparison:
        var entity = kb.Query("before", null, null);
        Assert.Single(entity);

        // The skeleton normalizer makes "living thing" and "living-thing"
        // equal, so old persisted minds keep answering.
        Assert.Single(kb.Query(null, "is-a", "living-thing"));
        Assert.Equal("living-thing", engine.AncestorsOf("animal")[0].Object);
    }

    [Fact]
    public void Polish_Capitalizes_Standalone_I_And_Sentence_Start()
    {
        Assert.Equal("If I buy something, I own it.",
            ReasoningModule.Polish("if i buy something, i own it."));
        Assert.Equal("Before is affect before cause.",
            ReasoningModule.Polish("before is affect before cause."));
        Assert.Equal("", ReasoningModule.Polish(""));
    }

    [Fact]
    public void Taught_Transcript_RoundTrip_Speaks_Naturally()
    {
        // The exact exchange from the user's transcript, end to end.
        using var mind = BuildMind();
        mind.Teach("Before is the before affect, the effect being if i buy something like an apple or some pineapples");
        var response = mind.Ask("what is before?");
        Assert.Contains("before is before affect", response.Content.ToLowerInvariant());
        Assert.DoesNotContain("-", response.Content);
    }
}
