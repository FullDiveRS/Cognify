using Cognify.Core;
using Cognify.Thinking;
using Cognify.Tools;

namespace Cognify.Tests;

/// <summary>
/// Tests for the conversational + web batch: the /searchweb focus
/// suggestion (arguments, not just a bare query), Mind.ExploreUrl (her own
/// WebFetch + site walk), the knowledge-gap AUTO lookup (WithAutoWebFetch),
/// THINKING AUTOMATIC (she picks her own effort per topic) and the
/// Conversational options (rate → chattier replies, understand → topic
/// shift notes).
/// </summary>
public class ConversationalWebTests
{
    /// <summary>A deterministic offline backend for tests.</summary>
    private sealed class FakeBackend : IWebSearchBackend
    {
        public int Calls;
        public string LastQuery = "";
        public List<WebSearchResult> Results { get; set; } = new();
        public bool FailWithException;

        public IReadOnlyList<WebSearchResult> Search(string query, int maxResults)
        {
            Calls++;
            LastQuery = query;
            if (FailWithException)
                throw new WebSearchException("web search could not complete (offline)");
            return Results.Take(maxResults).ToList();
        }
    }

    private static readonly List<WebSearchResult> TwoHits = new()
    {
        new WebSearchResult("Quokka habitat", "https://a.example/quokka",
            "Quokkas live on small islands off the coast."),
        new WebSearchResult("Quokka diet", "https://b.example/quokka-diet",
            "Quokkas eat grass and leaves. The quokka diet is mostly plants."),
    };

    // ------------------------------------------------------------------
    // /searchweb "query" suggestion — the focus argument
    // ------------------------------------------------------------------

    [Fact]
    public void SearchWeb_Focus_ReordersReportedResults()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithWebSearch(t => t.WebSearchBackend = backend)
            .Build();

        var result = mind.SearchWeb("quokkas", "what do quokkas eat");
        Assert.True(result.Success, result.Output);
        Assert.Equal("quokkas", backend.LastQuery);          // quoted part stays the query

        // The hit matching the focus (diet/eat) is reported FIRST.
        int dietIndex = result.Output.IndexOf("Quokka diet", StringComparison.Ordinal);
        int habitatIndex = result.Output.IndexOf("Quokka habitat", StringComparison.Ordinal);
        Assert.True(dietIndex >= 0 && habitatIndex >= 0);
        Assert.True(dietIndex < habitatIndex, "focus-matching hit should be reported first");
        Assert.Equal("what do quokkas eat", result.Data["focus"]);
    }

    [Fact]
    public void SearchWeb_Focus_LearnsFocusedHitFirst()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithWebSearch(t => t.WebSearchBackend = backend)
            .Build();

        var result = mind.SearchWeb("quokkas", "what do quokkas eat");
        Assert.True(result.Success, result.Output);

        var memories = mind.Memory!.LongTermMemories()
            .Where(m => m.Source == "web").ToList();
        Assert.NotEmpty(memories);
        Assert.Contains("Quokka diet", memories[0].Content);
    }

    [Fact]
    public void SearchWeb_NoFocus_KeepsEngineOrder()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithWebSearch(t => t.WebSearchBackend = backend)
            .Build();

        var result = mind.SearchWeb("quokkas");
        Assert.True(result.Success, result.Output);
        int habitatIndex = result.Output.IndexOf("Quokka habitat", StringComparison.Ordinal);
        int dietIndex = result.Output.IndexOf("Quokka diet", StringComparison.Ordinal);
        Assert.True(habitatIndex < dietIndex, "no focus → engine order preserved");
    }

    // ------------------------------------------------------------------
    // /explore — her own WebFetch + site walk
    // ------------------------------------------------------------------

    [Fact]
    public void ExploreUrl_InvalidUrl_FailsGracefully()
    {
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithWebSearch()
            .Build();

        var result = mind.ExploreUrl("not-a-url", "flowers");
        Assert.False(result.Success);
        Assert.Contains("not a web url", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ExploreUrl_UnreachableHost_FailsGracefully()
    {
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithWebSearch()
            .Build();

        // .invalid TLD never resolves (RFC 2606) — deterministic offline path.
        var result = await Task.Run(() =>
            mind.ExploreUrl("https://cognify-tests.invalid/page", "flowers"));
        Assert.False(result.Success);
        Assert.Contains("could not read anything", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExploreUrl_WithoutTools_ThrowsUnlessLenient()
    {
        using var strict = MindBuilder.Create("Strict").WithReasoning().Build();
        Assert.Throws<MindConfigurationException>(() => strict.ExploreUrl("https://a.example/x"));

        using var lenient = MindBuilder.Create("Lenient").WithReasoning().Lenient().Build();
        var result = lenient.ExploreUrl("https://a.example/x");
        Assert.False(result.Success);
    }

    // ------------------------------------------------------------------
    // THINKING AUTOMATIC — she picks her own effort per cycle
    // ------------------------------------------------------------------

    [Fact]
    public void Thinking_Automatic_ShortSmalltalk_ThinksLow()
    {
        using var mind = MindBuilder.Create("Thinker")
            .WithReasoning().WithMemory()
            .WithThinking(t => t.AutomaticOn())
            .Build();

        var response = mind.Process("hi");
        Assert.Contains(response.Chain!.Steps,
            s => s.Text.Contains("Automatic: chose to think at Low"));
    }

    [Fact]
    public void Thinking_Automatic_LongWhyQuestion_ThinksHard()
    {
        using var mind = MindBuilder.Create("Thinker")
            .WithReasoning().WithMemory()
            .WithThinking(t => t.AutomaticOn())
            .Build();

        var response = mind.Process(
            "Why does the sky look blue in the day and red at sunset, and how " +
            "does the light scatter through the atmosphere to make those colors happen?");
        Assert.Contains(response.Chain!.Steps,
            s => s.Text.Contains("Automatic: chose to think at VeryHigh") ||
                 s.Text.Contains("Automatic: chose to think at Max"));
    }

    [Fact]
    public void Thinking_Automatic_RespectsCeiling()
    {
        using var mind = MindBuilder.Create("Thinker")
            .WithReasoning().WithMemory()
            .WithThinking(t => t.AutomaticOn(), ceiling: ThinkingCeiling.Only1)
            .Build();

        var response = mind.Process(
            "Why does the sky look blue in the day and red at sunset, and how " +
            "does the light scatter through the atmosphere to make those colors happen?");
        // The choice happened, but the ceiling clamps it down to Low.
        Assert.Contains(response.Chain!.Steps,
            s => s.Text.Contains("Automatic: chose to think at Low"));
    }

    [Fact]
    public void Thinking_Automatic_Off_KeepsFixedLevel()
    {
        using var mind = MindBuilder.Create("Thinker")
            .WithReasoning().WithMemory()
            .WithThinking(t => t.At(ThinkingLevel.Medium))
            .Build();

        var response = mind.Process(
            "Why does the sky look blue in the day and red at sunset, and how " +
            "does the light scatter through the atmosphere to make those colors happen?");
        Assert.DoesNotContain(response.Chain!.Steps,
            s => s.Text.Contains("Automatic:"));
    }

    // ------------------------------------------------------------------
    // WithConversational — rate + understand
    // ------------------------------------------------------------------

    [Fact]
    public void Conversational_Understand_RemembersTopicShifts()
    {
        using var mind = MindBuilder.Create("Chatty")
            .WithReasoning().WithMemory()
            .WithConversational(c => c.understand())
            .Build();

        mind.Process("tell me about quokkas");
        mind.Process("let us talk about music now");

        var notes = mind.Memory!.LongTermMemories()
            .Where(m => m.Source == "conversation").ToList();
        Assert.NotEmpty(notes);
        Assert.Contains(notes, m =>
            m.Content.Contains("quokkas") && m.Content.Contains("music"));
    }

    [Fact]
    public void Conversational_UnderstandOff_NoTopicNotes()
    {
        using var mind = MindBuilder.Create("Quiet")
            .WithReasoning().WithMemory()
            .WithConversational(c => c.understand(false))
            .Build();

        mind.Process("tell me about quokkas");
        mind.Process("let us talk about music now");

        Assert.Empty(mind.Memory!.LongTermMemories()
            .Where(m => m.Source == "conversation"));
    }

    [Fact]
    public void Conversational_Rate_MakesRepliesLonger()
    {
        string[] inputs =
        {
            "tell me about quokkas",
            "let us talk about music",
            "I love painting",
            "rain is nice today",
            "books make me happy",
            "tell me about oceans",
        };

        double TotalWords(double rate)
        {
            using var mind = MindBuilder.Create("Chatty")
                .WithReasoning().WithMemory()
                .WithConversational(c => c.rate(rate))
                .Build();
            double words = 0;
            foreach (var line in inputs)
                words += mind.Process(line).Content.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
            return words;
        }

        double baseWords = TotalWords(1.0);
        double chattyWords = TotalWords(2.0);
        Assert.True(chattyWords > baseWords,
            $"rate 2.0 should speak more than rate 1.0 ({chattyWords} vs {baseWords})");
    }

    // ------------------------------------------------------------------
    // WithAutoWebFetch — the knowledge-gap lookup
    // ------------------------------------------------------------------

    [Fact]
    public void AutoWebFetch_UnknownTopic_SearchesLearnsAndAnswers()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithAutoWebFetch(t => t.WebSearchBackend = backend)
            .Build();

        var response = mind.Process("what is a quokka?");

        Assert.Equal(1, backend.Calls);
        Assert.Contains("searched the web", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Quokka", response.Content);
        // She learned from what she found.
        Assert.Contains(mind.Memory!.LongTermMemories(), m => m.Source == "web");
        // Explainable: the auto path is recorded on the response.
        Assert.Equal("auto-web-lookup", response.StyleName);
    }

    [Fact]
    public void AutoWebFetch_SameTopic_LooksUpOnlyOnce()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithAutoWebFetch(t => t.WebSearchBackend = backend)
            .Build();

        mind.Process("what is a quokka?");
        int afterFirst = backend.Calls;
        Assert.Equal(1, afterFirst);

        mind.Process("what is a quokka?");
        Assert.Equal(1, backend.Calls); // one lookup per unfamiliar topic
    }

    [Fact]
    public void AutoWebFetch_KnownTopic_NeverSearches()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithAutoWebFetch(t => t.WebSearchBackend = backend)
            .Build();

        mind.Teach("quokkas are marsupials");
        var response = mind.Process("what is a quokka?");

        Assert.Equal(0, backend.Calls);   // she knew the answer — no web trip
        Assert.DoesNotContain("searched the web", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AutoWebFetch_PersonalAndMathTalk_NeverSearches()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithAutoWebFetch(t => t.WebSearchBackend = backend)
            .Build();

        mind.Process("what do you like to talk about?");
        mind.Process("what is 2 + 2?");
        Assert.Equal(0, backend.Calls);
    }

    [Fact]
    public void AutoWebFetch_FailedLookup_StaysSilent()
    {
        var backend = new FakeBackend { Results = TwoHits, FailWithException = true };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithAutoWebFetch(t => t.WebSearchBackend = backend)
            .Build();

        var response = mind.Process("what is a quokka?");

        Assert.Equal(1, backend.Calls);   // she TRIED
        // …but a failed lookup never hijacks the reply — the normal honest
        // composition runs, no pretend search report.
        Assert.DoesNotContain("searched the web", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AutoWebFetch_Off_NeverSearches()
    {
        var backend = new FakeBackend { Results = TwoHits };
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning().WithMemory()
            .WithWebSearch(t => t.WebSearchBackend = backend) // manual-only, no auto
            .Build();

        var response = mind.Process("what is a quokka?");
        Assert.Equal(0, backend.Calls);
        Assert.DoesNotContain("searched the web", response.Content, StringComparison.OrdinalIgnoreCase);
    }
}
