using Cognify.Core;
using Cognify.Tools;

namespace Cognify.Tests;

/// <summary>
/// Tests for the manual-only web-search tool: explicit invocation works,
/// auto-invocation is impossible, and successful searches feed the mind's
/// memory, knowledge and language model.
/// </summary>
public class WebSearchTests
{
    /// <summary>A deterministic offline backend for tests.</summary>
    private sealed class FakeBackend : IWebSearchBackend
    {
        public int Calls;
        public string LastQuery = "";
        public List<WebSearchResult> Results { get; set; } = new()
        {
            new WebSearchResult("Quokka", "https://en.wikipedia.org/wiki/Quokka",
                "The quokka is a small macropod about the size of a domestic cat. It is the only terrestrial mammal on Rottnest Island."),
        };
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

    private static Mind BuildMind(FakeBackend backend, bool exploration = false) =>
        MindBuilder.Create("Curious")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithWebSearch(t =>
            {
                t.WebSearchBackend = backend;
                // Exploration (reading INTO result pages) gets its own offline
                // test file — here we test the search half without a network.
                t.Exploration(e => e.Explore(exploration));
            })
            .Build();

    [Fact]
    public void SearchWeb_ReturnsResults()
    {
        var backend = new FakeBackend();
        using var mind = BuildMind(backend);

        var result = mind.SearchWeb("what is a quokka");
        Assert.True(result.Success, result.Output);
        Assert.Equal(1, backend.Calls);
        Assert.Contains("quokka", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SearchWeb_Learns_MemoriesFactsAndLanguage()
    {
        var backend = new FakeBackend();
        using var mind = BuildMind(backend);

        mind.SearchWeb("what is a quokka");

        // Memories recallable later.
        Assert.Contains(mind.Memory!.LongTermMemories(),
            m => m.Content.Contains("quokka", StringComparison.OrdinalIgnoreCase)
                 && m.Source == "web");
        // Facts distilled from the snippet ("The quokka is a small macropod...").
        Assert.True(mind.Knowledge!.FactCount > 0, "snippet facts should be distilled");
        // Language model saw the results.
        Assert.True(mind.LanguageModel.IsTrained);
    }

    [Fact]
    public void SearchWeb_MarksQueryWordsAsKnown()
    {
        var backend = new FakeBackend();
        using var mind = BuildMind(backend);
        var lexicon = mind.Lexicon!;
        lexicon.Observe("quokka", 1);          // heard once, never spoken
        Assert.NotNull(lexicon.FreshWord());   // would normally be asked about

        mind.SearchWeb("what is a quokka");

        Assert.Null(lexicon.FreshWord());      // looked up instead of asked
    }

    [Fact]
    public void SearchWeb_NeverAutoFires_FromConversation()
    {
        var backend = new FakeBackend();
        using var mind = BuildMind(backend);

        // Talking about searching must NOT trigger the tool — manual-only.
        mind.Ask("can you search the web for quokkas?");
        mind.Ask("web-search would be useful now");
        Assert.Equal(0, backend.Calls);
    }

    [Fact]
    public void WebSearchTool_IsManualOnly_AndHasNoTriggers()
    {
        ITool tool = new WebSearchTool();
        Assert.True(tool.ManualOnly);
        Assert.Empty(tool.Triggers);

        var registry = new ToolRegistry();
        registry.Register(tool);
        Assert.Empty(registry.Match("please web-search quokka for me"));
    }

    [Fact]
    public void SearchWeb_ReportsFailureCleanly()
    {
        var backend = new FakeBackend { FailWithException = true };
        using var mind = BuildMind(backend);

        var result = mind.SearchWeb("anything");
        Assert.False(result.Success);
        Assert.Contains("could not search the web", result.Output, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("internet connection", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SearchWeb_EmptyQueryFails()
    {
        using var mind = BuildMind(new FakeBackend());
        Assert.False(mind.SearchWeb("   ").Success);
    }

    [Fact]
    public void SearchWeb_ExplicitUseToolCommand_StillWorks()
    {
        var backend = new FakeBackend();
        using var mind = BuildMind(backend);

        // The explicit "use tool" channel keeps working — that is still
        // EXPLICIT, the user asked for it.
        var response = mind.Ask("use tool web-search query=quokka");
        Assert.Equal(1, backend.Calls);
        Assert.Contains("quokka", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SearchWeb_Throws_WhenToolsDisabled()
    {
        using var mind = MindBuilder.Create("NoTools").Build();
        Assert.Throws<MindConfigurationException>(() => mind.SearchWeb("hello"));
    }
}
