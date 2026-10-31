using Cognify.Core;
using Cognify.Tools;

namespace Cognify.Tests;

/// <summary>
/// Tests for web EXPLORATION — reading INTO websites. The explorer fetches
/// pages, extracts readable text (no scripts, styles or chrome), walks to a
/// more relevant in-page link, and the mind learns from what she read: after
/// a search she opens the best result page, and ExploreWeb(url) opens any
/// page directly. Everything runs on offline fixtures — no network.
/// </summary>
public class WebExplorerTests
{
    // A site front door: heavy chrome, scripts that MUST NOT leak into the
    // readable text, and one genuinely relevant link ("Flower").
    private const string PortalHtml = """
        <html><head><title>Wikipedia</title>
        <style>.menu { color: red; font-size: 14px }</style>
        <script>var tracking = "flowers should never leak from scripts";</script>
        </head>
        <body>
        <nav><a href="/menu">Menu item words about navigation chrome only</a></nav>
        <header><p>Banner text lines that live in the page header chrome area.</p></header>
        <main>
        <h1>Wikipedia</h1>
        <p>Welcome to Wikipedia, the free encyclopedia that anyone can edit.</p>
        <p>This portal lists languages for readers: English, Fran&ccedil;ais, Deutsch and more.</p>
        <ul>
          <li><a href="/wiki/Flower">Flower</a> &mdash; the blossom of a plant</li>
          <li><a href="/wiki/Rose">Rose</a></li>
          <li><a href="javascript:void(0)">do nothing</a></li>
        </ul>
        </main>
        <footer><p>Site footer legal text lines that nobody ever reads carefully.</p></footer>
        </body></html>
        """;

    // The actual article — Wikipedia-style mw-content-text markup.
    private const string FlowerHtml = """
        <html><head><title>Flower - Wikipedia</title></head>
        <body>
        <div id="mw-content-text">
        <p>A flower is the reproductive structure of flowering plants, and bright petals attract pollinators.</p>
        <p>The flower contains the male stamen and the female pistil, and colorful petals draw insects and birds.</p>
        <p>Flowers have long been admired by humans as objects of beauty and as gifts for loved ones.</p>
        <p><a href="/wiki/Rose">Rose</a> is a popular garden flower species.</p>
        </div>
        </body></html>
        """;

    /// <summary>Serves fixture pages; records every fetch so tests can prove
    /// how many pages the explorer actually opened.</summary>
    private sealed class FakeFetcher : IWebPageFetcher
    {
        public int Calls;
        public string LastUrl = "";
        private readonly Dictionary<string, string> _pages;

        public FakeFetcher(Dictionary<string, string>? pages = null) =>
            _pages = (pages ?? new Dictionary<string, string>())
                .ToDictionary(kv => Normalize(kv.Key), kv => kv.Value);

        public string Fetch(string url)
        {
            Calls++;
            LastUrl = url;
            if (_pages.TryGetValue(Normalize(url), out var html)) return html;
            throw new WebSearchException($"the page could not be opened (no fixture for {url})",
                WebSearchFailure.NoResults);
        }

        // Same normalization the explorer uses for its visited set.
        private static string Normalize(string url) => url.TrimEnd('/').ToLowerInvariant();
    }

    /// <summary>Deterministic offline search backend with two results.</summary>
    private sealed class FakeBackend : IWebSearchBackend
    {
        public int Calls;
        public List<WebSearchResult> Results { get; set; } = new();
        public IReadOnlyList<WebSearchResult> Search(string query, int maxResults)
        {
            Calls++;
            return Results.Take(maxResults).ToList();
        }
    }

    private static FakeFetcher DefaultFixtures() => new(new Dictionary<string, string>
    {
        ["https://site.example/"] = PortalHtml,
        ["https://site.example/wiki/flower"] = FlowerHtml,
    });

    // ------------------------------------------------------------------
    // Page reading: readable text, titles, links.
    // ------------------------------------------------------------------

    [Fact]
    public void FetchPage_ExtractsTitleTextAndLinks_WithoutChrome()
    {
        var explorer = new WebExplorer(DefaultFixtures());

        var page = explorer.FetchPage("https://site.example/");

        Assert.Equal("Wikipedia", page.Title);
        // Readable text keeps real content...
        Assert.Contains("free encyclopedia that anyone can edit", page.Text);
        // ...and never leaks scripts, styles or site chrome.
        Assert.DoesNotContain("flowers should never leak", page.Text);
        Assert.DoesNotContain("font-size", page.Text);
        Assert.DoesNotContain("navigation chrome", page.Text);
        Assert.DoesNotContain("footer legal text", page.Text);
        // Links resolve to absolute urls; javascript: links are dropped.
        Assert.Contains(page.Links, l => l.Url == "https://site.example/wiki/Flower" && l.Text == "Flower");
        Assert.DoesNotContain(page.Links, l => l.Url.StartsWith("javascript:"));
    }

    [Fact]
    public void FetchPage_WikipediaContent_IsFoundInsideMwContentText()
    {
        var explorer = new WebExplorer(DefaultFixtures());

        var page = explorer.FetchPage("https://site.example/wiki/Flower");

        Assert.Equal("Flower - Wikipedia", page.Title);
        Assert.Contains("reproductive structure of flowering plants", page.Text);
        Assert.Contains(page.Links, l => l.Url.EndsWith("/wiki/Rose"));
    }

    [Fact]
    public void HttpFetcher_RejectsBinaryBodies()
    {
        Assert.True(HttpWebPageFetcher.LooksBinary("%PDF-1.7 not html"));
        Assert.True(HttpWebPageFetcher.LooksBinary("GIF89a\u0000\u0000 rest"));
        Assert.False(HttpWebPageFetcher.LooksBinary("<html><body>a normal page</body></html>"));
    }

    // ------------------------------------------------------------------
    // Relevance: which page, which link, which sentences.
    // ------------------------------------------------------------------

    [Fact]
    public void Relevance_RanksSearchResults_ByQueryMatch()
    {
        var results = new List<WebSearchResult>
        {
            new("Wikipedia", "https://www.wikipedia.org/",
                "Wikipedia is a free online encyclopedia hosted by volunteers."),
            new("Flower - Wikipedia", "https://en.wikipedia.org/wiki/Flower",
                "Flowers are the reproductive structures of flowering plants."),
        };

        var ranked = WebRelevance.RankResults("flowers", results);

        Assert.Same(results[1], ranked[0]);   // the flower article, not the brand hit
    }

    [Fact]
    public void Relevance_KeySentences_SurfaceTheQueryTopic()
    {
        var text = "Welcome to our site about many things. " +
                   "A flower is the reproductive structure of flowering plants. " +
                   "We also sell gardening tools and boots. " +
                   "Flowers brighten rooms and help people express feelings.";
        var keywords = WebRelevance.Keywords("flowers");

        var sentences = WebRelevance.KeySentences(text, keywords, maxSentences: 2);

        Assert.All(sentences, s => Assert.Contains("flower", s, StringComparison.OrdinalIgnoreCase));
        Assert.Equal(2, sentences.Count);
    }

    // ------------------------------------------------------------------
    // Exploration: the walk from a front door into the real article.
    // ------------------------------------------------------------------

    [Fact]
    public void Explore_FollowsTheMoreRelevantInPageLink()
    {
        var fetcher = DefaultFixtures();
        var explorer = new WebExplorer(fetcher, new WebExplorationOptions());   // MaxPages = 2

        var exploration = explorer.Explore("flowers", "https://site.example/");

        // She started on the portal, saw the "Flower" link matched the query
        // better, and stepped through it.
        Assert.Equal(2, exploration.Pages.Count);
        Assert.Equal(2, fetcher.Calls);
        Assert.Contains("reproductive structure", exploration.CombinedText);
        Assert.Contains(exploration.KeySentences,
            s => s.Contains("flower", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void Explore_RespectsMaxPagesAndFollowSwitch()
    {
        var fetcher = DefaultFixtures();

        var capped = new WebExplorer(fetcher, new WebExplorationOptions().Pages(1));
        Assert.Single(capped.Explore("flowers", "https://site.example/").Pages);
        Assert.Equal(1, fetcher.Calls);

        var noFollow = new WebExplorer(fetcher, new WebExplorationOptions().Follow(false));
        Assert.Single(noFollow.Explore("flowers", "https://site.example/").Pages);
        Assert.Equal(2, fetcher.Calls);   // the second call came from the first explorer
    }

    [Fact]
    public void Explore_WithoutQuery_ReadsExactlyOnePage()
    {
        var fetcher = DefaultFixtures();
        var explorer = new WebExplorer(fetcher);

        var exploration = explorer.Explore("", "https://site.example/wiki/Flower");

        Assert.Single(exploration.Pages);
        Assert.Equal(1, fetcher.Calls);
    }

    [Fact]
    public void Explore_StaysOnPage_WhenNothingBetterIsLinked()
    {
        // The portal has no flower link at all — she must not wander off.
        var fetcher = new FakeFetcher(new Dictionary<string, string>
        {
            ["https://site.example/"] = PortalHtml.Replace(
                """<a href="/wiki/Flower">Flower</a>""",
                """<a href="/wiki/Portal">Portal</a>"""),
        });
        var explorer = new WebExplorer(fetcher);

        var exploration = explorer.Explore("flowers", "https://site.example/");

        Assert.Single(exploration.Pages);
        Assert.Equal(1, fetcher.Calls);
    }

    // ------------------------------------------------------------------
    // The tool and the mind-level API.
    // ------------------------------------------------------------------

    [Fact]
    public void WebExploreTool_IsManualOnly_AndReadsPages()
    {
        ITool tool = new WebExploreTool(DefaultFixtures());
        Assert.True(tool.ManualOnly);
        Assert.Empty(tool.Triggers);

        var result = tool.Execute(new ToolInvocation
        {
            ToolName = "web-explore",
            Arguments = new Dictionary<string, string> { ["url"] = "site.example/wiki/Flower" },
        });
        Assert.True(result.Success, result.Output);
        Assert.Contains("Here is what I read on", result.Output);
        Assert.Contains("reproductive structure", result.Output);
        // Scheme-less addresses get https:// for free.
        Assert.Equal("https://site.example/wiki/Flower", result.Data["url"]);
    }

    [Fact]
    public void WebExploreTool_FailsCleanly_OnUnreadablePages()
    {
        var tool = new WebExploreTool(new FakeFetcher());   // serves nothing

        var result = tool.Execute(new ToolInvocation
        {
            ToolName = "web-explore",
            Arguments = new Dictionary<string, string> { ["url"] = "https://offline.example/x" },
        });

        Assert.False(result.Success);
        Assert.Contains("could not read the page", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExploreWeb_LearnsFromThePageSheReads()
    {
        var backend = new FakeBackend();
        var fetcher = DefaultFixtures();
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithWebSearch(t => { t.WebSearchBackend = backend; t.WebPageFetcher = fetcher; })
            .Build();

        var result = mind.ExploreWeb("site.example/wiki/Flower");

        Assert.True(result.Success, result.Output);
        Assert.Equal(0, backend.Calls);                    // no search happened — the page only
        Assert.Contains(mind.Memory!.LongTermMemories(),
            m => m.Content.Contains("reproductive structure", StringComparison.OrdinalIgnoreCase)
                 && m.Source == "web");
        Assert.True(mind.Knowledge!.FactCount > 0, "page facts should be distilled");
        Assert.True(mind.LanguageModel.IsTrained);
    }

    [Fact]
    public void SearchWeb_Exploration_ReadsIntoTheBestResult()
    {
        var backend = new FakeBackend
        {
            Results =
            {
                new WebSearchResult("Flower delivery and bouquets", "https://site.example/",
                    "Fresh flower arrangements handcrafted by local florists."),
            },
        };
        var fetcher = new FakeFetcher(new Dictionary<string, string>
        {
            ["https://site.example/"] = FlowerHtml,   // the florist page is, really, the article
        });
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithWebSearch(t => { t.WebSearchBackend = backend; t.WebPageFetcher = fetcher; })
            .Build();

        var result = mind.SearchWeb("flowers");

        Assert.True(result.Success, result.Output);
        Assert.Equal(1, fetcher.Calls);                    // exactly the one relevant page
        Assert.Contains("I also read into", result.Output);
        Assert.Contains("reproductive structure", result.Output);
        // The page text is in her knowledge now — not just the snippet.
        Assert.Contains(mind.Memory!.LongTermMemories(),
            m => m.Content.Contains("Web page:", StringComparison.Ordinal));
        Assert.Contains(mind.Memory.LongTermMemories(),
            m => m.Content.Contains("reproductive structure", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void SearchWeb_Exploration_SecondResultGetsItsChance()
    {
        // A brand-dominated SERP: the top hit mentions flowers only on the
        // surface (title, url, snippet) — its page says nothing about them.
        // The second result must get the chance: this is the
        // "she ignores flowers" fix.
        var backend = new FakeBackend
        {
            Results =
            {
                new WebSearchResult("Flowers & Gifts — Home", "https://site.example/flowers",
                    "Welcome to Flowers and Gifts, your online portal."),
                new WebSearchResult("Flower - Wikipedia", "https://site.example/wiki/Flower",
                    "Flowers are the reproductive structures of flowering plants."),
            },
        };
        var fetcher = new FakeFetcher(new Dictionary<string, string>
        {
            ["https://site.example/flowers"] = PortalHtml,       // the brand page: readable, but NOT about flowers
            ["https://site.example/wiki/flower"] = FlowerHtml,   // the real article
        });
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithWebSearch(t => { t.WebSearchBackend = backend; t.WebPageFetcher = fetcher; })
            .Build();

        var result = mind.SearchWeb("flowers");

        Assert.True(result.Success, result.Output);
        // She opened the brand page first (it ties on surface score and ranks
        // first), found nothing about flowers in its text, then read the
        // article page and learned from it:
        Assert.Equal(2, fetcher.Calls);
        Assert.Contains("reproductive structure", result.Output);
        Assert.Contains(mind.Memory!.LongTermMemories(),
            m => m.Content.Contains("reproductive structure", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void SearchWeb_ExplorationOff_NeverFetchesPages()
    {
        var backend = new FakeBackend
        {
            Results = { new WebSearchResult("Flowers", "https://site.example/", "Flower snippets.") },
        };
        var fetcher = DefaultFixtures();
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithWebSearch(t =>
            {
                t.WebSearchBackend = backend;
                t.WebPageFetcher = fetcher;
                t.Exploration(e => e.Explore(false));
            })
            .Build();

        var result = mind.SearchWeb("flowers");

        Assert.True(result.Success, result.Output);
        Assert.Equal(0, fetcher.Calls);
        Assert.DoesNotContain("I also read into", result.Output);
    }

    [Fact]
    public void SearchWeb_ExplorationSurvivesUnreadablePages()
    {
        // The best result is unreadable (no fixture → NoResults); the search
        // itself must still succeed and the next candidate gets the chance.
        var backend = new FakeBackend
        {
            Results =
            {
                new WebSearchResult("Flowers offline", "https://offline.example/flowers",
                    "A flowers page that cannot be fetched at all."),
                new WebSearchResult("Flower - Wikipedia", "https://site.example/wiki/Flower",
                    "Flowers are the reproductive structures of flowering plants."),
            },
        };
        var fetcher = DefaultFixtures();
        using var mind = MindBuilder.Create("Curious")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .WithWebSearch(t => { t.WebSearchBackend = backend; t.WebPageFetcher = fetcher; })
            .Build();

        var result = mind.SearchWeb("flowers");

        Assert.True(result.Success, result.Output);
        Assert.Equal(2, fetcher.Calls);   // the offline attempt, then the article
        Assert.Contains("reproductive structure", result.Output);
    }

    [Fact]
    public void ExploreWeb_ThrowWhenToolsDisabled()
    {
        using var mind = MindBuilder.Create("NoTools").Build();
        Assert.Throws<MindConfigurationException>(() => mind.ExploreWeb("https://site.example/"));
    }
}
