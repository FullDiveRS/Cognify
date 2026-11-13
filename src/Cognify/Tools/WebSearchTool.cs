using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Cognify.Core;

namespace Cognify.Tools;

/// <summary>One web search hit: title, url and snippet.</summary>
public sealed record WebSearchResult(string Title, string Url, string Snippet);

/// <summary>Why a web search could not complete.</summary>
public enum WebSearchFailure
{
    /// <summary>The machine could not reach the network at all.</summary>
    Offline,
    /// <summary>The network worked, but the engine answered with a bot check instead of results.</summary>
    Blocked,
    /// <summary>The network worked, the page parsed cleanly — it just has no results.</summary>
    NoResults
}

/// <summary>Raised when a web search could not complete (offline, blocked, parse failure).</summary>
public sealed class WebSearchException : Exception
{
    /// <summary>Creates the exception.</summary>
    public WebSearchException(string message, WebSearchFailure reason = WebSearchFailure.Offline,
        Exception? inner = null) : base(message, inner)
    {
        Reason = reason;
    }

    /// <summary>The failure category, so callers can give honest advice.</summary>
    public WebSearchFailure Reason { get; }
}

/// <summary>
/// One pluggable search engine probe: fetches a results page and parses it.
/// Implementations throw <see cref="WebSearchException"/> with the right
/// <see cref="WebSearchFailure"/> reason when they cannot produce results.
/// </summary>
public interface IWebSearchEngine
{
    /// <summary>Human-readable engine name (used in diagnostics).</summary>
    string Name { get; }

    /// <summary>Fetches and parses results; empty lists are allowed, failures throw.</summary>
    IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults);
}

/// <summary>
/// A pluggable web search backend. The default implementation tries several
/// keyhole-free engines in order (Google News, then Wikipedia, then Bing),
/// rejecting any engine whose results do not actually MATCH the query's
/// words (the relevance gate keeps junk out — see
/// <see cref="HttpWebSearchBackend"/>); plug in your own (SerpAPI, a local
/// index...) by implementing this interface and passing it to
/// <see cref="WebSearchTool"/>.
/// </summary>
public interface IWebSearchBackend
{
    /// <summary>Searches the web and returns up to <paramref name="maxResults"/> hits.</summary>
    IReadOnlyList<WebSearchResult> Search(string query, int maxResults);
}

/// <summary>Shared HTTP plumbing for the built-in engines (browser-like headers).</summary>
public abstract class HttpSearchEngineBase : IWebSearchEngine
{
    /// <summary>A browser-like User-Agent: search engines bot-check unfamiliar clients,
    /// which is exactly what broke the old "CognifyMind/1.0" client.</summary>
    public const string BrowserUserAgent =
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

    internal const string AnomalyMarker = "anomaly";

    private static readonly Regex TagRegex = new(@"<[^>]+>", RegexOptions.Compiled);

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults);

    /// <summary>Performs a GET and returns the body; classifies HTTP-level failures.</summary>
    protected string Get(HttpClient http, string url)
    {
        try
        {
            using var response = http.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
            var body = response.Content.ReadAsStringAsync().ConfigureAwait(false)
                           .GetAwaiter().GetResult();
            // Some engines answer bot checks with HTTP 202 + a challenge page —
            // "success" by status code, but there is nothing to parse.
            if ((int)response.StatusCode == 202 || IsAnomalyPage(body))
                throw new WebSearchException($"{Name} answered with a bot-check page",
                    WebSearchFailure.Blocked);
            if (response.StatusCode is HttpStatusCode.Forbidden or HttpStatusCode.TooManyRequests)
                throw new WebSearchException($"{Name} refused the request ({(int)response.StatusCode})",
                    WebSearchFailure.Blocked);
            response.EnsureSuccessStatusCode();
            return body;
        }
        catch (WebSearchException) { throw; }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or WebException)
        {
            throw new WebSearchException($"{Name} could not be reached ({RootCause(ex)})",
                WebSearchFailure.Offline, ex);
        }
    }

    /// <summary>True when the page is a bot-check/challenge instead of results.</summary>
    internal static bool IsAnomalyPage(string html)
    {
        if (html.Length < 200) return true; // a real results page is never this small
        return html.Contains(AnomalyMarker, StringComparison.OrdinalIgnoreCase)
            && !html.Contains("result", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Strips tags, decodes entities and collapses whitespace.</summary>
    internal static string StripTags(string html)
    {
        var text = TagRegex.Replace(html, " ");
        text = WebUtility.HtmlDecode(text);
        return Regex.Replace(text, @"\s+", " ").Trim();
    }

    /// <summary>Unwraps Google News redirect links (/rss/articles/... keep as-is; the
    /// link resolves to the real article when opened). Kept for symmetry.</summary>
    internal static string UnwrapGoogleUrl(string url) => url;

    /// <summary>Short, human-readable network error ("no such host", "timed out"...).</summary>
    internal static string RootCause(Exception ex)
    {
        var message = ex.Message;
        var inner = ex.InnerException;
        while (inner is not null)
        {
            message = inner.Message;
            inner = inner.InnerException;
        }
        return message.Length > 120 ? message[..120] : message;
    }
}

/// <summary>
/// Google web search (first engine in the default chain). Google's regular
/// results page requires JavaScript, but Google NEWS offers a clean RSS
/// feed (<c>news.google.com/rss/search</c>) — keyless, bot-resistant, and
/// genuinely relevant: a query about flowers returns flower articles, never
/// the randomized junk the other scrapes serve to lightweight clients.
/// Items carry the real publisher in <c>&lt;source url="..."&gt;</c>; the
/// article link is a Google redirect that resolves when opened.
/// </summary>
public sealed class GoogleNewsSearchEngine : HttpSearchEngineBase
{
    private static readonly Regex ItemRegex =
        new(@"<item>(.*?)</item>", RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex TitleRegex =
        new(@"<title>(.*?)</title>", RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex LinkRegex =
        new(@"<link>(.*?)</link>", RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex SourceUrlRegex =
        new(@"<source[^>]*url=""([^""]+)""[^>]*>(.*?)</source>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex DescriptionRegex =
        new(@"<description>(.*?)</description>", RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private readonly HttpClient _http;
    private readonly string _edition;

    /// <summary>Creates the engine (owner of its own HttpClient).</summary>
    public GoogleNewsSearchEngine(TimeSpan? timeout = null, string edition = "en-US")
    {
        _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(10) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(BrowserUserAgent);
        _http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
        _edition = edition;
    }

    /// <inheritdoc />
    public override string Name => "Google";

    /// <inheritdoc />
    public override IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults)
    {
        var url = $"https://news.google.com/rss/search?q={Uri.EscapeDataString(query)}" +
                  $"&hl={Uri.EscapeDataString(_edition)}&gl=US&ceid=US:en";
        var html = Get(_http, url);
        return Parse(html, maxResults);
    }

    /// <summary>Parses a Google News RSS results page (exposed for offline tests).</summary>
    public static IReadOnlyList<WebSearchResult> Parse(string html, int maxResults)
    {
        var results = new List<WebSearchResult>();
        foreach (Match item in ItemRegex.Matches(html))
        {
            if (results.Count >= maxResults) break;
            var body = item.Groups[1].Value;
            var title = RssText(TitleRegex.Match(body).Groups[1].Value);
            var link = RssText(LinkRegex.Match(body).Groups[1].Value);
            var source = SourceUrlRegex.Match(body);
            var publisher = source.Success ? StripTags(source.Groups[2].Value) : "";
            var description = RssText(DescriptionRegex.Match(body).Groups[1].Value);
            if (title.Length == 0 || link.Length == 0) continue;
            // Snippet: the publisher name grounds the hit; the description is
            // usually a related-articles list, so it only fills in when short.
            var snippet = publisher.Length > 0
                ? publisher + (description.Length is > 0 and < 160 ? " — " + description : "")
                : description;
            results.Add(new WebSearchResult(title, link, snippet));
        }
        return results;
    }

    private static string RssText(string raw)
    {
        var cdata = Regex.Match(raw, @"<!\[CDATA\[(.*?)\]\]>", RegexOptions.Singleline);
        if (cdata.Success) raw = cdata.Groups[1].Value;
        return StripTags(raw);
    }
}

/// <summary>
/// Wikipedia search (second engine — the knowledge fallback). Wikipedia's
/// public API is keyless, never bot-checked, and ideal for an AI that wants
/// to UNDERSTAND a topic: results arrive as JSON with a clean text snippet
/// and a stable article URL. For "what is X" style questions this engine is
/// the most reliable answer on the internet.
/// </summary>
public sealed class WikipediaSearchEngine : HttpSearchEngineBase
{
    private readonly HttpClient _http;
    private readonly string _language;

    /// <summary>Creates the engine (owner of its own HttpClient).</summary>
    public WikipediaSearchEngine(TimeSpan? timeout = null, string language = "en")
    {
        _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(10) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(BrowserUserAgent);
        _language = language;
    }

    /// <inheritdoc />
    public override string Name => "Wikipedia";

    /// <inheritdoc />
    public override IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults)
    {
        var url = $"https://{_language}.wikipedia.org/w/api.php" +
                  "?action=query&list=search&format=json&srlimit=" +
                  Math.Max(maxResults, 5) +
                  $"&srsearch={Uri.EscapeDataString(query)}";
        var json = Get(_http, url);
        return Parse(json, maxResults, _language);
    }

    /// <summary>Parses a Wikipedia API search response (exposed for offline tests).</summary>
    public static IReadOnlyList<WebSearchResult> Parse(string json, int maxResults, string language = "en")
    {
        var results = new List<WebSearchResult>();
        using var doc = JsonDocument.Parse(json);
        if (!doc.RootElement.TryGetProperty("query", out var query) ||
            !query.TryGetProperty("search", out var hits) ||
            hits.ValueKind != JsonValueKind.Array)
            return results;

        foreach (var hit in hits.EnumerateArray())
        {
            if (results.Count >= maxResults) break;
            if (!hit.TryGetProperty("title", out var titleEl)) continue;
            var title = titleEl.GetString() ?? "";
            if (title.Length == 0) continue;

            string snippet = "";
            if (hit.TryGetProperty("snippet", out var snippetEl))
            {
                // Snippets carry <span class="searchmatch"> highlights — strip them.
                snippet = StripTags(snippetEl.GetString() ?? "");
            }
            var slug = Uri.EscapeDataString(title.Replace(' ', '_'));
            var url = $"https://{language}.wikipedia.org/wiki/{slug}";
            results.Add(new WebSearchResult(title, url, snippet));
        }
        return results;
    }
}

/// <summary>
/// Bing web search (third engine — general web when the others cannot
/// answer). Uses Bing's RSS output (<c>format=rss</c>) — a small, stable
/// page with direct links; the HTML scrape of b_algo blocks serves
/// randomized junk to lightweight clients. NOTE: Bing's RSS endpoint is
/// known to serve locally-personalized junk to datacenter IPs — the
/// backend's relevance gate catches that and moves on to the next engine.
/// RSS descriptions are tag-stripped and entity-decoded; /ck/a-style links
/// are unwrapped defensively.
/// </summary>
public sealed class BingSearchEngine : HttpSearchEngineBase
{
    private static readonly Regex ItemRegex =
        new(@"<item>(.*?)</item>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex TitleRegex =
        new(@"<title>(.*?)</title>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex LinkRegex =
        new(@"<link>(.*?)</link>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex DescriptionRegex =
        new(@"<description>(.*?)</description>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private readonly HttpClient _http;
    private readonly string _market;

    /// <summary>Creates the engine (owner of its own HttpClient).</summary>
    public BingSearchEngine(TimeSpan? timeout = null, string market = "en-US")
    {
        _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(10) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(BrowserUserAgent);
        _http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
        _market = market;
    }

    /// <inheritdoc />
    public override string Name => "Bing";

    /// <inheritdoc />
    public override IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults)
    {
        var url = $"https://www.bing.com/search?q={Uri.EscapeDataString(query)}" +
                  $"&format=rss&count={Math.Max(maxResults, 10)}&mkt={_market}";
        var html = Get(_http, url);
        return Parse(html, maxResults);
    }

    /// <summary>Parses a Bing RSS results page (exposed for offline tests).</summary>
    public static IReadOnlyList<WebSearchResult> Parse(string html, int maxResults)
    {
        var results = new List<WebSearchResult>();
        foreach (Match item in ItemRegex.Matches(html))
        {
            if (results.Count >= maxResults) break;
            var title = RssText(TitleRegex.Match(item.Groups[1].Value).Groups[1].Value);
            var link = RssText(LinkRegex.Match(item.Groups[1].Value).Groups[1].Value);
            var description = RssText(DescriptionRegex.Match(item.Groups[1].Value).Groups[1].Value);
            if (title.Length == 0 || link.Length == 0) continue;
            results.Add(new WebSearchResult(title, UnwrapBingUrl(link), description));
        }
        return results;
    }

    private static string RssText(string raw)
    {
        // CDATA first, then strip tags, then decode entities.
        var cdata = Regex.Match(raw, @"<!\[CDATA\[(.*?)\]\]>", RegexOptions.Singleline);
        if (cdata.Success) raw = cdata.Groups[1].Value;
        return StripTags(raw);
    }

    /// <summary>Unwraps Bing /ck/a redirect links; passes plain links through.</summary>
    public static string UnwrapBingUrl(string url)
    {
        var idx = url.IndexOf("u=a1", StringComparison.Ordinal);
        if (idx < 0) return url;
        var token = url[(idx + 4)..];
        var end = token.IndexOf('&');
        if (end > 0) token = token[..end];
        try
        {
            var padded = token.PadRight(token.Length + (4 - token.Length % 4) % 4, '=');
            return Encoding.UTF8.GetString(Convert.FromBase64String(
                padded.Replace('-', '+').Replace('_', '/')));
        }
        catch (FormatException)
        {
            return url;
        }
    }
}

/// <summary>
/// Real HTTP web search across a chain of engines (Google News first, then
/// Wikipedia, then Bing — DuckDuckGo was removed by design). Two guards
/// keep the results honest:
///
/// 1. THE RELEVANCE GATE. An engine's output is only accepted when the
///    results actually MATCH the query: every content word of the query
///    (3+ letters) must appear in at least one result's title/snippet/host.
///    This is the fix for "wikipedia flowers" returning nothing but
///    Wikipedia-site pages (the word "flowers" silently vanished) and for
///    Bing's datacenter junk ("flowers" answering with pizza restaurants) —
///    an engine that misses part of the question is skipped, the next one
///    gets a try.
/// 2. THE TOPIC RETRY. When every engine missed part of the query, the
///    search retries once with the query's substantial topic words only
///    (short function words and site-name noise dropped), so a clunky
///    query still finds its subject.
///
/// When every engine fails, the reported reason is honest — "offline" only
/// when the network truly failed.
/// </summary>
public sealed class HttpWebSearchBackend : IWebSearchBackend
{
    private readonly IReadOnlyList<IWebSearchEngine> _engines;
    private readonly int _attempts;

    /// <summary>Creates the backend with the default engine chain
    /// (Google → Wikipedia → Bing).</summary>
    public HttpWebSearchBackend(TimeSpan? timeout = null, int attempts = 1,
        IReadOnlyList<IWebSearchEngine>? engines = null)
    {
        _attempts = Math.Max(1, attempts);
        _engines = engines ?? new IWebSearchEngine[]
        {
            new GoogleNewsSearchEngine(timeout),
            new WikipediaSearchEngine(timeout),
            new BingSearchEngine(timeout),
        };
    }

    /// <summary>The engine chain, for diagnostics.</summary>
    public IReadOnlyList<IWebSearchEngine> Engines => _engines;

    /// <inheritdoc />
    public IReadOnlyList<WebSearchResult> Search(string query, int maxResults)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new WebSearchException("Empty web search query.");

        bool sawResponse = false;   // at least one engine answered at HTTP level
        bool sawBlocked = false;    // at least one bot-check
        string detail = "no results";
        IReadOnlyList<WebSearchResult>? bestEffort = null;

        var run = RunOnce;
        var words = ContentWords(query);
        var (results, why) = run(query, maxResults, words);
        Collect(results, why);

        if (bestEffort is null && words.Count > 1)
        {
            // Topic retry: keep the query's substantial words only.
            var topic = string.Join(' ', words.OrderByDescending(w => w.Length).Take(3));
            if (!string.Equals(topic, query.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                (results, why) = run(topic, maxResults, ContentWords(topic));
                Collect(results, why);
            }
        }

        if (bestEffort is not null)
            return bestEffort;

        var reason = sawBlocked ? WebSearchFailure.Blocked
                   : sawResponse ? WebSearchFailure.NoResults
                   : WebSearchFailure.Offline;
        throw new WebSearchException($"web search could not complete ({detail})", reason);

        // ---- local helpers -------------------------------------------------
        void Collect(IReadOnlyList<WebSearchResult>? rs, string whyLocal)
        {
            if (rs is not null && rs.Count > 0 && bestEffort is null)
                bestEffort = rs;
            if (whyLocal.Length > 0) detail = whyLocal;
        }
    }

    private (IReadOnlyList<WebSearchResult>?, string) RunOnce(
        string query, int maxResults, IReadOnlyList<string> words)
    {
        bool sawResponse = false, sawBlocked = false;
        string detail = "no results";
        IReadOnlyList<WebSearchResult>? best = null;

        for (int attempt = 0; attempt < _attempts; attempt++)
        {
            foreach (var engine in _engines)
            {
                try
                {
                    var results = engine.Fetch(query, maxResults);
                    if (results.Count == 0)
                    {
                        sawResponse = true;
                        detail = $"{engine.Name} returned a page with no readable results";
                        continue;
                    }
                    sawResponse = true;
                    best ??= results; // keep as last-resort output
                    if (IsRelevant(words, results))
                        return (results, "");
                    detail = $"{engine.Name} results did not match the full query";
                }
                catch (WebSearchException ex)
                {
                    if (ex.Reason == WebSearchFailure.Blocked)
                    {
                        sawResponse = true;
                        sawBlocked = true;
                        detail = ex.Message;
                    }
                    else if (ex.Reason == WebSearchFailure.NoResults)
                    {
                        sawResponse = true;
                        detail = ex.Message;
                    }
                    else
                    {
                        detail = ex.Message;
                    }
                }
            }
        }

        if (sawBlocked) detail = detail.Length > 0 ? detail : "blocked";
        return (null, detail);
    }

    /// <summary>
    /// The relevance gate: every content word of the query (3+ letters,
    /// stop-words removed) must show up in at least one result's
    /// title/snippet/host. A query like "wikipedia flowers" answered only
    /// with Wikipedia-site pages fails the "flowers" half and is rejected.
    /// Queries with no content words always pass.
    /// </summary>
    internal static bool IsRelevant(IReadOnlyList<string> contentWords,
        IReadOnlyList<WebSearchResult> results)
    {
        if (contentWords.Count == 0) return true;
        var haystacks = results.Select(r =>
            $"{r.Title} {r.Snippet} {SafeHost(r.Url)}").ToList();
        foreach (var word in contentWords)
            if (!haystacks.Any(h => h.Contains(word, StringComparison.OrdinalIgnoreCase)))
                return false;
        return true;
    }

    /// <summary>Query's content words: 3+ letters, stop-words removed, deduped.</summary>
    internal static IReadOnlyList<string> ContentWords(string query) =>
        Regex.Matches(query.ToLowerInvariant(), @"[a-z0-9']{3,}")
             .Select(m => m.Value)
             .Where(w => !StopWords.Contains(w))
             .Distinct()
             .ToList();

    private static readonly HashSet<string> StopWords = new()
    {
        "the", "and", "for", "with", "that", "this", "what", "who", "when",
        "where", "why", "how", "are", "was", "were", "you", "your", "about",
        "any", "can", "could", "does", "did", "from", "have", "has", "into",
        "look", "please", "search", "find", "tell", "give", "some", "info",
        "information", "she", "her", "his", "him", "its", "our", "out",
        "get", "got", "all", "not", "but", "just", "like", "want", "need"
    };

    private static string SafeHost(string url)
    {
        try { return new Uri(url).Host; }
        catch (UriFormatException) { return ""; }
    }
}

/// <summary>
/// The web-search tool. MANUAL-ONLY by design: it carries no trigger phrases
/// and <see cref="ManualOnly"/> keeps it out of auto-invocation, so the mind
/// never goes online unprompted unless <c>Tools.AutoFetchWeb</c> turned the
/// knowledge-gap auto-lookup on (a separate, explicit builder decision).
/// A user (or program) must normally explicitly ask for it.
/// </summary>
public sealed class WebSearchTool : ITool
{
    private readonly IWebSearchBackend _backend;
    private readonly int _maxResults;

    /// <summary>Creates the tool over a backend (default: the multi-engine HTTP chain).</summary>
    public WebSearchTool(IWebSearchBackend? backend = null, int maxResults = 4)
    {
        _backend = backend ?? new HttpWebSearchBackend();
        _maxResults = Math.Clamp(maxResults, 1, 8);
    }

    /// <inheritdoc />
    public string Name => "web-search";

    /// <inheritdoc />
    public string Description => "Searches the web for a topic and returns the top results with snippets.";

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Parameters =>
        new Dictionary<string, string> { ["query"] = "what to search the web for", ["focus"] = "optional suggestion — which results matter most" };

    /// <inheritdoc />
    public IReadOnlyList<string> Triggers => Array.Empty<string>();

    /// <inheritdoc />
    public bool ManualOnly => true; // never fires on its own

    /// <inheritdoc />
    public ToolResult Execute(ToolInvocation invocation)
    {
        var query = invocation.Arguments.TryGetValue("query", out var q) && !string.IsNullOrWhiteSpace(q)
            ? q
            : invocation.RawInput;
        query = query?.Trim().TrimEnd('?', '!', '.') ?? "";
        var focus = invocation.Arguments.TryGetValue("focus", out var f) && !string.IsNullOrWhiteSpace(f)
            ? f.Trim()
            : null;
        if (query.Length == 0)
            return ToolResult.Fail("No search query given. Usage: /searchweb \"<what to look up>\" [suggestion].");

        try
        {
            var results = _backend.Search(query, _maxResults);
            if (results.Count == 0)
                return ToolResult.Fail($"Nothing readable came back from the web for \"{query}\".");

            // The suggestion argument steers the report: hits matching the
            // focus words come first, so what the user asked FOR leads.
            if (!string.IsNullOrWhiteSpace(focus))
                results = RankByFocus(results, focus);

            var lines = new List<string>
            {
                $"Here is what I found on the web for \"{query}\"" +
                (focus is not null ? $" (focusing on: {Mind.Truncate(focus, 60)})" : "") + ":"
            };
            for (int i = 0; i < results.Count; i++)
            {
                var host = SafeHost(results[i].Url);
                var snippet = string.IsNullOrWhiteSpace(results[i].Snippet)
                    ? ""
                    : " — " + Mind.Truncate(results[i].Snippet, 140);
                lines.Add($" {i + 1}. {results[i].Title}{snippet}" +
                          (host.Length > 0 ? $" ({host})" : ""));
            }
            return ToolResult.Ok(string.Join("\n", lines),
                new Dictionary<string, object>
                {
                    ["query"] = query,
                    ["focus"] = focus ?? "",
                    ["results"] = results,
                });
        }
        catch (WebSearchException ex)
        {
            var advice = ex.Reason switch
            {
                WebSearchFailure.Offline =>
                    "Web search needs an internet connection — check the network and try again, or teach me directly.",
                WebSearchFailure.Blocked =>
                    "The search engine sent a bot-check instead of results. I can try again in a minute, or you can teach me directly.",
                _ =>
                    "The web answered, but nothing matched. Try different words, or teach me directly.",
            };
            return ToolResult.Fail($"I could not search the web for \"{query}\": {ex.Message}. {advice}");
        }
    }

    /// <summary>
    /// Re-ranks results by the focus suggestion: hits whose title/snippet
    /// mention more of the focus's content words come first. Stable — equal
    /// scores keep their engine order. No focus (or no content words) leaves
    /// the order untouched.
    /// </summary>
    internal static IReadOnlyList<WebSearchResult> RankByFocus(
        IReadOnlyList<WebSearchResult> results, string? focus)
    {
        var words = HttpWebSearchBackend.ContentWords(focus ?? "");
        if (words.Count == 0 || results.Count <= 1) return results;
        return results
            .Select((r, i) => (r, i, score: words.Count(w =>
                $"{r.Title} {r.Snippet}".Contains(w, StringComparison.OrdinalIgnoreCase))))
            .OrderByDescending(x => x.score).ThenBy(x => x.i)
            .Select(x => x.r)
            .ToList();
    }

    private static string SafeHost(string url)
    {
        try { return new Uri(url).Host; }
        catch (UriFormatException) { return ""; }
    }
}
