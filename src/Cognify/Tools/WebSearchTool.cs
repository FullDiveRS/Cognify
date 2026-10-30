using System.Net;
using System.Net.Http;
using System.Text;
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
/// keyhole-free engines in order (Bing, then DuckDuckGo lite, then DuckDuckGo
/// html); plug in your own (SerpAPI, a local index...) by implementing this
/// interface and passing it to <see cref="WebSearchTool"/>.
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
            // DuckDuckGo answers bot checks with HTTP 202 + a challenge page —
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

    /// <summary>Performs a form POST and returns the body; classifies HTTP-level failures.</summary>
    protected string Post(HttpClient http, string url, Dictionary<string, string> form)
    {
        try
        {
            using var content = new FormUrlEncodedContent(form);
            using var response = http.PostAsync(url, content).ConfigureAwait(false)
                                     .GetAwaiter().GetResult();
            var body = response.Content.ReadAsStringAsync().ConfigureAwait(false)
                           .GetAwaiter().GetResult();
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

    /// <summary>Unwraps DuckDuckGo redirect links (/l/?uddg=&lt;encoded real url&gt;).</summary>
    internal static string UnwrapDdgUrl(string url)
    {
        var idx = url.IndexOf("uddg=", StringComparison.Ordinal);
        if (idx >= 0)
        {
            var encoded = url[(idx + 5)..];
            var end = encoded.IndexOf('&');
            if (end > 0) encoded = encoded[..end];
            try { return Uri.UnescapeDataString(encoded); }
            catch (UriFormatException) { return url; }
        }
        return url;
    }

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
/// Bing web search (first engine in the default chain). Uses Bing's RSS
/// output (<c>format=rss</c>) — a small, stable, bot-resistant page with
/// direct links and real snippets; the HTML scrape of b_algo blocks served
/// randomized junk to lightweight clients. RSS descriptions are tag-stripped
/// and entity-decoded; /ck/a-style links are unwrapped defensively.
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

/// <summary>DuckDuckGo "lite" HTML endpoint (GET, no key, second in the chain).</summary>
public sealed class DuckDuckGoLiteEngine : HttpSearchEngineBase
{
    private static readonly Regex LinkRegex =
        new(@"<a[^>]*class=""result-link""[^>]*href=""([^""]+)""[^>]*>(.*?)</a>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex SnippetRegex =
        new(@"<td[^>]*class=""result-snippet""[^>]*>(.*?)</td>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private readonly HttpClient _http;

    /// <summary>Creates the engine (owner of its own HttpClient).</summary>
    public DuckDuckGoLiteEngine(TimeSpan? timeout = null)
    {
        _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(10) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(BrowserUserAgent);
        _http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
    }

    /// <inheritdoc />
    public override string Name => "DuckDuckGo";

    /// <inheritdoc />
    public override IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults)
    {
        var html = Get(_http, "https://lite.duckduckgo.com/lite/?q=" +
                                 Uri.EscapeDataString(query));
        return Parse(html, maxResults);
    }

    /// <summary>Parses a DuckDuckGo lite page (exposed for offline tests).</summary>
    public static IReadOnlyList<WebSearchResult> Parse(string html, int maxResults)
    {
        var titles = new List<(string Url, string Title)>();
        foreach (Match m in LinkRegex.Matches(html))
            titles.Add((UnwrapDdgUrl(m.Groups[1].Value), StripTags(m.Groups[2].Value)));
        var snippets = new List<string>();
        foreach (Match m in SnippetRegex.Matches(html))
            snippets.Add(StripTags(m.Groups[1].Value));

        var results = new List<WebSearchResult>();
        for (int i = 0; i < titles.Count && results.Count < maxResults; i++)
        {
            if (titles[i].Title.Length == 0) continue;
            results.Add(new WebSearchResult(titles[i].Title, titles[i].Url,
                i < snippets.Count ? snippets[i] : ""));
        }
        return results;
    }
}

/// <summary>DuckDuckGo full HTML endpoint (POST form; last resort in the chain).</summary>
public sealed class DuckDuckGoHtmlEngine : HttpSearchEngineBase
{
    private static readonly Regex LinkRegex =
        new(@"<a[^>]*class=""result__a""[^>]*href=""([^""]+)""[^>]*>(.*?)</a>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex SnippetRegex =
        new(@"<a[^>]*class=""result__snippet""[^>]*>(.*?)</a>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private readonly HttpClient _http;

    /// <summary>Creates the engine (owner of its own HttpClient).</summary>
    public DuckDuckGoHtmlEngine(TimeSpan? timeout = null)
    {
        _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(10) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(BrowserUserAgent);
        _http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
    }

    /// <inheritdoc />
    public override string Name => "DuckDuckGo-html";

    /// <inheritdoc />
    public override IReadOnlyList<WebSearchResult> Fetch(string query, int maxResults)
    {
        var html = Post(_http, "https://html.duckduckgo.com/html/",
            new Dictionary<string, string> { ["q"] = query });
        var titles = new List<(string Url, string Title)>();
        foreach (Match m in LinkRegex.Matches(html))
            titles.Add((UnwrapDdgUrl(m.Groups[1].Value), StripTags(m.Groups[2].Value)));
        var snippets = new List<string>();
        foreach (Match m in SnippetRegex.Matches(html))
            snippets.Add(StripTags(m.Groups[1].Value));

        var results = new List<WebSearchResult>();
        for (int i = 0; i < titles.Count && results.Count < maxResults; i++)
        {
            if (titles[i].Title.Length == 0) continue;
            results.Add(new WebSearchResult(titles[i].Title, titles[i].Url,
                i < snippets.Count ? snippets[i] : ""));
        }
        return results;
    }
}

/// <summary>
/// Real HTTP web search across a chain of engines (Bing first, DuckDuckGo as
/// fallback). An engine that answers with a bot-check no longer poisons the
/// whole search: the next engine gets a try. When every engine fails, the
/// reported reason is honest — "offline" only when the network truly failed.
/// </summary>
public sealed class HttpWebSearchBackend : IWebSearchBackend
{
    private readonly IReadOnlyList<IWebSearchEngine> _engines;
    private readonly int _attempts;

    /// <summary>Creates the backend with the default engine chain (Bing → DDG lite → DDG html).</summary>
    public HttpWebSearchBackend(TimeSpan? timeout = null, int attempts = 1,
        IReadOnlyList<IWebSearchEngine>? engines = null)
    {
        _attempts = Math.Max(1, attempts);
        _engines = engines ?? new IWebSearchEngine[]
        {
            new BingSearchEngine(timeout),
            new DuckDuckGoLiteEngine(timeout),
            new DuckDuckGoHtmlEngine(timeout),
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

        for (int attempt = 0; attempt < _attempts; attempt++)
        {
            foreach (var engine in _engines)
            {
                try
                {
                    var results = engine.Fetch(query, maxResults);
                    if (results.Count > 0) return results;
                    sawResponse = true;
                    detail = $"{engine.Name} returned a page with no readable results";
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

        var reason = sawBlocked ? WebSearchFailure.Blocked
                   : sawResponse ? WebSearchFailure.NoResults
                   : WebSearchFailure.Offline;
        throw new WebSearchException($"web search could not complete ({detail})", reason);
    }
}

/// <summary>
/// The web-search tool. MANUAL-ONLY by design: it carries no trigger phrases
/// and <see cref="ManualOnly"/> keeps it out of auto-invocation, so the mind
/// never goes online unprompted — a user (or program) must explicitly ask
/// for it. This is the guard the conversation asked for: "she is able to
/// complete it and not do it whenever she feels like it".
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
        new Dictionary<string, string> { ["query"] = "what to search the web for" };

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
        if (query.Length == 0)
            return ToolResult.Fail("No search query given. Usage: /searchweb <what to look up>.");

        try
        {
            var results = _backend.Search(query, _maxResults);
            if (results.Count == 0)
                return ToolResult.Fail($"Nothing readable came back from the web for \"{query}\".");

            var lines = new List<string>
            {
                $"Here is what I found on the web for \"{query}\":"
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

    private static string SafeHost(string url)
    {
        try { return new Uri(url).Host; }
        catch (UriFormatException) { return ""; }
    }
}
