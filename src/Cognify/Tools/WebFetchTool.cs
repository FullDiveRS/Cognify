using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;

namespace Cognify.Tools;

/// <summary>One fetched web page: its url, title and readable text.</summary>
public sealed record WebPage(string Url, string Title, string Text)
{
    /// <summary>Short single-line summary used in reports.</summary>
    public string Excerpt(int max = 260) =>
        Text.Length <= max ? Text : Text[..max].TrimEnd() + "...";
}

/// <summary>
/// Reads a web page the way a person reads: fetches the HTML, throws away
/// the scripts, styles and navigation chrome, and keeps the readable text
/// plus the page title. Failures are classified with the same honest
/// categories as search (offline / blocked / nothing readable).
/// </summary>
public static class WebPageReader
{
    private static readonly Regex ScriptStyleRegex =
        new(@"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex CommentRegex =
        new(@"<!--.*?-->", RegexOptions.Compiled | RegexOptions.Singleline);
    private static readonly Regex TagRegex = new(@"<[^>]+>", RegexOptions.Compiled);
    private static readonly Regex TitleRegex =
        new(@"<title[^>]*>(.*?)</title>", RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex AnchorRegex =
        new(@"<a[^>]*href=""([^""#]+)""[^>]*>(.*?)</a>",
            RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex WhitespaceRegex = new(@"\s+", RegexOptions.Compiled);

    /// <summary>A browser-like User-Agent (sites bot-check unfamiliar clients).</summary>
    public const string BrowserUserAgent = HttpSearchEngineBase.BrowserUserAgent;

    /// <summary>
    /// A descriptive bot User-Agent used as a FALLBACK. Some sites
    /// (Wikimedia among them) 403 the fake-Chrome UA from datacenter IPs but
    /// happily serve an honest, descriptive bot agent — their policy prefers
    /// identifiable clients over spoofed browsers.
    /// </summary>
    public const string BotUserAgent =
        "Mozilla/5.0 (compatible; CognifyBot/1.0; +https://github.com/cognify-ai)";

    /// <summary>Creates an HttpClient with browser-like headers (caller owns it).</summary>
    public static HttpClient CreateClient(TimeSpan? timeout = null)
    {
        var http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(12) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd(BrowserUserAgent);
        http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
        return http;
    }

    /// <summary>
    /// Fetches one page and extracts (title, readable text). Throws
    /// <see cref="WebSearchException"/> with an honest reason on failure.
    /// When the site REFUSES the browser-like client (403/429), the read is
    /// retried once with the descriptive bot agent — same page, honest UA.
    /// </summary>
    public static (string Title, string Text) Read(HttpClient http, string url)
    {
        try
        {
            return ReadOnce(http, url);
        }
        catch (WebSearchException ex) when (ex.Reason == WebSearchFailure.Blocked)
        {
            // One honest-UA retry before giving up.
            using var botClient = new HttpClient { Timeout = http.Timeout };
            botClient.DefaultRequestHeaders.UserAgent.ParseAdd(BotUserAgent);
            botClient.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
            return ReadOnce(botClient, url);
        }
    }

    private static (string Title, string Text) ReadOnce(HttpClient http, string url)
    {
        string html;
        try
        {
            using var response = http.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
            if (response.StatusCode is HttpStatusCode.Forbidden or HttpStatusCode.TooManyRequests
                or HttpStatusCode.ServiceUnavailable)
                throw new WebSearchException($"the site refused the request ({(int)response.StatusCode})",
                    WebSearchFailure.Blocked);
            response.EnsureSuccessStatusCode();
            html = response.Content.ReadAsStringAsync().ConfigureAwait(false)
                       .GetAwaiter().GetResult();
        }
        catch (WebSearchException) { throw; }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or WebException)
        {
            throw new WebSearchException($"the page could not be reached ({ex.Message})",
                WebSearchFailure.Offline, ex);
        }

        var (title, text) = Parse(html);
        if (text.Length < 80)
            throw new WebSearchException("the page had no readable text (scripts-only or a bot-check)",
                WebSearchFailure.Blocked);
        return (title, text);
    }

    /// <summary>Extracts (title, readable text) from raw HTML (exposed for offline tests).</summary>
    public static (string Title, string Text) Parse(string html)
    {
        var title = TitleRegex.Match(html) is { Success: true } m
            ? WebUtility.HtmlDecode(WhitespaceRegex.Replace(TagRegex.Replace(m.Groups[1].Value, " "), " ").Trim())
            : "";

        var body = ScriptStyleRegex.Replace(html, " ");
        body = CommentRegex.Replace(body, " ");
        body = TagRegex.Replace(body, " ");
        var text = WebUtility.HtmlDecode(body);
        text = WhitespaceRegex.Replace(text, " ").Trim();
        return (title, text);
    }

    /// <summary>
    /// Extracts internal links from a page as (absoluteUrl, anchorText),
    /// same-host only, deduped, anchors stripped of tags. Exposed for tests.
    /// </summary>
    public static IReadOnlyList<(string Url, string Anchor)> ExtractLinks(
        string html, string pageUrl, string? host = null)
    {
        Uri baseUri;
        try { baseUri = new Uri(pageUrl); }
        catch (UriFormatException) { return Array.Empty<(string, string)>(); }
        host ??= baseUri.Host;

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var links = new List<(string Url, string Anchor)>();
        foreach (Match m in AnchorRegex.Matches(html))
        {
            var href = WebUtility.HtmlDecode(m.Groups[1].Value).Trim();
            if (href.Length == 0 || href.StartsWith("javascript:", StringComparison.OrdinalIgnoreCase)
                || href.StartsWith("mailto:", StringComparison.OrdinalIgnoreCase)) continue;
            Uri resolved;
            try { resolved = new Uri(baseUri, href); }
            catch (UriFormatException) { continue; }
            if (!resolved.Host.Equals(host, StringComparison.OrdinalIgnoreCase)) continue;
            if (resolved.Scheme != Uri.UriSchemeHttp && resolved.Scheme != Uri.UriSchemeHttps) continue;
            // Skip pure anchors/files; keep pages.
            var path = resolved.AbsoluteUri;
            if (path.EndsWith(".png") || path.EndsWith(".jpg") || path.EndsWith(".gif")
                || path.EndsWith(".css") || path.EndsWith(".js") || path.EndsWith(".ico")) continue;
            if (!seen.Add(path)) continue;
            var anchor = WhitespaceRegex.Replace(TagRegex.Replace(m.Groups[2].Value, " "), " ").Trim();
            links.Add((path, anchor));
        }
        return links;
    }
}

/// <summary>
/// The site explorer behind <c>/explore</c> and <see cref="Core.Mind.ExploreUrl"/>:
/// reads a starting page, then FOLLOWS internal links (same host, breadth
/// first) so the mind is not locked to one page or one search engine. When a
/// focus suggestion is given ("look for any information on flowers"), the
/// links whose address/anchor text best match the focus words are followed
/// first — the exploration steers toward what was asked.
/// </summary>
public sealed class WebExplorer
{
    private readonly HttpClient _http;
    private readonly HttpClient _botHttp;   // honest-UA fallback (Wikimedia & friends)
    private readonly int _maxPages;

    /// <summary>Creates the explorer (owner of its own HttpClients).</summary>
    public WebExplorer(TimeSpan? timeout = null, int maxPages = 3)
    {
        _http = WebPageReader.CreateClient(timeout);
        _botHttp = WebPageReader.CreateClient(timeout);
        _botHttp.DefaultRequestHeaders.UserAgent.Clear();
        _botHttp.DefaultRequestHeaders.UserAgent.ParseAdd(WebPageReader.BotUserAgent);
        _maxPages = Math.Clamp(maxPages, 1, 6);
    }

    /// <summary>How many pages one exploration reads at most.</summary>
    public int MaxPages => _maxPages;

    /// <summary>
    /// Explores a starting page plus up to <see cref="MaxPages"/>-1 internal
    /// links. The first page is always included; exploration stops early on
    /// failures (a page that cannot be read is skipped, not fatal). Focus
    /// words (from the user's suggestion) rank which links get followed.
    /// </summary>
    public IReadOnlyList<WebPage> Explore(string startUrl, string? focus = null)
    {
        if (!Uri.TryCreate(startUrl, UriKind.Absolute, out var start)
            || (start.Scheme != Uri.UriSchemeHttp && start.Scheme != Uri.UriSchemeHttps))
            throw new WebSearchException($"not a web url: {startUrl}", WebSearchFailure.NoResults);

        var pages = new List<WebPage>();
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var queue = new Queue<string>();
        queue.Enqueue(start.AbsoluteUri);

        var focusWords = HttpWebSearchBackend.ContentWords(focus ?? "");

        while (queue.Count > 0 && pages.Count < _maxPages)
        {
            var url = queue.Dequeue();
            if (!visited.Add(url)) continue;

            string html;
            string title;
            string text;
            try
            {
                html = FetchHtml(url);
                (title, text) = WebPageReader.Parse(html);
            }
            catch (Exception)
            {
                continue; // one unreadable page never kills the exploration
            }
            if (text.Length < 80) continue;

            pages.Add(new WebPage(url, title, text));

            if (pages.Count >= _maxPages) break;

            // Follow the most promising internal links next.
            var links = WebPageReader.ExtractLinks(html, url)
                .Where(l => !visited.Contains(l.Url));
            if (focusWords.Count > 0)
                links = links.OrderByDescending(l => ScoreLink(l, focusWords));
            foreach (var (linkUrl, _) in links.Take(4))
                queue.Enqueue(linkUrl);
        }

        return pages;
    }

    internal static int ScoreLink((string Url, string Anchor) link, IReadOnlyList<string> focusWords)
    {
        var haystack = $"{link.Url} {link.Anchor}";
        return focusWords.Count(w => haystack.Contains(w, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>GET with the browser-like client; on a refusal (403/429 —
    /// Wikimedia 403s spoofed Chrome from datacenter IPs) one retry with the
    /// descriptive bot agent, then the failure propagates.</summary>
    private string FetchHtml(string url)
    {
        try
        {
            using var response = _http.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
            if (response.StatusCode is HttpStatusCode.Forbidden or HttpStatusCode.TooManyRequests)
                throw new WebSearchException($"the site refused the request ({(int)response.StatusCode})",
                    WebSearchFailure.Blocked);
            response.EnsureSuccessStatusCode();
            return response.Content.ReadAsStringAsync().ConfigureAwait(false)
                       .GetAwaiter().GetResult();
        }
        catch (WebSearchException ex) when (ex.Reason == WebSearchFailure.Blocked)
        {
            // One honest-UA retry before the page is skipped.
            using var response = _botHttp.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            return response.Content.ReadAsStringAsync().ConfigureAwait(false)
                       .GetAwaiter().GetResult();
        }
    }
}

/// <summary>
/// Her own WebFetch tool: reads ONE web page and returns its readable text.
/// MANUAL-ONLY (like <see cref="WebSearchTool"/>): the mind never fetches a
/// page inside the cognitive cycle unless the knowledge-gap auto-lookup was
/// explicitly enabled by the developer. Explicit use: the <c>/explore</c>
/// command, <c>use tool web-fetch</c>, or <c>mind.FetchPage(url)</c>.
/// </summary>
public sealed class WebFetchTool : ITool
{
    private readonly HttpClient _http;
    private readonly int _maxChars;

    /// <summary>Creates the tool (owner of its own HttpClient).</summary>
    public WebFetchTool(TimeSpan? timeout = null, int maxChars = 1200)
    {
        _http = WebPageReader.CreateClient(timeout);
        _maxChars = Math.Clamp(maxChars, 200, 4000);
    }

    /// <inheritdoc />
    public string Name => "web-fetch";

    /// <inheritdoc />
    public string Description => "Reads one web page and returns its title and readable text.";

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Parameters =>
        new Dictionary<string, string> { ["url"] = "the page to read" };

    /// <inheritdoc />
    public IReadOnlyList<string> Triggers => Array.Empty<string>();

    /// <inheritdoc />
    public bool ManualOnly => true;

    /// <inheritdoc />
    public ToolResult Execute(ToolInvocation invocation)
    {
        var url = invocation.Arguments.TryGetValue("url", out var u) && !string.IsNullOrWhiteSpace(u)
            ? u.Trim()
            : invocation.RawInput?.Trim() ?? "";
        if (url.Length == 0 || !Uri.TryCreate(url, UriKind.Absolute, out var uri)
            || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            return ToolResult.Fail("No web url given. Usage: /explore <url> [suggestion].");

        try
        {
            var (title, text) = WebPageReader.Read(_http, url);
            var body = text.Length > _maxChars ? text[.._maxChars].TrimEnd() + "..." : text;
            return ToolResult.Ok($"I read \"{title}\":\n{body}",
                new Dictionary<string, object> { ["url"] = url, ["title"] = title, ["text"] = text });
        }
        catch (WebSearchException ex)
        {
            var advice = ex.Reason switch
            {
                WebSearchFailure.Offline =>
                    "Reading a page needs an internet connection — check the network and try again.",
                WebSearchFailure.Blocked =>
                    "That site would not let me read it (bot-check or scripts-only page). Try another page.",
                _ => "That page had nothing readable on it. Try another page.",
            };
            return ToolResult.Fail($"I could not read {url}: {ex.Message}. {advice}");
        }
    }
}
