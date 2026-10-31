using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using Cognify.Core;

namespace Cognify.Tools;

/// <summary>One link found on a web page: its anchor text and absolute url.</summary>
public sealed record WebLink(string Text, string Url);

/// <summary>
/// One visited web page, reduced to what a mind can actually learn from:
/// the page title, the readable text (scripts, styles, navigation and other
/// chrome stripped out) and the outbound links.
/// </summary>
public sealed record WebPage(string Url, string Title, string Text, IReadOnlyList<WebLink> Links);

/// <summary>
/// The result of exploring the web starting at one url: every page read,
/// their combined text, and the sentences that best answer the query.
/// </summary>
public sealed record WebExploration(
    string StartUrl,
    IReadOnlyList<WebPage> Pages,
    string CombinedText,
    IReadOnlyList<string> KeySentences)
{
    /// <summary>Every page visited after the first one (the "walk" the explorer took).</summary>
    public IEnumerable<WebPage> FollowedPages => Pages.Count > 1 ? Pages.Skip(1) : Enumerable.Empty<WebPage>();
}

/// <summary>
/// Pluggable page fetcher for web exploration. The default is
/// <see cref="HttpWebPageFetcher"/>; implement this to serve fixtures in
/// tests, a caching proxy, or an offline mirror.
/// </summary>
public interface IWebPageFetcher
{
    /// <summary>Fetches the raw HTML (or other text) at <paramref name="url"/>.
    /// Throws <see cref="WebSearchException"/> with the right reason when the
    /// page cannot be read.</summary>
    string Fetch(string url);
}

/// <summary>
/// The default page fetcher: a browser-like GET (search-engine-grade
/// headers), with the same honest failure classification as the search
/// engines — Offline when the network is down, Blocked when the site answers
/// with a bot-check, NoResults when the page is not readable text (a PDF, an
/// image...).
/// </summary>
public sealed class HttpWebPageFetcher : IWebPageFetcher
{
    private readonly HttpClient _http;

    /// <summary>Creates the fetcher (owner of its own HttpClient).</summary>
    public HttpWebPageFetcher(TimeSpan? timeout = null)
    {
        _http = new HttpClient { Timeout = timeout ?? TimeSpan.FromSeconds(12) };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd(HttpSearchEngineBase.BrowserUserAgent);
        _http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
    }

    /// <inheritdoc />
    public string Fetch(string url)
    {
        try
        {
            using var response = _http.GetAsync(url).ConfigureAwait(false).GetAwaiter().GetResult();
            var body = response.Content.ReadAsStringAsync().ConfigureAwait(false)
                           .GetAwaiter().GetResult();
            if ((int)response.StatusCode == 202 ||
                response.StatusCode is HttpStatusCode.Forbidden or HttpStatusCode.TooManyRequests ||
                HttpSearchEngineBase.IsAnomalyPage(body))
                throw new WebSearchException($"the site answered with a bot-check page",
                    WebSearchFailure.Blocked);
            response.EnsureSuccessStatusCode();
            if (LooksBinary(body))
                throw new WebSearchException("the page is not readable text (a PDF, image or download)",
                    WebSearchFailure.NoResults);
            return body;
        }
        catch (WebSearchException) { throw; }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or WebException)
        {
            throw new WebSearchException($"the page could not be opened ({HttpSearchEngineBase.RootCause(ex)})",
                WebSearchFailure.Offline, ex);
        }
    }

    /// <summary>True when the body is a binary file rather than readable text.</summary>
    public static bool LooksBinary(string body)
    {
        if (body.Length == 0) return true;
        var head = body.Length > 512 ? body[..512] : body;
        if (head.Contains('\0')) return true;
        return head.StartsWith("%PDF", StringComparison.Ordinal)
            || head.StartsWith("GIF8", StringComparison.Ordinal)
            || head.StartsWith("\u0089PNG", StringComparison.Ordinal)
            || head.StartsWith("\u00FF\u00D8", StringComparison.Ordinal)   // JPEG
            || head.StartsWith("PK\u0003\u0004", StringComparison.Ordinal); // zip/ooxml
    }
}

/// <summary>
/// Readability-lite HTML extraction: picks the main content fragment
/// (Wikipedia's mw-content-text, then &lt;main&gt;, &lt;article&gt;,
/// &lt;body&gt;), strips scripts/styles/navigation, and turns the rest into
/// plain paragraph text plus a link list. No HTML parser dependency —
/// deliberately regex-based like the rest of the web stack.
/// </summary>
internal static class HtmlRead
{
    private static readonly Regex CommentRegex = new(@"<!--.*?-->",
        RegexOptions.Compiled | RegexOptions.Singleline);
    private static readonly Regex ScriptishRegex = new(
        @"<(script|style|noscript|svg|iframe|select|button|form|header|footer|nav)\b[^>]*>.*?</\1\s*>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex TitleRegex = new(@"<title[^>]*>(.*?)</title>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex OgTitleRegex = new(
        @"<meta[^>]*property=""og:title""[^>]*>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex AnchorRegex = new(
        @"<a\b[^>]*href=(?:""(?<href>[^""]+)""|'(?<href>[^']+)')[^>]*>(?<text>.*?)</a>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex BlockSplitRegex = new(
        @"</p>|<br\s*/?>|</h[1-6]>|</li>|</blockquote>|</figcaption>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    /// <summary>Removes comments, scripts, styles and site chrome, then keeps
    /// the most content-like fragment of the page.</summary>
    public static string MainFragment(string html)
    {
        var cleaned = CommentRegex.Replace(html, " ");
        cleaned = ScriptishRegex.Replace(cleaned, " ");

        // Wikipedia articles: everything worth reading lives in mw-content-text.
        var marker = cleaned.IndexOf("mw-content-text", StringComparison.OrdinalIgnoreCase);
        if (marker >= 0)
            cleaned = cleaned[marker..];

        foreach (var (open, close) in new[] { ("<main", "</main>"), ("<article", "</article>"), ("<body", "</body>") })
        {
            var start = cleaned.IndexOf(open, StringComparison.OrdinalIgnoreCase);
            if (start < 0) continue;
            var end = cleaned.IndexOf(close, StringComparison.OrdinalIgnoreCase);
            if (end > start) cleaned = cleaned[start..(end + close.Length)];
            break;
        }
        return cleaned;
    }

    /// <summary>Extracts the page title (&lt;title&gt;, then og:title).</summary>
    public static string Title(string html)
    {
        var m = TitleRegex.Match(html);
        if (!m.Success)
        {
            var tag = OgTitleRegex.Match(html);
            if (tag.Success)
            {
                var cm = Regex.Match(tag.Value, @"content=""([^""]+)""", RegexOptions.IgnoreCase);
                if (cm.Success) return HttpSearchEngineBase.StripTags(cm.Groups[1].Value);
            }
            return "";
        }
        return HttpSearchEngineBase.StripTags(m.Groups[1].Value);
    }

    /// <summary>
    /// Readable text from the main fragment: block elements become lines,
    /// short lines (menus, breadcrumbs, buttons) are dropped, and the result
    /// is capped at <paramref name="maxChars"/>.
    /// </summary>
    public static string Text(string fragment, int maxChars)
    {
        var lines = new List<string>();
        foreach (var block in BlockSplitRegex.Split(fragment))
        {
            var line = HttpSearchEngineBase.StripTags(block);
            if (line.Length < 40 || !line.Contains(' ')) continue;   // menus, buttons, crumbs
            if (lines.Count > 0 && lines[^1] == line) continue;      // duplicated blocks
            lines.Add(line);
            if (lines.Sum(l => l.Length + 1) >= maxChars) break;
        }

        var text = string.Join("\n", lines);
        if (text.Length == 0)
        {
            // No paragraph structure at all — fall back to stripped body text.
            text = HttpSearchEngineBase.StripTags(fragment);
        }
        if (text.Length > maxChars)
        {
            text = text[..maxChars];
            var lastStop = text.LastIndexOfAny(new[] { '.', '!', '?' });
            if (lastStop > maxChars / 2) text = text[..(lastStop + 1)];
        }
        return text.Trim();
    }

    /// <summary>Outbound links of the page, resolved to absolute urls, chrome
    /// links (javascript:, mailto:, #fragments) filtered out.</summary>
    public static IReadOnlyList<WebLink> Links(string fragment, string baseUrl)
    {
        var links = new List<WebLink>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        Uri? baseUri = null;
        try { baseUri = new Uri(baseUrl); } catch (UriFormatException) { return links; }

        foreach (Match m in AnchorRegex.Matches(fragment))
        {
            var href = m.Groups["href"].Value.Trim();
            if (href.Length == 0 || href.StartsWith('#') ||
                href.StartsWith("javascript:", StringComparison.OrdinalIgnoreCase) ||
                href.StartsWith("mailto:", StringComparison.OrdinalIgnoreCase) ||
                href.StartsWith("tel:", StringComparison.OrdinalIgnoreCase) ||
                href.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
                continue;

            var anchorText = HttpSearchEngineBase.StripTags(m.Groups["text"].Value);
            if (anchorText.Length == 0) continue;

            string absolute;
            try
            {
                absolute = baseUri != null ? new Uri(baseUri, href).AbsoluteUri : href;
            }
            catch (UriFormatException) { continue; }
            if (!absolute.StartsWith("http://", StringComparison.OrdinalIgnoreCase) &&
                !absolute.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                continue;
            if (!seen.Add(absolute)) continue;

            links.Add(new WebLink(anchorText, absolute));
            if (links.Count >= 200) break;
        }
        return links;
    }
}

/// <summary>
/// Query-vs-page relevance scoring, shared by the search-result picker and
/// the in-page link walker. Plain keyword presence, weighted: title/anchor
/// text 3, url 2, body text 1 — with a light plural/singular fold so
/// "flowers" matches "Flower" and vice versa.
/// </summary>
public static class WebRelevance
{
    private const double TitleWeight = 3, UrlWeight = 2, TextWeight = 1;

    /// <summary>Query keywords worth matching: words over two letters, deduplicated.</summary>
    public static IReadOnlyList<string> Keywords(string query) =>
        query.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Select(w => w.Trim().Trim('*', '"', '\'', '!', '?', '.', ',', ':', ';').ToLowerInvariant())
            .Where(w => w.Length > 2)
            .Distinct()
            .ToList();

    /// <summary>Case-insensitive keyword match with a trailing-s fold
    /// ("flowers" also matches "flower", "flower" matches inside "flowers").</summary>
    public static bool Has(string haystack, string keyword)
    {
        if (haystack.Length == 0 || keyword.Length == 0) return false;
        if (haystack.IndexOf(keyword, StringComparison.OrdinalIgnoreCase) >= 0) return true;
        return keyword.EndsWith('s') &&
               haystack.IndexOf(keyword[..^1], StringComparison.OrdinalIgnoreCase) >= 0;
    }

    /// <summary>Relevance of a page (or candidate) to the query keywords.</summary>
    public static double Score(IReadOnlyList<string> keywords, string title, string url, string text)
    {
        if (keywords.Count == 0) return 0;
        double score = 0;
        var lowerUrl = url.ToLowerInvariant();
        foreach (var kw in keywords)
        {
            if (Has(title, kw)) score += TitleWeight;
            if (lowerUrl.Contains(kw, StringComparison.Ordinal) ||
                (kw.EndsWith('s') && lowerUrl.Contains(kw[..^1], StringComparison.Ordinal)))
                score += UrlWeight;
            if (Has(text, kw)) score += TextWeight;
        }
        return score;
    }

    /// <summary>The link on the page whose anchor text + url best matches the
    /// query — the step from a site's front door to the real article.</summary>
    public static WebLink? BestLink(WebPage page, IReadOnlyList<string> keywords,
        IReadOnlySet<string>? excludeUrls = null)
    {
        WebLink? best = null;
        var bestScore = 0.0;
        foreach (var link in page.Links)
        {
            if (excludeUrls is not null && excludeUrls.Contains(link.Url)) continue;
            var score = Score(keywords, link.Text, link.Url, "");
            if (score > bestScore)
            {
                bestScore = score;
                best = link;
            }
        }
        return best;
    }

    /// <summary>Re-orders search results by relevance to the query (stable:
    /// ties keep the engine's original ranking).</summary>
    public static IReadOnlyList<Cognify.Tools.WebSearchResult> RankResults(string query,
        IReadOnlyList<Cognify.Tools.WebSearchResult> results)
    {
        var keywords = Keywords(query);
        return results
            .Select((r, i) => (r, i, Score(keywords, r.Title, r.Url, r.Snippet)))
            .OrderByDescending(t => t.Item3)
            .ThenBy(t => t.i)
            .Select(t => t.r)
            .ToList();
    }

    /// <summary>
    /// The sentences of a text that best answer the query — this is what the
    /// mind reads closely, remembers and distills facts from when a page is
    /// far larger than the learning budget. Returned in reading order, capped
    /// at <paramref name="maxChars"/> characters in total.
    /// </summary>
    public static IReadOnlyList<string> KeySentences(string text, IReadOnlyList<string> keywords,
        int maxSentences = 8, int maxChars = 700)
    {
        if (text.Length == 0) return Array.Empty<string>();
        var sentences = Regex.Split(text.Replace('\n', ' '), @"(?<=[.!?])\s+")
            .Select(s => s.Trim())
            .Where(s => s.Length >= 25 && s.Length <= 320)
            .ToList();
        if (sentences.Count == 0) return Array.Empty<string>();

        var best = sentences.Select((s, i) =>
            {
                double score = 0;
                foreach (var kw in keywords)
                    if (Has(s, kw)) score += 1;
                if (i < sentences.Count * 0.4) score += 0.5;   // lead bias: intros matter
                return (s, i, score);
            })
            .OrderByDescending(t => t.score)
            .ThenBy(t => t.i)
            .Take(maxSentences)
            .OrderBy(t => t.i)   // restore reading order
            .ToList();

        var picked = new List<string>();
        var budget = maxChars;
        foreach (var (s, _, _) in best)
        {
            if (s.Length > budget) break;
            picked.Add(s);
            budget -= s.Length + 1;
        }
        return picked;
    }
}

/// <summary>
/// Reads INTO websites: fetches pages, extracts the readable text and links,
/// and — when the starting page is a poor match for the query — walks through
/// the in-page link that matches the query best (a site's front door becomes
/// the actual article). Bounded by <c>WebExplorationOptions.MaxPages</c> so
/// an exploration can never crawl away.
/// </summary>
public sealed class WebExplorer
{
    private readonly IWebPageFetcher _fetcher;
    private readonly WebExplorationOptions _options;

    /// <summary>Creates the explorer (default fetcher: the HTTP one).</summary>
    public WebExplorer(IWebPageFetcher? fetcher = null, WebExplorationOptions? options = null)
    {
        _fetcher = fetcher ?? new HttpWebPageFetcher();
        _options = options ?? new WebExplorationOptions();
    }

    /// <summary>The fetcher this explorer reads through (for diagnostics).</summary>
    public IWebPageFetcher Fetcher => _fetcher;

    /// <summary>Fetches one page and reduces it to title + readable text + links.</summary>
    public WebPage FetchPage(string url)
    {
        var html = _fetcher.Fetch(url);
        var fragment = HtmlRead.MainFragment(html);
        return new WebPage(
            url,
            HtmlRead.Title(html),
            HtmlRead.Text(fragment, _options.MaxPageChars),
            HtmlRead.Links(fragment, url));
    }

    /// <summary>
    /// Explores the web for <paramref name="query"/> starting at
    /// <paramref name="url"/>: reads the page, and while the pages budget
    /// allows, steps through the most query-relevant in-page link when that
    /// is more relevant than the current page. An empty query reads exactly
    /// the one page.
    /// </summary>
    public WebExploration Explore(string query, string url)
    {
        var keywords = WebRelevance.Keywords(query);
        var pages = new List<WebPage>();
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { Normalize(url) };

        var current = FetchPage(url);
        pages.Add(current);

        while (pages.Count < Math.Max(1, _options.MaxPages) && _options.FollowLinks && keywords.Count > 0)
        {
            var here = WebRelevance.Score(keywords, current.Title, current.Url, current.Text);
            var next = WebRelevance.BestLink(current, keywords, visited);
            if (next is null) break;
            var prospect = WebRelevance.Score(keywords, next.Text, next.Url, "");
            if (prospect <= here) break;   // only step onto strictly better ground

            current = FetchPage(next.Url);
            pages.Add(current);
            visited.Add(Normalize(next.Url));
        }

        var combined = string.Join("\n", pages.Select(p => p.Text));
        return new WebExploration(url, pages, combined,
            WebRelevance.KeySentences(combined, keywords));
    }

    private static string Normalize(string url) =>
        url.TrimEnd('/').ToLowerInvariant();
}

/// <summary>
/// The web-exploration tool: opens a page and reads it — the readable text,
/// not just what a search snippet said. MANUAL-ONLY like web-search: it has
/// no trigger phrases and never fires inside the cognitive cycle; the mind
/// goes to a specific page only when the user (or program) asks for it.
/// </summary>
public sealed class WebExploreTool : ITool
{
    private readonly WebExplorer _explorer;

    /// <summary>Creates the tool over an explorer (default: HTTP fetcher + the mind's options).</summary>
    public WebExploreTool(IWebPageFetcher? fetcher = null, WebExplorationOptions? options = null)
    {
        _explorer = new WebExplorer(fetcher, options);
    }

    /// <inheritdoc />
    public string Name => "web-explore";

    /// <inheritdoc />
    public string Description =>
        "Opens a web page and reads it — the readable text of the page itself, not just a search snippet.";

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Parameters =>
        new Dictionary<string, string> { ["url"] = "the page to open and read" };

    /// <inheritdoc />
    public IReadOnlyList<string> Triggers => Array.Empty<string>();

    /// <inheritdoc />
    public bool ManualOnly => true; // never fires on its own

    /// <inheritdoc />
    public ToolResult Execute(ToolInvocation invocation)
    {
        var url = invocation.Arguments.TryGetValue("url", out var u) && !string.IsNullOrWhiteSpace(u)
            ? u.Trim()
            : invocation.RawInput.Trim();
        if (url.Length == 0)
            return ToolResult.Fail("No page address given. Usage: /explore <url>.");
        if (!url.StartsWith("http://", StringComparison.OrdinalIgnoreCase) &&
            !url.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            url = "https://" + url;

        try
        {
            var exploration = _explorer.Explore("", url);
            var page = exploration.Pages[0];
            var host = SafeHost(page.Url);
            var preview = exploration.CombinedText.Length > 0
                ? Mind.Truncate(exploration.CombinedText.Replace('\n', ' '), 300)
                : "(the page had no readable text)";

            var lines = new List<string>
            {
                $"Here is what I read on {host}{(page.Title.Length > 0 ? $" — \"{page.Title}\"" : "")}:",
                $" {preview}",
                $" (readable text: {exploration.CombinedText.Length} characters, links found: {page.Links.Count})",
            };
            return ToolResult.Ok(string.Join("\n", lines), new Dictionary<string, object>
            {
                ["url"] = page.Url,
                ["pages"] = exploration.Pages,
                ["text"] = exploration.CombinedText,
            });
        }
        catch (WebSearchException ex)
        {
            var advice = ex.Reason switch
            {
                WebSearchFailure.Offline =>
                    "Opening a page needs an internet connection — check the network and try again.",
                WebSearchFailure.Blocked =>
                    "That site refused the visit (a bot-check). I can try again later, or you can paste the text for me.",
                _ => "I opened it, but there was nothing readable there. Is the address a text page?",
            };
            return ToolResult.Fail($"I could not read the page \"{url}\": {ex.Message}. {advice}");
        }
    }

    private static string SafeHost(string url)
    {
        try { return new Uri(url).Host; }
        catch (UriFormatException) { return url; }
    }
}
