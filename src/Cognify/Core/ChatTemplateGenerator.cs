using System.Text;

namespace Cognify.Core;

/// <summary>Value written to <c>cognify.chat_template.generator</c> next to
/// every generated chat template, identifying how the file's
/// <c>tokenizer.chat_template</c> was produced.</summary>
public static class ChatTemplateMeta
{
    public const string Generator = "cognify-builtin-jinja/1";
}

/// <summary>
/// Generates a mind's OWN llama.cpp chat template from its live state — the
/// mind is the model, so the template it ships is composed from what it
/// actually carries, not a hand-written constant: a provenance comment
/// naming the mind, its thinking switch/level/ceiling, capabilities, tools
/// and description (renders nothing, but is extractable from llama-server's
/// GET /props next to the manifest), then the format core: the optional
/// manifest <c>set</c>, a tools branch ONLY when the mind has the tools
/// capability, <c>role: content</c> lines, and a thinking-aware assistant
/// cue. Deterministic: the same mind state always produces the same template
/// bytes. The built-in engine (<see cref="ChatJinja"/>) renders the result,
/// llama.cpp's minja accepts it (comment/set/if/for are all in the subset),
/// and the exporter self-verifies the render before writing it into the GGUF.
/// </summary>
public static class ChatTemplateGenerator
{
    /// <summary>Synthesize the template for <paramref name="mind"/>.
    /// Pass <paramref name="manifestJson"/> (the escaped capability-manifest
    /// JSON) to also embed the <c>set</c> line the exported file carries;
    /// the runtime default omits it.</summary>
    public static string Generate(Mind mind, string? manifestJson = null)
    {
        bool thinking = mind.Thinking?.Enabled == true;
        var capabilities = mind.Modules.Select(m => m.Name)
            .OrderBy(n => n, StringComparer.Ordinal).ToList();
        var toolNames = mind.Tools?.Registry.All().Select(t => t.Name)
            .OrderBy(n => n, StringComparer.Ordinal).ToList() ?? new List<string>();
        bool hasTools = capabilities.Contains("tools") || toolNames.Count > 0;
        var description = mind.Config.Description ?? string.Empty;

        var comment = new StringBuilder("{#- cognify generated chat template: ");
        comment.Append("mind=").Append(Sanitize(mind.Name));
        comment.Append(" | thinking=").Append(thinking ? "on" : "off")
            .Append('/').Append(mind.Config.Thinking.Level.ToString().ToLowerInvariant())
            .Append('/').Append(mind.Config.Thinking.Ceiling.ToString().ToLowerInvariant());
        comment.Append(" | capabilities=")
            .Append(capabilities.Count > 0 ? string.Join(",", capabilities) : "none");
        comment.Append(" | tools=")
            .Append(toolNames.Count > 0 ? string.Join(",", toolNames) : "none");
        comment.Append(" | description=").Append(Sanitize(description));
        comment.Append(" -#}");   // strips the source newline: renders clean
        return comment + "\n" +
               ChatJinja.CognifyChatTemplate(manifestJson, thinking, hasTools);
    }

    /// <summary>Jinja comments end at the first <c>#}</c> — keep template
    /// text and newlines out of the generated provenance comment.</summary>
    private static string Sanitize(string text)
    {
        var cleaned = string.Join(' ',
            text.Split((char[])null!, StringSplitOptions.RemoveEmptyEntries))
            .Replace("#}", "# }");
        return cleaned.Length > 0 ? cleaned : "-";
    }

    /// <summary>Export self-verification: the GENERATED chat template must
    /// render the mind's own format through the built-in engine byte-exactly
    /// before it is written into a GGUF — a manifest <c>set</c> or provenance
    /// comment that leaks into the render, or a broken tools/thinking branch,
    /// aborts the export.</summary>
    public static void VerifyExportedTemplate(string template, bool thinkingEnabled)
    {
        var expected = thinkingEnabled
            ? "user: hello\nassistant: thought:"
            : "user: hello\nassistant:";
        var rendered = ChatJinja.Render(template,
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["messages"] = new List<object?>
                {
                    new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["role"] = "user",
                        ["content"] = "hello",
                    },
                },
            });
        if (rendered != expected)
            throw new JinjaException(
                "generated chat template failed export self-verification: " +
                $"expected {expected}, got {rendered}");
    }
}
