using System.Globalization;
using System.Text.Json;
using Cognify.Core;
using Cognify.Export;
using Xunit;

namespace Cognify.Tests;

/// <summary>
/// Tests for the built-in mini-Jinja engine (ChatJinja) — the C# mirror of
/// CognifyPy's test_jinja.py. Unit tests cover the minja-compatible subset;
/// the golden tests render REAL llama.cpp-served templates — ChatML
/// (OpenHermes), Zephyr, Qwen2.5, Qwen3, Llama-3.1 — fetched from HuggingFace
/// (see Templates/README.md) against canonical contexts, and compare the
/// output byte-for-byte with Goldens/*.txt, the same files the Python engine
/// must reproduce, so both engines stay in lockstep.
/// </summary>
public class ChatJinjaExpressionTests
{
    private static string Render(string template,
        IReadOnlyDictionary<string, object?>? vars = null) =>
        ChatJinja.Render(template, vars);

    private static Dictionary<string, object?> V(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    [Fact]
    public void Literals_and_arithmetic()
    {
        Assert.Equal("7", Render("{{ 1 + 2 * 3 }}"));
        Assert.Equal("9", Render("{{ (1 + 2) * 3 }}"));
        Assert.Equal("3 3.5 1", Render("{{ 7 // 2 }} {{ 7 / 2 }} {{ 7 % 3 }}"));
        Assert.Equal("2.5 -1 4", Render("{{ 2.5 }} {{ -1 }} {{ +4 }}"));
        Assert.Equal("2.0", Render("{{ 2.0 }}"));
    }

    [Fact]
    public void Strings_and_concat()
    {
        Assert.Equal("aba\"b", Render("{{ 'a' + 'b' }}{{ \"a\\\"b\" }}"));
        Assert.Equal("ab5None", Render("{{ 'a' ~ 'b' ~ 5 ~ none }}"));
    }

    [Fact]
    public void Logic_and_comparisons()
    {
        Assert.Equal("True", Render("{{ 1 < 2 and 'a' == 'a' or false }}"));
        Assert.Equal("True", Render("{{ not (1 > 2) }}"));
        Assert.Equal("(True, False, True)", Render("{{ 3 >= 3, 3 <= 2, 1 != 2 }}"));
    }

    [Fact]
    public void Membership()
    {
        Assert.Equal("True", Render("{{ 'ell' in 'hello' }}"));
        Assert.Equal("(True, True)", Render("{{ 2 in [1, 2, 3], 'x' not in 'hi' }}"));
        Assert.Equal("True", Render("{{ 'a' in {'a': 1} }}"));
    }

    [Fact]
    public void Ternary()
    {
        Assert.Equal("yes", Render("{{ 'yes' if 1 > 0 else 'no' }}"));
        Assert.Equal("no", Render("{{ 'yes' if false else 'no' }}"));
    }

    [Fact]
    public void Attribute_and_subscript()
    {
        var ctx = D(("m", D(("role", "user"), ("tags", new List<object?> { "a", "b" }),
            ("deep", D(("x", D(("y", "z"))))))));
        Assert.Equal("user user", Render("{{ m.role }} {{ m['role'] }}", ctx));
        Assert.Equal("b z", Render("{{ m.tags[1] }} {{ m.deep.x.y }}", ctx));
        Assert.Equal("b a", Render("{{ m.tags[-1] }} {{ m.tags[-2] }}", ctx));
        Assert.Equal("ab", Render("{{ m.tags[0:2]|join('') }}", ctx));
    }

    [Fact]
    public void Slices()
    {
        var ctx = D(("xs", new List<object?> { 0L, 1L, 2L, 3L, 4L }));
        Assert.Equal("4 4,3,2,1,0", Render("{{ xs[1:]|length }} {{ xs[::-1]|join(',') }}", ctx));
        Assert.Equal("el", Render("{{ 'hello'[1:3] }}"));
        Assert.Equal("1,3", Render("{{ xs[1:4:2]|join(',') }}", ctx));
    }

    [Fact]
    public void List_and_dict_literals()
    {
        Assert.Equal("1,a,True", Render("{{ [1, 'a', true]|join(',') }}"));
        Assert.Equal("1", Render("{{ {'b': 2, 'a': 1}['a'] }}"));
    }

    [Fact]
    public void Missing_is_undefined()
    {
        Assert.Equal("[]", Render("[{{ nothing }}]"));
        Assert.Equal("False False", Render("{{ nothing is defined }} {{ nothing.deeper is defined }}"));
    }

    [Fact]
    public void Truthiness()
    {
        Assert.Equal("empty zero", Render("{{ '' if '' else 'empty' }} {{ 0 or 'zero' }}"));
        Assert.Equal("[]|2", Render("{{ [] and 'x' }}|{{ 'y' and 2 }}"));
        Assert.Equal("fallback", Render("{{ none or 'fallback' }}"));
    }
}

public class ChatJinjaFilterTests
{
    private static string Render(string template,
        IReadOnlyDictionary<string, object?>? vars = null) =>
        ChatJinja.Render(template, vars);

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    [Fact]
    public void Tojson_sorted_compact_like_minja()
    {
        // llama.cpp renders with nlohmann::json: sorted keys, no spaces.
        Assert.Equal("{\"a\":[1,\"x\"],\"b\":2}",
            Render("{{ d | tojson }}", D(("d", D(("b", 2L), ("a", new List<object?> { 1L, "x" }))))));
        Assert.Equal("{\"s\":\"q\\\"uote\"}", Render("{{ d | tojson }}", D(("d", D(("s", "q\"uote"))))));
    }

    [Fact]
    public void Tojson_indent()
    {
        Assert.Equal("{\n    \"a\": 1\n}",
            Render("{{ d | tojson(indent=4) }}", D(("d", D(("a", 1L))))));
    }

    [Fact]
    public void Trim_first_last()
    {
        Assert.Equal("x|", Render("{{ '  x\\n ' | trim }}|"));
        Assert.Equal("3 2", Render("{{ xs | first }} {{ xs | last }}",
            D(("xs", new List<object?> { 3L, 1L, 2L }))));
    }

    [Fact]
    public void Length_join_items()
    {
        Assert.Equal("4 2", Render("{{ 'abcd' | length }} {{ [1, 2] | count }}"));
        Assert.Equal("a-b", Render("{{ ['a', 'b'] | join('-') }}"));
        Assert.Equal("['a', 1];['b', 2]",
            Render("{{ d | items | join(';') }}", D(("d", D(("b", 2L), ("a", 1L))))));
    }

    [Fact]
    public void Reject_select_with_tests()
    {
        // The exact construct Llama-3.1 uses for builtin tools.
        var ctx = D(("tools", new List<object?> { "code_interpreter", "python", "web_search" }));
        Assert.Equal("code_interpreter, web_search",
            Render("{{ tools | reject('equalto', 'python') | join(', ') }}", ctx));
        Assert.Equal("python", Render("{{ tools | select('equalto', 'python') | join }}", ctx));
        Assert.Equal("1,2", Render("{{ [1, 2, 'a'] | select('number') | join(',') }}"));
    }

    [Fact]
    public void Default()
    {
        Assert.Equal("dflt", Render("{{ missing | default('dflt') }}"));
        Assert.Equal("0 x", Render("{{ none | default(0) }} {{ '' | default('x', true) }}"));
        Assert.Equal("", Render("{{ '' | default('x') }}"));
    }

    [Fact]
    public void String_casing_and_replace()
    {
        Assert.Equal("ABC abc", Render("{{ 'aBc' | upper }} {{ 'aBc' | lower }}"));
        Assert.Equal("Hello world", Render("{{ 'hello world' | capitalize }}"));
        Assert.Equal("bbb", Render("{{ 'aaa' | replace('a', 'b') }}"));
        Assert.Equal("cba 2,1", Render("{{ 'abc' | reverse }} {{ [1, 2] | reverse | join(',') }}"));
    }

    [Fact]
    public void Conversions()
    {
        Assert.Equal("42 4.5 1",
            Render("{{ '42' | int }} {{ '4.5' | float }} {{ 5 | string | length }}"));
        Assert.Equal("7 0", Render("{{ 'bad' | int(7) }} {{ missing | int }}"));
        Assert.Equal("a-b", Render("{{ 'ab' | list | join('-') }}"));
    }

    [Fact]
    public void Unknown_filter_raises()
    {
        Assert.Throws<JinjaException>(() =>
            Render("{{ x | selectattr('a') }}", D(("x", new List<object?> { 1L }))));
    }
}

public class ChatJinjaTestOperatorTests
{
    private static string Render(string template,
        IReadOnlyDictionary<string, object?>? vars = null) =>
        ChatJinja.Render(template, vars);

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    [Fact]
    public void Defined_family()
    {
        var ctx = D(("s", "x"), ("n", null), ("d", D(("a", 1L))));
        Assert.Equal("True False True",
            Render("{{ s is defined }} {{ nope is defined }} {{ nope is undefined }}", ctx));
        Assert.Equal("True False", Render("{{ n is none }} {{ s is none }}", ctx));
    }

    [Fact]
    public void Type_tests()
    {
        var ctx = D(("s", "x"), ("i", 3L), ("f", 1.5), ("b", true),
            ("d", D(("a", 1L))), ("l", new List<object?> { 1L }));
        Assert.Equal("True True True",
            Render("{{ s is string }} {{ i is number }} {{ b is boolean }}", ctx));
        Assert.Equal("True True True True",
            Render("{{ d is mapping }} {{ l is iterable }} {{ i is integer }} {{ f is float }}", ctx));
        Assert.Equal("True False True",
            Render("{{ i is not string }} {{ s is false }} {{ b is true }}", ctx));
    }

    [Fact]
    public void Equalto()
    {
        Assert.Equal("True False True", Render("{{ 5 is equalto 5 }} {{ 5 is eq 4 }} {{ 5 is ne 4 }}"));
    }
}

public class ChatJinjaStatementTests
{
    private static string Render(string template,
        IReadOnlyDictionary<string, object?>? vars = null) =>
        ChatJinja.Render(template, vars);

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    private static Dictionary<string, object?> Loop(int index0, int length) =>
        D(("index", (long)(index0 + 1)), ("index0", (long)index0),
          ("first", index0 == 0), ("last", index0 == length - 1),
          ("length", (long)length), ("revindex", (long)(length - index0)),
          ("revindex0", (long)(length - index0 - 1)));

    [Fact]
    public void If_elif_else()
    {
        var tpl = "{% if x == 1 %}one{% elif x == 2 %}two{% else %}many{% endif %}";
        Assert.Equal("one", Render(tpl, D(("x", 1L))));
        Assert.Equal("two", Render(tpl, D(("x", 2L))));
        Assert.Equal("many", Render(tpl, D(("x", 9L))));
    }

    [Fact]
    public void Loop_variables()
    {
        var tpl = "{% for x in xs %}{{ loop.index }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}";
        Assert.Equal("1:a,2:b,3:c",
            Render(tpl, D(("xs", new List<object?> { "a", "b", "c" }))));
        tpl = "{% for x in xs %}{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }} {% endfor %}";
        Assert.Equal("0/2/1 1/1/0 ",
            Render(tpl, D(("xs", new List<object?> { 7L, 8L }))));
    }

    [Fact]
    public void For_else()
    {
        Assert.Equal("empty", Render("{% for x in xs %}{{ x }}{% else %}empty{% endfor %}",
            D(("xs", new List<object?>()))));
        Assert.Equal("1", Render("{% for x in xs %}{{ x }}{% else %}empty{% endfor %}",
            D(("xs", new List<object?> { 1L }))));
    }

    [Fact]
    public void Loop_scoping_set_does_not_leak()
    {
        // Jinja rule: assignments inside a loop stay in the iteration scope.
        Assert.Equal("12[False]", Render(
            "{% for x in xs %}{% set y = x %}{{ y }}{% endfor %}[{{ y is defined }}]",
            D(("xs", new List<object?> { 1L, 2L }))));
    }

    [Fact]
    public void Namespace_accumulates()
    {
        // The Qwen3 pattern: a namespace() bag survives across iterations.
        var tpl = "{%- set ns = namespace(total=0, seen='') %}"
                  + "{% for x in xs %}{% set ns.total = ns.total + x %}"
                  + "{% set ns.seen = ns.seen ~ x %}{% endfor %}"
                  + "{{ ns.total }} {{ ns.seen }}";
        Assert.Equal("6 123", Render(tpl, D(("xs", new List<object?> { 1L, 2L, 3L }))));
    }

    [Fact]
    public void Tuple_unpack()
    {
        Assert.Equal("a=1 b=2 ", Render("{% for k, v in d | items %}{{ k }}={{ v }} {% endfor %}",
            D(("d", D(("a", 1L), ("b", 2L))))));
        Assert.Equal("12", Render("{% set a, b = [1, 2] %}{{ a }}{{ b }}"));
    }

    [Fact]
    public void Set_at_loop_level_leaks_within_loop()
    {
        // if/else create no scope: a set inside `if` inside `for` persists
        // for the rest of the iteration (Llama-3.1 relies on this).
        Assert.Equal("1", Render("{% for x in [1] %}{% if true %}{% set z = x %}{% endif %}{{ z }}{% endfor %}"));
    }

    [Fact]
    public void Generation_tag_renders_body()
    {
        Assert.Equal("AB", Render("{% generation %}AB{% endgeneration %}"));
    }

    [Fact]
    public void Comments_render_nothing()
    {
        Assert.Equal("ab", Render("a{# note #}b"));
        Assert.Equal("ab", Render("a{# it's fine #}b"));
    }
}

public class ChatJinjaWhitespaceMethodTests
{
    private static string Render(string template,
        IReadOnlyDictionary<string, object?>? vars = null) =>
        ChatJinja.Render(template, vars);

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    [Fact]
    public void Default_keeps_newlines()
    {
        Assert.Equal("\n  x\n", Render("{% if true %}\n  x\n{% endif %}"));
    }

    [Fact]
    public void Strip_flags()
    {
        Assert.Equal("ab c", Render("a  {{- 'b' }} c"));
        Assert.Equal("a bc", Render("a {{ 'b' -}}  c"));   // strips ALL following ws
        Assert.Equal("abc", Render("a\n{%- if true %}b{% endif -%}\nc"));
        Assert.Equal("x 1 y", Render("x {% for i in [1] -%}\n  {{ i }}\n{%- endfor %} y"));
    }

    [Fact]
    public void String_methods()
    {
        var ctx = D(("c", "<think>\nX\n</think>\nY"));
        Assert.Equal("<think>\nX\n", Render("{{ c.split('</think>')[0] }}", ctx));
        Assert.Equal("\nY", Render("{{ c.split('</think>')[-1] }}", ctx));
        Assert.Equal("<think>\nX\n</think>\nY", Render("{{ c.rstrip('\\n') }}", ctx));
        Assert.Equal("True False", Render("{{ 'hi'.startswith('h') }} {{ 'hi'.endswith('x') }}"));
        Assert.Equal("aab == a", Render("{{ 'aab' | trim }} == {{ '  a  '.strip() }}"));
        Assert.Equal("a+b", Render("{{ 'a,b'.replace(',', '+') }}"));
        Assert.Equal("a\nb", Render("{{ '\\n'.join(['a', 'b']) }}"));
    }

    [Fact]
    public void Dict_get_and_items_methods()
    {
        var ctx = D(("d", D(("a", 1L))));
        Assert.Equal("1 dflt", Render("{{ d.get('a') }} {{ d.get('z', 'dflt') }}", ctx));
        Assert.Equal("a 1", Render("{{ d.keys() | join(',') }} {{ d.values() | join(',') }}", ctx));
    }

    [Fact]
    public void Global_functions()
    {
        var ex = Assert.Throws<JinjaException>(() => Render("{{ raise_exception('boom') }}"));
        Assert.Contains("boom", ex.Message);
        Assert.Matches("\\d{4}", Render("{{ strftime_now('%Y') }}"));
        Assert.Equal("0,1,2 1,2,3", Render("{{ range(3) | join(',') }} {{ range(1, 4) | join(',') }}"));
    }

    [Fact]
    public void Errors()
    {
        Assert.Throws<JinjaException>(() => Render("{{ 'x' }} {{ oops"));
        Assert.Contains("without {% endif %}",
            Assert.Throws<JinjaException>(() => Render("{{ 'x' }} {% if true %}")).Message);
        Assert.Contains("unsupported statement 'macro'",
            Assert.Throws<JinjaException>(() => Render("{% macro x() %}{% endmacro %}")).Message);
        Assert.Contains("index 5 out of range",
            Assert.Throws<JinjaException>(() =>
                Render("{{ xs[5] }}", D(("xs", new List<object?> { 1L })))).Message);
    }
}

/// <summary>Golden renders: every real llama.cpp-served template must
/// reproduce the committed bytes — the same files the Python engine is
/// pinned to (Goldens/*.txt, generated once, hand-verified).</summary>
public class ChatJinjaGoldenTests
{
    private static readonly string TemplateDir =
        Path.Combine(AppContext.BaseDirectory, "Templates");
    private static readonly string GoldenDir =
        Path.Combine(AppContext.BaseDirectory, "Goldens");

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    private static Dictionary<string, object?> Msg(string role, string content,
        Dictionary<string, object?>? extra = null)
    {
        var m = D(("role", role), ("content", content));
        if (extra is not null)
            foreach (var p in extra) m[p.Key] = p.Value;
        return m;
    }

    private static List<object?> Tools() => new()
    {
        D(("type", "function"), ("function", D(("name", "get_weather"),
            ("description", "Get the current weather for a city."),
            ("parameters", D(("type", "object"),
                ("properties", D(("city", D(("type", "string"))))),
                ("required", new List<object?> { "city" })))))),
        D(("type", "function"), ("function", D(("name", "get_time"),
            ("description", "Get the current local time."),
            ("parameters", D(("type", "object"), ("properties", D())))))),
    };

    private static readonly string Date = "26 Jul 2024";
    private static readonly string Bos = "<|begin_of_text|>";

    public static TheoryData<string, string, IReadOnlyDictionary<string, object?>> Contexts()
    {
        var qwen3Messages = new List<object?>
        {
            Msg("system", "You are Qwen, created by Alibaba Cloud."),
            Msg("user", "Weather in Tokyo?"),
            Msg("assistant", "<think>\nI should call a tool for this.\n</think>\nLet me check."),
            Msg("user", "And the time?"),
        };
        var data = new TheoryData<string, string, IReadOnlyDictionary<string, object?>>
        {
            { "chatml_gen", "chatml.jinja", D(("messages", new List<object?>
                {
                    Msg("user", "Hello!"),
                    Msg("assistant", "Hi there!"),
                    Msg("user", "How are you?"),
                }), ("add_generation_prompt", true)) },
            { "zephyr_gen", "zephyr.jinja", D(("messages", new List<object?>
                {
                    Msg("system", "You are a friendly chatbot."),
                    Msg("user", "Hello!"),
                    Msg("assistant", "Hi there!"),
                }), ("eos_token", "</s>"), ("add_generation_prompt", true)) },
            { "qwen25_tools_gen", "qwen25.jinja", D(("messages", new List<object?>
                {
                    Msg("system", "You are Qwen, created by Alibaba Cloud."),
                    Msg("user", "Weather in Tokyo?"),
                }), ("tools", Tools()), ("add_generation_prompt", true)) },
            { "qwen3_tools_gen_think", "qwen3.jinja", D(("messages", qwen3Messages),
                ("tools", Tools()), ("add_generation_prompt", true)) },
            { "qwen3_tools_gen_nothink", "qwen3.jinja", D(("messages", qwen3Messages),
                ("tools", Tools()), ("add_generation_prompt", true), ("enable_thinking", false)) },
            { "qwen3_notools_gen", "qwen3.jinja", D(("messages", qwen3Messages.GetRange(0, 2)),
                ("add_generation_prompt", true)) },
            { "llama31_tools_gen", "llama31.jinja", D(("messages", new List<object?>
                {
                    Msg("system", "  You are a helpful assistant.  "),
                    Msg("user", "What is the weather in Tokyo?"),
                }), ("tools", Tools()), ("date_string", Date), ("bos_token", Bos),
                ("add_generation_prompt", true)) },
            { "llama31_round_gen", "llama31.jinja", D(("messages", new List<object?>
                {
                    Msg("user", "hi"),
                    Msg("assistant", "", D(("tool_calls", new List<object?>
                    {
                        D(("function", D(("name", "get_weather"),
                            ("arguments", D(("city", "Tokyo")))))),
                    }))),
                    Msg("tool", "", D(("content", D(("temp_c", 21L))))),
                }), ("tools", Tools()), ("date_string", Date), ("bos_token", Bos),
                ("add_generation_prompt", true)) },
        };
        return data;
    }

    [Theory]
    [MemberData(nameof(Contexts))]
    public void Real_templates_match_goldens(string goldenName, string fixture,
        IReadOnlyDictionary<string, object?> variables)
    {
        var template = File.ReadAllText(Path.Combine(TemplateDir, fixture));
        var golden = File.ReadAllText(Path.Combine(GoldenDir, goldenName + ".txt"));
        Assert.Equal(golden, ChatJinja.Render(template, variables));
    }

    /// <summary>The capability manifest rides inside the exported template as
    /// a Jinja set (renders nothing) — extractable for GET /props readers.</summary>
    private static string ManifestEscaped() =>
        JsonSerializer.Serialize(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["version"] = 1L,
            ["description"] = "Yume's mind",
            ["thinking"] = D(("enabled", true), ("level", "balanced"), ("ceiling", "high")),
        }).Replace("'", "\\u0027");

    [Fact]
    public void Cognify_template_goldens()
    {
        var variables = D(("messages", new List<object?> { Msg("user", "hello") }));
        Assert.Equal(Golden("cognify_plain"),
            ChatJinja.Render(ChatJinja.CognifyChatTemplate(), variables));
        Assert.Equal(Golden("cognify_think_on"),
            ChatJinja.Render(ChatJinja.CognifyChatTemplate(ManifestEscaped(), true),
                new Dictionary<string, object?>(variables, StringComparer.Ordinal)));
        var noThink = new Dictionary<string, object?>(variables, StringComparer.Ordinal);
        noThink["enable_thinking"] = false;
        Assert.Equal(Golden("cognify_think_off_kw"),
            ChatJinja.Render(ChatJinja.CognifyChatTemplate(ManifestEscaped(), true), noThink));
        var tools = new Dictionary<string, object?>(variables, StringComparer.Ordinal);
        tools["tools"] = new List<object?>
            { D(("name", "clock"), ("description", "tells the time")) };
        tools["add_generation_prompt"] = true;
        Assert.Equal(Golden("cognify_tools"),
            ChatJinja.Render(ChatJinja.CognifyChatTemplate(null, true), tools));
    }

    private static string Golden(string name) =>
        File.ReadAllText(Path.Combine(GoldenDir, name + ".txt"));

    [Fact]
    public void Manifest_set_line_renders_nothing_and_is_extractable()
    {
        var template = ChatJinja.CognifyChatTemplate(ManifestEscaped(), true);
        var rendered = ChatJinja.Render(template,
            D(("messages", new List<object?>())));
        Assert.DoesNotContain("cognify_manifest", rendered);
        Assert.Equal("assistant: thought:", rendered);
        var start = template.IndexOf("cognify_manifest = '", StringComparison.Ordinal)
                    + "cognify_manifest = '".Length;
        var end = template.IndexOf("' %}", start, StringComparison.Ordinal);
        var embedded = template[start..end].Replace("\\u0027", "'");
        using var doc = JsonDocument.Parse(embedded);
        Assert.Equal("Yume's mind", doc.RootElement.GetProperty("description").GetString());
        Assert.True(doc.RootElement.GetProperty("thinking").GetProperty("enabled").GetBoolean());
    }

    [Fact]
    public void Qwen3_reasoning_content_message()
    {
        // Qwen3 renders message.reasoning_content into the <think> block.
        var messages = new List<object?>
        {
            Msg("user", "Hi"),
            Msg("assistant", "Answer.", D(("reasoning_content", "Chain of thought"))),
        };
        var out_ = ChatJinja.Render(
            File.ReadAllText(Path.Combine(TemplateDir, "qwen3.jinja")),
            D(("messages", messages), ("add_generation_prompt", true)));
        Assert.Contains("<think>\nChain of thought\n</think>\n\nAnswer.", out_);
    }

    [Fact]
    public void Llama31_raise_exception_on_tool_roundtrip()
    {
        // Llama-3.1 raises when tools must ride in the first user message
        // but there is none left: a lone system message is consumed first,
        // leaving messages empty at the guard.
        var template = File.ReadAllText(Path.Combine(TemplateDir, "llama31.jinja"));
        var ex = Assert.Throws<JinjaException>(() => ChatJinja.Render(template,
            D(("messages", new List<object?> { Msg("system", "sys") }),
              ("tools", Tools()), ("date_string", Date), ("bos_token", Bos))));
        Assert.Contains("no first user message", ex.Message);
    }
}

public class ChatJinjaMindTests
{
    private static readonly string TemplateDir =
        Path.Combine(AppContext.BaseDirectory, "Templates");

    private static Mind MakeMind(bool thinking) =>
        MindBuilder.Create("Yume").WithThinking(enabled: thinking).Build();

    [Fact]
    public void Default_template_is_the_exported_format()
    {
        var mind = MakeMind(thinking: false);
        Assert.Equal("user: hello\nassistant:",
            mind.ApplyChatTemplate(new object?[] { ("user", "hello") }));
        Assert.Equal("user: hello\nassistant:",
            mind.ApplyChatTemplate(new object?[]
            {
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["role"] = "user", ["content"] = "hello",
                },
            }));
    }

    [Fact]
    public void Thinking_mind_cue_and_kwarg()
    {
        var mind = MakeMind(thinking: true);
        Assert.Equal("user: hello\nassistant: thought:",
            mind.ApplyChatTemplate(new object?[] { ("user", "hello") }));
        Assert.Equal("user: hello\nassistant:",
            mind.ApplyChatTemplate(new object?[] { ("user", "hello") },
                extraVariables: new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["enable_thinking"] = false,
                }));
    }

    [Fact]
    public void Tuple_and_dict_messages_normalize()
    {
        var mind = MakeMind(thinking: false);
        var mixed = new List<object?>
        {
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["role"] = "system", ["content"] = "be nice",
            },
            ("user", "hi"),
        };
        Assert.Equal("system: be nice\nuser: hi\nassistant:",
            mind.ApplyChatTemplate(mixed));
    }

    [Fact]
    public void Bos_eos_default_to_special_tokens()
    {
        var mind = MakeMind(thinking: false);
        var chatml = "{%- for m in messages %}{{ bos_token }}{{ m['role'] }}:{{ eos_token }}{% endfor %}";
        Assert.Equal("<bos>user:<eos>",
            mind.ApplyChatTemplate(new object?[] { ("user", "hi") }, template: chatml));
    }

    [Fact]
    public void Custom_llama_cpp_template()
    {
        var mind = MakeMind(thinking: false);
        var chatml = File.ReadAllText(Path.Combine(TemplateDir, "chatml.jinja"));
        var out_ = mind.ApplyChatTemplate(new object?[] { ("user", "Hello!") },
            template: chatml, addGenerationPrompt: true);
        Assert.EndsWith("<|im_start|>assistant\n", out_);
    }

    [Fact]
    public void Tools_block()
    {
        // tools-capable minds generate a template WITH the tools branch...
        var mind = MindBuilder.Create("Yume").WithThinking(enabled: false)
            .WithTools().Build();
        var tools = new List<Dictionary<string, object?>>
        {
            new(StringComparer.Ordinal) { ["name"] = "clock" },
        };
        var out_ = mind.ApplyChatTemplate(new object?[] { ("user", "hi") }, tools: tools);
        Assert.StartsWith("tools: [{\"name\":\"clock\"}]\nuser: hi\n", out_);
        // ...and plain minds generate one WITHOUT it: runtime tools are
        // ignored, matching the capability the GGUF documents
        var plain = MindBuilder.Create("Yume").WithThinking(enabled: false).Build();
        Assert.Equal("user: hi\nassistant:",
            plain.ApplyChatTemplate(new object?[] { ("user", "hi") }, tools: tools));
    }

    [Fact]
    public void Add_generation_prompt_is_always_defined()
    {
        var mind = MakeMind(thinking: false);
        var tpl = "[{% if add_generation_prompt %}G{% endif %}]";
        Assert.Equal("[]", mind.ApplyChatTemplate(Array.Empty<object?>(), template: tpl));
        Assert.Equal("[G]", mind.ApplyChatTemplate(Array.Empty<object?>(),
            template: tpl, addGenerationPrompt: true));
    }

    [Fact]
    public void Exported_template_renders_with_builtin_engine()
    {
        var dir = Path.Combine(Path.GetTempPath(), "cognify-chatjinja-export");
        Directory.CreateDirectory(dir);
        var mind = MindBuilder.Create("Yume").WithThinking().Build();
        string corpus = Path.Combine(dir, "corpus.txt");
        File.WriteAllLines(corpus, new[]
            { "Yume is friendly.", "Birds are animals.", "The sky is blue." });
        mind.TrainFile(corpus);
        mind.TrainDeepModel(1);

        var path = Path.Combine(dir, "yume.gguf");
        MindExporter.ExportGgufForLlamaCpp(mind, path);
        var gguf = new GgufReader(path);
        var template = gguf.Metadata["tokenizer.chat_template"].AsString();

        // The built-in engine renders the file it ships with.
        Assert.Equal("user: hello\nassistant: thought:",
            ChatJinja.Render(template,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["messages"] = new List<object?>
                    {
                        new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["role"] = "user", ["content"] = "hello",
                        },
                    },
                }));
        Assert.Equal("user: hello\nassistant:",
            ChatJinja.Render(template,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["messages"] = new List<object?>
                    {
                        new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["role"] = "user", ["content"] = "hello",
                        },
                    },
                    ["enable_thinking"] = false,
                }));

        // Mind default (no manifest line) matches the exported tail.
        Assert.Equal("user: hello\nassistant: thought:",
            mind.ApplyChatTemplate(new object?[] { ("user", "hello") }));
    }
}

/// <summary>The mind generates its OWN chat template from its live state;
/// the exporter embeds that generated template and self-verifies its render.
/// Mirrors CognifyPy's TestTemplateGeneration.</summary>
public class ChatTemplateGenerationTests
{
    private static Mind MakeMind(bool thinking = true, bool tools = false)
    {
        var builder = MindBuilder.Create("Yume").WithThinking(enabled: thinking);
        if (tools) builder = builder.WithTools();
        return builder.Build();
    }

    private static Dictionary<string, object?> D(params (string K, object? V)[] pairs)
    {
        var d = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (k, v) in pairs) d[k] = v;
        return d;
    }

    private static string Render(string template,
        IReadOnlyDictionary<string, object?>? vars = null) =>
        ChatJinja.Render(template, vars);

    [Fact]
    public void Thinking_variant_shapes_the_cue()
    {
        var on = MakeMind(thinking: true).GenerateChatTemplate();
        Assert.Contains("enable_thinking is defined", on);
        Assert.Contains("assistant: thought:", on);
        var off = MakeMind(thinking: false).GenerateChatTemplate();
        Assert.DoesNotContain("enable_thinking", off);
        Assert.EndsWith("assistant:", off);
        Assert.DoesNotContain("thought:", off);
    }

    [Fact]
    public void Tools_block_only_for_tools_minds()
    {
        var plain = MakeMind(tools: false).GenerateChatTemplate();
        Assert.DoesNotContain("tools | tojson", plain);   // branch not even emitted
        var rich = MakeMind(tools: true).GenerateChatTemplate();
        Assert.Contains("{%- if tools %}tools: {{ tools | tojson }}", rich);
        var out_ = Render(rich, D(("messages", new List<object?>
        {
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["role"] = "user", ["content"] = "hi",
            },
        }), ("tools", new List<object?>
        {
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["name"] = "clock",
            },
        })));
        Assert.StartsWith("tools: [{\"name\":\"clock\"}]\n", out_);
    }

    [Fact]
    public void Provenance_comment_carries_the_mind()
    {
        var mind = MakeMind(tools: true);
        var template = mind.GenerateChatTemplate();
        Assert.StartsWith("{#- cognify generated chat template: ", template);
        Assert.Contains("mind=Yume", template);
        Assert.Contains("tools=", template);              // builtin registry
        Assert.Contains("clock", template);
        Assert.Contains("tools", template);               // capability names
    }

    [Fact]
    public void Comment_is_sanitized()
    {
        var mind = MindBuilder.Create("Yume")
            .Describe("ends with #} and\nnewlines\ttabs")
            .WithThinking().Build();
        var template = mind.GenerateChatTemplate();
        Assert.Equal(1, template.Split("#}").Length - 1);  // closes exactly once
        Assert.Contains("and newlines tabs", template);
        // the sanitized text cannot terminate the comment early: it renders
        Assert.Equal(thinkingCue(mind),
            Render(template, D(("messages", new List<object?>()))));
    }

    private static string thinkingCue(Mind mind) =>
        mind.Thinking?.Enabled == true ? "assistant: thought:" : "assistant:";

    [Fact]
    public void Deterministic_bytes()
    {
        var manifest = "{\"version\": 1}";
        var a = MakeMind(tools: true).GenerateChatTemplate(manifest);
        var b = MakeMind(tools: true).GenerateChatTemplate(manifest);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Generated_renders_identical_to_the_format_core()
    {
        // The provenance comment and conditional branches render nothing:
        // for the same conversation the generated template must produce the
        // exact bytes of the hand-written format core.
        foreach (var thinking in new[] { true, false })
        {
            var mind = MakeMind(thinking: thinking, tools: true);
            var generated = mind.GenerateChatTemplate("{\"version\": 1}");
            var core = ChatJinja.CognifyChatTemplate("{\"version\": 1}", thinking);
            var variables = D(("messages", new List<object?>
            {
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["role"] = "user", ["content"] = "hello",
                },
            }));
            Assert.Equal(Render(core, variables), Render(generated, variables));
        }
    }

    [Fact]
    public void Mind_default_is_its_own_generated_template()
    {
        var mind = MakeMind();
        var out_ = mind.ApplyChatTemplate(new object?[] { ("user", "hello") });
        Assert.Equal("user: hello\nassistant: thought:", out_);
        Assert.Equal(out_, Render(mind.GenerateChatTemplate(),
            D(("messages", new List<object?>
            {
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["role"] = "user", ["content"] = "hello",
                },
            }), ("bos_token", "<bos>"), ("eos_token", "<eos>"),
                ("add_generation_prompt", false))));
    }

    [Fact]
    public void Export_self_verification()
    {
        var template = MakeMind().GenerateChatTemplate();
        ChatTemplateGenerator.VerifyExportedTemplate(template, thinkingEnabled: true);
        var doctored = template.Replace("assistant: thought:", "assistant:");
        Assert.Contains("self-verification",
            Assert.Throws<JinjaException>(() =>
                ChatTemplateGenerator.VerifyExportedTemplate(doctored,
                    thinkingEnabled: true)).Message);
    }

    [Fact]
    public void Leading_newline_cannot_leak_from_the_set_line()
    {
        // regression: with no tools block, the newline after the manifest
        // `set` had nothing to strip it and leaked into the render
        var mind = MakeMind(thinking: false, tools: false);
        var template = mind.GenerateChatTemplate("{\"version\": 1}");
        Assert.Equal("user: hello\nassistant:",
            Render(template, D(("messages", new List<object?>
            {
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["role"] = "user", ["content"] = "hello",
                },
            }))));
    }

    [Fact]
    public void Exported_template_is_generated_and_self_verified()
    {
        var dir = Path.Combine(Path.GetTempPath(), "cognify-gen-template-export");
        Directory.CreateDirectory(dir);
        var mind = MindBuilder.Create("Yume").WithThinking().WithTools().Build();
        string corpus = Path.Combine(dir, "corpus.txt");
        File.WriteAllLines(corpus, new[]
            { "Yume is friendly.", "Birds are animals.", "The sky is blue." });
        mind.TrainFile(corpus);
        mind.TrainDeepModel(1);

        var path = Path.Combine(dir, "yume.gguf");
        MindExporter.ExportGgufForLlamaCpp(mind, path);
        var gguf = new GgufReader(path);
        var template = gguf.Metadata["tokenizer.chat_template"].AsString();

        // the embedded template IS the mind's own generated one (with the
        // escaped manifest line) — and the generator KV marker travels along
        Assert.Equal("cognify-builtin-jinja/1",
            gguf.Metadata["cognify.chat_template.generator"].AsString());
        Assert.StartsWith("{#- cognify generated chat template: mind=Yume", template);
        Assert.Contains("clock", template);

        // it still renders the mind's own format through the built-in engine
        Assert.Equal("user: hello\nassistant: thought:",
            ChatJinja.Render(template,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["messages"] = new List<object?>
                    {
                        new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["role"] = "user", ["content"] = "hello",
                        },
                    },
                }));
    }
}
