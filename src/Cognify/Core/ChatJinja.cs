using System.Globalization;
using System.Text;

namespace Cognify.Core;

/// <summary>Thrown when a Jinja chat template cannot be parsed or rendered.</summary>
public sealed class JinjaException : CognifyException
{
    public JinjaException(string message) : base(message) { }
}

/// <summary>Jinja's undefined value: falsy, renders empty, silently propagates
/// (attribute/item access on it returns undefined again, so
/// <c>message.reasoning_content is string</c> works without a key check).</summary>
public sealed class JUndefined
{
    public static readonly JUndefined Instance = new();
    private JUndefined() { }
    public override string ToString() => "";
    public override bool Equals(object? other) => other is JUndefined;
    public override int GetHashCode() => "undefined".GetHashCode();
}

/// <summary>The <c>namespace(...)</c> template function result: a mutable
/// attribute bag, the Jinja-sanctioned way to accumulate state across loop
/// iterations (Qwen3 uses exactly this for its last-query tracking).</summary>
public sealed class JNamespace
{
    private readonly Dictionary<string, object?> _fields =
        new(StringComparer.Ordinal);

    public JNamespace(IEnumerable<KeyValuePair<string, object?>> fields)
    {
        foreach (var (k, v) in fields) _fields[k] = v;
    }

    public bool Has(string name) => _fields.ContainsKey(name);
    public object? Get(string name) => _fields.TryGetValue(name, out var v) ? v : JUndefined.Instance;
    public void Set(string name, object? value) => _fields[name] = value;
    public IReadOnlyDictionary<string, object?> Fields => _fields;
    public override string ToString()
        => "namespace(" + string.Join(", ", _fields.OrderBy(p => p.Key, StringComparer.Ordinal)
               .Select(p => $"{p.Key}={ChatJinja.Repr(p.Value)}")) + ")";
}

/// <summary>
/// Built-in mini-Jinja engine for chat templates — a dependency-free Jinja
/// subset that renders the templates llama.cpp's <c>minja</c> accepts (the
/// same subset real model templates — ChatML, Qwen2.5, Qwen3, Llama-3.1,
/// Zephyr — actually use). Mirrors cognify/core/jinja.py feature-for-feature:
/// if/elif/else, for with the loop variable and for-else, set (plain, tuple
/// and <c>ns.attr</c> targets), <c>{% generation %}</c> blocks, comments,
/// <c>-</c> whitespace control, filters (tojson with sorted keys like
/// llama.cpp's nlohmann dump, trim, length, first, last, join, items,
/// reject/select, default, casing, replace, reverse, conversions), tests
/// (defined/none/string/number/mapping/iterable/equalto/…), the functions
/// <c>raise_exception</c>/<c>strftime_now</c>/<c>namespace</c>/<c>range</c>,
/// slices with negative steps, string/dict methods, and Jinja scoping
/// (assignments inside a loop stay in that iteration — hence namespace()).
/// Macros, includes and auto-escaping are not supported and raise.
/// </summary>
public static class ChatJinja
{
    // ------------------------------------------------------------------ API
    /// <summary>Render <paramref name="source"/> with the built-in engine.
    /// <paramref name="variables"/> supplies the template context — messages,
    /// tools, add_generation_prompt, bos_token, … Missing names behave like
    /// Jinja's undefined (render "").</summary>
    public static string Render(string source,
        IReadOnlyDictionary<string, object?>? variables = null)
    {
        var scopes = new List<Dictionary<string, object?>> { new(variables ?? EmptyVars, StringComparer.Ordinal) };
        var renderer = new Renderer(source, Lexer(source));
        return renderer.Run(scopes);
    }

    private static readonly IReadOnlyDictionary<string, object?> EmptyVars =
        new Dictionary<string, object?>();

    /// <summary>Cognify's own llama.cpp chat template — the format core the
    /// generator composes minds' templates from (<see cref="ChatTemplateGenerator"/>).
    /// The shape the GGUF exporter used to embed by hand: an optional
    /// capability-manifest <c>set</c> (renders nothing; extractable from
    /// llama-server's GET /props), an optional OpenAI-style tools block,
    /// <c>role: content</c> lines, and the assistant cue — which honors
    /// Qwen3's <c>enable_thinking</c> kwarg when the mind thinks.
    /// <paramref name="manifestJson"/> = null (the runtime default) omits
    /// the manifest line; <paramref name="toolsBlock"/>=false drops the
    /// tools branch (minds without the tools capability get a template that
    /// cannot even express tools).</summary>
    public static string CognifyChatTemplate(string? manifestJson = null,
        bool thinkingEnabled = false, bool toolsBlock = true)
    {
        var sb = new StringBuilder();
        if (manifestJson is not null)
        {
            // The newline after the set is source formatting only — emit it
            // only when a left-stripping tag follows; with no tools block
            // the bare newline would otherwise leak into the render.
            sb.Append("{%- set cognify_manifest = '").Append(manifestJson)
                .Append("' %}").Append(toolsBlock ? "\n" : "");
        }
        if (toolsBlock)
            sb.Append("{%- if tools %}tools: {{ tools | tojson }}\n{% endif %}");
        sb.Append("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}");
        if (thinkingEnabled)
            sb.Append("{%- if enable_thinking is defined and not enable_thinking %}" +
                      "assistant:{% else %}assistant: thought:{% endif %}");
        else
            sb.Append("assistant:");
        return sb.ToString();
    }

    // ------------------------------------------------------- value helpers
    internal static bool IsNumber(object? v) => v is long or double;

    internal static bool Truthy(object? v) => v switch
    {
        null => false,
        bool b => b,
        JUndefined => false,
        string s => s.Length > 0,
        long n => n != 0,
        double d => d != 0,
        List<object?> l => l.Count > 0,
        Dictionary<string, object?> d => d.Count > 0,
        _ => true,
    };

    internal static bool EqualsValue(object? a, object? b)
    {
        if (a is JUndefined || b is JUndefined) return a is JUndefined && b is JUndefined;
        if (a is bool || b is bool) return a is bool x && b is bool y && x == y;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) == ToDouble(b);
        if (a is string sa && b is string sb) return sa == sb;
        if (a is null || b is null) return a is null && b is null;
        if (a is List<object?> la && b is List<object?> lb)
            return la.Count == lb.Count && la.Zip(lb).All(p => EqualsValue(p.First, p.Second));
        if (a is Dictionary<string, object?> da && b is Dictionary<string, object?> db)
            return da.Count == db.Count && da.All(p => db.TryGetValue(p.Key, out var v) && EqualsValue(p.Value, v));
        return ReferenceEquals(a, b);
    }

    internal static double ToDouble(object? v) => v switch
    {
        long n => n,
        double d => d,
        _ => throw new JinjaException("expected a number"),
    };

    /// <summary>Python-style str(): True/False/None, lists as ['a', 'b'],
    /// dicts as {'k': 'v'} — what {{ expr }} emits.</summary>
    internal static string Str(object? v) => v switch
    {
        null => "None",
        JUndefined => "",
        true => "True",
        false => "False",
        string s => s,
        long n => n.ToString(CultureInfo.InvariantCulture),
        double d => FormatDouble(d),
        JNamespace ns => ns.ToString(),
        JTuple t => "(" + string.Join(", ", t.Items.Select(Repr))
                    + (t.Items.Count == 1 ? "," : "") + ")",
        List<object?> l => "[" + string.Join(", ", l.Select(Repr)) + "]",
        IReadOnlyList<object?> l => "[" + string.Join(", ", l.Select(Repr)) + "]",
        Dictionary<string, object?> d => "{" + string.Join(", ",
            d.Select(p => $"{Repr(p.Key)}: {Repr(p.Value)}")) + "}",
        _ => v.ToString() ?? "",
    };

    internal static string Repr(object? v) => v is string s
        ? "'" + s.Replace("\\", "\\\\").Replace("'", "\\'") + "'"
        : Str(v);

    internal static string FormatDouble(double d)
    {
        if (double.IsNaN(d)) return "nan";
        if (double.IsPositiveInfinity(d)) return "inf";
        if (double.IsNegativeInfinity(d)) return "-inf";
        // Python repr: integral doubles keep a trailing ".0", others take
        // the shortest round-trip form.
        if (d == Math.Floor(d) && Math.Abs(d) < 1e15)
            return d.ToString("0.0", CultureInfo.InvariantCulture);
        return d.ToString("R", CultureInfo.InvariantCulture);
    }

    // ------------------------------------------------------------- tojson
    /// <summary>minja-compatible tojson: nlohmann::json dump semantics —
    /// dict keys sorted, compact (no spaces) unless an indent is given.</summary>
    private static string ToJson(object? value, int? indent)
    {
        var sb = new StringBuilder();
        WriteJson(sb, value, indent, 0);
        return sb.ToString();
    }

    private static void WriteJson(StringBuilder sb, object? value, int? indent, int depth)
    {
        switch (value)
        {
            case null or JUndefined:
                sb.Append("null");
                break;
            case bool b:
                sb.Append(b ? "true" : "false");
                break;
            case long n:
                sb.Append(n.ToString(CultureInfo.InvariantCulture));
                break;
            case double d:
                sb.Append(FormatDouble(d));
                break;
            case string s:
                WriteJsonString(sb, s);
                break;
            case JNamespace ns:
                WriteJsonSorted(sb, ns.Fields.OrderBy(p => p.Key, StringComparer.Ordinal),
                    indent, depth);
                break;
            case Dictionary<string, object?> dict:
                WriteJsonSorted(sb, dict.OrderBy(p => p.Key, StringComparer.Ordinal),
                    indent, depth);
                break;
            case List<object?> list:
                WriteJsonSeq(sb, list, indent, depth);
                break;
            case IReadOnlyList<object?> list:
                WriteJsonSeq(sb, list, indent, depth);
                break;
            default:
                throw new JinjaException($"tojson cannot serialize {value.GetType().Name}");
        }
    }

    private static void WriteJsonSorted(StringBuilder sb,
        IEnumerable<KeyValuePair<string, object?>> pairs, int? indent, int depth)
    {
        var list = pairs.ToList();
        if (list.Count == 0) { sb.Append("{}"); return; }
        sb.Append('{');
        bool first = true;
        foreach (var (k, v) in list)
        {
            if (!first) sb.Append(',');
            first = false;
            Newline(sb, indent, depth + 1);
            WriteJsonString(sb, k);
            sb.Append(indent is null ? ":" : ": ");
            WriteJson(sb, v, indent, depth + 1);
        }
        Newline(sb, indent, depth);
        sb.Append('}');
    }

    private static void WriteJsonSeq(StringBuilder sb, IEnumerable<object?> items,
        int? indent, int depth)
    {
        var list = items.ToList();
        if (list.Count == 0) { sb.Append("[]"); return; }
        sb.Append('[');
        bool first = true;
        foreach (var v in list)
        {
            if (!first) sb.Append(',');
            first = false;
            Newline(sb, indent, depth + 1);
            WriteJson(sb, v, indent, depth + 1);
        }
        Newline(sb, indent, depth);
        sb.Append(']');
    }

    private static void Newline(StringBuilder sb, int? indent, int depth)
    {
        if (indent is { } n) sb.Append('\n').Append(' ', n * depth);
    }

    internal static void WriteJsonPublic(StringBuilder sb, object? value, int? indent, int depth)
        => WriteJson(sb, value, indent, depth);

    private static void WriteJsonString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                    else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }

    // --------------------------------------------------------------- lexer
    // Tokens: (kind, payload, leftStrip, rightStrip) — kind: text/var/stmt/comment
    private static List<(string Kind, string Payload, bool Left, bool Right)> Lexer(string source)
    {
        var tokens = new List<(string, string, bool, bool)>();
        var openers = new[] { ("{{", "var"), ("{%", "stmt"), ("{#", "comment") };
        var closers = new Dictionary<string, string> { ["var"] = "}}", ["stmt"] = "%}", ["comment"] = "#}" };
        int i = 0, n = source.Length;
        while (i < n)
        {
            int best = -1;
            string kind = "", opener = "";
            foreach (var (open, k) in openers)
            {
                int k2 = source.IndexOf(open, i, StringComparison.Ordinal);
                if (k2 != -1 && (best == -1 || k2 < best)) { best = k2; kind = k; opener = open; }
            }
            if (best == -1)
            {
                tokens.Add(("text", source[i..], false, false));
                break;
            }
            if (best > i)
                tokens.Add(("text", source[i..best], false, false));
            bool left = best + 2 < n && source[best + 2] == '-';
            int start = left ? best + 3 : best + 2;
            string closer = closers[kind];
            int q = start, payloadEnd = -1;
            bool right = false;
            char? quote = null;
            while (q < n)
            {
                char c = source[q];
                if (kind != "comment" && quote is { } qc)   // inside a string
                {                                           // literal: tag
                    if (c == '\\') { q += 2; continue; }    // closers never
                    if (c == qc) quote = null;              // count; comments
                    q += 1;                                 // are plain text
                    continue;
                }
                if (kind != "comment" && (c == '\'' || c == '"')) { quote = c; q += 1; continue; }
                if (q + closer.Length <= n && source[q..(q + closer.Length)] == closer)
                {
                    if (q > start && source[q - 1] == '-') { right = true; payloadEnd = q - 1; }
                    else payloadEnd = q;
                    break;
                }
                q += 1;
            }
            if (payloadEnd < 0)
                throw new JinjaException($"unterminated {kind} tag starting at offset {best}");
            tokens.Add((kind, source[start..payloadEnd].Trim(), left, right));
            i = q + closer.Length;
        }
        return tokens;
    }

    // -------------------------------------------------------- expression lexer
    internal static List<Tok> LexExpr(string src)
    {
        var toks = new List<Tok>();
        int i = 0, n = src.Length;
        while (i < n)
        {
            char c = src[i];
            if (char.IsWhiteSpace(c)) { i += 1; continue; }
            if (char.IsDigit(c) || (c == '.' && i + 1 < n && char.IsDigit(src[i + 1])))
            {
                int j = i;
                while (j < n && (char.IsDigit(src[j]) || src[j] == '.')) j += 1;
                string text = src[i..j];
                toks.Add(new Tok("num", text.Contains('.')
                    ? (object?)double.Parse(text, CultureInfo.InvariantCulture)
                    : long.Parse(text, CultureInfo.InvariantCulture)));
                i = j;
                continue;
            }
            if (c == '\'' || c == '"')
            {
                var sb = new StringBuilder();
                char quote = c;
                i += 1;
                while (i < n && src[i] != quote)
                {
                    if (src[i] == '\\' && i + 1 < n)
                    {
                        char e = src[i + 1];
                        switch (e)
                        {
                            case 'n': sb.Append('\n'); break;
                            case 't': sb.Append('\t'); break;
                            case 'r': sb.Append('\r'); break;
                            case 'b': sb.Append('\b'); break;
                            case 'f': sb.Append('\f'); break;
                            case 'u':
                                sb.Append((char)Convert.ToInt32(src[(i + 2)..(i + 6)], 16));
                                i += 4;
                                break;
                            default: sb.Append(e); break;
                        }
                        i += 2;
                        continue;
                    }
                    sb.Append(src[i]);
                    i += 1;
                }
                if (i >= n) throw new JinjaException($"unterminated string in expression: {src}");
                i += 1;
                toks.Add(new Tok("str", sb.ToString()));
                continue;
            }
            if (char.IsLetter(c) || c == '_')
            {
                int j = i;
                while (j < n && (char.IsLetterOrDigit(src[j]) || src[j] == '_')) j += 1;
                toks.Add(new Tok("name", src[i..j]));
                i = j;
                continue;
            }
            // multi-char operators first
            string two = i + 1 < n ? src[i..(i + 2)] : "";
            if (two is "==" or "!=" or "<=" or ">=" or "//")
            {
                toks.Add(new Tok("op", two));
                i += 2;
                continue;
            }
            if ("+-*/%~|[](){}.,:<>=(".Contains(c))
            {
                toks.Add(new Tok("op", c.ToString()));
                i += 1;
                continue;
            }
            throw new JinjaException($"bad character '{c}' in expression: {src}");
        }
        toks.Add(new Tok("end", null));
        return toks;
    }
}

// [PART2]

/// <summary>A string/list/dict method awaiting a call (m.content.split(x)).</summary>
internal sealed class BoundMethod
{
    public BoundMethod(object? target, string name) { Target = target; Name = name; }
    public object? Target; public string Name;
}

/// <summary>Explicit tuple output node value: renders like Python's
/// "(1, 2)" (single element "(1,"), iterates as a sequence.</summary>
internal sealed class JTuple
{
    public JTuple(List<object?> items) { Items = items; }
    public List<object?> Items;
}

/// <summary>One expression-lexer token (num/str/name/op/end).</summary>
internal sealed class Tok
{
    public Tok(string kind, object? value) { Kind = kind; Value = value; }
    public string Kind; public object? Value;
    public override string ToString() => $"<{Kind} {Value}>";
}

/// <summary>Expression/statement AST nodes: Kind names the node, A–E
/// carry children (boxed Expr/Stmt structs, lists or literal values) —
/// mirrors cognify/core/jinja.py node-for-node.</summary>
internal readonly record struct Expr(string Kind, object? A = null, object? B = null,
    object? C = null, object? D = null, object? E = null);

internal readonly record struct Stmt(string Kind, object? A = null, object? B = null,
    object? C = null, object? D = null);

internal sealed class ExprParser
{
    private readonly List<Tok> _toks;
    private int _i;
    private readonly string _source;

    public ExprParser(string source)
    {
        _source = source;
        _toks = ChatJinja.LexExpr(source);
    }

    // Raw token access for the template parser's for/set target scanning.
    public Tok NextRaw() => Next();
    public string? AcceptOpRaw(string op) => AcceptOp(op);
    public string? AcceptNameRaw(string name) => AcceptName(name);

    public Expr ParseRest()
    {
        var node = Ternary();
        if (Peek().Kind != "end")
            throw new JinjaException($"unexpected '{Peek().Value}' in expression: {_source}");
        return node;
    }

    private Tok Peek() => _toks[_i];
    private Tok Next() => _toks[_i++];

    private string? AcceptOp(params string[] ops)
    {
        var t = Peek();
        if (t.Kind == "op" && ops.Contains((string)t.Value!)) { _i += 1; return (string)t.Value!; }
        return null;
    }

    private string? AcceptName(params string[] names)
    {
        var t = Peek();
        if (t.Kind == "name" && names.Contains((string)t.Value!)) { _i += 1; return (string)t.Value!; }
        return null;
    }

    private void ExpectOp(string op)
    {
        if (AcceptOp(op) is null)
            throw new JinjaException($"expected '{op}' in expression: {_source}");
    }

    public Expr Parse()
    {
        var node = Ternary();
        if (Peek().Kind == "op" && Peek().Value is ",")
        {
            var items = new List<object?> { node };
            while (AcceptOp(",") is not null)
            {
                if (Peek().Kind == "end") break;
                items.Add(Ternary());
            }
            node = new Expr("tuple", items);
        }
        if (Peek().Kind != "end")
            throw new JinjaException($"unexpected '{Peek().Value}' in expression: {_source}");
        return node;
    }

    private Expr Ternary()
    {
        var expr = Or();
        if (AcceptName("if") is not null)
        {
            var cond = Or();
            if (AcceptName("else") is null)
                throw new JinjaException($"expected 'else' in expression: {_source}");
            return new Expr("ternary", cond, expr, Ternary());
        }
        return expr;
    }

    private Expr Or()
    {
        var node = And();
        while (AcceptName("or") is not null) node = new Expr("or", node, And());
        return node;
    }

    private Expr And()
    {
        var node = Not();
        while (AcceptName("and") is not null) node = new Expr("and", node, Not());
        return node;
    }

    private Expr Not()
    {
        if (AcceptName("not") is not null) return new Expr("not", Not());
        return Compare();
    }

    private Expr Compare()
    {
        var node = Concat();
        while (true)
        {
            string? op = AcceptOp("==", "!=", "<=", ">=", "<", ">");
            if (op is not null)
            {
                node = new Expr("binop", op, node, Concat());
                continue;
            }
            if (AcceptName("in") is not null)
            {
                node = new Expr("binop", "in", node, Concat());
                continue;
            }
            if (Peek().Kind == "name" && Peek().Value is "not"
                && _i + 1 < _toks.Count && _toks[_i + 1].Kind == "name"
                && _toks[_i + 1].Value is "in")
            {
                _i += 2;
                node = new Expr("not", new Expr("binop", "in", node, Concat()));
                continue;
            }
            if (AcceptName("is") is not null)
            {
                bool negated = AcceptName("not") is not null;
                var t = Next();
                if (t.Kind != "name")
                    throw new JinjaException($"expected test name after 'is': {_source}");
                var args = new List<object?>();
                if (AcceptOp("(") is not null) args = CallArgs(")").Args;
                else if (TestArgFollows()) args.Add(Unary());
                node = new Expr("test", node, t.Value, negated, args);
                continue;
            }
            return node;
        }
    }

    private static readonly HashSet<string> ReservedAfterTest = new(StringComparer.Ordinal)
    { "defined", "undefined", "none", "string", "number", "integer", "float",
      "boolean", "true", "false", "mapping", "iterable", "sequence",
      "equalto", "eq", "ne", "in", "divisibleby",
      "and", "or", "not", "if", "else", "is" };

    private bool TestArgFollows()
    {
        var t = Peek();
        if (t.Kind is "num" or "str") return true;
        if (t.Kind == "op" && (t.Value is "[" or "{")) return true;
        if (t.Kind == "name") return !ReservedAfterTest.Contains((string)t.Value!);
        return false;
    }

    private Expr Concat()
    {
        var parts = new List<object?> { Math1() };
        while (AcceptOp("~") is not null) parts.Add(Math1());
        return parts.Count == 1 ? (Expr)parts[0]! : new Expr("concat", parts);
    }

    private Expr Math1()
    {
        var node = Math2();
        while (true)
        {
            string? op = AcceptOp("+", "-");
            if (op is null) return node;
            node = new Expr("binop", op, node, Math2());
        }
    }

    private Expr Math2()
    {
        var node = Unary();
        while (true)
        {
            string? op = AcceptOp("*", "/", "//", "%");
            if (op is null) return node;
            node = new Expr("binop", op, node, Unary());
        }
    }

    private Expr Unary()
    {
        if (AcceptOp("-") is not null) return new Expr("neg", Unary());
        if (AcceptOp("+") is not null) return new Expr("pos", Unary());
        return Postfix();
    }

    private Expr Postfix()
    {
        var node = Atom();
        while (true)
        {
            if (AcceptOp(".") is not null)
            {
                var t = Next();
                if (t.Kind != "name")
                    throw new JinjaException($"expected name after '.': {_source}");
                if (Peek().Kind == "op" && Peek().Value is "(")
                {
                    var (args, kwargs) = CallWithParens();
                    node = new Expr("call", new Expr("attr", node, t.Value), args, kwargs);
                }
                else node = new Expr("attr", node, t.Value);
                continue;
            }
            if (AcceptOp("[") is not null) { node = Subscript(node); continue; }
            if (AcceptOp("|") is not null) { node = Filter(node); continue; }
            if (Peek().Kind == "op" && Peek().Value is "(")
            {
                var (args, kwargs) = CallWithParens();
                node = new Expr("call", node, args, kwargs);
                continue;
            }
            return node;
        }
    }

    private (List<object?>, List<KeyValuePair<string, object?>>) CallWithParens()
    {
        ExpectOp("(");
        return CallArgs(")");
    }

    private (List<object?> Args, List<KeyValuePair<string, object?>> Kwargs) CallArgs(string closer)
    {
        var args = new List<object?>();
        var kwargs = new List<KeyValuePair<string, object?>>();
        if (AcceptOp(closer) is not null) return (args, kwargs);
        while (true)
        {
            if (Peek().Kind == "name" && _i + 1 < _toks.Count
                && _toks[_i + 1].Kind == "op" && _toks[_i + 1].Value is "=")
            {
                var name = (string)Next().Value!;
                _i += 1;                                  // '='
                kwargs.Add(new(name, Ternary()));
            }
            else args.Add(Ternary());
            if (AcceptOp(",") is not null)
            {
                if (Peek().Kind == "op" && Peek().Value is var pv
                    && (string?)pv == closer) break;      // trailing comma
                continue;
            }
            ExpectOp(closer);
            return (args, kwargs);
        }
        ExpectOp(closer);
        return (args, kwargs);
    }

    private Expr Subscript(Expr node)
    {
        // '[' already consumed: [expr] | [start:] | [start:stop] |
        // [start:stop:step] | [:stop] | [:stop:step] | [::step] | [::]
        object? start = null, stop = null, step = null;
        if (AcceptOp(":") is not null)
        {
            if (AcceptOp(":") is not null)                 // [::...]
            {
                if (Peek().Kind != "op" || Peek().Value is not "]") step = Ternary();
            }
            else
            {
                if (Peek().Kind != "op" || Peek().Value is not "]") stop = Ternary();
                if (AcceptOp(":") is not null &&
                    (Peek().Kind != "op" || Peek().Value is not "]")) step = Ternary();
            }
            ExpectOp("]");
            return new Expr("slice", node, start, stop, step);
        }
        start = Ternary();
        if (AcceptOp("]") is not null) return new Expr("item", node, start);
        ExpectOp(":");
        if (AcceptOp(":") is not null)                     // [start::...]
        {
            if (Peek().Kind != "op" || Peek().Value is not "]") step = Ternary();
        }
        else
        {
            if (Peek().Kind != "op" || Peek().Value is not "]") stop = Ternary();
            if (AcceptOp(":") is not null &&
                (Peek().Kind != "op" || Peek().Value is not "]")) step = Ternary();
        }
        ExpectOp("]");
        return new Expr("slice", node, start, stop, step);
    }

    private Expr Filter(Expr node)
    {
        var t = Next();
        if (t.Kind != "name")
            throw new JinjaException($"expected filter name after '|': {_source}");
        var args = new List<object?>();
        var kwargs = new List<KeyValuePair<string, object?>>();
        if (Peek().Kind == "op" && Peek().Value is "(")
            (args, kwargs) = CallWithParens();
        return new Expr("filter", node, t.Value, args, kwargs);
    }

    private Expr Atom()
    {
        var t = Peek();
        if (t.Kind is "num" or "str") { _i += 1; return new Expr("lit", t.Value); }
        if (t.Kind == "name")
        {
            _i += 1;
            return (string)t.Value! switch
            {
                "true" => new Expr("lit", true),
                "false" => new Expr("lit", false),
                "none" => new Expr("lit", null),
                _ => new Expr("name", t.Value),
            };
        }
        string? op = AcceptOp("[", "{", "(");
        if (op == "[")
        {
            var items = new List<object?>();
            if (AcceptOp("]") is null)
            {
                while (true)
                {
                    items.Add(Ternary());
                    if (AcceptOp(",") is not null)
                    {
                        if (AcceptOp("]") is not null) break;
                        continue;
                    }
                    ExpectOp("]");
                    break;
                }
            }
            return new Expr("list", items);
        }
        if (op == "{")
        {
            var pairs = new List<KeyValuePair<string, object?>>();
            if (AcceptOp("}") is null)
            {
                while (true)
                {
                    var key = Ternary();
                    ExpectOp(":");
                    if (key.Kind != "lit" || key.A is not string ks)
                        throw new JinjaException("dict literal keys must be strings");
                    pairs.Add(new(ks, Ternary()));
                    if (AcceptOp(",") is not null)
                    {
                        if (AcceptOp("}") is not null) break;
                        continue;
                    }
                    ExpectOp("}");
                    break;
                }
            }
            return new Expr("dict", pairs);
        }
        if (op == "(")
        {
            var inner = Ternary();
            ExpectOp(")");
            return inner;
        }
        throw new JinjaException($"unexpected '{t.Value}' in expression: {_source}");
    }
}

/// <summary>Token stream → node tree. Strip flags follow Jinja: a tag's
/// <c>-</c> left flag right-trims the preceding text, its <c>-</c> right
/// flag left-trims the next text token (cleared by any intervening tag).</summary>
internal sealed class TemplateParser
{
    private readonly List<(string Kind, string Payload, bool Left, bool Right)> _tokens;
    private int _i;
    private bool _pendingLstrip;

    public TemplateParser(List<(string, string, bool, bool)> tokens)
    {
        _tokens = tokens;
    }

    public List<Stmt> Parse()
    {
        var (body, stop, _) = Body(Array.Empty<string>());
        if (stop is not null)
            throw new JinjaException($"unexpected {{% {stop} %}}");
        return body;
    }

    private (List<Stmt> Body, string? Stop, string? Tail) Body(params string[] stops)
    {
        var body = new List<Stmt>();
        while (_i < _tokens.Count)
        {
            var (kind, payload, left, right) = _tokens[_i];
            _i += 1;
            if (kind == "text")
            {
                EmitText(body, payload);
                continue;
            }
            _pendingLstrip = false;              // an intervening tag absorbs
            if (left && body.Count > 0 && body[^1].Kind == "text")
                body[^1] = new Stmt("text", ((string)body[^1].A!).TrimEnd());
            if (kind == "comment")
            {
                _pendingLstrip = right;
                continue;
            }
            if (kind == "var")
            {
                body.Add(new Stmt("out", ParseExpr(payload)));
                _pendingLstrip = right;
                continue;
            }
            // stmt
            var word = payload.Split(' ', 2)[0];
            var rest = payload.Length > word.Length ? payload[(word.Length + 1)..].Trim() : "";
            if (stops.Contains(word))
            {
                _pendingLstrip = right;
                return (body, word, rest);
            }
            // This tag's right-strip applies to the first text INSIDE the
            // block it opens — set before dispatching nested bodies.
            _pendingLstrip = right;
            switch (word)
            {
                case "if":
                    body.Add(If(ParseExpr(rest)));
                    break;
                case "for":
                    body.Add(For(rest));
                    break;
                case "set":
                    body.Add(Set(rest));
                    break;
                case "generation":
                    var (inner, stop, _) = Body("endgeneration");
                    if (stop != "endgeneration")
                        throw new JinjaException("{% generation %} without {% endgeneration %}");
                    body.Add(new Stmt("generation", inner));
                    break;
                case "elif" or "else" or "endif" or "endfor" or "endgeneration":
                    throw new JinjaException($"unexpected {{% {word} %}}");
                default:
                    throw new JinjaException(
                        $"unsupported statement '{word}' (built-in engine supports " +
                        "if/for/set/generation, not macros or includes)");
            }
        }
        return (body, null, null);
    }

    private void EmitText(List<Stmt> body, string payload)
    {
        if (_pendingLstrip)
        {
            payload = payload.TrimStart();
            _pendingLstrip = false;
        }
        if (payload.Length > 0) body.Add(new Stmt("text", payload));
    }

    private static Expr ParseExpr(string src) =>
        new ExprParser(src).Parse();

    private Stmt If(Expr cond)
    {
        var branches = new List<object?>();
        var (body, stop, rest) = Body("elif", "else", "endif");
        branches.Add(new KeyValuePair<Expr, List<Stmt>>(cond, body));
        while (stop == "elif")
        {
            var cond2 = ParseExpr(rest ?? "");
            var (body2, stop2, rest2) = Body("elif", "else", "endif");
            branches.Add(new KeyValuePair<Expr, List<Stmt>>(cond2, body2));
            (stop, rest) = (stop2, rest2);
        }
        if (stop == "else")
        {
            var (elseBody, stop3, _) = Body("endif");
            branches.Add(new KeyValuePair<Expr, List<Stmt>>(new Expr("lit", true), elseBody));
            stop = stop3;
        }
        if (stop != "endif")
            throw new JinjaException("{% if %} without {% endif %}");
        return new Stmt("if", branches);
    }

    private Stmt For(string rest)
    {
        // targets 'in' expression — targets: name | (a, b) | a, b
        var parser = new ExprParser(rest);
        var targets = new List<string>();
        bool parenthesized = parser.AcceptOpRaw("(") is not null;
        while (true)
        {
            var tok = parser.NextRaw();
            if (tok.Kind != "name")
                throw new JinjaException($"bad loop target in '{{% for {rest} %}}'");
            targets.Add((string)tok.Value!);
            if (parser.AcceptOpRaw(",") is null) break;
        }
        if (parenthesized && parser.AcceptOpRaw(")") is null)
            throw new JinjaException($"bad loop target in '{{% for {rest} %}}'");
        if (parser.AcceptNameRaw("in") is null)
            throw new JinjaException($"missing 'in' in '{{% for {rest} %}}'");
        var iterExpr = parser.ParseRest();
        var (body, stop, _) = Body("else", "endfor");
        var elseBody = new List<Stmt>();
        if (stop == "else")
        {
            (elseBody, stop, _) = Body("endfor");
        }
        if (stop != "endfor")
            throw new JinjaException("{% for %} without {% endfor %}");
        return new Stmt("for", targets, iterExpr, body, elseBody);
    }

    private Stmt Set(string rest)
    {
        // target = expression — target: name | ns.attr | a, b | (a, b)
        var parser = new ExprParser(rest);
        var targets = new List<string>();
        bool parenthesized = parser.AcceptOpRaw("(") is not null;
        while (true)
        {
            var tok = parser.NextRaw();
            if (tok.Kind != "name")
                throw new JinjaException($"bad assignment target in '{{% set {rest} %}}'");
            targets.Add((string)tok.Value!);
            if (parser.AcceptOpRaw(",") is null) break;
        }
        if (parenthesized && parser.AcceptOpRaw(")") is null)
            throw new JinjaException($"bad assignment target in '{{% set {rest} %}}'");
        Expr target;
        if (targets.Count == 1 && !parenthesized)
        {
            target = new Expr("name", targets[0]);
            while (parser.AcceptOpRaw(".") is not null)
            {
                var tok = parser.NextRaw();
                if (tok.Kind != "name")
                    throw new JinjaException("bad attribute assignment target");
                target = new Expr("attr", target, tok.Value);
            }
        }
        else
        {
            target = new Expr("tuple",
                targets.Select(n => (object?)new Expr("name", n)).ToList());
        }
        if (parser.AcceptOpRaw("=") is null)
            throw new JinjaException($"missing '=' in '{{% set {rest} %}}'");
        var value = parser.ParseRest();
        return new Stmt("set", target, value);
    }
}

/// <summary>Executes the parsed tree. Jinja scoping: assignments inside a
/// for loop stay in that iteration's scope; if/else create no scope.</summary>
internal sealed class Renderer
{
    private readonly List<Stmt> _tree;
    private readonly StringBuilder _out = new();

    public Renderer(string template, List<(string, string, bool, bool)> tokens)
    {
        _tree = new TemplateParser(tokens).Parse();
    }

    public string Run(List<Dictionary<string, object?>> scopes)
    {
        Exec(_tree, scopes);
        return _out.ToString();
    }

    private static readonly HashSet<string> BuiltinNames = new(StringComparer.Ordinal)
    { "raise_exception", "strftime_now", "namespace", "range" };

    private static object? Lookup(List<Dictionary<string, object?>> scopes, string name)
    {
        for (int i = scopes.Count - 1; i >= 0; i--)
            if (scopes[i].ContainsKey(name)) return scopes[i][name];
        return BuiltinNames.Contains(name) ? name : JUndefined.Instance;
    }

    private static void Assign(List<Dictionary<string, object?>> scopes, string name, object? value)
        => scopes[^1][name] = value;

    private void Exec(List<Stmt> body, List<Dictionary<string, object?>> scopes)
    {
        foreach (var stmt in body)
        {
            switch (stmt.Kind)
            {
                case "text":
                    _out.Append((string)stmt.A!);
                    break;
                case "out":
                    _out.Append(ChatJinja.Str(Eval((Expr)stmt.A!, scopes)));
                    break;
                case "if":
                    ExecIf(stmt, scopes);
                    break;
                case "for":
                    ExecFor(stmt, scopes);
                    break;
                case "set":
                    ExecSet(stmt, scopes);
                    break;
                case "generation":
                    Exec((List<Stmt>)stmt.A!, scopes);
                    break;
                default:
                    throw new JinjaException($"unknown node {stmt.Kind}");
            }
        }
    }

    private void ExecIf(Stmt node, List<Dictionary<string, object?>> scopes)
    {
        foreach (var branch in (List<object?>)node.A!)
        {
            var kv = (KeyValuePair<Expr, List<Stmt>>)branch;
            if (ChatJinja.Truthy(Eval(kv.Key, scopes)))
            {
                Exec(kv.Value, scopes);
                return;
            }
        }
    }

    private void ExecFor(Stmt node, List<Dictionary<string, object?>> scopes)
    {
        var targets = (List<string>)node.A!;
        var seq = Eval((Expr)node.B!, scopes);
        if (seq is null or JUndefined)
            throw new JinjaException("cannot iterate over None/undefined");
        List<object?> items;
        if (seq is Dictionary<string, object?> d) items = d.Keys.Select(k => (object?)k).ToList();
        else if (seq is List<object?> l) items = new List<object?>(l);
        else if (seq is JTuple t) items = new List<object?>(t.Items);
        else if (seq is string s) items = s.Select(c => (object?)c.ToString()).ToList();
        else throw new JinjaException($"cannot iterate over {seq.GetType().Name}");
        if (items.Count == 0)
        {
            Exec((List<Stmt>)node.D!, scopes);
            return;
        }
        int length = items.Count;
        for (int index = 0; index < length; index++)
        {
            var loopScope = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["loop"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["index"] = (long)(index + 1), ["index0"] = (long)index,
                    ["first"] = index == 0, ["last"] = index == length - 1,
                    ["length"] = (long)length,
                    ["revindex"] = (long)(length - index),
                    ["revindex0"] = (long)(length - index - 1),
                },
            };
            var item = items[index];
            if (targets.Count == 1)
            {
                loopScope[targets[0]] = item;
            }
            else
            {
                var unpacked = item as IReadOnlyList<object?> ?? new List<object?> { item };
                if (unpacked.Count != targets.Count)
                    throw new JinjaException(
                        $"cannot unpack loop item into {targets.Count} targets");
                for (int t = 0; t < targets.Count; t++)
                    loopScope[targets[t]] = unpacked[t];
            }
            scopes.Add(loopScope);
            Exec((List<Stmt>)node.C!, scopes);
            scopes.RemoveAt(scopes.Count - 1);
        }
    }

    private void ExecSet(Stmt node, List<Dictionary<string, object?>> scopes)
    {
        var target = (Expr)node.A!;
        var value = Eval((Expr)node.B!, scopes);
        if (target.Kind == "tuple")
        {
            var values = value as IReadOnlyList<object?>
                         ?? (value as JTuple)?.Items
                         ?? new List<object?> { value };
            var targets = (List<object?>)target.A!;
            if (values.Count != targets.Count)
                throw new JinjaException("tuple assignment length mismatch");
            for (int t = 0; t < targets.Count; t++)
                Assign(scopes, (string)((Expr)targets[t]!).A!, values[t]);
            return;
        }
        if (target.Kind == "attr")
        {
            var baseValue = Eval((Expr)target.A!, scopes);
            switch (baseValue)
            {
                case JNamespace ns:
                    ns.Set((string)target.B!, value);
                    return;
                case Dictionary<string, object?> d:
                    d[(string)target.B!] = value;
                    return;
                default:
                    throw new JinjaException(
                        "{% set %} on attributes needs a namespace() or dict target");
            }
        }
        Assign(scopes, (string)target.A!, value);
    }

    private object? Eval(Expr e, List<Dictionary<string, object?>> scopes) => e.Kind switch
    {
        "lit" => e.A,
        "name" => Lookup(scopes, (string)e.A!),
        "attr" => Attr(Eval((Expr)e.A!, scopes), (string)e.B!),
        "item" => IndexOf(Eval((Expr)e.A!, scopes), Eval((Expr)e.B!, scopes)),
        "slice" => SliceOf(Eval((Expr)e.A!, scopes),
            e.B is null ? null : Eval((Expr)e.B!, scopes),
            e.C is null ? null : Eval((Expr)e.C!, scopes),
            e.D is null ? null : Eval((Expr)e.D!, scopes)),
        "call" => Call(Eval((Expr)e.A!, scopes),
            ((List<object?>)e.B!).Select(a => Eval((Expr)a!, scopes)).ToList(),
            ((List<KeyValuePair<string, object?>>)e.C!)
                .Select(p => new KeyValuePair<string, object?>(p.Key, Eval((Expr)p.Value!, scopes))).ToList()),
        "filter" => Filter(Eval((Expr)e.A!, scopes), (string)e.B!,
            ((List<object?>)e.C!).Select(a => Eval((Expr)a!, scopes)).ToList(),
            ((List<KeyValuePair<string, object?>>)e.D!)
                .Select(p => new KeyValuePair<string, object?>(p.Key, Eval((Expr)p.Value!, scopes))).ToList()),
        "not" => !ChatJinja.Truthy(Eval((Expr)e.A!, scopes)),
        "neg" => Neg(Eval((Expr)e.A!, scopes), -1),
        "pos" => Neg(Eval((Expr)e.A!, scopes), 1),
        "and" => And(e, scopes),
        "or" => Or(e, scopes),
        "ternary" => ChatJinja.Truthy(Eval((Expr)e.A!, scopes))
            ? Eval((Expr)e.B!, scopes) : Eval((Expr)e.C!, scopes),
        "concat" => string.Join("", ((List<object?>)e.A!)
            .Select(p => ChatJinja.Str(Eval((Expr)p!, scopes)))),
        "binop" => BinOp((string)e.A!, Eval((Expr)e.B!, scopes), Eval((Expr)e.C!, scopes)),
        "test" => TestValue(e, scopes),
        "list" => (object?)((List<object?>)e.A!).Select(i => Eval((Expr)i!, scopes)).ToList(),
        "tuple" => new JTuple(((List<object?>)e.A!).Select(i => Eval((Expr)i!, scopes)).ToList()),
        "dict" => ((List<KeyValuePair<string, object?>>)e.A!)
            .ToDictionary(p => p.Key, p => Eval((Expr)p.Value!, scopes), StringComparer.Ordinal),
        _ => throw new JinjaException($"unknown expression node {e.Kind}"),
    };

    private object? TestValue(Expr e, List<Dictionary<string, object?>> scopes)
    {
        var value = Eval((Expr)e.A!, scopes);
        var args = ((List<object?>?)e.D ?? new List<object?>())
            .Select(a => Eval((Expr)a!, scopes)).ToList();
        var result = RunTest((string)e.B!, value, args);
        return (bool)e.C! ? !result : result;
    }

    private object? And(Expr e, List<Dictionary<string, object?>> scopes)
    {
        var left = Eval((Expr)e.A!, scopes);
        return !ChatJinja.Truthy(left) ? left : Eval((Expr)e.B!, scopes);
    }

    private object? Or(Expr e, List<Dictionary<string, object?>> scopes)
    {
        var left = Eval((Expr)e.A!, scopes);
        return ChatJinja.Truthy(left) ? left : Eval((Expr)e.B!, scopes);
    }

    private static object? Neg(object? v, int sign)
    {
        if (!ChatJinja.IsNumber(v)) throw new JinjaException("unary sign needs a number");
        return v is long n ? (object?)(sign * n) : sign * ChatJinja.ToDouble(v);
    }

    private object? BinOp(string op, object? left, object? right) => op switch
    {
        "==" => ChatJinja.EqualsValue(left, right),
        "!=" => !ChatJinja.EqualsValue(left, right),
        "in" => Contains(left, right),
        "<" or ">" or "<=" or ">=" => CompareValues(op, left, right),
        "+" => Add(left, right),
        "-" or "*" or "/" or "//" or "%" => Arithmetic(op, left, right),
        _ => throw new JinjaException($"unknown operator {op}"),
    };

    private static object? Add(object? left, object? right)
    {
        if (ChatJinja.IsNumber(left) && ChatJinja.IsNumber(right))
        {
            if (left is long a && right is long b) return (object?)(a + b);
            return ChatJinja.ToDouble(left) + ChatJinja.ToDouble(right);
        }
        if (left is string sa && right is string sb) return sa + sb;
        if (left is List<object?> la && right is List<object?> lb)
            return la.Concat(lb).ToList();
        throw new JinjaException(
            $"cannot add {TypeName(left)} and {TypeName(right)}");
    }

    private static object? Arithmetic(string op, object? left, object? right)
    {
        if (!ChatJinja.IsNumber(left) || !ChatJinja.IsNumber(right))
            throw new JinjaException(
                $"{op} needs numbers, got {TypeName(left)} and {TypeName(right)}");
        if (op == "*")
        {
            if (left is string s && ChatJinja.IsNumber(right))
            {
                var times = (long)ChatJinja.ToDouble(right);
                return times <= 0 ? "" : string.Concat(Enumerable.Repeat(s, (int)times));
            }
            if (right is string s2 && ChatJinja.IsNumber(left))
            {
                var times = (long)ChatJinja.ToDouble(left);
                return times <= 0 ? "" : string.Concat(Enumerable.Repeat(s2, (int)times));
            }
        }
        double a = ChatJinja.ToDouble(left), b = ChatJinja.ToDouble(right);
        if (left is long la && right is long lb)
        {
            switch (op)
            {
                case "-": return la - lb;
                case "*": return checked(la * lb);
                case "//": return b == 0 ? throw new JinjaException("division by zero") : la / lb;
                case "%": return b == 0 ? throw new JinjaException("modulo by zero") : PythonMod(la, lb);
            }
        }
        return op switch
        {
            "-" => a - b,
            "*" => a * b,
            "/" => b == 0 ? throw new JinjaException("division by zero") : a / b,
            "//" => b == 0 ? throw new JinjaException("division by zero") : Math.Floor(a / b),
            "%" => b == 0 ? throw new JinjaException("modulo by zero") : a - b * Math.Floor(a / b),
            _ => throw new JinjaException($"unknown operator {op}"),
        };
    }

    private static long PythonMod(long a, long b)
    {
        var m = a % b;
        return m != 0 && (m < 0) != (b < 0) ? m + b : m;
    }

    private static bool CompareValues(string op, object? a, object? b)
    {
        if (a is JUndefined || b is JUndefined || a is null || b is null)
            throw new JinjaException("cannot order-compare with None/undefined");
        bool numbers = ChatJinja.IsNumber(a) && ChatJinja.IsNumber(b);
        bool strings = a is string && b is string;
        if (!numbers && !strings)
            throw new JinjaException(
                $"cannot order-compare {TypeName(a)} and {TypeName(b)}");
        int cmp = numbers
            ? ChatJinja.ToDouble(a).CompareTo(ChatJinja.ToDouble(b))
            : string.CompareOrdinal((string)a, (string)b);
        return op switch
        {
            "<" => cmp < 0, ">" => cmp > 0, "<=" => cmp <= 0, ">=" => cmp >= 0,
            _ => throw new JinjaException($"unknown operator {op}"),
        };
    }

    private static bool Contains(object? item, object? container) => container switch
    {
        string s => item is string needle && s.Contains(needle, StringComparison.Ordinal),
        List<object?> l => l.Any(v => ChatJinja.EqualsValue(item, v)),
        JTuple t => t.Items.Any(v => ChatJinja.EqualsValue(item, v)),
        Dictionary<string, object?> d => d.Keys.Any(k => ChatJinja.EqualsValue(item, k)),
        _ => throw new JinjaException($"'in' not supported on {TypeName(container)}"),
    };

    internal static object? IndexOf(object? seq, object? key)
    {
        if (seq is Dictionary<string, object?> d)
        {
            foreach (var k in d.Keys)
                if (ChatJinja.EqualsValue(k, key)) return d[k];
            return JUndefined.Instance;
        }
        if (seq is List<object?> l) return SeqAt(l, key);
        if (seq is string s) return SeqAt(s.ToCharArray().Select(c => (object?)c.ToString()).ToList(), key);
        if (seq is JUndefined) return JUndefined.Instance;
        throw new JinjaException($"{TypeName(seq)} is not subscriptable");
    }

    private static object? SeqAt(List<object?> seq, object? key)
    {
        if (!ChatJinja.IsNumber(key))
            throw new JinjaException("indices must be integers");
        int i = (int)(long)key;
        int original = i;
        if (i < 0) i += seq.Count;
        if (i < 0 || i >= seq.Count)
            throw new JinjaException($"list index {original} out of range (length {seq.Count})");
        return seq[i];
    }

    private static object? SliceOf(object? seq, object? start, object? stop, object? step)
    {
        if (seq is not (List<object?> or string))
            throw new JinjaException($"cannot slice {TypeName(seq)}");
        int stepI = step is null ? 1 : (int)(long)step;
        if (stepI == 0) throw new JinjaException("slice step cannot be zero");
        int len = seq is string s ? s.Length : ((List<object?>)seq).Count;
        int startI, stopI;
        if (stepI > 0)
        {
            startI = start is null ? 0 : ClampIndex((long)start, len, 0, len);
            stopI = stop is null ? len : ClampIndex((long)stop, len, 0, len);
        }
        else
        {
            startI = start is null ? len - 1 : (int)Math.Min((long)start < 0 ? len + (long)start : (long)start, len - 1);
            stopI = stop is null ? -1 : (int)((long)stop < 0 ? len + (long)stop : Math.Min((long)stop, len));
        }
        var result = new List<object?>();
        for (int i = startI; stepI > 0 ? i < stopI : i > stopI; i += stepI)
        {
            if (i < 0 || i >= len) break;
            result.Add(seq is string s2 ? (object?)s2[i].ToString() : ((List<object?>)seq)[i]);
        }
        return seq is string ? string.Concat(result.Select(v => (string)v!)) : result;
    }

    private static int ClampIndex(long index, int len, int lo, int hi)
    {
        if (index < 0) index += len;
        return (int)Math.Clamp(index, lo, hi);
    }

    internal static object? Attr(object? value, string name) => value switch
    {
        JNamespace ns => ns.Get(name),
        Dictionary<string, object?> d => DictAttr(d, name),
        JUndefined => JUndefined.Instance,
        string s when ChatJinjaMethods.Contains(name) => new BoundMethod(s, name),
        List<object?> when name == "length" => (object?)AttrListLength(value),
        List<object?> when name == "index" => new BoundMethod(value, name),
        _ => throw new JinjaException($"{TypeName(value)} has no attribute '{name}'"),
    };

    private static object? AttrListLength(object? value) =>
        (long)((List<object?>)value!).Count;

    private static readonly HashSet<string> ChatJinjaMethods = new(StringComparer.Ordinal)
    { "upper", "lower", "strip", "lstrip", "rstrip", "split", "startswith",
      "endswith", "replace", "capitalize", "join" };

    private static object? DictAttr(Dictionary<string, object?> d, string name)
    {
        if (d.TryGetValue(name, out var v)) return v;
        if (name is "get" or "keys" or "values" or "items")
            return new BoundMethod(d, name);
        return JUndefined.Instance;
    }

    private object? Call(object? callee, List<object?> args,
        List<KeyValuePair<string, object?>> kwargs)
    {
        if (callee is BoundMethod bm) return CallBound(bm, args, kwargs);
        if (callee is string name)
        {
            switch (name)
            {
                case "raise_exception":
                    if (args.Count != 1)
                        throw new JinjaException("raise_exception() takes 1 argument");
                    throw new JinjaException(ChatJinja.Str(args[0]));
                case "strftime_now":
                    if (args.Count != 1)
                        throw new JinjaException("strftime_now() takes 1 argument");
                    return StrftimeNow((string)args[0]!);
                case "namespace":
                    return new JNamespace(kwargs);
                case "range":
                    if (args.Count is < 1 or > 3)
                        throw new JinjaException("range() takes 1-3 arguments");
                    var nums = args.Select(a => (long)ToLong(a)).ToList();
                    var seq = nums.Count switch
                    {
                        1 => Enumerable.Range(0, (int)Math.Max(0, nums[0])),
                        2 => Enumerable.Range((int)nums[0], (int)Math.Max(0, nums[1] - nums[0])),
                        _ => RangeStep(nums[0], nums[1], nums[2]),
                    };
                    return seq.Select(x => (object?)(long)x).ToList();
            }
        }
        throw new JinjaException($"not callable: {ChatJinja.Repr(callee)}");
    }

    private static IEnumerable<int> RangeStep(long start, long stop, long step)
    {
        if (step == 0) throw new JinjaException("range() step cannot be zero");
        if (step > 0)
            for (long i = start; i < stop; i += step) yield return (int)i;
        else
            for (long i = start; i > stop; i += step) yield return (int)i;
    }

    private static long ToLong(object? v)
    {
        if (!ChatJinja.IsNumber(v))
            throw new JinjaException("range() needs integers");
        return (long)ChatJinja.ToDouble(v);
    }

    private static string StrftimeNow(string format)
    {
        var now = DateTime.Now;
        var sb = new StringBuilder();
        for (int i = 0; i < format.Length; i++)
        {
            if (format[i] != '%') { sb.Append(format[i]); continue; }
            if (i + 1 >= format.Length) { sb.Append('%'); continue; }
            i += 1;
            sb.Append(format[i] switch
            {
                'Y' => now.ToString("yyyy"),
                'm' => now.ToString("MM"),
                'd' => now.ToString("dd"),
                'e' => now.ToString("dd"),
                'H' => now.ToString("HH"),
                'M' => now.ToString("mm"),
                'S' => now.ToString("ss"),
                'I' => now.ToString("hh"),
                'p' => now.ToString("tt"),
                'B' => now.ToString("MMMM"),
                'b' => now.ToString("MMM"),
                'A' => now.ToString("dddd"),
                'a' => now.ToString("ddd"),
                'F' => now.ToString("yyyy-MM-dd"),
                'T' => now.ToString("HH:mm:ss"),
                '%' => "%",
                var c => throw new JinjaException($"strftime_now: unsupported directive %{c}"),
            });
        }
        return sb.ToString();
    }

    private static object? CallBound(BoundMethod bm, List<object?> args,
        List<KeyValuePair<string, object?>> kwargs)
    {
        if (kwargs.Count > 0)
            throw new JinjaException($"method '{bm.Name}' takes no keyword arguments");
        var target = bm.Target;
        string Name() => bm.Name;
        switch (target)
        {
            case string s:
                switch (bm.Name)
                {
                    case "startswith":
                        return s.StartsWith(ExpectStr(args, bm.Name), StringComparison.Ordinal);
                    case "endswith":
                        return s.EndsWith(ExpectStr(args, bm.Name), StringComparison.Ordinal);
                    case "split":
                        if (args.Count == 0) return s.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList<object?>();
                        return s.Split(ExpectStr(args, bm.Name)).ToList<object?>();
                    case "rstrip":
                        return s.TrimEnd(args.Count > 0 ? ExpectStr(args).ToCharArray() : null);
                    case "lstrip":
                        return s.TrimStart(args.Count > 0 ? ExpectStr(args).ToCharArray() : null);
                    case "strip":
                        return s.Trim(args.Count > 0 ? ExpectStr(args).ToCharArray() : null);
                    case "replace":
                        Arity(args, 2, bm.Name);
                        return s.Replace((string)args[0]!, (string)args[1]!);
                    case "upper": return s.ToUpperInvariant();
                    case "lower": return s.ToLowerInvariant();
                    case "capitalize":
                        return s.Length == 0 ? s
                            : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();
                    case "join":
                        if (args.Count != 1 || args[0] is not List<object?> list)
                            throw new JinjaException("str.join() needs a list argument");
                        return string.Join(s, list.Select(ChatJinja.Str));
                }
                break;
            case Dictionary<string, object?> d:
                switch (bm.Name)
                {
                    case "get":
                        Arity(args, 1, bm.Name, 2);
                        if (d.Keys.Any(k => ChatJinja.EqualsValue(k, args[0])))
                            return d[d.Keys.First(k => ChatJinja.EqualsValue(k, args[0]))];
                        return args.Count > 1 ? args[1] : null;
                    case "keys": return d.Keys.Select(k => (object?)k).ToList();
                    case "values": return d.Values.ToList();
                    case "items":
                        return d.Select(p => (object?)new List<object?> { p.Key, p.Value }).ToList();
                }
                break;
            case List<object?> l when bm.Name == "index":
                Arity(args, 1, bm.Name);
                for (int i = 0; i < l.Count; i++)
                    if (ChatJinja.EqualsValue(l[i], args[0])) return (long)i;
                throw new JinjaException("value not in list");
        }
        throw new JinjaException($"unsupported method '{Name()}'");
    }

    private static string ExpectStr(List<object?> args, string name = "")
    {
        if (args.Count == 0 || args[0] is not string s)
            throw new JinjaException($"method '{(name.Length > 0 ? name : "str")}' needs a string argument");
        return s;
    }

    private static void Arity(List<object?> args, int exact, string name, int? max = null)
    {
        bool ok = args.Count == exact || (max is { } m && args.Count >= exact && args.Count <= m);
        if (!ok) throw new JinjaException($"method '{name}' takes {exact} argument(s)");
    }

    // ------------------------------------------------------------- filters
    private object? Filter(object? value, string name, List<object?> args,
        List<KeyValuePair<string, object?>> kwargs)
    {
        var kw = new Dictionary<string, object?>(kwargs, StringComparer.Ordinal);
        switch (name)
        {
            case "tojson":
            {
                int? indent = null;
                if (kw.TryGetValue("indent", out var iv) && ChatJinja.IsNumber(iv))
                    indent = (int)(long)iv;
                else if (args.Count > 0 && ChatJinja.IsNumber(args[0]))
                    indent = (int)(long)args[0];
                return ToJsonPublic(value, indent);
            }
            case "trim":
                NoArgs(args, kwargs, name);
                return ExpectStrValue(value, name).Trim();
            case "length" or "count":
                NoArgs(args, kwargs, name);
                return (long)LengthOf(value, name);
            case "first":
                NoArgs(args, kwargs, name);
                return IndexOf(ExpectSeq(value, name), (long)0);
            case "last":
                NoArgs(args, kwargs, name);
                return IndexOf(ExpectSeq(value, name), (long)-1);
            case "join":
            {
                if (value is not List<object?> list)
                    throw new JinjaException($"filter 'join' needs a list, got {TypeName(value)}");
                var sep = args.Count > 0 ? ChatJinja.Str(args[0]) : kw.TryGetValue("d", out var dv) ? ChatJinja.Str(dv) : "";
                if (kw.TryGetValue("attribute", out var attr) && attr is not null)
                    return string.Join(sep, list.Select(v => ChatJinja.Str(Attr(v, (string)attr))));
                return string.Join(sep, list.Select(ChatJinja.Str));
            }
            case "items":
            {
                NoArgs(args, kwargs, name);
                if (value is not Dictionary<string, object?> d)
                    throw new JinjaException($"filter 'items' needs a dict, got {TypeName(value)}");
                return d.OrderBy(p => p.Key, StringComparer.Ordinal)
                    .Select(p => (object?)new List<object?> { p.Key, p.Value }).ToList();
            }
            case "reject":
                return SelectOrReject(value, args, keep: false);
            case "select":
                return SelectOrReject(value, args, keep: true);
            case "default":
            {
                var dv = args.Count > 0 ? args[0] : kw.TryGetValue("d", out var d) ? d : "";
                var boolean = kw.TryGetValue("boolean", out var b) && b is true
                              || args.Count > 1 && args[1] is true;
                if (value is null or JUndefined || (boolean && !ChatJinja.Truthy(value)))
                    return dv;
                return value;
            }
            case "upper": NoArgs(args, kwargs, name); return ExpectStrValue(value, name).ToUpperInvariant();
            case "lower": NoArgs(args, kwargs, name); return ExpectStrValue(value, name).ToLowerInvariant();
            case "capitalize":
                NoArgs(args, kwargs, name);
                var cap = ExpectStrValue(value, name);
                return cap.Length == 0 ? cap : char.ToUpperInvariant(cap[0]) + cap[1..].ToLowerInvariant();
            case "replace":
                Arity(args, 2, name);
                return ExpectStrValue(value, name).Replace((string)args[0]!, (string)args[1]!);
            case "reverse":
                NoArgs(args, kwargs, name);
                if (value is string rs) return new string(rs.Reverse().ToArray());
                if (value is List<object?> rl) return rl.AsEnumerable().Reverse().ToList();
                throw new JinjaException("filter 'reverse' needs a list or string");
            case "string": NoArgs(args, kwargs, name); return ChatJinja.Str(value);
            case "list":
                NoArgs(args, kwargs, name);
                return value switch
                {
                    string s => s.Select(c => (object?)c.ToString()).ToList(),
                    List<object?> l => new List<object?>(l),
                    JTuple t => new List<object?>(t.Items),
                    Dictionary<string, object?> d => d.Keys.Select(k => (object?)k).ToList(),
                    _ => throw new JinjaException($"filter 'list' cannot convert {TypeName(value)}"),
                };
            case "int": return ToNumber(value, args, whole: true);
            case "float": return ToNumber(value, args, whole: false);
            case "sort":
            {
                NoArgs(args, kwargs, name);
                if (value is not List<object?> sl)
                    throw new JinjaException("filter 'sort' needs a list");
                var keys = sl.Select(SortKey).ToList();
                if (keys.All(k => k is double)) return sl.OrderBy(k => ChatJinja.ToDouble(k)).ToList();
                if (keys.All(k => k is string)) return sl.OrderBy(k => (string)SortKey(k)!, StringComparer.Ordinal).ToList();
                throw new JinjaException("sort cannot compare mixed types");
            }
            default:
                throw new JinjaException(
                    $"unsupported filter '{name}' (built-in engine supports: tojson, " +
                    "trim, length, count, first, last, join, items, reject, select, " +
                    "default, upper, lower, capitalize, replace, reverse, string, " +
                    "list, int, float, sort)");
        }
    }

    private static string ToJsonPublic(object? value, int? indent)
    {
        // Reuse ChatJinja's writer through a small shim: values are first
        // normalized (namespace → dict, undefined → null).
        object? Normalize(object? v) => v switch
        {
            JNamespace ns => ns.Fields.ToDictionary(p => p.Key, p => Normalize(p.Value), StringComparer.Ordinal),
            JUndefined => null,
            List<object?> l => l.Select(Normalize).ToList(),
            JTuple t => t.Items.Select(Normalize).ToList(),
            Dictionary<string, object?> d => d.ToDictionary(p => p.Key, p => Normalize(p.Value), StringComparer.Ordinal),
            _ => v,
        };
        return JsonWrite(Normalize(value), indent);
    }

    private static string JsonWrite(object? value, int? indent)
    {
        var sb = new System.Text.StringBuilder();
        ChatJinja.WriteJsonPublic(sb, value, indent, 0);
        return sb.ToString();
    }

    private object? SelectOrReject(object? value, List<object?> args, bool keep)
    {
        if (value is not List<object?> list)
            throw new JinjaException("reject/select need a list");
        if (args.Count == 0 || args[0] is not string testName)
            throw new JinjaException("reject/select need a test name");
        var testArgs = args.Skip(1).ToList();
        return list.Where(v => RunTest(testName, v, testArgs) == keep).ToList();
    }

    private static object? SortKey(object? v) =>
        ChatJinja.IsNumber(v) ? ChatJinja.ToDouble(v) : ChatJinja.Str(v);

    private static int LengthOf(object? value, string name) => value switch
    {
        string s => s.Length,
        List<object?> l => l.Count,
        JTuple t => t.Items.Count,
        Dictionary<string, object?> d => d.Count,
        _ => throw new JinjaException($"object of type {TypeName(value)} has no length"),
    };

    private static string ExpectStrValue(object? value, string name) => value switch
    {
        string s => s,
        null or JUndefined => "",
        _ => throw new JinjaException($"filter '{name}' needs a string, got {TypeName(value)}"),
    };

    private static object? ExpectSeq(object? value, string name) => value switch
    {
        List<object?> or string or JTuple => value,
        _ => throw new JinjaException($"filter '{name}' needs a list or string"),
    };

    private static void NoArgs(List<object?> args, List<KeyValuePair<string, object?>> kwargs, string name)
    {
        if (args.Count > 0 || kwargs.Count > 0)
            throw new JinjaException($"filter '{name}' takes no arguments");
    }

    private static object? ToNumber(object? value, List<object?> args, bool whole)
    {
        object? fallback = args.Count > 0 ? args[0] : (whole ? (object?)(long)0 : 0.0);
        try
        {
            switch (value)
            {
                case string s when whole:
                    return long.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var l) ? l : fallback;
                case string s:
                    return double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : fallback;
                case long n when whole: return n;
                case long n: return (double)n;
                case double dd when whole: return (long)dd;
                case double dd: return dd;
                case null or JUndefined: return fallback;
                case bool bo: return whole ? (object?)(bo ? 1L : 0L) : (bo ? 1.0 : 0.0);
                default:
                    throw new JinjaException($"filter 'int' cannot convert {TypeName(value)}");
            }
        }
        catch (JinjaException)
        {
            return fallback;
        }
    }

    private bool RunTest(string name, object? value, List<object?> args) => name switch
    {
        "defined" => value is not JUndefined,
        "undefined" => value is JUndefined,
        "none" => value is null,
        "string" => value is string,
        "number" => ChatJinja.IsNumber(value),
        "integer" => value is long,
        "float" => value is double,
        "boolean" => value is bool,
        "true" => value is true,
        "false" => value is false,
        "mapping" => value is Dictionary<string, object?>,
        "iterable" or "sequence" => value is List<object?> or string or Dictionary<string, object?> or JTuple,
        "equalto" or "eq" => Arity1(args, name) && ChatJinja.EqualsValue(value, args[0]),
        "ne" => Arity1(args, name) && !ChatJinja.EqualsValue(value, args[0]),
        "in" => Arity1(args, name) && Contains(value, args[0]),
        "divisibleby" => Arity1(args, name) && ChatJinja.IsNumber(value)
                          && ChatJinja.IsNumber(args[0])
                          && (long)ChatJinja.ToDouble(value) % (long)ChatJinja.ToDouble(args[0]) == 0,
        _ => throw new JinjaException($"unsupported test '{name}'"),
    };

    private static bool Arity1(List<object?> args, string name)
    {
        if (args.Count != 1)
            throw new JinjaException($"test '{name}' takes 1 argument");
        return true;
    }

    private static string TypeName(object? v) => v switch
    {
        null => "None",
        JUndefined => "undefined",
        bool => "bool",
        long => "int",
        double => "float",
        string => "str",
        List<object?> or IReadOnlyList<object?> => "list",
        JTuple => "tuple",
        Dictionary<string, object?> => "dict",
        JNamespace => "namespace",
        _ => v.GetType().Name,
    };
}

