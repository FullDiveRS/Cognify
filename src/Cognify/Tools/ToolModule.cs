using Cognify.Core;
using Cognify.Reasoning;

namespace Cognify.Tools;

/// <summary>
/// The tool-use module: turns the mind into an agent that can act.
/// When a stimulus looks like a tool request — either by trigger phrase or by
/// naming a registered tool — the module extracts arguments from natural
/// language, invokes the tool and turns its result into the response.
/// Custom tools can be registered at any time via <see cref="Registry"/>.
/// </summary>
public sealed class ToolModule : CognitiveModuleBase
{
    private readonly ToolOptions _options;

    /// <summary>Creates the tool module with an optional prebuilt registry.</summary>
    public ToolModule(ToolOptions options, ToolRegistry? registry = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Registry = registry ?? new ToolRegistry();
        if (_options.UseBuiltinTools)
            BuiltinTools.RegisterAll(Registry);
    }

    /// <inheritdoc />
    public override string Name => "tools";

    /// <inheritdoc />
    public override int Order => 45; // action band: response set here is kept by thinking (order 50)

    /// <summary>The registry of available tools.</summary>
    public ToolRegistry Registry { get; }

    /// <summary>The tool invoked in the most recent cycle, if any.</summary>
    public string? LastInvokedTool { get; private set; }

    /// <summary>Total number of successful tool invocations.</summary>
    public int SuccessfulInvocations { get; private set; }

    /// <summary>Total number of failed tool invocations.</summary>
    public int FailedInvocations { get; private set; }

    /// <summary>
    /// Programmatically invokes a tool by name (also available as
    /// <c>mind.InvokeTool(...)</c>).
    /// </summary>
    public ToolResult Invoke(string toolName, IReadOnlyDictionary<string, string>? arguments = null, string rawInput = "")
    {
        var tool = Registry.Get(toolName);
        if (tool is null)
            return ToolResult.Fail($"Unknown tool '{toolName}'. Available: {string.Join(", ", Registry.All().Select(t => t.Name))}.");

        var invocation = new ToolInvocation
        {
            ToolName = tool.Name,
            Arguments = arguments ?? new Dictionary<string, string>(),
            RawInput = rawInput
        };

        ToolResult result;
        try
        {
            result = tool.Execute(invocation);
        }
        catch (Exception ex)
        {
            result = ToolResult.FromException(ex);
        }

        if (result.Success) SuccessfulInvocations++;
        else FailedInvocations++;

        LastInvokedTool = tool.Name;
        Host?.RaiseToolInvoked(tool.Name, invocation.Arguments, result);
        Host?.Stream?.Add("tools", $"{(result.Success ? "used" : "tried")} '{tool.Name}': {Core.Mind.Truncate(result.Output, 80)}");
        return result;
    }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        if (!_options.AutoInvoke) return;
        if (context.Stimulus.Kind is StimulusKind.Image or StimulusKind.Audio or StimulusKind.Video or StimulusKind.Feedback)
            return; // perception and feedback are not tool requests

        var text = context.EffectiveContent;
        if (string.IsNullOrWhiteSpace(text)) return;

        // Reasoning-first arithmetic: when the reasoning module already
        // answered a math question from taught knowledge, the calculator must
        // NOT auto-fire on top of it. The user can still explicitly demand
        // the calculator ("use the calculator", "calculate 2 + 3").
        bool mathAnswered = context.Bag.ContainsKey("reasoning.mathAnswered");
        bool mathAttempted = context.Bag.ContainsKey("reasoning.mathAttempted");
        bool calcExplicitlyRequested = ExplicitCalculatorRequest(text);

        // 1. Explicit "use tool <name>" commands take priority.
        var explicitTool = MatchExplicitCommand(text);
        if (explicitTool is { } command)
        {
            var result = Invoke(command.name, command.args, text);
            if (result.Success)
            {
                context.Response = new ResponseDraft(result.Output, 0.85, 850, "tool-use");
                context.Bag["tools.invoked"] = command.name;
                return;
            }
        }

        // 2. Trigger-phrase matching against the registry. A tool is only
        //    invoked when its required arguments can actually be extracted —
        //    e.g. the calculator is skipped when there is no real math in the
        //    input ("what is a bird?" stays with the normal cognitive cycle).
        foreach (var tool in Registry.Match(text))
        {
            // Taught arithmetic already answered this — no calculator crutch.
            if (mathAnswered && tool.Name == "calculator" && !calcExplicitlyRequested)
                continue;

            var args = ExtractArguments(tool, text);
            if (args is null) continue;

            var result = Invoke(tool.Name, args, text);
            context.Bag["tools.invoked"] = tool.Name;

            if (result.Success)
            {
                context.Response = new ResponseDraft(WithFallbackNote(tool.Name, result.Output, mathAttempted, calcExplicitlyRequested), 0.8, 850, "tool-use");
            }
            else
            {
                context.Response = new ResponseDraft(
                    $"{result.Output} (I can use: {string.Join(", ", Registry.All().Select(t => t.Name))})",
                    0.4, 851, "tool-failure");
            }
            return;
        }

        // 3. Bare arithmetic like "2+2*3" (no trigger phrase needed).
        if ((!mathAnswered || calcExplicitlyRequested) && TryBareExpression(text, out var bare))
        {
            var result = Invoke("calculator",
                new Dictionary<string, string> { ["expression"] = bare }, text);
            if (result.Success)
            {
                context.Bag["tools.invoked"] = "calculator";
                context.Response = new ResponseDraft(WithFallbackNote("calculator", result.Output, mathAttempted, calcExplicitlyRequested), 0.8, 850, "tool-use");
            }
            return;
        }

        // 4. Word problems the mind could not reason out from taught
        //    knowledge ("if i have 2 apples and buy 3 more..."): the
        //    calculator fills in transparently, computing the extracted
        //    quantities mechanically. Reasoning still gets first crack —
        //    this only fires when the reasoning module flagged mathAttempted.
        if (mathAttempted && !mathAnswered && !calcExplicitlyRequested)
        {
            var engine = Host?.Module<ReasoningModule>()?.Arithmetic;
            var mechanical = engine?.TryWordProblemMechanically(text);
            if (mechanical is not null)
            {
                context.Bag["tools.invoked"] = "calculator";
                context.Response = new ResponseDraft(
                    WithFallbackNote("calculator", mechanical.Reply, mathAttempted, calcExplicitlyRequested),
                    0.7, 850, "tool-use");
            }
        }
    }

    /// <summary>When the calculator fills in for missing arithmetic knowledge,
    /// say so — transparency beats a silent crutch.</summary>
    private static string WithFallbackNote(string toolName, string output, bool mathAttempted, bool explicitRequest) =>
        toolName == "calculator" && mathAttempted && !explicitRequest
            ? $"I have not been taught that one yet, so I used the calculator: {output}"
            : output;

    /// <summary>True when the user clearly asked for the calculator by name.</summary>
    private static bool ExplicitCalculatorRequest(string text)
    {
        var lower = text.ToLowerInvariant();
        return lower.Contains("calculator") ||
               lower.StartsWith("calculate", StringComparison.Ordinal) ||
               lower.StartsWith("compute", StringComparison.Ordinal);
    }

    private (string name, Dictionary<string, string> args)? MatchExplicitCommand(string text)
    {
        var triggers = new[] { "use tool", "call tool", "run tool", "invoke tool" };
        foreach (var trigger in triggers)
        {
            int index = text.IndexOf(trigger, StringComparison.OrdinalIgnoreCase);
            if (index < 0) continue;

            var rest = text[(index + trigger.Length)..].Trim();
            // Optional "named <tool>" or "<tool>" with optional key=value pairs.
            var parts = rest.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;

            var name = parts[0].Trim(',', '.', '!');
            var tool = Registry.Get(name);
            if (tool is null) continue;

            var args = new Dictionary<string, string>();
            foreach (var part in parts.Skip(1))
            {
                var kv = part.Split('=', 2);
                if (kv.Length == 2) args[kv[0].Trim()] = kv[1].Trim(',', '.', '!');
            }
            return (tool.Name, args);
        }
        return null;
    }

    private Dictionary<string, string>? ExtractArguments(ITool tool, string text)
    {
        var args = new Dictionary<string, string>();
        foreach (var parameter in tool.Parameters.Keys)
        {
            switch (parameter)
            {
                case "expression":
                    var expression = ExtractExpression(text);
                    if (string.IsNullOrWhiteSpace(expression) || !LooksLikeMath(expression))
                        return null; // not a real math request — let normal cognition handle it
                    args["expression"] = expression;
                    break;

                case "text":
                    // The whole input is the text to analyze.
                    args["text"] = text;
                    break;

                default:
                    args[parameter] = text;
                    break;
            }
        }
        return args;
    }

    private static bool LooksLikeMath(string expression) =>
        expression.All(c => char.IsDigit(c) || "+-*/%(). ".Contains(c)) &&
        expression.Any(char.IsDigit);

    private static bool TryBareExpression(string text, out string expression)
    {
        expression = string.Empty;
        var trimmed = text.Trim().TrimEnd('?');
        if (trimmed.Length < 3) return false;
        if (LooksLikeMath(trimmed) && trimmed.Any(c => "+-*/%".Contains(c)))
        {
            expression = trimmed;
            return true;
        }
        return false;
    }

    private static string? ExtractExpression(string text)
    {
        var trimmed = text.Trim().TrimEnd('?', '!', '.');
        var lower = trimmed.ToLowerInvariant();
        foreach (var prefix in new[] { "calculate ", "compute ", "what is ", "what's ", "how much is " })
        {
            if (lower.StartsWith(prefix, StringComparison.Ordinal))
            {
                var rest = trimmed[prefix.Length..].Trim();
                // "what is a bird?" -> "a bird" is not math; fall through to
                // the embedded scan so "what is the answer to 2 + 2?" works.
                if (LooksLikeMath(rest)) return rest;
                break;
            }
        }
        // Greetings and filler words before the math: scan the whole sentence
        // for an embedded expression ("Hello Yume, what is 2 + 2?" -> "2 + 2").
        return MathSniffer.FindExpression(trimmed);
    }
}
