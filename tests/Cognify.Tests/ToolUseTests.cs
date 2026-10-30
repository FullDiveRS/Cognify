using Cognify.Core;
using Cognify.Tools;

namespace Cognify.Tests;

/// <summary>Tests for the tool registry, built-in tools and the tool module.</summary>
public class ToolUseTests
{
    // ---- Calculator: the real expression parser --------------------------

    [Theory]
    [InlineData("2 + 2 * 3", 8)]
    [InlineData("(1 + 2) / 4", 0.75)]
    [InlineData("10 % 3", 1)]
    [InlineData("-5 + 3", -2)]
    [InlineData("2 * (3 + 4)", 14)]
    [InlineData("1,000 + 24", 1024)]
    [InlineData("3.5 * 2", 7)]
    public void Calculator_EvaluatesArithmetic(string expression, double expected)
    {
        var tool = new CalculatorTool();
        var result = tool.Execute(new ToolInvocation
        {
            ToolName = "calculator",
            Arguments = new Dictionary<string, string> { ["expression"] = expression }
        });

        Assert.True(result.Success, result.Output);
        Assert.Equal(expected, (double)result.Data["value"], precision: 6);
    }

    [Fact]
    public void Calculator_RejectsDivisionByZero()
    {
        var tool = new CalculatorTool();
        var result = tool.Execute(new ToolInvocation
        {
            Arguments = new Dictionary<string, string> { ["expression"] = "1 / 0" }
        });
        Assert.False(result.Success);
        Assert.Contains("division by zero", result.Output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Calculator_RejectsGarbage()
    {
        var tool = new CalculatorTool();
        var result = tool.Execute(new ToolInvocation
        {
            Arguments = new Dictionary<string, string> { ["expression"] = "2 + banana" }
        });
        Assert.False(result.Success);
    }

    [Fact]
    public void Calculator_RejectsUnbalancedParentheses()
    {
        var tool = new CalculatorTool();
        var result = tool.Execute(new ToolInvocation
        {
            Arguments = new Dictionary<string, string> { ["expression"] = "(2 + 3" }
        });
        Assert.False(result.Success);
    }

    // ---- Other built-ins ---------------------------------------------------

    [Fact]
    public void Clock_ReturnsTime()
    {
        var result = new ClockTool().Execute(new ToolInvocation { ToolName = "clock" });
        Assert.True(result.Success);
        Assert.Contains("UTC", result.Output);
        Assert.True(result.Data.ContainsKey("utc"));
    }

    [Fact]
    public void TextStats_CountsWords()
    {
        var result = new TextStatsTool().Execute(new ToolInvocation
        {
            Arguments = new Dictionary<string, string> { ["text"] = "one two three four. Five!" }
        });

        Assert.True(result.Success);
        Assert.Equal(5, (int)result.Data["words"]!);
        Assert.Equal(2, (int)result.Data["sentences"]!);
    }

    // ---- Registry ------------------------------------------------------------

    [Fact]
    public void Registry_RegisterAndLookup()
    {
        var registry = new ToolRegistry();
        registry.Register(new ClockTool());
        Assert.Same(registry.Get("CLOCK"), registry.Get("clock")); // case-insensitive
        Assert.Equal(1, registry.Count);

        registry.Unregister("clock");
        Assert.Null(registry.Get("clock"));
    }

    [Fact]
    public void Registry_Match_RanksTriggerHits()
    {
        var registry = new ToolRegistry();
        BuiltinTools.RegisterAll(registry);

        var matches = registry.Match("calculate 2+2 for me");
        Assert.NotEmpty(matches);
        Assert.Equal("calculator", matches[0].Name);
    }

    [Fact]
    public void Registry_Match_EmptyForUnrelatedText()
    {
        var registry = new ToolRegistry();
        BuiltinTools.RegisterAll(registry);
        Assert.Empty(registry.Match("tell me about dolphins"));
    }

    // ---- ToolModule in the cognitive cycle --------------------------------------

    [Fact]
    public void Module_AutoInvokesCalculator()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var response = mind.Ask("calculate 2 + 2 * 3");

        Assert.Contains("8", response.Content);
        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Equal(1, mind.Tools.SuccessfulInvocations);
    }

    [Fact]
    public void Module_AutoInvokesCalculator_OnBareExpression()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var response = mind.Ask("(1 + 2) / 4");
        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Contains("0.75", response.Content);
    }

    [Fact]
    public void Module_AutoInvokesClock()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var response = mind.Ask("what time is it?");

        Assert.Equal("clock", mind.Tools!.LastInvokedTool);
        Assert.Contains("UTC", response.Content);
    }

    [Fact]
    public void Module_SkipsCalculator_ForPlainQuestions()
    {
        // "what is a bird?" must NOT be hijacked by the calculator.
        using var mind = MindBuilder.Create("Agent")
            .WithTools()
            .WithReasoning()
            .WithThinking()
            .Build();

        mind.Teach("birds are animals");
        var response = mind.Ask("what is a bird?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("animal", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_TextStats_Trigger()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var response = mind.Ask("count words: one two three");
        Assert.Equal("text-stats", mind.Tools!.LastInvokedTool);
        Assert.Contains("words", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_ExplicitCommand_WithArguments()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var response = mind.Ask("use tool calculator expression=6*7");
        Assert.True(response.Confidence >= 0.8);
        Assert.Contains("42", response.Content);
    }

    [Fact]
    public void Invoke_UnknownTool_Fails()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var result = mind.InvokeTool("nonexistent");
        Assert.False(result.Success);
        Assert.Contains("Unknown tool", result.Output);
    }

    [Fact]
    public void InvokeTool_Convenience_Works()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        var result = mind.InvokeTool("calculator",
            new Dictionary<string, string> { ["expression"] = "5 * 5" });
        Assert.True(result.Success);
        Assert.Equal(25, (double)result.Data["value"], precision: 6);
    }

    [Fact]
    public void InvokeTool_Throws_WhenToolsDisabled()
    {
        using var mind = MindBuilder.Create("NoAgent").Build();
        Assert.Throws<MindConfigurationException>(() => mind.InvokeTool("clock"));
    }

    [Fact]
    public void CustomTool_CanBeRegistered()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        mind.Tools!.Registry.Register(new EchoTool());

        var result = mind.InvokeTool("echo",
            new Dictionary<string, string> { ["message"] = "cognify" });
        Assert.True(result.Success);
        Assert.Equal("echo: cognify", result.Output);
    }

    [Fact]
    public void ToolInvoked_EventIsRaised()
    {
        using var mind = MindBuilder.Create("Agent").WithTools().Build();
        string? invoked = null;
        mind.ToolInvoked += (_, e) => invoked = e.ToolName;

        mind.InvokeTool("clock");
        Assert.Equal("clock", invoked);
    }

    private sealed class EchoTool : ITool
    {
        public string Name => "echo";
        public string Description => "Echoes back the message argument.";
        public IReadOnlyDictionary<string, string> Parameters =>
            new Dictionary<string, string> { ["message"] = "text to echo" };
        public IReadOnlyList<string> Triggers => new[] { "echo" };
        public ToolResult Execute(ToolInvocation invocation) =>
            ToolResult.Ok($"echo: {invocation.Arguments.GetValueOrDefault("message")}");
    }
}
