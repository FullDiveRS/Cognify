using Cognify.Code;
using Cognify.Core;

namespace Cognify.Tests;

/// <summary>Tests for code analysis, generation and the code module.</summary>
public class CodeTests
{
    // ---- Static analyzer ---------------------------------------------------

    [Fact]
    public void Analyzer_CountsLinesAndComments()
    {
        const string source = """
            // a comment
            public class Foo {
                public int Bar() { return 1; }
            }
            """;

        var metrics = CodeAnalyzer.Analyze(source.Trim());
        Assert.Equal(4, metrics.Lines);
        Assert.Equal(1, metrics.CommentLines);
        Assert.Equal(3, metrics.CodeLines);
        Assert.True(metrics.BracketsBalanced);
    }

    [Fact]
    public void Analyzer_DetectsUnbalancedBrackets()
    {
        var metrics = CodeAnalyzer.Analyze("def f(:\n    return 1");
        Assert.False(metrics.BracketsBalanced);
    }

    [Fact]
    public void Analyzer_MeasuresNestingDepth()
    {
        const string source = """
            public class A {
                public void M() {
                    if (true) {
                        while (true) {
                            Do();
                        }
                    }
                }
            }
            """;

        var metrics = CodeAnalyzer.Analyze(source);
        Assert.True(metrics.MaxNestingDepth >= 3, $"expected depth >= 3, got {metrics.MaxNestingDepth}");
        Assert.True(metrics.ComplexityScore > 0);
    }

    [Fact]
    public void Analyzer_GuessesLanguages()
    {
        Assert.Equal("csharp", CodeAnalyzer.GuessLanguage("using System;\npublic class A { }"));
        Assert.Equal("python", CodeAnalyzer.GuessLanguage("def main():\n    print('hi')"));
        Assert.Equal("javascript", CodeAnalyzer.GuessLanguage("function main() {\n  console.log('hi');\n}"));
        Assert.Equal("sql", CodeAnalyzer.GuessLanguage("SELECT id, name FROM users"));
        Assert.Equal("unknown", CodeAnalyzer.GuessLanguage("this is just prose, nothing else"));
    }

    [Fact]
    public void Analyzer_EmptyInput_IsSafe()
    {
        var metrics = CodeAnalyzer.Analyze("");
        Assert.Equal(0, metrics.Lines);
        Assert.Equal(0, metrics.ComplexityScore);
    }

    // ---- Template backend ------------------------------------------------------

    [Fact]
    public void Backend_GeneratesCSharpClass()
    {
        var backend = new TemplateCodeBackend();
        var result = backend.Generate(new CodeRequest("a robot class", CodeRequestKind.Class, "csharp"));

        Assert.Equal("csharp", result.Language);
        Assert.Contains("public sealed class", result.Source);
        Assert.Contains("namespace Generated;", result.Source);
        Assert.NotEmpty(result.Notes);
    }

    [Fact]
    public void Backend_GeneratesFunction()
    {
        var backend = new TemplateCodeBackend();
        var result = backend.Generate(new CodeRequest("compute average", CodeRequestKind.Function, "csharp"));

        Assert.Contains("public static double", result.Source);
        Assert.Equal(CodeRequestKind.Function, result.Kind);
    }

    [Fact]
    public void Backend_GeneratesPython()
    {
        var backend = new TemplateCodeBackend();
        var result = backend.Generate(new CodeRequest("a data loader class", CodeRequestKind.Class, "python"));

        Assert.Equal("python", result.Language);
        Assert.Contains("class ", result.Source);
        Assert.Contains("def __init__", result.Source);
    }

    [Fact]
    public void Backend_GeneratesJavaScript()
    {
        var backend = new TemplateCodeBackend();
        var result = backend.Generate(new CodeRequest("a fetch helper function", CodeRequestKind.Function, "javascript"));

        Assert.Equal("javascript", result.Language);
        Assert.Contains("function ", result.Source);
        Assert.Contains("module.exports", result.Source);
    }

    [Fact]
    public void Backend_GeneratesSql()
    {
        var backend = new TemplateCodeBackend();
        var result = backend.Generate(new CodeRequest("orders table report", CodeRequestKind.Query, "sql"));

        Assert.Equal("sql", result.Language);
        Assert.Contains("SELECT", result.Source);
        Assert.Contains("FROM", result.Source);
    }

    [Fact]
    public void Backend_IsDeterministic()
    {
        var backend = new TemplateCodeBackend();
        var a = backend.Generate(new CodeRequest("a widget class", CodeRequestKind.Class, "csharp"));
        var b = backend.Generate(new CodeRequest("a widget class", CodeRequestKind.Class, "csharp"));
        Assert.Equal(a.Source, b.Source);
    }

    // ---- Module in the cognitive cycle ---------------------------------------------

    [Fact]
    public void Module_GeneratesOnRequest()
    {
        using var mind = MindBuilder.Create("Coder").WithCode().Build();
        var response = mind.Ask("write a csharp class for a robot");

        Assert.Equal(1, mind.CodeEngine!.RequestsHandled);
        Assert.Contains("class", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("code-generation", response.StyleName);
    }

    [Fact]
    public void Module_AnalyzesFencedSnippet()
    {
        using var mind = MindBuilder.Create("Coder").WithCode().Build();
        var snippet = "Here is my code:\n```csharp\npublic class A {\n  // hello\n  public int X() { return 1; }\n}\n```";
        var response = mind.Ask(snippet);

        Assert.Equal(1, mind.CodeEngine!.SnippetsAnalyzed);
        Assert.Equal("code-analysis", response.StyleName);
        Assert.Contains("csharp", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Module_IgnoresPlainText()
    {
        using var mind = MindBuilder.Create("Coder").WithCode().Build();
        mind.Ask("what is the capital of France?");
        Assert.Equal(0, mind.CodeEngine!.RequestsHandled);
        Assert.Equal(0, mind.CodeEngine.SnippetsAnalyzed);
    }

    [Fact]
    public void Module_RespondsInThinkingBand()
    {
        // The code module sets the response at order 46; the thinking module
        // (order 50) must keep it and record the conclusion.
        using var mind = MindBuilder.Create("Coder").WithCode().WithThinking().Build();
        var response = mind.Ask("write a python function for sorting");

        Assert.Equal("code-generation", response.StyleName);
        Assert.NotNull(response.Chain);
        Assert.Contains(response.Chain.Steps, s =>
            s.Text.Contains("already answered", StringComparison.OrdinalIgnoreCase));
    }
}
