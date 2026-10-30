using Cognify.Core;
using Cognify.Reasoning;

namespace Cognify.Tests;

/// <summary>
/// Regression tests: arithmetic questions must route to the calculator tool,
/// questions must never be "learned" as garbage facts, and identity questions
/// ("introduce yourself") must answer from the mind's own description.
/// </summary>
public class ArithmeticRoutingTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Yume")
            .Describe("Yume (夢, 'dream') — a warm, curious companion AI.")
            .WithEmotions(e => e.Intensity(0.85))
            .WithReasoning()
            .WithThinking()
            .WithMemory()
            .WithTools()
            .Build();

    // ------------------------------------------------------------------ calculator

    [Fact]
    public void Arithmetic_Question_With_Greeting_Routes_To_Calculator()
    {
        using var mind = BuildMind();
        var response = mind.Ask("Hello Yume, what is 2 + 2?");

        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Equal("tool-use", response.StyleName);
        Assert.Contains("2 + 2 = 4", response.Content);
    }

    [Fact]
    public void Answer_To_Arithmetic_Found_Mid_Sentence()
    {
        using var mind = BuildMind();
        var response = mind.Ask("what is the answer to 12 * 3?");

        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Contains("36", response.Content);
    }

    [Fact]
    public void Casual_Multiplication_With_X_Is_Supported()
    {
        using var mind = BuildMind();
        var response = mind.Ask("hey, what is 3 x 4?");

        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Contains("12", response.Content);
    }

    [Fact]
    public void Plain_Question_Is_Still_Not_Hijacked()
    {
        using var mind = BuildMind();
        mind.Teach("birds are animals");
        var response = mind.Ask("what is a bird?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.NotEqual("tool-use", response.StyleName);
        Assert.Contains("animal", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void No_Digits_Means_No_Calculator()
    {
        using var mind = BuildMind();
        mind.Ask("what is your favorite food?");
        Assert.Null(mind.Tools!.LastInvokedTool);
    }

    // ------------------------------------------------------------------ no garbage facts

    [Fact]
    public void Questions_Are_Never_Parsed_Into_Facts()
    {
        Assert.Empty(FactParser.Parse("Hello Yume, what is 2 + 2?"));
        Assert.Empty(FactParser.Parse("what is the answer to 2 + 2?"));
        Assert.Empty(FactParser.Parse("is a bird an animal?"));
        Assert.Empty(FactParser.Parse("do you know math?"));

        // Statements still parse.
        Assert.Single(FactParser.Parse("birds are animals"));
        Assert.Single(FactParser.Parse("2 + 2 is 4"));
    }

    [Fact]
    public void Arithmetic_Question_Does_Not_Claim_Learning()
    {
        using var mind = BuildMind();
        var before = mind.Knowledge!.FactCount;

        var response = mind.Ask("Hello Yume, what is 2 + 2?");

        Assert.Equal(before, mind.Knowledge.FactCount);
        Assert.DoesNotContain("learned", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    // ------------------------------------------------------------------ identity

    [Fact]
    public void Introduce_Yourself_Answers_With_Identity()
    {
        using var mind = BuildMind();
        var response = mind.Ask("Hello Yume, Introduce yourself.");

        Assert.Contains("Yume", response.Content);
        Assert.Contains("warm, curious companion", response.Content);
        Assert.NotEqual("tool-use", response.StyleName);
    }

    [Fact]
    public void Who_Are_You_Answers_With_Identity()
    {
        using var mind = BuildMind();
        var response = mind.Ask("who are you?");

        Assert.Contains("Yume", response.Content);
        // Identity answers never echo the question or claim a memory.
        Assert.DoesNotContain("remember", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Identity_Answer_Includes_Current_Mood()
    {
        using var mind = BuildMind();
        mind.Process("I love this, this is wonderful!");

        var response = mind.Ask("introduce yourself");

        Assert.Contains("Yume", response.Content);
        // Feelings color the identity answer silently by default: the
        // mechanical "Right now ..." read is developer introspection and
        // stays hidden until ShowFeelingReads is on.
        Assert.DoesNotContain("Right now", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("Right now", response.Reasoning!.Answer!, StringComparison.OrdinalIgnoreCase);

        mind.Config.Expression.Feelings(true);   // developer flips the switch
        var read = mind.Ask("introduce yourself");
        Assert.Contains("Yume", read.Content);
        // The full identity answer (with mood) lives in the reasoning result;
        // the chosen style may legitimately render only its first sentence.
        Assert.Contains("Right now", read.Reasoning!.Answer!, StringComparison.OrdinalIgnoreCase);
    }
}
