using Cognify.Core;
using Cognify.Reasoning;

namespace Cognify.Tests;

/// <summary>
/// Taught-arithmetic reasoning: Yume answers math questions from what she
/// was TAUGHT (direct fact, or derived from related taught facts), uses the
/// calculator only as a transparent fallback, and never learns garbage from
/// unlabeled questions.
/// </summary>
public class ArithmeticReasoningTests
{
    private static Mind BuildMind(bool tools = true) =>
        MindBuilder.Create("Yume")
            .Describe("Yume (夢, 'dream') — a warm, curious companion AI.")
            .WithEmotions(e => e.Intensity(0.85))
            .WithReasoning()
            .WithThinking()
            .WithMemory()
            .WithTools(enabled: tools)
            .Build();

    private static void Teach(Mind mind, params string[] lines)
    {
        foreach (var line in lines)
            Assert.True(mind.Teach(line) > 0, $"teaching failed: {line}");
    }

    // ------------------------------------------------------------- taught facts

    [Fact]
    public void Taught_Fact_Is_Answered_Without_Calculator()
    {
        using var mind = BuildMind();
        Teach(mind, "2 + 2 = 4");

        var response = mind.Ask("Hello Yume, what is 2 + 2?");

        Assert.Null(mind.Tools!.LastInvokedTool);           // no crutch needed
        Assert.Contains("4", response.Content);
        Assert.Contains("taught", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.NotNull(response.Reasoning!.Answer);         // sanity: answer present
    }

    [Fact]
    public void Taught_Fact_Is_Stored_Structured()
    {
        var fact = FactParser.Parse("2 + 2 is 4").Single();
        Assert.Equal("2+2", fact.Subject);
        Assert.Equal("equals", fact.Predicate);
        Assert.Equal("4", fact.Object);

        var worded = FactParser.Parse("two plus two is four").Single();
        Assert.Equal(("2+2", "equals", "4"), (worded.Subject, worded.Predicate, worded.Object));
    }

    [Fact]
    public void Wrong_Teaching_Is_Rejected()
    {
        using var mind = BuildMind();
        var before = mind.Knowledge!.FactCount;

        mind.Process("2 + 2 is 5");

        Assert.Equal(before, mind.Knowledge.FactCount);
    }

    // ------------------------------------------------------------- derivation

    [Fact]
    public void Untaught_Sum_Is_Derived_By_Counting_On()
    {
        using var mind = BuildMind();
        Teach(mind, "2 + 1 = 3", "3 + 1 = 4");

        var response = mind.Ask("what is 2 + 2?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("4", response.Content);
        Assert.Contains("counting on", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Commutative_Taught_Fact_Is_Recognized()
    {
        using var mind = BuildMind();
        Teach(mind, "2 + 3 = 5");

        var response = mind.Ask("what is 3 + 2?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("5", response.Content);
        Assert.Contains("same as", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Subtraction_Is_Derived_From_Taught_Addition()
    {
        using var mind = BuildMind();
        Teach(mind, "2 + 1 = 3");

        var response = mind.Ask("what is 3 - 1?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("2", response.Content);
    }

    [Fact]
    public void Multiplication_Is_Derived_As_Repeated_Addition()
    {
        using var mind = BuildMind();
        Teach(mind, "1 + 1 = 2", "2 + 1 = 3", "3 + 1 = 4", "4 + 1 = 5", "5 + 1 = 6");

        var response = mind.Ask("what is 2 * 3?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("6", response.Content);
        Assert.Contains("added together 3 times", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Number_Words_Are_Understood()
    {
        using var mind = BuildMind();
        Teach(mind, "2 + 2 = 4");

        var response = mind.Ask("what is two plus two?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("4", response.Content);
    }

    // ------------------------------------------------------------- word problems

    [Fact]
    public void Word_Problem_Is_Reasoned_From_Taught_Arithmetic()
    {
        using var mind = BuildMind();
        Teach(mind, "1 + 1 = 2");

        var response = mind.Ask("if I have 1 apple and buy another one, how many apples do I have?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("2 apples", response.Content);
    }

    [Fact]
    public void Word_Problem_With_Taught_Subtraction()
    {
        using var mind = BuildMind();
        Teach(mind, "3 + 2 = 5");

        var response = mind.Ask("I have 5 cookies and I eat 2, how many cookies do I have?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("3 cookies", response.Content);
    }

    [Fact]
    public void Unanswerable_Word_Problem_Falls_Back_To_Calculator_Transparently()
    {
        using var mind = BuildMind(); // nothing taught at all

        var response = mind.Ask("if I have 4 apples and buy 3 more, how many apples do I have?");

        // Taught reasoning could not work this out, so the calculator fills
        // in — but it must SAY so openly instead of passing mechanical math
        // off as knowledge the mind reasoned out itself.
        Assert.Contains("not been taught", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("7", response.Content);
    }

    [Fact]
    public void Unanswerable_Word_Problem_Without_Tools_Admits_Not_Knowing()
    {
        using var mind = BuildMind(tools: false); // no tool module: no calculator crutch

        var response = mind.Ask("if I have 4 apples and buy 3 more, how many apples do I have?");

        Assert.DoesNotContain("7", response.Content);
    }

    // ------------------------------------------------------------- calculator as fallback

    [Fact]
    public void UnTaught_Question_Falls_Back_To_Calculator_Transparently()
    {
        using var mind = BuildMind();
        var response = mind.Ask("what is 2 + 3?");

        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Contains("not been taught", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("2 + 3 = 5", response.Content);
    }

    [Fact]
    public void UnTaught_Question_Without_Tools_Admits_Not_Knowing()
    {
        using var mind = BuildMind(tools: false);

        var response = mind.Ask("what is 2 + 3?");

        Assert.Null(mind.Tools);
        Assert.DoesNotContain("5", response.Content);
        Assert.DoesNotContain("learned", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void User_Can_Explicitly_Demand_The_Calculator()
    {
        using var mind = BuildMind();
        Teach(mind, "2 + 2 = 4");

        var response = mind.Ask("use the calculator to check what is 2 + 2");

        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Equal("tool-use", response.StyleName);
        Assert.Contains("2 + 2 = 4", response.Content);
    }

    // ------------------------------------------------------------- unlabeled questions

    [Fact]
    public void Unlabeled_Questions_Never_Become_Facts()
    {
        Assert.Empty(FactParser.Parse("yume what is 2 + 2"));
        Assert.Empty(FactParser.Parse("tell me what 2 + 2 is"));
        Assert.Empty(FactParser.Parse("hello yume what is 2 plus 2"));
    }

    // ------------------------------------------------------------- chained premises

    [Fact]
    public void Chained_Premise_Question_Checks_The_Trailing_Claim()
    {
        using var mind = BuildMind(tools: false);
        mind.Teach("birds are animals");
        mind.Teach("animals are living things");

        var response = mind.Ask("If a bird is an animal, and animals are living things, is a bird living?");

        Assert.Contains("Yes", response.Content);
        Assert.DoesNotContain("remember our earlier exchange", response.Content,
                              StringComparison.OrdinalIgnoreCase);
    }

    // ------------------------------------------------------------- no verbatim repeats

    [Fact]
    public void Consecutive_Teaching_Confirmations_Never_Repeat_Verbatim()
    {
        using var mind = BuildMind();

        var first = mind.Process("1 + 1 = 2");
        var second = mind.Process("2 + 1 = 3");
        var third = mind.Process("3 + 1 = 4");

        Assert.NotEqual(first.Content, second.Content);
        Assert.NotEqual(second.Content, third.Content);
        Assert.NotEqual(first.Content, third.Content);
    }
}
