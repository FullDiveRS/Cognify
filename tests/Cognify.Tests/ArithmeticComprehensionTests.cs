using Cognify.Core;
using Cognify.Reasoning;

namespace Cognify.Tests;

/// <summary>
/// Comprehension regressions from the apple-and-number-system transcript:
/// natural quantities ("no apples", "an apple"), the "how may" typo, the
/// taught counting ladder (reasoning along "0, 1, 2, ..., 9" instead of the
/// calculator), honest learned counts (conditionals / existentials / "can
/// be" are not facts), template-level repeat avoidance and topic salience.
/// </summary>
public class ArithmeticComprehensionTests
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

    private const string NumberSystemLesson =
        "there is a 10 based number system, starts at 0, 0 can be nothing or can be a holder for something. " +
        "the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. " +
        "if i have no apples and i buy an apple. how many do i currently have?";

    // ------------------------------------------------------------- counting ladder

    [Fact]
    public void Taught_Number_Line_Lets_Her_Count_On_Without_Calculator()
    {
        using var mind = BuildMind();
        mind.Process("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");

        var response = mind.Ask("what is 3 + 4?");

        Assert.Null(mind.Tools!.LastInvokedTool);            // counting, not calculating
        Assert.Contains("7", response.Content);
        Assert.Contains("counting on", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Beyond_The_Taught_Line_She_Uses_The_Calculator_Openly()
    {
        using var mind = BuildMind();
        mind.Process("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");

        var response = mind.Ask("what is 8 + 5?");           // 13 lies beyond the taught line

        Assert.Equal("calculator", mind.Tools!.LastInvokedTool);
        Assert.Contains("13", response.Content);
    }

    [Fact]
    public void Taught_Line_Also_Counts_Backward()
    {
        using var mind = BuildMind();
        mind.Process("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");

        var response = mind.Ask("if i have 3 apples and i eat an apple, how many apples do i have?");

        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("2 apples", response.Content);
    }

    // ------------------------------------------------------------- natural quantities

    [Fact]
    public void No_Apples_And_An_Apple_Are_Understood()
    {
        using var mind = BuildMind();

        var response = mind.Process(NumberSystemLesson);

        Assert.Contains("1 apple", response.Content);        // answered, not just "learned"
        Assert.Null(mind.Tools!.LastInvokedTool);
    }

    [Fact]
    public void Quoted_Word_Problem_Is_Answered_Not_Echoed()
    {
        using var mind = BuildMind();
        mind.Process(NumberSystemLesson);

        var response = mind.Ask(
            "what is the answer to \"if i have no apples and i buy an apple. how many do i currently have?\"");

        Assert.Contains("1 apple", response.Content);
        Assert.DoesNotContain("remember our earlier exchange", response.Content,
                              StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void How_May_Typo_Is_Still_A_Question()
    {
        using var mind = BuildMind();
        mind.Process(NumberSystemLesson);

        var response = mind.Ask("if i have 0 apples, and i buy an apple. how may apples do i have now?");

        Assert.Contains("1 apple", response.Content);
        Assert.DoesNotContain("learned", response.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Stated_Quantity_Alone_Is_Answerable()
    {
        using var mind = BuildMind();

        var response = mind.Ask("i have 3 apples, how many apples do i have?");

        Assert.Contains("3 apples", response.Content);
    }

    // ------------------------------------------------------------- honest learned counts

    [Fact]
    public void Conditionals_Existentials_And_Can_Be_Are_Not_Facts()
    {
        Assert.Empty(FactParser.Parse("if i have no apples and i buy an apple"));
        Assert.Empty(FactParser.Parse("there is a 10 based number system, starts at 0"));
        Assert.Empty(FactParser.Parse("0 can be nothing or can be a holder for something"));

        // Real knowledge still parses.
        Assert.Single(FactParser.Parse("birds can fly"));
        Assert.Single(FactParser.Parse("the system is base 10"));
    }

    [Fact]
    public void Lesson_Learns_One_Honest_Fact_Not_Garbage()
    {
        using var mind = BuildMind();
        var before = mind.Knowledge!.FactCount;

        mind.Process(NumberSystemLesson);

        Assert.Equal(before + 1, mind.Knowledge.FactCount);  // only "the system is 0-1-...-9"
    }

    // ------------------------------------------------------------- template-level repeats

    [Fact]
    public void Teaching_Turns_Never_Reuse_The_Same_Sentence_Template()
    {
        using var mind = BuildMind();

        var replies = new[]
        {
            mind.Process("2 + 2 is 4").Content,
            mind.Process("3 + 3 is 6").Content,
            mind.Process("10 + 10 is 20").Content,
            mind.Process("5 + 5 is 10").Content,
        };

        for (int i = 0; i < replies.Length; i++)
            for (int j = i + 1; j < replies.Length; j++)
                Assert.NotEqual(replies[i], replies[j]);
    }

    // ------------------------------------------------------------- topic salience

    [Fact]
    public void Time_Fillers_Never_Become_The_Topic()
    {
        using var mind = BuildMind();
        mind.Process("my favorite fruit is apples");

        var response = mind.Ask("what do you recall about my apples currently?");

        Assert.DoesNotContain("about currently", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("currently", response.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("apples", response.Content, StringComparison.OrdinalIgnoreCase);
    }
}
