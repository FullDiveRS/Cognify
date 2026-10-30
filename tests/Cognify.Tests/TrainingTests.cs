using Cognify.Core;

namespace Cognify.Tests;

/// <summary>
/// Tests for the train option: facts become knowledge, counting sequences
/// become the number line, and plain conversational lines become sayings —
/// the mind's spoken vocabulary. A bigger phrase pool directly widens reply
/// variety, which is the structural fix for "she keeps repeating the same
/// sentence".
/// </summary>
public class TrainingTests : IDisposable
{
    private readonly string _path = Path.Combine(
        Path.GetTempPath(), $"cognify-train-{Guid.NewGuid():N}.db");

    public void Dispose()
    {
        if (File.Exists(_path)) File.Delete(_path);
    }

    private static Mind BuildMind() =>
        MindBuilder.Create("Yume")
            .Describe("training test mind")
            .WithPersonality(p => p.Traits(openness: 0.9, conscientiousness: 0.6,
                                           extraversion: 0.75, agreeableness: 0.85,
                                           neuroticism: 0.3))
            .WithEmotions(e => e.Intensity(0.85))
            .WithReasoning()
            .WithThinking(t => t.Steps(5))
            .WithSelfLearning(l => l.Rate(0.12))
            .WithMemory(m => m.Capacity(20))
            .WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness"))
            .WithTools()
            .Build();

    // ------------------------------------------------------------- the split

    [Fact]
    public void Train_Splits_Facts_Sayings_And_Number_Lines()
    {
        using var mind = BuildMind();

        var report = mind.TrainAll(new[]
        {
            "Birds are animals.",
            "1 + 1 = 2.",
            "The number system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.",
            "I find the stars endlessly fascinating.",
        });

        Assert.Equal(4, report.Lines);
        Assert.Equal(3, report.FactsLearned);   // birds/animals + 1+1=2 + the number-line sentence
        Assert.Equal(1, report.SayingsAdded);   // only the conversational line
        Assert.Equal(1, report.SequencesTaught);
        Assert.Equal(1, mind.Phrases.Count);
        Assert.Equal(3, mind.Knowledge!.FactCount);
    }

    [Fact]
    public void Train_Skips_Questions_And_Conditionals_As_Sayings()
    {
        using var mind = BuildMind();

        var report = mind.TrainAll(new[]
        {
            "What is a bird?",
            "if i have no apples and i buy an apple, how many do i have?",
            "Tell me about the weather.",
            "I enjoy rainy days.",
        });

        Assert.Equal(4, report.Lines);
        Assert.Equal(0, report.FactsLearned);
        Assert.Equal(1, report.SayingsAdded);   // only "I enjoy rainy days."
        Assert.Equal("I enjoy rainy days", mind.Phrases.All()[0]);
    }

    [Fact]
    public void Train_Never_Duplicates_A_Saying()
    {
        using var mind = BuildMind();

        mind.Train("I love learning new things.");
        var second = mind.Train("i love learning new things!"); // same, different case/punct

        Assert.Equal(0, second.SayingsAdded);
        Assert.Equal(1, mind.Phrases.Count);
    }

    // ------------------------------------------------------------- file input

    [Fact]
    public void TrainFile_Reads_Corpus_And_Ignores_Comments_And_Blanks()
    {
        var corpus = Path.Combine(Path.GetTempPath(), $"corpus-{Guid.NewGuid():N}.txt");
        File.WriteAllLines(corpus, new[]
        {
            "# a comment line",
            "",
            "Cats are animals.",
            "What is love?",
            "I enjoy quiet mornings.",
        });
        try
        {
            using var mind = BuildMind();
            var report = mind.TrainFile(corpus);

            Assert.Equal(3, report.Lines);
            Assert.Equal(1, report.FactsLearned);
            Assert.Equal(1, report.SayingsAdded);
        }
        finally
        {
            File.Delete(corpus);
        }
    }

    [Fact]
    public void TrainFile_Missing_File_Throws_FileNotFound()
    {
        using var mind = BuildMind();
        Assert.Throws<FileNotFoundException>(() => mind.TrainFile("no/such/corpus.txt"));
    }

    // ------------------------------------------------------------- persistence

    [Fact]
    public async Task Sayings_Survive_Save_And_Restore()
    {
        using (var mind = MindBuilder.Create("Trained").WithAllModules().Build())
        {
            mind.TrainAll(new[] { "I enjoy quiet mornings.", "Rainy afternoons put me in a quiet mood." });
            await mind.SaveAsync(_path);
        }

        using var restored = MindBuilder.Restore(_path);
        Assert.Equal(2, restored.Phrases.Count);
        Assert.Contains(restored.Phrases.All(),
                        s => s.StartsWith("I enjoy quiet", StringComparison.OrdinalIgnoreCase));
    }

    // ------------------------------------------------------------- vocabulary in action

    [Fact]
    public void Trained_Sayings_Are_Spoken_In_The_Minds_Own_Voice()
    {
        using var mind = BuildMind();
        // Cross-implementation sayings: none of these parse as facts in
        // either the C# or the Python fact parser (no copulas / fact verbs).
        string[] sayings =
        {
            "I find the stars endlessly fascinating.",
            "Old stories open doors into other lives.",
            "I enjoy quiet mornings.",
            "Rainy afternoons put me in a quiet mood.",
            "The ocean sounds endless and calm to me.",
            "Bread baking reminds me of a happy home.",
        };
        Assert.Equal(sayings.Length, mind.TrainAll(sayings).SayingsAdded);

        // Unanswerable small talk — nothing in the knowledge base, no math.
        // Style rotation pushes deliberation through the whole candidate set,
        // so within a handful of turns a trained saying must surface verbatim.
        string[] prompts =
        {
            "tell me something you think about",
            "what do you wonder about",
            "share a thought with me",
            "i am curious what you like",
            "say whatever is on your mind",
            "what feels nice to think about",
            "go on, surprise me",
            "is there something you find lovely",
            "what occupies your dreams",
            "talk to me about anything",
        };
        var spoken = new HashSet<string>();
        foreach (var prompt in prompts)
            spoken.Add(mind.Ask(prompt).Content.TrimEnd('.', '!').ToLowerInvariant());

        Assert.Contains(spoken, s => sayings.Any(
            sy => string.Equals(sy.TrimEnd('.', '!'), s, StringComparison.OrdinalIgnoreCase)));
    }

    // ------------------------------------------------------------- regression

    [Fact]
    public void Word_Problem_Starting_From_Nothing_Is_Reasoned_Out()
    {
        using var mind = BuildMind();
        Teach(mind, "1 + 1 = 2"); // taught, but NO "0 + 1" anywhere

        var response = mind.Ask("if i have no apples and i buy an apple, how many apples do i have?");

        // "no apples" = 0 is stated by the problem itself: 0 + 1 is an
        // identity the mind can reason out — no calculator, no refusal.
        Assert.Null(mind.Tools!.LastInvokedTool);
        Assert.Contains("1 apple", response.Content);
    }

    [Fact]
    public void Quoted_Word_Problem_Is_Answered()
    {
        using var mind = BuildMind();
        Teach(mind, "1 + 1 = 2");

        var response = mind.Ask(
            "what is the answer to \"if i have no apples and i buy an apple. how many do i currently have?\"");

        Assert.Contains("1", response.Content);
        Assert.Contains("apple", response.Content);
    }

    // ------------------------------------------------------------- word meanings

    [Fact]
    public void Means_Lines_Become_Recallable_Knowledge()
    {
        using var mind = BuildMind();
        var report = mind.TrainAll(new[]
        {
            "Joy means a bright feeling of gladness deep inside.",
            "Happiness means feeling glad about being alive.",
        });

        Assert.Equal(2, report.Lines);
        Assert.Equal(2, report.FactsLearned);   // dictionary knowledge, not sayings
        Assert.Equal(0, report.SayingsAdded);
        Assert.NotNull(mind.Knowledge!.Query("joy", "means",
            "bright feeling of gladness deep inside"));

        var spoken = mind.Ask("what does joy mean?").Content;
        Assert.Contains("gladness", spoken, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Meaning_Of_And_Define_Questions_Route_To_The_Dictionary()
    {
        using var mind = BuildMind();
        mind.TrainAll(new[] { "Patience means the calm strength of being able to wait." });

        var meaning = mind.Ask("what is the meaning of patience?").Content;
        Assert.Contains("wait", meaning, StringComparison.OrdinalIgnoreCase);

        var defined = mind.Ask("define patience").Content;
        Assert.Contains("wait", defined, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Describe_Leads_With_The_Meaning()
    {
        using var mind = BuildMind();
        mind.TrainAll(new[] { "A fox is an animal.",
                              "Fox means a clever red rustler of the night." });

        var spoken = mind.Ask("what is a fox?").Content;
        Assert.Contains("rustler", spoken, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("animal", spoken, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Ness_Words_Keep_Their_Tail()
    {
        using var mind = BuildMind();
        mind.TrainAll(new[] { "Kindness means the gentle art of caring for people." });

        var facts = mind.Knowledge!.Query("kindness", "means", null);
        // The old singularizer stripped the trailing "s": "kindnes" — mangled.
        Assert.Single(facts);
        Assert.Equal("kindness", facts[0].Subject);

        var spoken = mind.Ask("what does kindness mean?").Content;
        Assert.Contains("caring", spoken, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Words_And_Meanings_Corpus_Trains_All_Lines_As_Facts()
    {
        var corpus = FindCorpus();
        if (corpus is null) return; // corpus ships with the demos, not the tests

        using var mind = BuildMind();
        var report = mind.TrainFile(corpus);

        // Parity with CognifyPy: every shipped entry must land as knowledge —
        // nothing leaks into the sayings pool, nothing fails to parse.
        Assert.Equal(report.Lines, report.FactsLearned);
        Assert.Equal(0, report.SayingsAdded);
        Assert.Equal(report.FactsLearned, mind.Knowledge!.FactCount);
        Assert.Contains("wait", mind.Ask("what does patience mean?").Content,
                        StringComparison.OrdinalIgnoreCase);
        Assert.Contains("caring", mind.Ask("what does kindness mean?").Content,
                        StringComparison.OrdinalIgnoreCase);
    }

    private static string? FindCorpus()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir is not null; i++, dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName,
                "demo", "Cognify.Yume", "corpus", "words-and-meanings.txt");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    private static void Teach(Mind mind, string sentence)
    {
        foreach (var fact in Reasoning.FactParser.Parse(sentence))
            mind.Knowledge!.Add(fact);
    }
}
