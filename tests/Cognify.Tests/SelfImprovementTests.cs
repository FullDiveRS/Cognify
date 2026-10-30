using Cognify.Core;
using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>
/// Tests for the self-improvement module: implied feedback detection,
/// template demotion/promotion, reinforcement of the actually-spoken style,
/// review passes that retire disliked sayings, persistence.
/// </summary>
public class SelfImprovementTests
{
    // Local stand-in for the composer's template normalization (the real one
    // is internal): case, whitespace and numbers collapse the same way.
    private static string TemplateKey(string text) =>
        System.Text.RegularExpressions.Regex.Replace(
            System.Text.RegularExpressions.Regex.Replace(
                text.Trim().ToLowerInvariant(), @"\d+(?:\.\d+)?", "#"),
            @"\s+", " ");

    [Fact]
    public void ClassifyFeedback_DetectsApprovalAndDisapproval()
    {
        Assert.Equal(1, SelfImprovementModule.ClassifyFeedback("good job, that helped a lot"));
        Assert.Equal(1, SelfImprovementModule.ClassifyFeedback("thank you!"));
        Assert.Equal(-1, SelfImprovementModule.ClassifyFeedback("no, that is wrong"));
        Assert.Equal(-1, SelfImprovementModule.ClassifyFeedback("stop saying that"));
        // Questions are not grades, even when they contain cue words.
        Assert.Equal(0, SelfImprovementModule.ClassifyFeedback("are you sure that is right?"));
        Assert.Equal(0, SelfImprovementModule.ClassifyFeedback("tell me about birds"));
    }

    [Fact]
    public void NegativeFeedback_DemotesTemplate_AndReinforcesPreviousStyle()
    {
        using var mind = MindBuilder.Create("Learner")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .Build();
        var improvement = mind.SelfImprovement!;

        // Turn 1: she answers something.
        mind.Ask("what is a bird?");
        var replyTemplate = TemplateKey(mind.LastResponse!.Content);

        // Turn 2: the user clearly dislikes the previous reply.
        mind.Ask("no, that is wrong, stop saying that");

        Assert.Equal(1, improvement.NegativeSignals);
        Assert.True(improvement.TemplatePenalty(replyTemplate) > 0,
            "the disliked wording should be demoted");

        // The reinforcement loop closed on the hybrid engine.
        Assert.True(mind.Learning!.Stats.TotalExperiences > 0);
    }

    [Fact]
    public void PositiveFeedback_PromotesTemplate()
    {
        using var mind = MindBuilder.Create("Learner")
            .WithReasoning()
            .WithSelfLearning()
            .WithMemory()
            .Build();
        var improvement = mind.SelfImprovement!;

        mind.Ask("what is a bird?");
        var replyTemplate = TemplateKey(mind.LastResponse!.Content);
        mind.Ask("good job, exactly right, thank you");

        Assert.Equal(1, improvement.PositiveSignals);
        Assert.True(improvement.TemplateBoost(replyTemplate) > 0,
            "the approved wording should be promoted");
    }

    [Fact]
    public void FeedbackOnlyGradesTheImmediatelyPreviousReply()
    {
        using var mind = MindBuilder.Create("Learner")
            .WithSelfLearning().WithMemory().Build();
        var improvement = mind.SelfImprovement!;

        mind.Ask("what is a bird?");
        var firstTemplate = TemplateKey(mind.LastResponse!.Content);
        mind.Ask("tell me about cats");            // a newer reply now exists
        mind.Ask("no, that is wrong");             // grades the CATS reply, not birds

        Assert.Equal(1, improvement.NegativeSignals);
        Assert.Equal(0, improvement.TemplatePenalty(firstTemplate));
    }

    [Fact]
    public void SelfImprove_RetiresDislikedSayings()
    {
        using var mind = MindBuilder.Create("Learner")
            .WithSelfLearning().WithMemory().Build();
        var improvement = mind.SelfImprovement!;

        mind.Phrases.Add("I am a boring broken record line");
        Assert.Contains("I am a boring broken record line", mind.Phrases.All());

        // The host grades this specific wording directly (deterministic);
        // three clear rejections push its reputation past the threshold.
        for (int i = 0; i < 3; i++)
            improvement.RecordFeedback("I am a boring broken record line", -1.0);

        var report = mind.SelfImprove();
        Assert.DoesNotContain("I am a boring broken record line", mind.Phrases.All());
        Assert.Equal(1, report.SayingsPruned);
    }

    [Fact]
    public void SelfImprove_KeepsApprovedSayings()
    {
        using var mind = MindBuilder.Create("Learner")
            .WithSelfLearning().WithMemory().Build();
        var improvement = mind.SelfImprovement!;
        mind.Phrases.Add("Users love this charming line");

        improvement.RecordFeedback("Users love this charming line", 1.0);
        mind.SelfImprove();

        Assert.Contains("Users love this charming line", mind.Phrases.All());
    }

    [Fact]
    public void ImprovementLedger_PersistsAcrossSnapshot()
    {
        var path = Path.Combine(Path.GetTempPath(), $"cognify-si-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilder.Create("Sleepy")
                       .WithSelfLearning().WithMemory()
                       .WithPersistence(path).Build())
            {
                mind.Ask("what is a bird?");
                mind.Ask("no, that is wrong");
                mind.SaveAsync(path).Wait();
            }

            using (var restored = MindBuilder.Restore(path))
            {
                Assert.Equal(1, restored.SelfImprovement!.NegativeSignals);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void SelfImprove_Throws_WhenLearningDisabled()
    {
        using var mind = MindBuilder.Create("NoLearner").Build();
        Assert.Throws<MindConfigurationException>(() => mind.SelfImprove());
    }
}
