using Cognify.Core;

namespace Cognify.Tests;

/// <summary>Tests for drives, goals, planning and autonomous motivation.</summary>
public class GoalTests
{
    private static Mind BuildMind(double driveGrowth = 0.05) =>
        MindBuilder.Create("Motivated")
            .WithGoals(g => { g.DriveGrowth = driveGrowth; })
            .Build();

    [Fact]
    public void AddGoal_SetsActiveStatus()
    {
        using var mind = BuildMind();
        var goal = mind.GoalSystem!.AddGoal("write-book", "write a novel", 0.8, new[] { "achievement" });
        Assert.Equal(Goals.GoalStatus.Active, goal.Status);
        Assert.Contains(goal, mind.GoalSystem.ActiveGoals());
    }

    [Fact]
    public void ReportProgress_AchievesGoal()
    {
        using var mind = BuildMind();
        var goal = mind.GoalSystem!.AddGoal("task", priority: 0.5, tags: new[] { "achievement" });
        bool achieved = false;
        mind.GoalUpdated += (_, e) => { if (e.Change == "achieved") achieved = true; };
        mind.GoalSystem.ReportProgress(goal, 1.0);
        Assert.Equal(Goals.GoalStatus.Achieved, goal.Status);
        Assert.True(achieved);
        Assert.DoesNotContain(goal, mind.GoalSystem.ActiveGoals());
    }

    [Fact]
    public void AchievingGoal_SatisfiesDrive()
    {
        using var mind = BuildMind();
        var drive = mind.GoalSystem!.Drives["achievement"];
        drive.SetLevel(0.8);
        var goal = mind.GoalSystem.AddGoal("task", priority: 0.5, tags: new[] { "achievement" });
        mind.GoalSystem.ReportProgress(goal, 1.0);
        Assert.True(drive.Level < 0.8);
    }

    [Fact]
    public void Urgency_PrefersHigherPriority()
    {
        using var mind = BuildMind();
        var low = mind.GoalSystem!.AddGoal("low", priority: 0.2);
        var high = mind.GoalSystem.AddGoal("high", priority: 0.9);
        Assert.True(high.Urgency(DateTimeOffset.UtcNow) > low.Urgency(DateTimeOffset.UtcNow));
        Assert.Equal(high, mind.GoalSystem.ActiveGoals()[0]);
    }

    [Fact]
    public void Deadline_IncreasesUrgency()
    {
        using var mind = BuildMind();
        var plain = mind.GoalSystem!.AddGoal("plain", priority: 0.5);
        var urgent = mind.GoalSystem.AddGoal("urgent", priority: 0.5,
            deadline: DateTimeOffset.UtcNow.AddDays(-1));
        Assert.True(urgent.Urgency(DateTimeOffset.UtcNow) > plain.Urgency(DateTimeOffset.UtcNow));
    }

    [Fact]
    public void Drives_GrowOverTicks()
    {
        using var mind = BuildMind(driveGrowth: 0.1);
        double before = mind.GoalSystem!.Drives["curiosity"].Level;
        for (int i = 0; i < 10; i++) mind.Tick();
        Assert.True(mind.GoalSystem.Drives["curiosity"].Level > before);
    }

    [Fact]
    public void HighDrive_AutonomouslySpawnsGoal()
    {
        using var mind = BuildMind(driveGrowth: 0.2);
        for (int i = 0; i < 5; i++) mind.Tick(); // curiosity reaches 0.15 + 5*0.2 > 0.75
        Assert.Contains(mind.GoalSystem!.AllGoals(), g => g.Name.Contains("satisfy-curiosity"));
    }

    [Fact]
    public void FocusGoal_PublishedToContext()
    {
        using var mind = BuildMind();
        mind.GoalSystem!.AddGoal("focus-me", priority: 0.95, tags: new[] { "calm" });
        mind.Process("hello");
        Assert.Equal("focus-me", mind.GoalSystem.FocusGoal?.Name);
    }

    [Fact]
    public void GoalRelevantStimulus_GivesProgress()
    {
        using var mind = BuildMind();
        var goal = mind.GoalSystem!.AddGoal("learn", priority: 0.9, tags: new[] { "curiosity" });
        double before = goal.Progress;
        mind.Process("let us talk about curiosity and learning");
        Assert.True(goal.Progress > before);
    }

    [Fact]
    public void AbandonedGoal_NotActive()
    {
        using var mind = BuildMind();
        var goal = mind.GoalSystem!.AddGoal("dropped", priority: 0.9);
        mind.GoalSystem.Abandon(goal);
        Assert.Equal(Goals.GoalStatus.Abandoned, goal.Status);
        Assert.DoesNotContain(goal, mind.GoalSystem.ActiveGoals());
    }
}
