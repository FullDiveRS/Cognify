using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Tests;

/// <summary>
/// Tests for Darwin's finer-grained emotions layered on top of Plutchik's
/// eight primaries (user requirement: "implement some other emotions" from
/// the Darwin taxonomy list — Joy, Anger, Disgust, Surprise and Fear already
/// existed and were kept).
/// </summary>
public class DarwinEmotionTests
{
    private static Mind BuildMind() =>
        MindBuilder.Create("Feeler")
            .WithEmotions(e => e.Intensity(1.0).Decay(0.0)) // no decay during tests
            .WithPersonality(p => p.Traits())               // all traits 0.5 → factor 1.0
            .Build();

    [Fact]
    public void Darwin_Emotion_Vocabulary_Is_Complete()
    {
        var names = Enum.GetNames<EmotionType>().ToHashSet();

        // The five the taxonomy shares with Plutchik's primaries...
        Assert.Contains("Joy", names);
        Assert.Contains("Anger", names);
        Assert.Contains("Disgust", names);
        Assert.Contains("Surprise", names);
        Assert.Contains("Fear", names);

        // ...plus the 29 finer-grained feelings, appended after the primaries.
        string[] darwin =
        {
            "Love", "Devotion", "Tenderness", "Suffering", "Weeping",
            "HighSpirits", "LowSpirits", "Anxiety", "Grief", "Dejection",
            "Despair", "Hatred", "Disdain", "Contempt", "Guilt", "Pride",
            "Helplessness", "Patience", "Affirmation", "Negation",
            "SelfAttention", "Shyness", "Modesty", "Blushing", "Reflection",
            "Meditation", "IllTemper", "Sulkiness", "Determination"
        };
        Assert.Equal(37, Enum.GetValues<EmotionType>().Length);
        Assert.Subset(names, darwin.ToHashSet());

        // Every emotion the state can feel must be tracked individually.
        var state = new EmotionalState();
        Assert.Equal(37, state.Intensities.Count);
    }

    [Fact]
    public void Despair_Words_Evoke_Despair()
    {
        using var mind = BuildMind();
        mind.Process("everything feels hopeless, I am in despair", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Despair] > 0.3);
        Assert.True(mind.EmotionalState.Valence < 0);
    }

    [Fact]
    public void Determination_Words_Evoke_Determination()
    {
        using var mind = BuildMind();
        mind.Process("I am determined and I will persist", intensity: 0.9);
        var state = mind.EmotionalState!;
        Assert.True(state.Intensities[EmotionType.Determination] > 0.3);
        Assert.True(state.Intensities[EmotionType.Determination] >
                    state.Intensities[EmotionType.Anticipation]);
    }

    [Fact]
    public void Weeping_Words_Evoke_Weeping()
    {
        using var mind = BuildMind();
        mind.Process("I am weeping and sobbing right now", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Weeping] > 0.3);
    }

    [Fact]
    public void Blushing_Words_Evoke_Blushing()
    {
        using var mind = BuildMind();
        mind.Process("I blush and feel flushed", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Blushing] > 0.3);
    }

    [Fact]
    public void Hatred_Words_Evoke_Hatred_Not_Anger()
    {
        using var mind = BuildMind();
        // Third-person hatred: "I hate you" would now be appraised as BLAME
        // (self-insult -> anger), which is the correct human response since
        // the social-appraisal triggers landed. This test is about the
        // Darwin "hatred" vocabulary specifically.
        mind.Process("watching them filled me with hatred and loathing", intensity: 0.9);
        var state = mind.EmotionalState!;
        Assert.True(state.Intensities[EmotionType.Hatred] > 0.3);
        Assert.True(state.Intensities[EmotionType.Hatred] >
                    state.Intensities[EmotionType.Anger]);
    }

    [Fact]
    public void Grief_Words_Evoke_Grief()
    {
        using var mind = BuildMind();
        mind.Process("I am grieving, my heart is broken with sorrow", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Grief] > 0.3);
    }

    [Fact]
    public void Anxiety_Words_Evoke_Anxiety()
    {
        using var mind = BuildMind();
        mind.Process("I feel uneasy and full of dread", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Anxiety] > 0.3);
    }

    [Fact]
    public void Affirmation_And_Negation_Words_Are_Feelings_Too()
    {
        using var mind = BuildMind();
        mind.Process("yes exactly absolutely", intensity: 0.9);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Affirmation] > 0.4);

        using var other = BuildMind();
        other.Process("no, that is wrong", intensity: 0.9);
        Assert.True(other.EmotionalState!.Intensities[EmotionType.Negation] > 0.3);
    }

    [Fact]
    public void HighSpirits_Are_Positive_And_Arousing()
    {
        var spirits = new EmotionalState();
        spirits.Feel(EmotionType.HighSpirits, 0.8);
        Assert.True(spirits.Valence > 0);
        Assert.True(spirits.Arousal > 0.1);

        var calm = new EmotionalState();
        calm.Feel(EmotionType.Patience, 0.8);
        Assert.True(spirits.Arousal > calm.Arousal);
    }

    [Fact]
    public void Humanize_Splits_PascalCase_Names()
    {
        Assert.Equal("joy", EmotionalState.Humanize(EmotionType.Joy));
        Assert.Equal("high spirits", EmotionalState.Humanize(EmotionType.HighSpirits));
        Assert.Equal("low spirits", EmotionalState.Humanize(EmotionType.LowSpirits));
        Assert.Equal("self attention", EmotionalState.Humanize(EmotionType.SelfAttention));
        Assert.Equal("ill temper", EmotionalState.Humanize(EmotionType.IllTemper));
        Assert.Equal("tenderness", EmotionalState.Humanize(EmotionType.Tenderness));
    }

    [Fact]
    public void Describe_Uses_Humanized_Names()
    {
        var state = new EmotionalState();
        state.Feel(EmotionType.SelfAttention, 0.5);
        state.UpdateMood(1.0);
        Assert.Contains("self attention", state.Describe());
        Assert.DoesNotContain("selfattention", state.Describe());
    }

    [Fact]
    public void Composer_Reads_Out_Humanized_Dominant_Emotion()
    {
        using var mind = BuildMind();
        mind.Process("I feel elated, thrilled and upbeat", intensity: 0.9);
        mind.Process("tell me something");
        var reply = mind.LastResponse;
        Assert.NotNull(reply);
        // PascalCase enum names never leak into speech: the mind says
        // "high spirits", never "highspirits".
        Assert.DoesNotContain("highspirits", reply.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task New_Emotion_Names_Survive_Save_And_Restore()
    {
        var path = Path.Combine(Path.GetTempPath(), $"darwin-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilder.Create("Sleeper")
                       .WithEmotions(e => e.Intensity(1.0).Decay(0.0))
                       .WithPersistence(path)
                       .Build())
            {
                mind.EmotionalState!.Feel(EmotionType.Despair, 0.6);
                mind.EmotionalState.Feel(EmotionType.SelfAttention, 0.4);
                await mind.SaveAsync();
            }

            using var restored = MindBuilder.Restore(path);
            // Emotions persist by NAME, so appended enum members round-trip.
            Assert.True(restored.EmotionalState!.Intensities[EmotionType.Despair] > 0.5);
            Assert.True(restored.EmotionalState.Intensities[EmotionType.SelfAttention] > 0.3);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
