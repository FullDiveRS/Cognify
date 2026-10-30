using Cognify.Core;
using Cognify.Emotions;

namespace Cognify.Tests;

/// <summary>Tests for emotional appraisal, mood and decay.</summary>
public class EmotionTests
{
    private static Mind BuildMind(double neuroticism = 0.5) =>
        MindBuilder.Create("Feeler")
            .WithEmotions(e => e.Intensity(1.0).Decay(0.0)) // no decay during tests
            .WithPersonality(p => p.Traits(neuroticism: neuroticism))
            .Build();

    [Fact]
    public void PositiveWords_EvokeJoy()
    {
        using var mind = BuildMind();
        mind.Process("I love this wonderful amazing day", intensity: 0.9);
        var state = mind.EmotionalState!;
        // "love" is its own Darwin emotion now; the joy words carry Joy.
        Assert.True(state.Intensities[EmotionType.Joy] > 0.2);
        Assert.True(state.Intensities[EmotionType.Love] > 0.1);
        Assert.True(state.Valence > 0);
    }

    [Fact]
    public void NegativeWords_EvokeSadness()
    {
        using var mind = BuildMind();
        mind.Process("I lost everything and I feel sad and hurt");
        var state = mind.EmotionalState!;
        Assert.True(state.Intensities[EmotionType.Sadness] > 0.2);
        Assert.True(state.Valence < 0);
    }

    [Fact]
    public void ThreatWords_EvokeFear()
    {
        using var mind = BuildMind();
        mind.Process("there is danger and I am scared");
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Fear] > 0.15);
    }

    [Fact]
    public void NeutralText_KeepsEmotionsLow()
    {
        using var mind = BuildMind();
        mind.Process("the table has four legs and is made of wood");
        Assert.True(mind.EmotionalState!.DominantIntensity < 0.2);
    }

    [Fact]
    public void Decay_ReducesIntensity()
    {
        var state = new EmotionalState();
        state.Feel(EmotionType.Joy, 0.8);
        state.Decay(0.5);
        Assert.Equal(0.4, state.Intensities[EmotionType.Joy], precision: 2);
    }

    [Fact]
    public void Feel_ClampsAtOne()
    {
        var state = new EmotionalState();
        state.Feel(EmotionType.Joy, 2.0);
        Assert.Equal(1.0, state.Intensities[EmotionType.Joy]);
    }

    [Fact]
    public void Valence_SignMatchesEmotion()
    {
        var joy = new EmotionalState();
        joy.Feel(EmotionType.Joy, 0.8);
        Assert.True(joy.Valence > 0);

        var fear = new EmotionalState();
        fear.Feel(EmotionType.Fear, 0.8);
        Assert.True(fear.Valence < 0);
    }

    [Fact]
    public void Arousal_HighForFear_LowForTrust()
    {
        var fear = new EmotionalState();
        fear.Feel(EmotionType.Fear, 0.8);
        var trust = new EmotionalState();
        trust.Feel(EmotionType.Trust, 0.8);
        Assert.True(fear.Arousal > trust.Arousal);
    }

    [Fact]
    public void Neuroticism_AmplifiesNegativeEmotion()
    {
        double GetSadness(double neuroticism)
        {
            using var mind = BuildMind(neuroticism);
            mind.Process("I feel sad and hurt and lost");
            return mind.EmotionalState!.Intensities[EmotionType.Sadness];
        }

        Assert.True(GetSadness(0.9) > GetSadness(0.1));
    }

    [Fact]
    public void Extraversion_AmplifiesJoy()
    {
        double GetJoy(double extraversion)
        {
            using var mind = MindBuilder.Create("E")
                .WithEmotions()
                .WithPersonality(p => p.Traits(extraversion: extraversion))
                .Build();
            mind.Process("I love this wonderful amazing great day");
            return mind.EmotionalState!.Intensities[EmotionType.Joy];
        }

        Assert.True(GetJoy(0.9) > GetJoy(0.1));
    }

    [Fact]
    public void Mood_FollowsEmotion()
    {
        var state = new EmotionalState();
        state.Feel(EmotionType.Joy, 0.9);
        state.UpdateMood(1.0); // full inertia = mood = emotion now
        Assert.True(state.MoodValence > 0);
    }

    [Fact]
    public void Feel_API_WorksOnMind()
    {
        using var mind = BuildMind();
        var emotions = mind.Module<EmotionModule>()!;
        emotions.Feel(EmotionType.Surprise, 0.7);
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Surprise] > 0.5);
    }

    [Fact]
    public void EmotionChanged_EventFires()
    {
        using var mind = BuildMind();
        EmotionType? changed = null;
        mind.EmotionChanged += (_, e) => changed = e.Emotion;
        mind.Process("I love this amazing wonderful great happy thing", intensity: 0.9);
        Assert.True(changed is EmotionType.Joy);
    }

    [Fact]
    public void Feedback_Stimulus_ProducesRewardEmotion()
    {
        using var mind = BuildMind();
        mind.Process(Stimulus.Feedback(0.8));
        Assert.True(mind.EmotionalState!.Intensities[EmotionType.Joy] > 0.2);
    }

    [Fact]
    public void Clone_IsDeep()
    {
        var a = new EmotionalState();
        a.Feel(EmotionType.Joy, 0.5);
        var b = a.Clone();
        a.Feel(EmotionType.Joy, 0.5);
        Assert.True(b.Intensities[EmotionType.Joy] < a.Intensities[EmotionType.Joy]);
    }
}
