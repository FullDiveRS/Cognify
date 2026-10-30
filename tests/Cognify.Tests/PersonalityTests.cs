using Cognify.Core;
using Cognify.Personality;

namespace Cognify.Tests;

/// <summary>Tests for Big Five personality traits and their influence.</summary>
public class PersonalityTests
{
    [Fact]
    public void BuilderConfiguresTraits()
    {
        using var mind = MindBuilder.Create("P")
            .WithPersonality(p => p.Traits(openness: 0.9, neuroticism: 0.2))
            .Build();
        var traits = mind.Personality!;
        Assert.Equal(0.9, traits.Openness);
        Assert.Equal(0.2, traits.Neuroticism);
        Assert.Equal(0.5, traits.Extraversion); // default
    }

    [Fact]
    public void EmotionFactor_HighNeuroticism_AmplifiesFear()
    {
        var sensitive = new BigFiveTraits { Neuroticism = 0.9 };
        var stoic = new BigFiveTraits { Neuroticism = 0.1 };
        Assert.True(sensitive.EmotionFactor(Emotions.EmotionType.Fear) >
                    stoic.EmotionFactor(Emotions.EmotionType.Fear));
    }

    [Fact]
    public void EmotionFactor_HighExtraversion_AmplifiesJoy()
    {
        var sunny = new BigFiveTraits { Extraversion = 0.9 };
        var shy = new BigFiveTraits { Extraversion = 0.1 };
        Assert.True(sunny.EmotionFactor(Emotions.EmotionType.Joy) >
                    shy.EmotionFactor(Emotions.EmotionType.Joy));
    }

    [Fact]
    public void DeliberationDepth_ConscientiousnessDeepensThinking()
    {
        var careful = new BigFiveTraits { Conscientiousness = 0.95 };
        var quick = new BigFiveTraits { Conscientiousness = 0.05 };
        Assert.True(careful.DeliberationDepth(6) > quick.DeliberationDepth(6));
    }

    [Fact]
    public void Describe_SummarizesDominantTraits()
    {
        var traits = new BigFiveTraits
        {
            Openness = 0.9,
            Extraversion = 0.8,
            Neuroticism = 0.9
        };
        string description = traits.Describe();
        Assert.Contains("curious", description);
        Assert.Contains("outgoing", description);
        Assert.Contains("sensitive", description);
    }

    [Fact]
    public void Nudge_MovesTraitGradually()
    {
        using var mind = MindBuilder.Create("P").WithPersonality().Build();
        var module = mind.Module<PersonalityModule>()!;
        module.Nudge("openness", 0.1);
        Assert.Equal(0.6, mind.Personality!.Openness, precision: 3);
        module.Nudge("openness", 2.0); // clamps at 1
        Assert.Equal(1.0, mind.Personality.Openness);
    }

    [Fact]
    public void SameInput_DifferentPersonalities_DifferentExploration()
    {
        using var open = MindBuilder.Create("Open")
            .WithPersonality(p => p.Traits(openness: 0.95))
            .WithThinking()
            .WithSelfLearning(l => l.Exploration(0.3))
            .Build();
        var module = open.Module<PersonalityModule>()!;
        var traits = module.Traits;
        Assert.True(traits.ExplorationRate(0.3) > 0.3);
    }
}
