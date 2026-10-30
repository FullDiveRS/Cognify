using Cognify.Core;

namespace Cognify.Personality;

/// <summary>
/// The personality module attaches Big Five traits to the mind and exposes
/// them to every other module. It runs early in the cycle and publishes
/// trait-derived parameters (exploration rate, deliberation depth) into the
/// context bag so downstream modules can shape behavior.
/// </summary>
public sealed class PersonalityModule : CognitiveModuleBase
{
    internal PersonalityModule(PersonalityOptions options) => Traits = new BigFiveTraits(options);

    /// <inheritdoc />
    public override string Name => "personality";

    /// <inheritdoc />
    public override int Order => 5;

    /// <summary>The mind's personality traits.</summary>
    public BigFiveTraits Traits { get; }

    /// <summary>Moves a trait toward a new value gradually (personality drift).</summary>
    public void Nudge(string trait, double delta)
    {
        switch (trait.ToLowerInvariant())
        {
            case "openness": Traits.Openness = Math.Clamp(Traits.Openness + delta, 0, 1); break;
            case "conscientiousness": Traits.Conscientiousness = Math.Clamp(Traits.Conscientiousness + delta, 0, 1); break;
            case "extraversion": Traits.Extraversion = Math.Clamp(Traits.Extraversion + delta, 0, 1); break;
            case "agreeableness": Traits.Agreeableness = Math.Clamp(Traits.Agreeableness + delta, 0, 1); break;
            case "neuroticism": Traits.Neuroticism = Math.Clamp(Traits.Neuroticism + delta, 0, 1); break;
        }
    }

    /// <inheritdoc />
    public override void Process(CognitiveContext context)
    {
        context.Bag["exploration.rate"] = Traits.ExplorationRate(
            context.Mind.Config.Learning.BaseExploration);
        context.Bag["deliberation.depth"] = Traits.DeliberationDepth(
            context.Mind.Config.Thinking.DeliberationCandidates);
        context.Bag["goal.focus"] = Traits.GoalFocusBias;
    }
}
