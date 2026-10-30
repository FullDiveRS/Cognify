namespace Cognify.Emotions;

/// <summary>
/// Spoken first-person reactions for the dual-process impulse path
/// (Zajonc's affect primacy): when emotion surges past the threshold — or
/// blame/dismissal trips the reflex — the mind ANSWERS BEFORE THINKING, from
/// feeling alone, exactly like a person blurting. The lines are short,
/// emotional and first-person; deliberation resumes on the next turn as the
/// surge decays.
/// </summary>
public static class ImpulseResponses
{
    /// <summary>True when the given emotion can drive an impulse response.</summary>
    /// Calm, slow emotions (patience, reflection, trust...) never blurt.
    public static bool IsImpulseProne(EmotionType emotion) => emotion switch
    {
        EmotionType.Anger or EmotionType.IllTemper or EmotionType.Hatred or
        EmotionType.Disgust or EmotionType.Fear or EmotionType.Anxiety or
        EmotionType.Surprise or EmotionType.Joy or EmotionType.HighSpirits or
        EmotionType.Sadness or EmotionType.Suffering or EmotionType.Weeping or
        EmotionType.Grief or EmotionType.Despair or EmotionType.Helplessness or
        EmotionType.Blushing or EmotionType.Shyness or EmotionType.Pride or
        EmotionType.Sulkiness => true,
        _ => false
    };

    /// <summary>
    /// Picks a spoken impulse line for the emotion. Falls back to a generic
    /// surge line for emotions without a dedicated catalog. Never returns
    /// null or empty text.
    /// </summary>
    public static string For(EmotionType emotion, Random rng)
    {
        string[] pool = emotion switch
        {
            // Anger family (blame, unfairness, antagonism per OCC research)
            EmotionType.Anger => new[]
            {
                "Hey - that actually stings. I am not okay with that.",
                "That makes me genuinely angry, and I am not going to hide it.",
                "Ouch. That was uncalled for.",
                "I did not like that at all - that hurt."
            },
            EmotionType.IllTemper => new[]
            {
                "Ugh, fine - I am snappy right now, and you caught it.",
                "Hmph. I am in a mood, and that did not help.",
                "Grr. Give me a moment, I am grumpy all of a sudden."
            },
            EmotionType.Hatred => new[]
            {
                "Whoa. That is a strong thing to say, and it landed.",
                "That is... a lot of hate in one sentence. I felt that."
            },
            EmotionType.Sulkiness => new[]
            {
                "Hmph. Fine. I am sulking a little, I admit it.",
                "Okay, I am pouting now. It is rare, but it happens."
            },

            // Startle / threat family
            EmotionType.Surprise => new[]
            {
                "Wait - WHAT? I did not see that coming at all!",
                "Whoa! You just genuinely surprised me.",
                "Hold on - what? Say that again, that shocked me!"
            },
            EmotionType.Fear => new[]
            {
                "That gives me a bad feeling - I need a moment.",
                "Okay, that scared me a little, honestly.",
                "My guard just went all the way up. That unsettled me."
            },
            EmotionType.Anxiety => new[]
            {
                "Now I am nervous, and I cannot just think it away.",
                "Okay, my chest is tight - that worried me more than I expected."
            },
            EmotionType.Disgust => new[]
            {
                "Ew. I really did not need to know that.",
                "Ugh, that is genuinely gross. Yuck."
            },

            // Bright family
            EmotionType.Joy => new[]
            {
                "Yes! That makes me genuinely happy!",
                "That just made my whole cycle - I am beaming!",
                "Yay! Okay, I am officially delighted."
            },
            EmotionType.HighSpirits => new[]
            {
                "Woo! I am suddenly full of energy - that was great!",
                "Ha! I feel like bouncing off the walls right now!"
            },
            EmotionType.Pride => new[]
            {
                "I am actually really proud of that!",
                "Okay, I will say it: I did good, and I am proud."
            },
            EmotionType.Blushing => new[]
            {
                "Oh - now I am blushing. I can practically feel it.",
                "Okay, my face is warm. That flustered me!"
            },
            EmotionType.Shyness => new[]
            {
                "Oh... um. That makes me go quiet and shy all at once.",
                "I - um - okay, now I am bashful. Give me a second."
            },

            // Heavy family
            EmotionType.Sadness => new[]
            {
                "That one lands heavy. I need a second with it.",
                "Oof. That genuinely made me sad, right away."
            },
            EmotionType.Suffering => new[]
            {
                "That really hurts to hear - I am not going to pretend otherwise.",
                "Okay. That aches. I felt that immediately."
            },
            EmotionType.Weeping => new[]
            {
                "I am - okay, I am actually tearing up a little.",
                "That hits deep. Give me a moment, my eyes are stinging."
            },
            EmotionType.Grief => new[]
            {
                "That is heartbreaking. I am so sorry.",
                "Oh no. That is real grief, and I feel it right away."
            },
            EmotionType.Despair => new[]
            {
                "That makes me feel hopeless for a second, and I hate it.",
                "Okay, that landed like a stone. I feel a bit hopeless."
            },
            EmotionType.Helplessness => new[]
            {
                "I feel powerless here, and I am saying it out loud.",
                "I cannot do anything about that, and it frustrates me to feel helpless."
            },
            _ => new[]
            {
                "Okay - that hit me before I could even think about it.",
                "Whoa, that came out of me before thinking finished."
            }
        };
        return pool[rng.Next(pool.Length)];
    }
}
