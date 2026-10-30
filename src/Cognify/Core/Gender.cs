namespace Cognify.Core;

/// <summary>
/// A pronoun set: the words other people use to talk ABOUT the mind
/// ("she is curious", "tell her", "that is hers"). Exactly two identity
/// presets exist — <see cref="Female"/> (she/her) and <see cref="Male"/>
/// (he/him) — plus <see cref="Unspecified"/> for a mind with no gender
/// configured at all.
///
/// "they/them" is deliberately NOT an option for a mind's own pronouns:
/// that set lives only in <see cref="UnknownPerson"/>, which is reserved
/// for describing OTHER people whose gender the mind does not know yet.
/// All fields are lowercase by convention.
/// </summary>
public sealed class PronounSet
{
    /// <summary>Display label of the set, e.g. "she/her" (used in self-description).</summary>
    public string Label { get; }

    /// <summary>Subject pronoun ("she is curious"). Empty for <see cref="Unspecified"/>.</summary>
    public string Subject { get; }

    /// <summary>Object pronoun ("talk to her").</summary>
    public string Object { get; }

    /// <summary>Possessive determiner ("her name").</summary>
    public string Possessive { get; }

    /// <summary>Independent possessive ("that book is hers").</summary>
    public string PossessivePronoun { get; }

    /// <summary>Reflexive pronoun ("she taught herself").</summary>
    public string Reflexive { get; }

    /// <summary>
    /// Creates a pronoun set. Meant for the built-in presets and for
    /// persistence round-trips; configuration code should pick one of the
    /// presets — a mind's own pronouns can only ever be she/her, he/him or
    /// none at all.
    /// </summary>
    public PronounSet(string label, string subject, string @object,
        string possessive, string possessivePronoun, string reflexive)
    {
        if (string.IsNullOrWhiteSpace(label)) throw new MindConfigurationException("A pronoun set needs a label.");
        Label = label.Trim().ToLowerInvariant();
        Subject = subject.Trim().ToLowerInvariant();
        Object = Empty(@object, Subject);
        Possessive = Empty(possessive, Subject);
        PossessivePronoun = Empty(possessivePronoun, Possessive);
        Reflexive = Empty(reflexive, $"{Subject}self");
    }

    private PronounSet(string label)
    {
        // Marker set with no pronoun words at all (Unspecified): a mind
        // without a gender simply speaks first-person "I" and nothing else.
        Label = label;
        Subject = Object = Possessive = PossessivePronoun = Reflexive = "";
    }

    private static string Empty(string? value, string fallback) =>
        string.IsNullOrWhiteSpace(value) ? fallback : value.Trim().ToLowerInvariant();

    /// <summary>She / her / her / hers / herself.</summary>
    public static PronounSet Female { get; } =
        new("she/her", "she", "her", "her", "hers", "herself");

    /// <summary>He / him / his / his / himself.</summary>
    public static PronounSet Male { get; } =
        new("he/him", "he", "him", "his", "his", "himself");

    /// <summary>
    /// They / them / their / theirs / themself — RESERVED for describing a
    /// third person whose gender the mind does not know ("someone is at the
    /// door — they are waiting"). This set must never be assigned as a
    /// mind's own pronouns; <see cref="GenderOptions"/> rejects it.
    /// </summary>
    public static PronounSet UnknownPerson { get; } =
        new("they/them", "they", "them", "their", "theirs", "themself");

    /// <summary>
    /// Pronouns left unspecified — the mind has no gender words at all and
    /// never volunteers pronouns in conversation; first-person "I" only.
    /// </summary>
    public static PronounSet Unspecified { get; } = new("unspecified");

    /// <summary>True when the mind was not given an explicit pronoun set.</summary>
    public bool IsUnspecified => Label == Unspecified.Label;

    /// <summary>True for the set reserved for unknown third parties.</summary>
    public bool IsUnknownPerson => ReferenceEquals(this, UnknownPerson) || Label == UnknownPerson.Label;

    /// <summary>"she/her" style display label.</summary>
    public override string ToString() => Label;

    /// <summary>
    /// Parses "she/her" style labels to a preset when one matches. Only the
    /// two identity presets (she/her, he/him) parse; "they/them" and
    /// non-binary labels are intentionally rejected — they are not valid
    /// settings for a mind's own pronouns.
    /// </summary>
    public static bool TryParse(string label, out PronounSet set)
    {
        switch ((label ?? "").Trim().ToLowerInvariant())
        {
            case "she/her" or "her/hers" or "female" or "f":
                set = Female; return true;
            case "he/him" or "him/his" or "male" or "m":
                set = Male; return true;
            default:
                set = Unspecified; return false;
        }
    }

    /// <summary>
    /// Labels that name a pronoun choice a mind may not adopt for itself
    /// (they/them and non-binary variants). <see cref="GenderOptions"/>
    /// rejects these with an explanatory error instead of silently
    /// misinterpreting them.
    /// </summary>
    public static bool IsReservedLabel(string label)
    {
        var key = (label ?? "").Trim().ToLowerInvariant();
        if (key.Length == 0) return false;
        // they/them and non-binary are not self-pronoun/identity options;
        // catch both exact labels and phrases that embed them.
        return key == "nb" || key == "enby"
            || key.Contains("non-binary") || key.Contains("nonbinary")
            || key.Contains("they/them") || key.Contains("them/their")
            || key.Contains("they them");
    }
}

/// <summary>
/// Options for the mind's gender identity: how it describes itself and which
/// pronouns fit it. A mind with no gender configured behaves exactly as
/// before (first-person only, gender never mentioned).
/// </summary>
public sealed class GenderOptions
{
    /// <summary>
    /// Identity word used in self-description, e.g. "girl", "boy", "woman".
    /// Leave empty to keep gender out of the mind's self-description. An
    /// article ("a"/"an") is added automatically unless the phrase already
    /// starts with one.
    /// </summary>
    public string Identity { get; set; } = string.Empty;

    /// <summary>The pronoun set others use for the mind (she/her, he/him, or none).</summary>
    public PronounSet Pronouns { get; set; } = PronounSet.Unspecified;

    /// <summary>True when no gender identity was configured.</summary>
    public bool IsUnset => string.IsNullOrWhiteSpace(Identity) && Pronouns.IsUnspecified;

    /// <summary>Fluent setter: the identity word ("girl", "woman"...).</summary>
    public GenderOptions Is(string identity)
    {
        var word = (identity ?? string.Empty).Trim();
        if (PronounSet.IsReservedLabel(word))
            throw new MindConfigurationException(
                $"'{word}' is not a pronoun set a mind can adopt for itself. " +
                "they/them is reserved for describing other people whose gender " +
                "the mind does not know. Use she/her or he/him, or leave the mind genderless.");
        Identity = word;
        return this;
    }

    /// <summary>
    /// Sets identity and pronouns in one call. When the identity word itself
    /// names a pronoun set ("she/her", "female"...) it is treated as a
    /// pronoun choice only — the label is not stored as the mind's identity
    /// word.
    /// </summary>
    public GenderOptions Set(string identity, PronounSet? pronouns = null)
    {
        if (pronouns is null && PronounSet.TryParse(identity, out var parsed))
        {
            With(parsed); // "she/her" / "male" / ... → pronouns only, no identity word
            return this;
        }
        Is(identity);
        if (pronouns is not null) With(pronouns);
        return this;
    }

    /// <summary>
    /// Fluent setter: the pronoun set. Only <see cref="PronounSet.Female"/>,
    /// <see cref="PronounSet.Male"/> and <see cref="PronounSet.Unspecified"/>
    /// are accepted — <see cref="PronounSet.UnknownPerson"/> (they/them)
    /// throws, because they/them is reserved for describing unknown people.
    /// </summary>
    public GenderOptions With(PronounSet pronouns)
    {
        if (pronouns is null) pronouns = PronounSet.Unspecified;
        if (pronouns.IsUnknownPerson)
            throw new MindConfigurationException(
                "they/them is reserved for referring to people whose gender the mind " +
                "does not know — it cannot be a mind's own pronoun set. " +
                "Use PronounSet.Female, PronounSet.Male or leave the mind genderless (Unspecified).");
        if (!pronouns.IsUnspecified &&
            !ReferenceEquals(pronouns, PronounSet.Female) &&
            !ReferenceEquals(pronouns, PronounSet.Male))
            throw new MindConfigurationException(
                "Only the she/her and he/him presets can be set as a mind's own pronouns.");
        Pronouns = pronouns;
        return this;
    }

    /// <summary>Convenience: female preset + identity in one call.</summary>
    public GenderOptions Female(string identity = "girl") => Is(identity).With(PronounSet.Female);

    /// <summary>Convenience: male preset + identity in one call.</summary>
    public GenderOptions Male(string identity = "boy") => Is(identity).With(PronounSet.Male);
}
