using Cognify.Core;
using Cognify.Persistence;

namespace Cognify.Tests;

/// <summary>
/// Tests for the gender identity system. The gender system offers exactly
/// two identity presets — she/her and he/him — plus a genderless mode.
/// they/them is reserved for describing OTHER people whose gender the mind
/// does not know; it can never be a mind's own pronoun set.
/// </summary>
public class GenderTests
{
    // ------------------------------------------------------------------
    // Pronoun presets
    // ------------------------------------------------------------------

    [Fact]
    public void PronounPresets_CarryCorrectWords()
    {
        Assert.Equal("she", PronounSet.Female.Subject);
        Assert.Equal("her", PronounSet.Female.Object);
        Assert.Equal("her", PronounSet.Female.Possessive);
        Assert.Equal("hers", PronounSet.Female.PossessivePronoun);
        Assert.Equal("herself", PronounSet.Female.Reflexive);

        Assert.Equal("he", PronounSet.Male.Subject);
        Assert.Equal("him", PronounSet.Male.Object);
        Assert.Equal("his", PronounSet.Male.Possessive);
        Assert.Equal("himself", PronounSet.Male.Reflexive);
    }

    [Fact]
    public void UnknownPerson_IsTheyThem_AndReserved()
    {
        Assert.Equal("they", PronounSet.UnknownPerson.Subject);
        Assert.Equal("them", PronounSet.UnknownPerson.Object);
        Assert.Equal("their", PronounSet.UnknownPerson.Possessive);
        Assert.Equal("themself", PronounSet.UnknownPerson.Reflexive);
        Assert.True(PronounSet.UnknownPerson.IsUnknownPerson);
    }

    [Fact]
    public void Unspecified_HasNoPronounWords_AtAll()
    {
        // A genderless mind has NO pronoun words: it only ever says "I".
        // they/them must not hide in the unspecified set.
        Assert.Equal("", PronounSet.Unspecified.Subject);
        Assert.Equal("", PronounSet.Unspecified.Object);
        Assert.Equal("", PronounSet.Unspecified.Possessive);
        Assert.True(PronounSet.Unspecified.IsUnspecified);
        Assert.False(PronounSet.Unspecified.IsUnknownPerson);
    }

    [Fact]
    public void TryParse_AcceptsOnlySheHerAndHeHim()
    {
        Assert.True(PronounSet.TryParse("she/her", out var female));
        Assert.Same(PronounSet.Female, female);
        Assert.True(PronounSet.TryParse("female", out var f2));
        Assert.Same(PronounSet.Female, f2);
        Assert.True(PronounSet.TryParse("male", out var male));
        Assert.Same(PronounSet.Male, male);
        Assert.True(PronounSet.TryParse("he/him", out var m2));
        Assert.Same(PronounSet.Male, m2);

        // they/them and non-binary labels are NOT settings for the mind's
        // own pronouns — parsing rejects them.
        Assert.False(PronounSet.TryParse("they/them", out _));
        Assert.False(PronounSet.TryParse("non-binary", out _));
        Assert.False(PronounSet.TryParse("nonbinary", out _));
        Assert.False(PronounSet.TryParse("xe/xem", out _));
    }

    [Fact]
    public void ReservedLabels_AreDetected()
    {
        Assert.True(PronounSet.IsReservedLabel("they/them"));
        Assert.True(PronounSet.IsReservedLabel("non-binary person"));
        Assert.True(PronounSet.IsReservedLabel("nonbinary"));
        Assert.False(PronounSet.IsReservedLabel("girl"));
        Assert.False(PronounSet.IsReservedLabel("she/her"));
    }

    // ------------------------------------------------------------------
    // Guard rails: they/them can never become the mind's own pronouns
    // ------------------------------------------------------------------

    [Fact]
    public void WithUnknownPerson_Throws()
    {
        using var mind = MindBuilder.Create("Yume").Build();
        var ex = Assert.Throws<MindConfigurationException>(() => mind.SetPronouns(PronounSet.UnknownPerson));
        Assert.Contains("reserved", ex.Message);
    }

    [Fact]
    public void SetReservedIdentity_Throws()
    {
        using var mind = MindBuilder.Create("Yume").Build();
        Assert.Throws<MindConfigurationException>(() => mind.SetGender("they/them"));
        Assert.Throws<MindConfigurationException>(() => mind.SetGender("non-binary person"));
        // The mind is unchanged after the rejected attempts.
        Assert.True(mind.Config.Gender.IsUnset);
    }

    // ------------------------------------------------------------------
    // Builder + mind accessors
    // ------------------------------------------------------------------

    [Fact]
    public void WithGender_SetsIdentityAndPronouns()
    {
        using var mind = MindBuilder.Create("Yume")
            .WithGender("girl", PronounSet.Female)
            .Build();
        Assert.Equal("girl", mind.GenderIdentity);
        Assert.Same(PronounSet.Female, mind.Pronouns);
        Assert.False(mind.Config.Gender.IsUnset);
    }

    [Fact]
    public void WithGender_PresetLabel_InfersPronouns()
    {
        using var mind = MindBuilder.Create("Sol").WithGender("he/him").Build();
        Assert.Equal(string.Empty, mind.GenderIdentity);
        Assert.Same(PronounSet.Male, mind.Pronouns);
    }

    [Fact]
    public void DefaultMind_HasNoGender()
    {
        using var mind = MindBuilder.Create("Plain").Build();
        Assert.Equal(string.Empty, mind.GenderIdentity);
        Assert.True(mind.Pronouns.IsUnspecified);
        Assert.True(mind.Config.Gender.IsUnset);
        Assert.Equal(string.Empty, mind.GenderClause);
    }

    [Fact]
    public void WithGender_FluentConfigure_Works()
    {
        using var mind = MindBuilder.Create("Sol")
            .WithGender(g => g.Is("boy").With(PronounSet.Male))
            .Build();
        Assert.Equal("boy", mind.GenderIdentity);
        Assert.Same(PronounSet.Male, mind.Pronouns);
    }

    // ------------------------------------------------------------------
    // Self-description
    // ------------------------------------------------------------------

    [Fact]
    public void GenderClause_CombinesIdentityAndPronouns()
    {
        using var mind = MindBuilder.Create("Yume").WithGender("girl", PronounSet.Female).Build();
        Assert.Equal("I am a girl, and I go by she/her pronouns.", mind.GenderClause);
    }

    [Fact]
    public void GenderClause_UsesAnBeforeVowel()
    {
        using var mind = MindBuilder.Create("Sol").WithGender("android", PronounSet.Male).Build();
        Assert.Equal("I am an android, and I go by he/him pronouns.", mind.GenderClause);
    }

    [Fact]
    public void GenderClause_KeepsCallerProvidedArticle()
    {
        using var mind = MindBuilder.Create("Sol").WithGender("a storyteller", PronounSet.Male).Build();
        Assert.StartsWith("I am a storyteller", mind.GenderClause);
    }

    [Fact]
    public void DescribeSelf_IncludesGenderWhenSet()
    {
        using var mind = MindBuilder.Create("Yume")
            .Describe("Yume (夢, 'dream') — a warm, curious companion AI.")
            .WithGender("girl", PronounSet.Female)
            .Build();
        var self = mind.DescribeSelf();
        Assert.StartsWith("Yume (夢, 'dream') — a warm, curious companion AI.", self);
        Assert.Contains("I am a girl, and I go by she/her pronouns.", self);
    }

    [Fact]
    public void DescribeSelf_WithoutGender_Unchanged()
    {
        using var mind = MindBuilder.Create("Plain").WithThinking().Build();
        mind.Process("hello");
        var self = mind.DescribeSelf();
        Assert.StartsWith("I am Plain", self);
        Assert.DoesNotContain("pronouns", self);
        Assert.DoesNotContain("I am a", self);
    }

    [Fact]
    public void DescribeSelf_NeverUsesTheyThemForTheMind()
    {
        // No configuration path may leave the mind describing itself with
        // they/them — those pronouns are for unknown OTHER people only.
        using var mind = MindBuilder.Create("Yume").WithThinking().Build();
        mind.Process("hello there");
        Assert.DoesNotContain("they", mind.DescribeSelf());
        Assert.DoesNotContain("them", mind.DescribeSelf());
    }

    [Fact]
    public void IdentityAnswer_IncludesGender_WhenAsked()
    {
        using var mind = MindBuilder.Create("Yume")
            .Describe("Yume — a warm, curious companion AI.")
            .WithGender("girl", PronounSet.Female)
            .WithReasoning()
            .WithThinking()
            .Build();
        var answer = mind.Ask("Yume, who are you?");
        Assert.Contains("I am a girl", answer.Content);
        Assert.Contains("she/her", answer.Content);
    }

    [Fact]
    public void IdentityAnswer_WithoutGender_NeverMentionsGender()
    {
        using var mind = MindBuilder.Create("Plain").WithReasoning().WithThinking().Build();
        var answer = mind.Ask("who are you?");
        Assert.StartsWith("I am Plain", answer.Content);
        Assert.DoesNotContain("pronouns", answer.Content);
    }

    // ------------------------------------------------------------------
    // Runtime changes
    // ------------------------------------------------------------------

    [Fact]
    public void SetGender_AppliesImmediately()
    {
        using var mind = MindBuilder.Create("Sage").Build();
        Assert.True(mind.Config.Gender.IsUnset);
        mind.SetGender("boy", PronounSet.Male);
        Assert.Equal("boy", mind.GenderIdentity);
        Assert.Contains("I am a boy", mind.DescribeSelf());
        mind.SetPronouns(PronounSet.Female);
        Assert.Same(PronounSet.Female, mind.Pronouns);
        Assert.Contains("she/her", mind.GenderClause);
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    [Fact]
    public void Gender_SurvivesSaveAndRestore()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-gender-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilder.Create("Yume")
                       .Describe("Yume — a warm, curious companion AI.")
                       .WithGender("girl", PronounSet.Female)
                       .WithPersistence(path)
                       .Build())
            {
                Assert.Equal("girl", mind.GenderIdentity);
            } // auto-save happened at build; disposing closes the store

            using var restored = MindBuilder.Restore(path);
            Assert.Equal("girl", restored.GenderIdentity);
            // JSON restore yields a value-equal copy of the pronoun set.
            Assert.Equal(PronounSet.Female.Label, restored.Pronouns.Label);
            Assert.Equal(PronounSet.Female.Reflexive, restored.Pronouns.Reflexive);
            Assert.Contains("I am a girl", restored.DescribeSelf());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Gender_SetAfterBuild_PersistsToSnapshot()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-gender-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilder.Create("Sage").WithPersistence(path).Build())
            {
                mind.SetGender("boy", PronounSet.Male);
                mind.Store!.Save(MindSnapshot.Capture(mind));
            }

            using var restored = MindBuilder.Restore(path);
            Assert.Equal("boy", restored.GenderIdentity);
            Assert.Equal(PronounSet.Male.Label, restored.Pronouns.Label);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void OldSnapshot_WithoutGender_RestoresUnset()
    {
        // A snapshot taken before the gender system existed must restore a
        // genderless mind without errors.
        using var mind = MindBuilder.Create("Legacy").WithThinking().Build();
        var snap = MindSnapshot.Capture(mind);
        snap.ConfigJson = snap.ConfigJson.Replace("\"Gender\":", "\"Gone\":"); // simulate pre-gender JSON

        using var target = MindBuilder.Create("Fresh").Build();
        snap.ApplyTo(target);
        Assert.True(target.Config.Gender.IsUnset);
        Assert.Equal(string.Empty, target.GenderClause);
    }

    [Fact]
    public void LegacySnapshot_WithNonBinaryGender_IsSanitized()
    {
        // A snapshot from a version that still offered they/them for the
        // mind itself restores clean: no they/them pronouns, no non-binary
        // identity word — the mind ends up genderless.
        using var mind = MindBuilder.Create("Legacy").WithThinking().Build();
        var snap = MindSnapshot.Capture(mind);
        snap.ConfigJson = snap.ConfigJson
            // the default (unspecified) pronoun set -> an old they/them set
            .Replace(
                "\"Label\":\"unspecified\",\"Subject\":\"\",\"Object\":\"\",\"Possessive\":\"\",\"PossessivePronoun\":\"\",\"Reflexive\":\"\"",
                "\"Label\":\"they/them\",\"Subject\":\"they\",\"Object\":\"them\",\"Possessive\":\"their\",\"PossessivePronoun\":\"theirs\",\"Reflexive\":\"themself\"")
            // an old non-binary identity word
            .Replace("\"Identity\":\"\"", "\"Identity\":\"non-binary person\"");

        using var target = MindBuilder.Create("Fresh").Build();
        snap.ApplyTo(target);
        Assert.NotEqual("they/them", target.Pronouns.Label);
        Assert.DoesNotContain("non-binary", target.GenderIdentity, StringComparison.OrdinalIgnoreCase);
    }
}
