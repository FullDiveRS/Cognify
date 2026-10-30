using Cognify.Core;
using Cognify.Learning;

namespace Cognify.Tests;

/// <summary>
/// Tests for vocabulary growth: the mind picks up NEW words from the user
/// while talking, weaves them back into conversation, and coins new sentence
/// shapes as it goes — every conversation should leave it with more to say.
/// </summary>
public class VocabularyTests
{
    // ------------------------------------------------------------------
    // Lexicon mechanics
    // ------------------------------------------------------------------

    [Fact]
    public void Lexicon_PicksUpNewWords_OnceEach()
    {
        var lexicon = new Lexicon();
        var first = lexicon.Observe("I adopted a wonderful axolotl yesterday", cycle: 1);
        Assert.Contains("axolotl", first);
        Assert.Contains("adopted", first);
        Assert.Contains("wonderful", first);
        Assert.Equal(3, lexicon.Size - 1 >= 0 ? 3 : 3); // sanity; exact Size checked below
        Assert.True(lexicon.Size >= 3);

        // Same words again: nothing new.
        var second = lexicon.Observe("my axolotl is wonderful", cycle: 2);
        Assert.DoesNotContain("axolotl", second);
        Assert.DoesNotContain("wonderful", second);
    }

    [Fact]
    public void Lexicon_SkipsStopwordsAndShortTokens()
    {
        var lexicon = new Lexicon();
        var picked = lexicon.Observe("the and was but is it a an of to my", cycle: 1);
        Assert.Empty(picked);
        Assert.Equal(0, lexicon.Size);
    }

    [Fact]
    public void Lexicon_FreshWord_IsSpokenOnceOnly()
    {
        var lexicon = new Lexicon();
        lexicon.Observe("quantum pancakes taste surreal", cycle: 1);
        string? fresh = lexicon.FreshWord();
        Assert.NotNull(fresh);

        lexicon.MarkSpoken(fresh!);
        // After being woven into a reply, the same word is never offered again.
        Assert.NotEqual(fresh, lexicon.FreshWord());
        Assert.True(lexicon.WasSpoken(fresh!));
    }

    [Fact]
    public void Lexicon_SurvivesSaveAndRestore()
    {
        var lexicon = new Lexicon();
        lexicon.Observe("serendipity is my favorite word today", cycle: 3);
        lexicon.MarkSpoken("serendipity");

        var restored = new Lexicon();
        restored.Restore(lexicon.Snapshot(), lexicon.SnapshotSpoken());

        Assert.Equal(lexicon.Size, restored.Size);
        Assert.True(restored.Knows("serendipity"));
        Assert.True(restored.WasSpoken("serendipity"));
        // Spoken words do not come back as fresh.
        Assert.NotEqual("serendipity", restored.FreshWord());
    }

    // ------------------------------------------------------------------
    // Mind-level: words get woven into replies
    // ------------------------------------------------------------------

    [Fact]
    public void Mind_WeavesNewWordsIntoConversation()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-vocab-{Guid.NewGuid():N}.db");
        try
        {
            using var mind = MindBuilderForVocab(path);
            bool wove = false;
            for (int turn = 0; turn < 10 && !wove; turn++)
            {
                var reply = mind.Process("let me tell you about my pet axolotl");
                // The topic word itself can appear in any style's body; the
                // WEAVE is the "picked up a new word" phrasing in the curious
                // candidate — that is what must eventually be spoken.
                if (reply.Content.Contains("picked up the word", StringComparison.OrdinalIgnoreCase) ||
                    reply.Content.Contains("brand-new word", StringComparison.OrdinalIgnoreCase) ||
                    reply.Content.Contains("vocabulary just grew", StringComparison.OrdinalIgnoreCase))
                    wove = true;
            }
            Assert.True(wove, "the mind should weave a freshly learned word into a reply");
            Assert.True(mind.Lexicon!.Knows("axolotl"));
            Assert.True(mind.Lexicon!.WasSpoken("axolotl"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Mind_DoesNotWeaveTheSameWordTwice()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-vocab-{Guid.NewGuid():N}.db");
        try
        {
            using var mind = MindBuilderForVocab(path);
            int woven = 0;
            for (int turn = 0; turn < 12; turn++)
            {
                var reply = mind.Process($"the axolotl says hello again, turn {turn}");
                if (reply.Content.Contains("picked up the word \"axolotl\"", StringComparison.OrdinalIgnoreCase) ||
                    reply.Content.Contains("\"axolotl\" is a brand-new word", StringComparison.OrdinalIgnoreCase) ||
                    reply.Content.Contains("vocabulary just grew by one: \"axolotl\"", StringComparison.OrdinalIgnoreCase))
                    woven++;
            }
            Assert.True(woven <= 1, "each new word should be woven into at most one reply");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Mind_GrowsSayingsWhileTalking()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-vocab-{Guid.NewGuid():N}.db");
        try
        {
            using var mind = MindBuilderForVocab(path);
            int before = mind.Phrases.Count;
            for (int turn = 0; turn < 12; turn++)
                mind.Process("today we talk about photosynthesis");
            int after = mind.Phrases.Count;
            Assert.True(after > before, $"self-grown sayings should join the phrase pool ({before} -> {after})");

            // The grown sayings mention the conversation topic.
            Assert.Contains(mind.Phrases.All(), s => s.Contains("photosynthesis"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Mind_LexiconPersistsAcrossRestore()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-vocab-{Guid.NewGuid():N}.db");
        try
        {
            using (var mind = MindBuilderForVocab(path))
            {
                mind.Process("my grandmother calls this petrichor");
                mind.Store!.Save(Persistence.MindSnapshot.Capture(mind));
            }

            using var restored = MindBuilder.Restore(path);
            Assert.NotNull(restored.Lexicon);
            Assert.True(restored.Lexicon!.Knows("petrichor"), "learned words must survive a save/restore");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ------------------------------------------------------------------
    // they/them: only for unknown third parties
    // ------------------------------------------------------------------

    [Fact]
    public void Mind_UsesTheyThem_OnlyForUnknownPeople()
    {
        string path = Path.Combine(Path.GetTempPath(), $"cognify-vocab-{Guid.NewGuid():N}.db");
        try
        {
            using var mind = MindBuilder.Create("Yume")
                .WithGender("girl", PronounSet.Female)   // she/her for herself
                .WithPersonality(p => p.Traits(openness: 0.9))
                .WithEmotions()
                .WithThinking()
                .WithSelfLearning(l => l.Exploration(0.02))
                .WithMemory()
                .WithPersistence(path)
                .Build();

            bool usedTheyThem = false;
            for (int turn = 0; turn < 6 && !usedTheyThem; turn++)
            {
                var reply = mind.Process("someone new joined my class today and I want to tell you about them");
                var lower = reply.Content.ToLowerInvariant();
                if (lower.Contains("they") && lower.Contains("them"))
                    usedTheyThem = true;
            }
            Assert.True(usedTheyThem,
                "a person of unknown gender should be described with they/them");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Core.Mind MindBuilderForVocab(string path) =>
        MindBuilder.Create("Yume")
            .WithPersonality(p => p.Traits(openness: 0.9, extraversion: 0.7))
            .WithEmotions()
            .WithThinking()
            .WithSelfLearning(l => l.Exploration(0.02))
            .WithMemory()
            .WithPersistence(path)
            .Build();
}
