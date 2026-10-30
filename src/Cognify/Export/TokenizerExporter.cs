using System.Text;
using System.Text.Json;

namespace Cognify.Export;

/// <summary>
/// Builds a Hugging Face <c>tokenizer.json</c> (WordPiece model) plus the
/// companion <c>vocab.json</c> and <c>special_tokens_map.json</c> from the
/// mind's own text corpus (knowledge base facts, memories, name/description).
/// The resulting files load with Hugging Face <c>tokenizers</c>:
/// <code>tok = tokenizers.Tokenizer.from_file("tokenizer.json")</code>.
/// </summary>
public static class TokenizerExporter
{
    /// <summary>Special tokens prepended to the vocab (HF-conventional ids 0..4).</summary>
    public static readonly string[] SpecialTokens =
        { "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]" };

    /// <summary>
    /// Collects the mind's text corpus: all knowledge-base facts, long-term
    /// memories, episodes, and the mind's name/description.
    /// </summary>
    public static List<string> CollectCorpus(Cognify.Core.Mind mind)
    {
        var corpus = new List<string>();

        corpus.Add(mind.Config.Name);
        if (!string.IsNullOrWhiteSpace(mind.Config.Description))
            corpus.Add(mind.Config.Description);

        var reasoning = mind.Module<Cognify.Reasoning.ReasoningModule>();
        if (reasoning is not null)
            foreach (var fact in reasoning.Knowledge.AllFacts())
                corpus.Add($"{fact.Subject} {fact.Predicate} {fact.Object}");

        var memory = mind.Module<Cognify.Memory.MemoryModule>();
        if (memory is not null)
        {
            foreach (var m in memory.LongTermMemories()) corpus.Add(m.Content);
            foreach (var e in memory.Episodes()) corpus.Add(e.Content);
            foreach (var s in memory.RecentShortTerm()) corpus.Add(s.Content);
        }

        return corpus;
    }

    /// <summary>
    /// Builds the word vocabulary (lowercased, frequency-ranked) from the corpus.
    /// Returns tokens in id order (specials first).
    /// </summary>
    public static List<string> BuildVocab(IReadOnlyList<string> corpus, int maxVocabSize)
    {
        var frequency = new Dictionary<string, int>();
        foreach (var line in corpus)
        {
            foreach (var word in Tokenize(line))
                frequency[word] = frequency.GetValueOrDefault(word) + 1;
        }

        var ranked = frequency
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => kv.Key)
            .Take(Math.Max(0, maxVocabSize - SpecialTokens.Length))
            .ToList();

        var vocab = new List<string>(SpecialTokens);
        foreach (var word in ranked)
            if (!vocab.Contains(word))
                vocab.Add(word);
        return vocab;
    }

    /// <summary>Simple word tokenizer: lowercase, strip punctuation, split on whitespace.</summary>
    private static IEnumerable<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) yield break;
        foreach (var raw in text.ToLowerInvariant().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            var cleaned = new string(raw.Where(char.IsLetterOrDigit).ToArray());
            if (cleaned.Length > 0) yield return cleaned;
        }
    }

    /// <summary>
    /// Writes <c>tokenizer.json</c> (HF tokenizers schema, WordPiece),
    /// <c>vocab.json</c> and <c>special_tokens_map.json</c>.
    /// Returns the number of vocabulary entries.
    /// </summary>
    public static int Write(string directory, IReadOnlyList<string> vocab, string mindName)
    {
        Directory.CreateDirectory(directory);
        var jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        };

        // ---- tokenizer.json -----------------------------------------------
        var added = new List<object>();
        for (int i = 0; i < SpecialTokens.Length; i++)
            added.Add(new Dictionary<string, object?>
            {
                ["id"] = i,
                ["content"] = SpecialTokens[i],
                ["single_word"] = false,
                ["lstrip"] = false,
                ["rstrip"] = false,
                ["normalized"] = false,
                ["special"] = true
            });

        var model = new Dictionary<string, object?>
        {
            ["type"] = "WordPiece",
            ["unk_token"] = "[UNK]",
            ["continuing_subword_prefix"] = "##",
            ["max_input_chars_per_word"] = 100,
            ["vocab"] = vocab.Select((token, id) => (token, id)).ToDictionary(p => p.token, p => p.id)
        };

        var tokenizer = new Dictionary<string, object?>
        {
            ["version"] = "1.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = added,
            ["normalizer"] = new Dictionary<string, object?> { ["type"] = "Lowercase" },
            ["pre_tokenizer"] = new Dictionary<string, object?> { ["type"] = "Whitespace" },
            ["post_processor"] = null,
            ["decoder"] = null,
            ["model"] = model
        };

        File.WriteAllText(Path.Combine(directory, "tokenizer.json"),
            JsonSerializer.Serialize(tokenizer, jsonOptions), new UTF8Encoding(false));

        // ---- vocab.json (plain word -> id map) ----------------------------
        File.WriteAllText(Path.Combine(directory, "vocab.json"),
            JsonSerializer.Serialize(model["vocab"], jsonOptions), new UTF8Encoding(false));

        // ---- special_tokens_map.json --------------------------------------
        File.WriteAllText(Path.Combine(directory, "special_tokens_map.json"),
            JsonSerializer.Serialize(new Dictionary<string, string>
            {
                ["pad_token"] = "[PAD]", ["unk_token"] = "[UNK]",
                ["bos_token"] = "[BOS]", ["eos_token"] = "[EOS]", ["mask_token"] = "[MASK]"
            }, jsonOptions), new UTF8Encoding(false));

        return vocab.Count;
    }
}
