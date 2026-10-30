using Cognify.Emotions;

namespace Cognify.Learning;

/// <summary>
/// The mind's growing word vocabulary. Every cognitive cycle the learner
/// listens to what the user says and picks up words it has never heard
/// before; each new word is remembered with the cycle it appeared in.
///
/// This is the "the AI should add words as it talks" mechanism: the response
/// composer weaves freshly picked words back into conversation ("I just
/// picked up the word ... — what does it mean to you?"), so every new word
/// the user uses becomes one more thing the mind can say — steadily widening
/// its replies beyond the built-in templates and directly fighting the
/// repetition problem.
/// </summary>
public sealed class Lexicon
{
    /// <summary>One known word and where it came from.</summary>
    public sealed class Lexeme
    {
        /// <summary>The word (lowercase).</summary>
        public string Word { get; init; } = "";

        /// <summary>How many times it has been observed.</summary>
        public int Count { get; internal set; }

        /// <summary>Cycle the word was first heard in.</summary>
        public int FirstCycle { get; internal set; }

        /// <summary>Cycle the word was last heard in.</summary>
        public int LastCycle { get; internal set; }
    }

    private readonly object _gate = new();
    private readonly Dictionary<string, Lexeme> _words = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _recent = new();      // newest last
    private readonly HashSet<string> _spoken = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>How many distinct words the mind has picked up.</summary>
    public int Size { get { lock (_gate) return _words.Count; } }

    /// <summary>
    /// Words picked up during the LAST observed cycle (what the demo shows
    /// as "she just picked up ..."). Empty when nothing was new.
    /// </summary>
    public IReadOnlyList<string> LastPicks { get { lock (_gate) return _lastPicks.ToArray(); } }
    private List<string> _lastPicks = new();

    /// <summary>All known words, oldest first.</summary>
    public IReadOnlyList<Lexeme> All()
    {
        lock (_gate) return _words.Values.OrderBy(w => w.FirstCycle).ToList();
    }

    /// <summary>
    /// The freshest not-yet-spoken word for the composer to weave into a
    /// reply, or null. "Fresh" = heard recently and never woven into a reply
    /// before, so each word gets used once and the same question never
    /// repeats. Peeking does not consume the word —
    /// <see cref="MarkSpoken"/> does that once the reply wins deliberation.
    /// </summary>
    public string? FreshWord()
    {
        lock (_gate)
        {
            for (int i = _recent.Count - 1; i >= 0; i--)
                if (!_spoken.Contains(_recent[i]))
                    return _recent[i];
            return null;
        }
    }

    /// <summary>
    /// Like <see cref="FreshWord"/> but also reports the cycle the word was
    /// first heard in, so the composer can tell "brand-new" from "picked up
    /// earlier and still unspoken" — an old word called brand-new sounds
    /// like a glitch.
    /// </summary>
    public (string Word, int FirstCycle)? FreshWordWithAge()
    {
        lock (_gate)
        {
            for (int i = _recent.Count - 1; i >= 0; i--)
                if (!_spoken.Contains(_recent[i]) &&
                    _words.TryGetValue(_recent[i], out var lexeme))
                    return (lexeme.Word, lexeme.FirstCycle);
            return null;
        }
    }

    /// <summary>Marks a word as having been spoken in a reply (consumes it).</summary>
    public void MarkSpoken(string word)
    {
        lock (_gate) _spoken.Add(word.Trim().ToLowerInvariant());
    }

    /// <summary>The most recently picked words, newest first.</summary>
    public IReadOnlyList<string> RecentNewWords(int max = 8)
    {
        lock (_gate)
        {
            var result = new List<string>();
            for (int i = _recent.Count - 1; i >= 0 && result.Count < max; i--)
                result.Add(_recent[i]);
            return result;
        }
    }

    /// <summary>True when the mind already used this word in one of its replies.</summary>
    public bool WasSpoken(string word)
    {
        lock (_gate) return _spoken.Contains(word.Trim().ToLowerInvariant());
    }

    // Time/degree words never make good vocabulary picks — weaving "I just
    // picked up the word 'currently'" into a reply would sound like a
    // glitch. Mirrors the topic-filler list the response composer skips.
    // Extended after real conversations showed the mind "picking up" speech
    // glue: okay, getting, doing, good, job, lets, try, slightly, vocab, fore.
    private static readonly HashSet<string> FillerWords = new()
    {
        "currently", "now", "then", "today", "recently", "still", "already",
        "again", "always", "never", "maybe", "perhaps", "really", "actually",
        "just", "very", "many", "much", "also", "too", "quite", "pretty",
        // speech glue and semi-words heard in real chat
        "okay", "ok", "good", "fine", "well", "getting", "doing", "lets",
        "let", "try", "trying", "tried", "job", "vocab", "fore", "slightly",
        "thing", "things", "stuff", "gonna", "wanna", "gotta", "kinda",
        "sorta", "yeah", "yep", "nah", "hmm", "huh", "wow", "um", "uh",
        "like", "lot", "lots", "bit", "way", "sure", "maybe", "please"
    };

    /// <summary>
    /// Listens to one utterance, registering every content word (3+ letters,
    /// no stopwords) it has never heard before. Returns the words picked up
    /// THIS cycle — new to the mind — in the order they appeared.
    /// </summary>
    public IReadOnlyList<string> Observe(string text, int cycle)
    {
        var picked = new List<string>();
        if (string.IsNullOrWhiteSpace(text)) return picked;

        lock (_gate)
        {
            foreach (var token in MoodEngine.Tokenize(text))
            {
                if (token.Length < 3 || MoodEngine.IsStopWord(token) ||
                    FillerWords.Contains(token) ||
                    !token.All(char.IsLetter))
                    continue;

                if (_words.TryGetValue(token, out var lexeme))
                {
                    lexeme.Count++;
                    lexeme.LastCycle = cycle;
                    continue;
                }

                var fresh = new Lexeme { Word = token, Count = 1, FirstCycle = cycle, LastCycle = cycle };
                _words[token] = fresh;
                _recent.Add(token);
                while (_recent.Count > 16) _recent.RemoveAt(0);
                picked.Add(token);
            }
            _lastPicks = picked;
        }
        return picked;
    }

    /// <summary>True when the mind knows this word.</summary>
    public bool Knows(string word)
    {
        lock (_gate) return _words.ContainsKey(word.Trim().ToLowerInvariant());
    }

    /// <summary>Serializes the lexicon for persistence (word, count, first, last).</summary>
    public List<string[]> Snapshot()
    {
        lock (_gate)
            return _words.Values
                .OrderBy(w => w.FirstCycle)
                .Select(w => new[]
                {
                    w.Word, w.Count.ToString(),
                    w.FirstCycle.ToString(), w.LastCycle.ToString()
                })
                .ToList();
    }

    /// <summary>Serializes the set of words already woven into replies.</summary>
    public List<string> SnapshotSpoken()
    {
        lock (_gate) return _spoken.ToList();
    }

    /// <summary>Restores a lexicon from a snapshot.</summary>
    public void Restore(IEnumerable<string[]> entries, IEnumerable<string>? spoken = null)
    {
        lock (_gate)
        {
            _words.Clear();
            _recent.Clear();
            foreach (var e in entries)
            {
                if (e.Length < 4 || e[0].Length == 0) continue;
                if (!int.TryParse(e[1], out var count)) continue;
                if (!int.TryParse(e[2], out var first)) continue;
                if (!int.TryParse(e[3], out var last)) continue;
                _words[e[0]] = new Lexeme { Word = e[0], Count = count, FirstCycle = first, LastCycle = last };
                _recent.Add(e[0]);
            }
            while (_recent.Count > 16) _recent.RemoveAt(0);
            _spoken.Clear();
            if (spoken is not null)
                foreach (var w in spoken)
                    if (!string.IsNullOrWhiteSpace(w)) _spoken.Add(w.Trim().ToLowerInvariant());
            _lastPicks = new List<string>();
        }
    }
}
