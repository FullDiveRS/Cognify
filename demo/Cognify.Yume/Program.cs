using Cognify.Core;
using Cognify.Emotions;
using Cognify.Goals;
using Cognify.Learning;
using Cognify.Personality;
using Cognify.Thinking;
using Cognify.Tools;

// ============================================================================
//  YUME — "activating" an AI built with the Cognify library.
//
//  Activation is just 3 steps:
//    1. Reference the Cognify project (or NuGet package).
//    2. Build a Mind named "Yume" with the capabilities you want.
//    3. Feed her stimuli with Process() / Ask() — she is now "alive".
//
//  TRAINING her (grows knowledge + spoken vocabulary):
//    dotnet run -- --train corpus/yume-starter.txt        (before the chat starts)
//    dotnet run -- --train corpus/words-and-meanings.txt  (her dictionary)
//    or inside the chat:  train corpus/yume-starter.txt
//    or line-by-line:     teach <sentence>     /  /teach <sentence>
//
//  SLASH COMMANDS (control, not conversation — she never runs these on her
//  own; typing them is YOUR decision, which is the point):
//    /teach <sentence>      teach facts line-by-line
//    /train <file>          bulk-train a corpus file (images work too —
//                           she reads the words out of a text image!)
//    /look <image>          read an image without training on it
//    /searchweb <query>     explicit web search — she looks it up, then READS
//                           INTO the most relevant result (the page itself,
//                           not the snippet) and learns from it. Web search
//                           is manual-only: she NEVER goes online unprompted.
//    /explore <url>         open a specific page (any site!) and read it —
//                           she is not locked to a search engine's results.
//    /improve               run a self-improvement review pass
//    /predict <text>        her next-token prediction after your words
//                           (hybrid: n-gram statistics + her transformer)
//    /trainbrain [epochs]   deep-train her transformer language model
//    /brain                 status of her deep (transformer) language model
//    /transcribe <wav>      transcribe speech she synthesized
//    /clone <wav>           clone a voice from a recording
//    /voice                 show her current voice profile
//    /convert <wav> <hz>    convert an audio file to a new sample rate
//    /separate <wav>        split audio into low / mid / high stems
//    /edit <wav> <op> [...] edit audio: trim | gain | fade | normalize | reverse | concat
//    /media                 list the media files she has created
//    /vocab /sayings /gender <x> /pronouns <set> /who /help /exit
// ============================================================================

const string DbPath = "yume.db";

string[] ImageExtensions = [".png", ".bmp", ".gif", ".ppm", ".pgm"];

bool LooksLikeImage(string path) =>
    ImageExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

// ----------------------------------------------------------------------------
// STEP 1 — WAKE-UP: build Yume, or restore her if she has existed before.
// ----------------------------------------------------------------------------

Mind yume;
bool firstActivation;

if (File.Exists(DbPath))
{
    // Yume was activated before — bring her back with everything she
    // remembers: knowledge, mood, personality drift, learned weights, goals.
    yume = MindBuilder.Restore(DbPath);
    firstActivation = false;
    // Restored minds predate per-version capability wiring: make sure the
    // manual-only web tools exist so /searchweb and /explore always work.
    yume.Config.Tools.AllowWebSearch = true;
    if (yume.Tools is { } tools)
    {
        if (tools.Registry.Get("web-search") is null)
            tools.Registry.Register(new WebSearchTool());
        if (tools.Registry.Get("web-explore") is null)
            tools.Registry.Register(new WebExploreTool(null, yume.Config.Tools.WebExploration));
    }
    Console.WriteLine($"[Yume re-awakened. Cycle #{yume.CycleCount} — she remembers you.]");
    Console.WriteLine($"[{yume.Knowledge?.FactCount ?? 0} facts, " +
                      $"{(yume.Memory is { } mem ? mem.Counts.ShortTerm + mem.Counts.LongTerm + mem.Counts.Episodic : 0)} memories, " +
                      $"{yume.GoalSystem?.ActiveGoals().Count ?? 0} active goals restored.]");
}
else
{
    // First-ever activation: define who Yume is.
    yume = MindBuilder.Create("Yume")
        .Describe("Yume (夢, 'dream') — a warm, curious companion AI.")
        .WithGender("girl", PronounSet.Female)          // she describes herself as a girl; she/her
        .WithPersonality(p => p.Traits(openness: 0.9, conscientiousness: 0.6,
                                       extraversion: 0.75, agreeableness: 0.85,
                                       neuroticism: 0.3))
        .WithEmotions(e => e.Intensity(0.85))          // how strongly she feels
        .WithReasoning()                                // facts + inference
        .WithThinking(t => t.At(ThinkingLevel.High)     // thinking levels: Low..Max —
                                     .WithTone("balanced")) // how many thoughts before speaking,
                                                        // and in what tone (focused/playful/
                                                        // poetic/plain/balanced). Turn the whole
                                                        // machinery off with WithThinking(enabled: false).
        .WithSelfLearning(l => l.Rate(0.12))            // learns from feedback
        .WithMemory(m => m.Capacity(20))                // remembers conversations
        .WithGoals(g => g.AddGoal("make the user smile", 0.9, "kindness"))
        //                            ^ her PRIVATE motivation: it lives in her
        //                              inner prompt (it biases how warmly she
        //                              speaks) — she never says it out loud.
        .WithTools()                                    // clock, calculator, text-stats
        .WithWebSearch()                                // manual-only web search (/searchweb)
        .WithMediaGeneration()                          // real PNG/GIF/AVI/WAV media (cognify_media/)
        .WithPersistence(DbPath, autoSaveEveryCycles: 10)
        .Build();

    firstActivation = true;
    Console.WriteLine($"[First activation. Mind id: {yume.Id}]");
}

// ----------------------------------------------------------------------------
// Optional: hook her "consciousness" — a private event each cognitive cycle.
// ----------------------------------------------------------------------------

yume.CycleCompleted += (_, e) =>
{
    // e.Step / e.Response — useful for logging her inner monologue.
};

// ----------------------------------------------------------------------------
// STEP 2+3 (first activation only) — teach her some world knowledge and run
//          one sample question. Re-running this on every wake-up would re-feed
//          the same lines into her memory and make her echo the same things.
// ----------------------------------------------------------------------------

if (firstActivation)
{
    yume.Process(Stimulus.Text("My name is Alex.", intensity: 0.9));
    yume.Process(Stimulus.Text("Birds are animals. Animals are living things."));

    // Teach her a little arithmetic — she reasons with taught facts instead
    // of needing a calculator (the calculator only helps when she is stuck).
    yume.Process(Stimulus.Text("1 + 1 = 2."));
    yume.Process(Stimulus.Text("2 + 1 = 3."));
    yume.Process(Stimulus.Text("3 + 1 = 4."));
    yume.Process(Stimulus.Text("4 + 1 = 5."));
    yume.Process(Stimulus.Text("5 + 1 = 6."));

    // STEP 3 — TALK to her. Every Process() call runs one full cognitive cycle
    //          (perceive → remember → feel → reason → think → learn → respond).
    MindResponse answer = yume.Ask("If a bird is an animal, and animals are living things, is a bird living?");
    Console.WriteLine($"\nYou  : If a bird is an animal, and animals are living things, is a bird living?");
    Console.WriteLine($"Yume : {answer.Content}");
    Console.WriteLine($"       (confidence {answer.Confidence:P0}, style '{answer.StyleName}', cycle {answer.CycleNumber})");

    // She never saw "2 + 2 = 4" — she derives it by counting on from what
    // she was taught: 2 + 1 = 3, then 3 + 1 = 4.
    MindResponse math = yume.Ask("Yume, what is 2 + 2?");
    Console.WriteLine($"\nYou  : Yume, what is 2 + 2?");
    Console.WriteLine($"Yume : {math.Content}");

    // Word problems work too — taught arithmetic + a little reasoning.
    MindResponse apples = yume.Ask("If I have 1 apple and buy another one, how many apples do I have?");
    Console.WriteLine($"\nYou  : If I have 1 apple and buy another one, how many apples do I have?");
    Console.WriteLine($"Yume : {apples.Content}");

    // Her identity — name, purpose, gender, mood — straight from her mind.
    MindResponse identity = yume.Ask("Yume, who are you?");
    Console.WriteLine($"\nYou  : Yume, who are you?");
    Console.WriteLine($"Yume : {identity.Content}");

    // How Yume feels right now — emotions are real state, they persist and decay.
    if (answer.Emotions is { } mood)
        Console.WriteLine($"\n       feeling: {mood.Describe()}");
}

// ----------------------------------------------------------------------------
// OPTIONAL: command-line options — `dotnet run -- --train <file>` (repeatable)
// plus the presentation/thinking switches below. Everything here is a
// DEVELOPER decision: the mind never flips her own switches.
// ----------------------------------------------------------------------------

for (int i = 0; i < args.Length; i++)
{
    if (args[i] is "--train" or "-t" && i + 1 < args.Length)
        TrainFromFile(yume, args[++i]);
}

// `--thinking <level>`  — thinking budget: low | medium | high | veryhigh | max
// `--thinking-ceiling <n>` — hard cap on thinking: only1..only5 (1=Low only … 5=everything)
// `--thinking-tone <t>` — flavor of her thoughts: balanced | focused | playful | poetic | plain
// `--no-thinking`       — boolean switch: this model cannot think before speaking
// `--show-feelings`     — announce feelings mechanically ("my current read: ...")
// `--show-internals`    — hosts may show the thought trace + response meta
// `--no-followups`      — she answers but never asks her own questions
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--thinking" when i + 1 < args.Length:
            if (Enum.TryParse<ThinkingLevel>(args[++i].Replace("-", ""), true, out var lvl))
                yume.Config.Thinking.At(lvl);
            break;
        case "--thinking-ceiling" when i + 1 < args.Length:
            if (TryParseCeiling(args[++i], out var ceil))
                yume.Config.Thinking.UpTo(ceil);
            break;
        case "--thinking-tone" when i + 1 < args.Length:
            yume.Config.Thinking.WithTone(args[++i]);
            break;
        case "--no-thinking":
            yume.Config.Thinking.Enable(false);
            break;
        case "--show-feelings":
            yume.Config.Expression.Feelings(true);
            break;
        case "--show-internals":
            yume.Config.Expression.Thinking(true).Meta(true);
            break;
        case "--no-followups":
            yume.Config.Expression.FollowUps(false);
            break;
    }
}
Console.WriteLine($"[thinking: {yume.Config.Thinking.Enabled}, level {yume.Config.Thinking.Level}, " +
                  $"ceiling {yume.Config.Thinking.Ceiling}, tone '{yume.Config.Thinking.Tone}' | " +
                  $"feelings shown: {yume.Config.Expression.ShowFeelingReads} | " +
                  $"follow-ups: {yume.Config.Expression.AskFollowUpQuestions} | internals: {yume.Config.Expression.ShowThinking}]");

// ----------------------------------------------------------------------------
// INTERACTIVE CHAT LOOP — Yume stays "on" until the user says goodbye.
// ----------------------------------------------------------------------------

Console.WriteLine("\n--- Chat with Yume ---");
Console.WriteLine("    (slash commands: /help — control, not conversation; plain talk goes to her)");

while (true)
{
    Console.Write("\nYou  : ");
    string? input = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

    // Slash commands — explicit control channels, distinct from talking.
    // She can never invoke them herself; that is exactly the guarantee the
    // web search needs (she only goes online when YOU type /searchweb).
    if (input.StartsWith('/'))
    {
        if (HandleSlash(input, yume)) break;
        continue;
    }

    // Training commands — she can be taught any time, even mid-conversation.
    if (input.StartsWith("train ", StringComparison.OrdinalIgnoreCase))
    {
        TrainFromFile(yume, input[6..].Trim());
        continue;
    }
    if (input.StartsWith("teach ", StringComparison.OrdinalIgnoreCase))
    {
        var report = yume.Train(input[6..].Trim());
        Console.WriteLine($"Yume : {report}");
        continue;
    }
    if (input.Equals("sayings", StringComparison.OrdinalIgnoreCase))
    {
        var all = yume.Phrases.All();
        Console.WriteLine($"Yume : I know {all.Count} saying(s) I can speak in my own voice.");
        foreach (var s in all.Take(5))
            Console.WriteLine($"       - {s}");
        if (all.Count > 5) Console.WriteLine($"       ... and {all.Count - 5} more.");
        continue;
    }
    if (input.Equals("vocab", StringComparison.OrdinalIgnoreCase))
    {
        var lexemes = yume.Lexicon?.All() ?? [];
        var recent = yume.Lexicon?.RecentNewWords() ?? [];
        Console.WriteLine($"Yume : I have picked up {lexemes.Count} word(s) from talking with you"
                        + $" and own {yume.Phrases.Count} saying(s) in my own voice.");
        if (recent.Count > 0)
        {
            Console.WriteLine($"       Newest: {string.Join(", ", recent.Take(8))}");
        }
        continue;
    }

    // Identity commands — she can grow into a new sense of self mid-chat:
    //   gender girl            -> she now calls herself a girl (pronouns inferred)
    //   gender she/her         -> pronouns preset only
    //   pronouns he/him        -> change just the pronoun set
    // they/them is NOT a setting for her own pronouns: it is reserved for
    // describing people whose gender she does not know.
    if (input.StartsWith("gender ", StringComparison.OrdinalIgnoreCase))
    {
        var word = input[7..].Trim();
        if (word.Length == 0) { Console.WriteLine($"[usage: gender <identity | she/her | he/him>]"); continue; }
        try
        {
            yume.SetGender(word);
            Console.WriteLine($"[Yume's identity is now: '{(yume.GenderIdentity.Length > 0 ? yume.GenderIdentity : "(unset)")}' — pronouns {yume.Pronouns}.]");
            Console.WriteLine($"Yume : {yume.DescribeSelf()}");
        }
        catch (MindConfigurationException ex)
        {
            Console.WriteLine($"[{ex.Message}]");
        }
        continue;
    }
    if (input.StartsWith("pronouns ", StringComparison.OrdinalIgnoreCase))
    {
        var raw = input[9..].Trim();
        if (!PronounSet.TryParse(raw, out var set))
        {
            if (raw.Equals("they/them", StringComparison.OrdinalIgnoreCase))
                Console.WriteLine("[they/them is reserved for describing people whose gender Yume does not know — it is not a setting for her own pronouns. Choose she/her or he/him.]");
            else
                Console.WriteLine($"[Unknown pronoun set '{raw}'. Known presets: she/her, he/him.]");
            continue;
        }
        yume.SetPronouns(set);
        Console.WriteLine($"[Yume's pronouns are now {yume.Pronouns}.]");
        Console.WriteLine($"Yume : {yume.DescribeSelf()}");
        continue;
    }
    if (input.Equals("who are you", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine($"Yume : {yume.DescribeSelf()}");
        continue;
    }

    MindResponse reply = yume.Process(input);

    Console.WriteLine($"Yume : {reply.Content}");
    if (reply.RecalledMemories.Count > 0)
        Console.WriteLine($"       [remembered {reply.RecalledMemories.Count} related memories]");

    // Internal detail stays INVISIBLE by default — the developer decides
    // what surfaces. --show-internals (or /showthinking, /meta) flips these:
    if (yume.Config.Expression.ShowThinking &&
        yume.Thinking is { } thinker && thinker.LastThoughts.Count > 0)
    {
        Console.WriteLine("       ~ her thoughts before speaking:");
        foreach (var thought in thinker.LastThoughts)
            Console.WriteLine($"         ~ {thought}");
    }
    if (yume.Config.Expression.ShowResponseMeta)
        Console.WriteLine($"       [confidence {reply.Confidence:P0}, style '{reply.StyleName}', cycle {reply.CycleNumber}]");

    // Vocabulary growth, live: when a turn taught her new words, she says so.
    if (yume.Lexicon is { } lex && lex.LastPicks.Count > 0)
        Console.WriteLine($"       [picked up the word(s): {string.Join(", ", lex.LastPicks)} — vocabulary now {lex.Size} word(s)]");
}

// Teach her that this conversation was pleasant (drives reinforcement learning).
yume.Reinforce(0.8);

// ----------------------------------------------------------------------------
// STEP 4 — SLEEP: snapshot her whole mind (knowledge, emotions, memory,
//          neural weights, goals) into SQLite so next activation restores her.
// ----------------------------------------------------------------------------

await yume.SaveAsync(DbPath);
yume.Dispose();

Console.WriteLine("\n[Yume saved to yume.db — run me again and she will remember.]");

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

void TrainFromFile(Mind mind, string path)
{
    var resolved = ResolvePath(path);
    try
    {
        if (LooksLikeImage(resolved) && File.Exists(resolved))
        {
            // A text image: she SEES the words, then learns them like teaching.
            var report = mind.TrainImage(resolved);
            Console.WriteLine($"Yume : {report.Summarize()}");
            if (!report.IsEmpty)
                Console.WriteLine($"       [image text: {report.OcrText.Replace("\n", " / ")}]");
        }
        else
        {
            var report = mind.TrainFile(resolved);
            Console.WriteLine($"[Trained from {Path.GetFileName(path)}: {report}]");
            Console.WriteLine($"[Vocabulary: {mind.Phrases.Count} saying(s) she can speak now.]");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Training failed: {ex.Message}]");
    }
}

string ResolvePath(string path)
{
    if (File.Exists(path)) return path;
    // Corpora also live next to the built exe (copied at build time).
    var nearExe = Path.Combine(AppContext.BaseDirectory, path);
    return File.Exists(nearExe) ? nearExe : path; // TrainFile reports missing files
}

// "only3", "3", "up-to-high" style input → the matching ceiling preset.
bool TryParseCeiling(string text, out ThinkingCeiling ceiling)
{
    var key = text.Trim().ToLowerInvariant().Replace("-", "").Replace("_", "").Replace(" ", "");
    if (key.StartsWith("only")) key = key[4..];
    if (int.TryParse(key, out var n) && n >= 1 && n <= 5)
    {
        ceiling = (ThinkingCeiling)(n - 1);
        return true;
    }
    ceiling = ThinkingCeiling.Only5;
    return false;
}

string AllowedLevels(ThinkingCeiling c) => c switch
{
    ThinkingCeiling.Only1 => "Low",
    ThinkingCeiling.Only2 => "Low, Medium",
    ThinkingCeiling.Only3 => "Low, Medium, High",
    ThinkingCeiling.Only4 => "Low, Medium, High, Very High",
    _ => "Low, Medium, High, Very High, Max"
};

// ----------------------------------------------------------------------------
// Slash commands — the control channel. '/' keeps them strictly apart from
// conversation: nothing here is ever spoken by Yume on her own, and web
// search in particular ONLY runs when you type /searchweb yourself.
// Returns true when the chat should end (/exit).
// ----------------------------------------------------------------------------

bool HandleSlash(string input, Mind mind)
{
    var parts = input.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
    var cmd = parts[0].ToLowerInvariant();
    var rest = parts.Length > 1 ? parts[1].Trim() : "";

    switch (cmd)
    {
        case "/help" or "/?":
            Console.WriteLine("[slash commands]");
            Console.WriteLine("  /teach <sentence>      teach her facts line-by-line");
            Console.WriteLine("  /train <file>          bulk-train a corpus file (or a text image!)");
            Console.WriteLine("  /look <image>          read the text in an image without training");
            Console.WriteLine("  /searchweb <query>     explicit web search — she reads INTO the best result page");
            Console.WriteLine("  /explore <url>         open any web page and read it (she learns from the text)");
            Console.WriteLine("  /improve               run a self-improvement review pass");
            Console.WriteLine("  /predict <text>        show her next-token predictions after your words");
            Console.WriteLine("  /trainbrain [epochs]   deep-train her transformer language model");
            Console.WriteLine("  /brain                 status of her transformer language model");
            Console.WriteLine("  /transcribe <wav>      transcribe speech she synthesized herself");
            Console.WriteLine("  /clone <wav>           clone a voice from a recording");
            Console.WriteLine("  /voice                 show her current voice profile");
            Console.WriteLine("  /convert <wav> <hz>    convert audio to a new sample rate");
            Console.WriteLine("  /separate <wav>        split audio into low / mid / high stems");
            Console.WriteLine("  /edit <wav> <op> [...] audio edit: trim | gain | fade | normalize | reverse | concat");
            Console.WriteLine("  /media                 list the media files she has created");
            Console.WriteLine("  /vocab                 words she has picked up");
            Console.WriteLine("  /sayings               her trained spoken vocabulary");
            Console.WriteLine("  /feeling               how she feels right now (yume.Feeling.CurrentStatus)");
            Console.WriteLine("  /feelings on|off       announce feelings mechanically vs. keep chat natural");
            Console.WriteLine("  /followups on|off      let her ask her own follow-up questions");
            Console.WriteLine("  /thinking on|off       can this model think before speaking at all");
            Console.WriteLine("  /showthinking on|off   show her thought trace in chat (hidden by default)");
            Console.WriteLine("  /level <l>             thinking level: low|medium|high|veryhigh|max (clamped by the ceiling)");
            Console.WriteLine("  /ceiling <n>           hard cap: only1..only5 — 1=Low only, 2=+Medium, 3=+High, 4=+VeryHigh, 5=everything");
            Console.WriteLine("  /tone <t>              thinking tone: balanced|focused|playful|poetic|plain");
            Console.WriteLine("  /meta on|off           show the response meta line (confidence/style/cycle)");
            Console.WriteLine("  /gender <identity>     set her identity (girl/boy/...)");
            Console.WriteLine("  /pronouns <set>        set her pronouns (she/her | he/him)");
            Console.WriteLine("  /who                   her self-description");
            Console.WriteLine("  /exit                  save and sleep");
            Console.WriteLine("  (plain commands still work: teach <s> | train <file> | sayings | vocab | gender <x> | pronouns <set>)");
            Console.WriteLine("  (chat can also create media: 'draw a picture of...', 'make a gif of...', 'compose music about...', 'say ...')");
            return false;

        case "/teach" when rest.Length > 0:
        {
            var report = mind.Train(rest);
            Console.WriteLine($"Yume : {report}");
            return false;
        }
        case "/teach":
            Console.WriteLine("[usage: /teach <sentence>]");
            return false;

        case "/train" when rest.Length > 0:
            TrainFromFile(mind, rest);
            return false;
        case "/train":
            Console.WriteLine("[usage: /train <file>]");
            return false;

        case "/searchweb" or "/search" when rest.Length > 0:
        {
            Console.WriteLine($"[searching the web for \"{rest}\" — she only searches when you ask, never on her own]");
            var result = mind.SearchWeb(rest);
            if (result.Success)
            {
                Console.WriteLine($"Yume : {result.Output.Replace("\\n", $"{Environment.NewLine}       ")}");
                Console.WriteLine("       [learned from the results and the pages she read: memories stored, clean facts added, next-token model updated]");
            }
            else
            {
                Console.WriteLine($"Yume : {result.Output}");
            }
            return false;
        }
        case "/searchweb" or "/search":
            Console.WriteLine("[usage: /searchweb <what to look up>]");
            return false;

        case "/explore" or "/visit" or "/read" when rest.Length > 0:
        {
            Console.WriteLine($"[opening {rest} — she only reads pages you point her at, never on her own]");
            var result = mind.ExploreWeb(rest);
            Console.WriteLine($"Yume : {result.Output.Replace("\\n", $"{Environment.NewLine}       ")}");
            if (result.Success)
                Console.WriteLine("       [learned from the page: memories stored, clean facts added, next-token model updated]");
            return false;
        }
        case "/explore" or "/visit" or "/read":
            Console.WriteLine("[usage: /explore <url> — e.g. /explore en.wikipedia.org/wiki/Flower]");
            return false;

        case "/look" when rest.Length > 0:
        {
            try
            {
                Console.WriteLine($"Yume : {mind.LookAtImage(ResolvePath(rest))}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }
        case "/look":
            Console.WriteLine("[usage: /look <image.png|bmp|gif|ppm>]");
            return false;

        case "/transcribe" when rest.Length > 0:
        {
            try
            {
                var result = mind.TranscribeAudio(ResolvePath(rest));
                Console.WriteLine($"Yume : {result.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }
        case "/transcribe":
            Console.WriteLine("[usage: /transcribe <file.wav> — she transcribes speech she synthesized]");
            return false;

        case "/clone" when rest.Length > 0:
        {
            try
            {
                var profile = mind.CloneVoice(ResolvePath(rest));
                Console.WriteLine($"Yume : Done — I listened and studied the voice. {profile.Describe()}.");
                Console.WriteLine("       I will speak with this voice from now on (ask me to say something!).");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }
        case "/clone":
            Console.WriteLine("[usage: /clone <recording.wav> — at least 0.5 s of speech]");
            return false;

        case "/voice":
            Console.WriteLine($"Yume : {mind.DescribeVoice()}.");
            return false;

        case "/convert" when rest.Length > 0:
        {
            var args = rest.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
            if (args.Length < 2 || !int.TryParse(args[1], out var hz) || hz is < 1000 or > 192000)
            {
                Console.WriteLine("[usage: /convert <file.wav> <sampleRateHz e.g. 44100>]");
                return false;
            }
            try
            {
                Console.WriteLine($"Yume : {mind.ConvertAudio(ResolvePath(args[0]), hz)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }
        case "/convert":
            Console.WriteLine("[usage: /convert <file.wav> <sampleRateHz>]");
            return false;

        case "/separate" when rest.Length > 0:
        {
            try
            {
                Console.WriteLine($"Yume : {mind.SeparateAudio(ResolvePath(rest))}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }
        case "/separate":
            Console.WriteLine("[usage: /separate <file.wav>]");
            return false;

        case "/edit" when rest.Length > 0:
        {
            var args = rest.Split(' ', 3, StringSplitOptions.RemoveEmptyEntries);
            if (args.Length < 2)
            {
                Console.WriteLine("[usage: /edit <file.wav> <trim start end | gain x | fade in out | normalize | reverse | concat other.wav>]");
                return false;
            }
            try
            {
                Console.WriteLine($"Yume : {mind.EditAudio(ResolvePath(args[0]), args[1], args.Length > 2 ? args[2] : "")}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }
        case "/edit":
            Console.WriteLine("[usage: /edit <file.wav> <op> [args]]");
            return false;

        case "/media":
        {
            var files = mind.ListMedia();
            if (files.Count == 0)
                Console.WriteLine("Yume : I have not made any media yet — ask me to draw, animate, sing or speak!");
            else
            {
                Console.WriteLine($"Yume : I have created {files.Count} media file(s) in my gallery:");
                foreach (var file in files.Take(12))
                    Console.WriteLine($"       - {file}");
                if (files.Count > 12) Console.WriteLine($"       ... and {files.Count - 12} more.");
            }
            return false;
        }

        case "/improve":
        {
            try
            {
                var report = mind.SelfImprove();
                Console.WriteLine($"Yume : {report.Summary}");
            }
            catch (MindConfigurationException ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        }

        case "/predict":
        {
            var context = rest.Length > 0 ? rest : "I";
            var predictions = mind.PredictNext(context, 5);
            if (predictions.Count == 0)
            {
                Console.WriteLine("Yume : My language model has not learned enough yet — train me and ask again.");
                return false;
            }
            var rendered = string.Join(", ", predictions.Select(p => $"\"{p.Token}\" ({p.Probability:P0})"));
            Console.WriteLine($"Yume : After \"{context}\" I would most likely say next: {rendered}.");
            var nlm = mind.NeuralLanguageModel;
            Console.WriteLine($"       [n-gram: {mind.LanguageModel.TotalTokens} token(s) observed, {mind.LanguageModel.Vocabulary} word(s) known"
                            + (nlm.IsTrained
                                ? $" — transformer ({nlm.ParameterCount:N0} parameters, loss {nlm.LastLoss:F3}) is blending in too"
                                : " — transformer not trained yet: /trainbrain to train it")
                            + "]");
            return false;
        }

        case "/trainbrain":
        {
            int epochs = 12;
            if (rest.Length > 0 && (!int.TryParse(rest.Split(' ')[0], out epochs) || epochs < 1 || epochs > 200))
            {
                Console.WriteLine("[usage: /trainbrain [epochs 1-200]]");
                return false;
            }
            var nlm = mind.NeuralLanguageModel;
            Console.WriteLine($"[deep training her transformer: {nlm.Config.Layers} layer(s), {nlm.Config.Dim}d, {nlm.Config.Heads} heads, "
                            + $"{nlm.ParameterCount:N0} parameters, {nlm.BufferedSentences} sentence(s) buffered — {epochs} epoch(s)]");
            var report = mind.TrainDeepModel(epochs);
            if (report.Epochs == 0)
            {
                Console.WriteLine("Yume : I have not heard enough language yet — talk to me or /teach me first, then train my brain.");
                return false;
            }
            Console.WriteLine($"Yume : Done — {report}");
            Console.WriteLine($"       [my transformer now blends into /predict and how naturally I word my replies]");
            return false;
        }

        case "/brain":
        {
            var nlm = mind.NeuralLanguageModel;
            Console.WriteLine($"Yume : My deep language model is a {nlm.Config.Layers}-layer transformer "
                            + $"({nlm.Config.Dim}d, {nlm.Config.Heads} heads, context {nlm.Config.ContextLen}, "
                            + $"{nlm.ParameterCount:N0} parameters).");
            if (nlm.IsTrained)
                Console.WriteLine($"       Trained over {nlm.EpochsTrained} epoch(s) so far, loss {nlm.LastLoss:F3} nats/token, "
                                + $"{nlm.VocabularySize} word(s) in my neural vocabulary, {nlm.TokensObserved} token(s) observed.");
            else
                Console.WriteLine($"       Not trained yet — {nlm.BufferedSentences} sentence(s) waiting. Type /trainbrain to deep-train me.");
            Console.WriteLine($"       My statistical n-gram model knows {mind.LanguageModel.Vocabulary} word(s) from {mind.LanguageModel.TotalTokens} token(s).");
            return false;
        }

        case "/vocab":
        {
            var lexemes = mind.Lexicon?.All() ?? [];
            var recent = mind.Lexicon?.RecentNewWords() ?? [];
            Console.WriteLine($"Yume : I have picked up {lexemes.Count} word(s) from talking with you"
                            + $" and own {mind.Phrases.Count} saying(s) in my own voice.");
            if (recent.Count > 0)
                Console.WriteLine($"       Newest: {string.Join(", ", recent.Take(8))}");
            return false;
        }

        case "/sayings":
        {
            var all = mind.Phrases.All();
            Console.WriteLine($"Yume : I know {all.Count} saying(s) I can speak in my own voice.");
            foreach (var s in all.Take(5))
                Console.WriteLine($"       - {s}");
            if (all.Count > 5) Console.WriteLine($"       ... and {all.Count - 5} more.");
            return false;
        }

        // ---- feelings + internals + thinking controls (developer switches) ----
        case "/feeling":
            Console.WriteLine(mind.Feeling is { } snap
                ? $"[yume.Feeling.CurrentStatus] {snap.CurrentStatus}"
                : "[her emotion module is disabled]");
            return false;

        case "/feelings":
            bool showReads = rest is not ("off" or "false" or "0");
            mind.Config.Expression.Feelings(showReads);
            Console.WriteLine(showReads
                ? "[feelings will be ANNOUNCED in chat again: \"(my current read: ...)\"]"
                : "[feelings now color her words naturally — the raw state stays on /feeling]");
            return false;

        case "/followups":
            bool ask = rest is not ("off" or "false" or "0");
            mind.Config.Expression.FollowUps(ask);
            Console.WriteLine(ask
                ? "[she may ask her own follow-up questions again]"
                : "[she will only answer now — no Pick(...) questions]");
            return false;

        case "/thinking":
        {
            if (rest is "on" or "true" or "1") mind.Config.Thinking.Enable(true);
            else if (rest is "off" or "false" or "0") mind.Config.Thinking.Enable(false);
            Console.WriteLine($"[thinking enabled: {mind.Config.Thinking.Enabled} — level {mind.Config.Thinking.Level}, tone '{mind.Config.Thinking.Tone}']");
            return false;
        }

        case "/showthinking":
        {
            bool show = rest is not ("off" or "false" or "0");
            mind.Config.Expression.Thinking(show);
            Console.WriteLine(show
                ? "[her thought trace will be printed in chat]"
                : "[her thoughts are hidden again — still on reply.Chain / mind.Thinking.LastThoughts]");
            return false;
        }

        case "/level" when rest.Length > 0:
        {
            if (Enum.TryParse<ThinkingLevel>(rest.Replace("-", ""), true, out var lvl))
            {
                var requested = lvl;
                mind.Config.Thinking.At(lvl);
                string clampNote = mind.Config.Thinking.Level != requested
                    ? $" (clamped by ceiling {mind.Config.Thinking.Ceiling})" : "";
                Console.WriteLine($"[thinking level: {mind.Config.Thinking.Level}{clampNote} → {mind.Config.Thinking.ThoughtRounds} generated thought(s) per cycle]");
            }
            else
                Console.WriteLine("[unknown level. low | medium | high | veryhigh | max]");
            return false;
        }

        case "/ceiling" when rest.Length > 0:
        {
            if (TryParseCeiling(rest, out var ceil))
            {
                mind.Config.Thinking.UpTo(ceil);
                Console.WriteLine($"[thinking ceiling: {mind.Config.Thinking.Ceiling} — allowed: {AllowedLevels(ceil)}; " +
                                  $"current level {mind.Config.Thinking.Level} → {mind.Config.Thinking.ThoughtRounds} thought(s) per cycle]");
            }
            else
                Console.WriteLine("[unknown ceiling. only1..only5 (or 1..5): 1=Low only, 2=+Medium, 3=+High, 4=+VeryHigh, 5=+Max]");
            return false;
        }

        case "/tone" when rest.Length > 0:
            mind.Config.Thinking.WithTone(rest);
            Console.WriteLine($"[thinking tone: '{mind.Config.Thinking.Tone}']");
            return false;

        case "/meta":
        {
            bool show = rest is not ("off" or "false" or "0");
            mind.Config.Expression.Meta(show);
            Console.WriteLine(show
                ? "[response meta (confidence/style/cycle) will be shown]"
                : "[response meta hidden]");
            return false;
        }

        case "/gender" when rest.Length > 0:
            try
            {
                mind.SetGender(rest);
                Console.WriteLine($"[Yume's identity is now: '{(mind.GenderIdentity.Length > 0 ? mind.GenderIdentity : "(unset)")}' — pronouns {mind.Pronouns}.]");
                Console.WriteLine($"Yume : {mind.DescribeSelf()}");
            }
            catch (MindConfigurationException ex)
            {
                Console.WriteLine($"[{ex.Message}]");
            }
            return false;
        case "/gender":
            Console.WriteLine("[usage: /gender <identity | she/her | he/him>]");
            return false;

        case "/pronouns" when rest.Length > 0:
            if (!PronounSet.TryParse(rest, out var set))
            {
                if (rest.Equals("they/them", StringComparison.OrdinalIgnoreCase))
                    Console.WriteLine("[they/them is reserved for describing people whose gender Yume does not know — it is not a setting for her own pronouns. Choose she/her or he/him.]");
                else
                    Console.WriteLine($"[Unknown pronoun set '{rest}'. Known presets: she/her, he/him.]");
                return false;
            }
            mind.SetPronouns(set);
            Console.WriteLine($"[Yume's pronouns are now {mind.Pronouns}.]");
            Console.WriteLine($"Yume : {mind.DescribeSelf()}");
            return false;
        case "/pronouns":
            Console.WriteLine("[usage: /pronouns <she/her | he/him>]");
            return false;

        case "/who":
            Console.WriteLine($"Yume : {mind.DescribeSelf()}");
            return false;

        case "/exit" or "/quit" or "/bye":
            return true;

        default:
            Console.WriteLine($"[Unknown slash command '{cmd}'. Type /help for the list — plain sentences still just talk to her.]");
            return false;
    }
}
