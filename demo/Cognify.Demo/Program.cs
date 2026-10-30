using Cognify.Core;
using Cognify.Emotions;
using Cognify.Generation;

namespace Cognify.Demo;

/// <summary>
/// Interactive demo of the Cognify cognitive architecture.
/// Run it to see reasoning, thinking, emotions, self-learning, memory,
/// goals, personality — plus vision, hearing, language understanding,
/// semantic memory, tool use, code skills and media generation — all
/// working together. Then talk to your creation.
/// </summary>
internal static class Program
{
    private static void Main()
    {
        Console.WriteLine("=====================================================");
        Console.WriteLine("  Cognify - cognitive architecture library for .NET");
        Console.WriteLine("=====================================================\n");

        // ------------------------------------------------------------------
        // 1. Build a mind with the capabilities you want.
        // ------------------------------------------------------------------
        using var mind = MindBuilder.Create("Aria")
            .Describe("A curious, warm demo mind with senses and hands")
            .WithPersonality(p => p.Traits(openness: 0.9, conscientiousness: 0.6,
                                           extraversion: 0.75, agreeableness: 0.8,
                                           neuroticism: 0.35))
            .WithEmotions(e => e.Intensity(0.9).Decay(0.05))
            .WithReasoning(r => r.Depth(5))
            .WithThinking()
            .WithSelfLearning(l => l.Rate(0.1).Exploration(0.15))
            .WithMemory(m => m.Capacity(15))
            .WithGoals(g => g.AddGoal("learn-things", priority: 0.8, "curiosity")
                             .AddGoal("be-helpful", priority: 0.7, "achievement"))
            .WithVision()
            .WithAudio()
            .WithEmbeddings()
            .WithMultilingual(m => m.Prefer("en"))
            .WithTools()
            .WithCode()
            .WithMediaGeneration()
            .WithPersistence("aria.mind.db")
            .Build();

        Console.WriteLine($"Built mind '{mind.Name}' with {mind.Modules.Count} cognitive modules:");
        foreach (var module in mind.Modules)
            Console.WriteLine($"  - {module.Name,-12} (order {module.Order})");
        Console.WriteLine();

        // ------------------------------------------------------------------
        // 2. Teach it facts - it reasons with what it knows.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Teaching ----------------------------------------");
        foreach (var fact in new[]
                 {
                     "birds are animals",
                     "animals are living things",
                     "a bird has wings",
                     "a bird can fly",
                     "dogs have fur",
                     "cats can climb"
                 })
        {
            int learned = mind.Teach(fact);
            Console.WriteLine($"  taught: \"{fact}\" -> {learned} fact(s)");
        }

        Console.WriteLine($"\nKnowledge base now holds {mind.Knowledge!.FactCount} facts.\n");

        // ------------------------------------------------------------------
        // 3. Ask it questions - watch deduction expand the is-a hierarchy.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Asking ------------------------------------------");
        foreach (var question in new[] { "what is a bird?", "what is a bird?" })
        {
            var response = mind.Ask(question);
            Console.WriteLine($"  Q: {question}");
            Console.WriteLine($"  A: {response.Content}\n");
        }

        // ------------------------------------------------------------------
        // 4. Vision - show it a scene.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Vision ------------------------------------------");
        var scene = mind.See("a bright sunny garden with a red flower, a small bird and a wooden table");
        Console.WriteLine($"  caption: {mind.Vision!.LastScene!.Caption}");
        Console.WriteLine($"  objects: {string.Join(", ", mind.Vision.LastScene.Objects.Select(o => o.ToString()))}");
        Console.WriteLine($"  colors: {string.Join("/", mind.Vision.LastScene.Colors)}, " +
                          $"brightness {mind.Vision.LastScene.Brightness:0.00}, scene type: {mind.Vision.LastScene.SceneType}");
        Console.WriteLine($"  Aria says: \"{FirstSentence(scene.Content)}\"\n");

        // ------------------------------------------------------------------
        // 5. Audio - let it hear speech and sounds.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Audio -------------------------------------------");
        var heard = mind.Hear("please tell me about dogs");
        Console.WriteLine($"  heard speech -> Aria: \"{FirstSentence(heard.Content)}\"");
        mind.Hear("loud thunder crash");
        Console.WriteLine($"  heard noise  -> classified as '{mind.Hearing!.LastHeard!.Category}': {mind.Hearing.LastHeard.Description}\n");

        // ------------------------------------------------------------------
        // 6. Multilingual - it knows which language you speak.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Multilingual ------------------------------------");
        foreach (var phrase in new[] { "gracias", "hello", "你好" })
        {
            var guess = mind.DetectLanguage(phrase)!;
            Console.WriteLine($"  \"{phrase}\" -> {guess.Language} ({guess.Script}, {guess.Confidence:P0})");
        }
        var translated = mind.Ask("gracias");
        Console.WriteLine($"  Aria: \"{translated.Content}\"\n");

        // ------------------------------------------------------------------
        // 7. Embeddings - semantic similarity, a "sense of aboutness".
        // ------------------------------------------------------------------
        Console.WriteLine("--- Embeddings --------------------------------------");
        double close = mind.SemanticSimilarity("the cat sat on the mat", "a cat sat on a mat");
        double far = mind.SemanticSimilarity("the cat sat on the mat", "quarterly revenue grew by twelve percent");
        Console.WriteLine($"  similarity(\"the cat sat on the mat\", \"a cat sat on a mat\") = {close:0.000}");
        Console.WriteLine($"  similarity(\"the cat sat on the mat\", \"quarterly revenue grew...\") = {far:0.000}");
        var neighbors = mind.SemanticMemory!.FindSimilar("cats and fish", topK: 1);
        if (neighbors.Count > 0)
            Console.WriteLine($"  most similar stored input: \"{neighbors[0].Entry.Text}\" ({neighbors[0].Similarity:P0})\n");

        // ------------------------------------------------------------------
        // 8. Tool use - the mind can act, not just talk.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Tool use ----------------------------------------");
        Console.WriteLine($"  tools available: {string.Join(", ", mind.Tools!.Registry.All().Select(t => t.Name))}");
        Console.WriteLine($"  \"calculate (21 + 21) / 2\" -> {mind.Ask("calculate (21 + 21) / 2").Content}");
        Console.WriteLine($"  \"what time is it?\"        -> {mind.Ask("what time is it?").Content}");
        Console.WriteLine($"  bare \"2+2*3\"             -> {mind.Ask("2+2*3").Content}\n");

        // ------------------------------------------------------------------
        // 9. Code - generation and analysis.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Code --------------------------------------------");
        var codeResponse = mind.Ask("write a python class for a robot vacuum");
        Console.WriteLine(Indent(codeResponse.Content, "  "));
        var analysis = mind.Ask("check this: ```csharp\npublic class A {\n  // demo\n  public int Get() { return 1; }\n}\n```");
        Console.WriteLine($"  analysis: {FirstSentence(analysis.Content)}\n");

        // ------------------------------------------------------------------
        // 10. Media generation - images, video specs and real WAV audio.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Media generation --------------------------------");
        var image = mind.GenerateMedia("a sunset over quiet mountains", MediaType.Image)!;
        Console.WriteLine($"  {image.Describe()}");

        var speech = mind.GenerateMedia("hello world, i am aria", MediaType.Speech)!;
        var wavPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "aria-hello.wav");
        System.IO.File.WriteAllBytes(wavPath, speech.Data!);
        Console.WriteLine($"  {speech.Describe()}");
        Console.WriteLine($"  -> real WAV bytes written to: {wavPath}\n");

        // ------------------------------------------------------------------
        // 11. Emotional appraisal - the mind FEELS its input.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Emotions ----------------------------------------");
        foreach (var text in new[] { "I love this wonderful day!", "I lost something and I feel sad.",
                                     "There is danger nearby and I am scared." })
        {
            var r = mind.Process(text);
            Console.WriteLine($"  \"{text}\"");
            Console.WriteLine($"    -> {mind.EmotionalState!.Describe()}");
            for (int i = 0; i < 10; i++) mind.Tick(); // time passes: emotions decay
        }
        Console.WriteLine($"    -> after 10 ticks: {mind.EmotionalState!.Describe()}\n");

        // ------------------------------------------------------------------
        // 12. Self-learning - reward good answers, the net learns from it.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Self-learning -----------------------------------");
        var beforeLoss = mind.Learning!.Stats.LastLoss;
        for (int i = 0; i < 10; i++)
        {
            mind.Ask("what is a bird?");
            mind.Reinforce(0.9); // the style it chose was good
        }
        Console.WriteLine($"  trained {mind.Learning.Stats.TrainingSteps} steps, " +
                          $"avg reward {mind.Learning.Stats.AverageReward:0.00}, " +
                          $"{mind.Learning.Stats.TotalExperiences} experiences");
        Console.WriteLine($"  network loss: {beforeLoss:0.000} -> {mind.Learning.Stats.LastLoss:0.000}\n");

        // ------------------------------------------------------------------
        // 13. Goals and drives - intrinsic motivation, Tick advances time.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Goals & drives ----------------------------------");
        for (int i = 0; i < 40; i++) mind.Tick(); // time passes: drives grow
        foreach (var drive in mind.GoalSystem!.Drives.Values)
            Console.WriteLine($"  drive {drive.Name,-12} level {drive.Level:0.00}");
        var focus = mind.GoalSystem.FocusGoal ?? mind.GoalSystem.ActiveGoals().FirstOrDefault();
        if (focus is not null)
            Console.WriteLine($"  focus goal: {focus}\n");

        // ------------------------------------------------------------------
        // 14. Explainability - the full chain of thought of the last cycle.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Chain of thought (last cycle) -------------------");
        var last = mind.Ask("what is a bird?");
        Console.WriteLine(Indent(last.Chain!.Render(), "  "));
        Console.WriteLine();

        // ------------------------------------------------------------------
        // 15. Persistence - save the whole mind, restore it later.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Persistence -------------------------------------");
        mind.SaveAsync().GetAwaiter().GetResult();
        Console.WriteLine($"  saved '{mind.Name}' to aria.mind.db " +
                          $"({mind.Knowledge.FactCount} facts, {mind.Memory!.Counts.LongTerm} long-term memories, " +
                          $"{mind.SemanticMemory!.Store.Count} semantic vectors)");

        using var revived = MindBuilder.Restore("aria.mind.db");
        var revivedAnswer = revived.Ask("what is a bird?");
        Console.WriteLine($"  restored mind '{revived.Name}' answered: {FirstSentence(revivedAnswer.Content)}");
        Console.WriteLine($"  (knowledge intact: {revived.Knowledge!.FactCount} facts, " +
                          $"learning intact: {revived.Learning!.Stats.TrainingSteps} training steps, " +
                          $"semantic memory intact: {revived.SemanticMemory!.Store.Count} vectors)\n");

        // ------------------------------------------------------------------
        // 16. Interactive chat - your turn.
        // ------------------------------------------------------------------
        Console.WriteLine("--- Interactive chat --------------------------------");
        Console.WriteLine("  Talk to the mind. Commands: /teach <fact>, /feel <emotion> <0-1>,");
        Console.WriteLine("  /reward <number>, /state, /goals, /memories, /see <scene>, /hear <words>,");
        Console.WriteLine("  /similar <text>, /generate <image|video|speech|sound> <prompt>, /save, /exit");
        ChatLoop(mind);
    }

    private static void ChatLoop(Mind mind)
    {
        while (true)
        {
            Console.Write("\nyou> ");
            var input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input)) continue;
            var parts = input.Split(' ', 2);
            var command = parts[0].ToLowerInvariant();

            switch (command)
            {
                case "/exit":
                    mind.SaveAsync().GetAwaiter().GetResult();
                    Console.WriteLine("aria> mind saved. Goodbye!");
                    return;

                case "/teach" when parts.Length > 1:
                    int learned = mind.Teach(parts[1]);
                    Console.WriteLine($"aria> learned {learned} fact(s).");
                    break;

                case "/feel" when parts.Length > 1:
                {
                    var args = parts[1].Split(' ');
                    if (args.Length == 2 && Enum.TryParse<EmotionType>(args[0], true, out var emotion) &&
                        double.TryParse(args[1], out var amount))
                    {
                        mind.EmotionalState!.Feel(emotion, amount);
                        Console.WriteLine($"aria> now feeling: {mind.EmotionalState.Describe()}");
                    }
                    else Console.WriteLine("aria> usage: /feel <Joy|Trust|Fear|Surprise|Sadness|Disgust|Anger|Anticipation> <0-1>");
                    break;
                }

                case "/reward" when parts.Length > 1 && double.TryParse(parts[1], out var reward):
                    mind.Reinforce(reward);
                    Console.WriteLine($"aria> learned from that (avg reward {mind.Learning!.Stats.AverageReward:0.00}).");
                    break;

                case "/see" when parts.Length > 1:
                {
                    var r = mind.See(parts[1]);
                    Console.WriteLine($"aria> {FirstSentence(r.Content)}");
                    if (mind.Vision!.LastScene is { } s)
                        Console.WriteLine($"      (objects: {string.Join(", ", s.Objects.Take(4).Select(o => o.Label))}; " +
                                          $"colors: {string.Join("/", s.Colors)}; brightness {s.Brightness:0.00})");
                    break;
                }

                case "/hear" when parts.Length > 1:
                {
                    var r = mind.Hear(parts[1]);
                    Console.WriteLine($"aria> {FirstSentence(r.Content)}");
                    break;
                }

                case "/similar" when parts.Length > 1:
                {
                    var hits = mind.SemanticMemory!.FindSimilar(parts[1], topK: 3);
                    if (hits.Count == 0) Console.WriteLine("aria> nothing semantically related stored yet.");
                    foreach (var (entry, similarity) in hits)
                        Console.WriteLine($"aria> {similarity:P0}  \"{entry.Text}\"");
                    break;
                }

                case "/generate" when parts.Length > 1:
                {
                    var args = parts[1].Split(' ', 2);
                    if (args.Length == 2 &&
                        Enum.TryParse<MediaType>(args[0], true, out var media))
                    {
                        var artifact = mind.GenerateMedia(args[1], media)!;
                        Console.WriteLine($"aria> {artifact.Describe()}");
                        if (artifact.Data is { } bytes)
                        {
                            var path = System.IO.Path.Combine(
                                System.IO.Directory.GetCurrentDirectory(),
                                $"aria-{artifact.Id}.wav");
                            System.IO.File.WriteAllBytes(path, bytes);
                            Console.WriteLine($"      WAV written to: {path}");
                        }
                    }
                    else Console.WriteLine("aria> usage: /generate <Image|Video|Speech|Audio> <prompt>");
                    break;
                }

                case "/state":
                    Console.WriteLine($"aria> emotions: {mind.EmotionalState?.Describe() ?? "n/a"}");
                    Console.WriteLine($"aria> personality: {mind.Personality?.Describe() ?? "n/a"}");
                    Console.WriteLine($"aria> cycles: {mind.CycleCount}, facts: {mind.Knowledge?.FactCount ?? 0}");
                    Console.WriteLine($"aria> language: {mind.Languages?.LastLanguage ?? "n/a"}, tools: {mind.Tools?.Registry.Count ?? 0}, " +
                                      $"scenes seen: {mind.Vision?.ScenesSeen ?? 0}");
                    break;

                case "/goals":
                    foreach (var goal in mind.GoalSystem!.ActiveGoals())
                        Console.WriteLine($"aria> {goal}");
                    foreach (var drive in mind.GoalSystem.Drives.Values)
                        Console.WriteLine($"aria> drive {drive.Name}: {drive.Level:0.00}");
                    break;

                case "/memories":
                    foreach (var m in mind.Memory!.LongTermMemories().Take(5))
                        Console.WriteLine($"aria> {m}");
                    foreach (var m in mind.Memory.Episodes().Take(3))
                        Console.WriteLine($"aria> (episodic) {m}");
                    break;

                case "/save":
                    mind.SaveAsync().GetAwaiter().GetResult();
                    Console.WriteLine("aria> saved.");
                    break;

                default:
                    var response = mind.Process(input);
                    Console.WriteLine($"aria> {response.Content}");
                    if (mind.EmotionalState is { } emo && emo.DominantIntensity > 0.3)
                        Console.WriteLine($"      (feeling: {emo.Describe()})");
                    break;
            }
        }
    }

    private static string FirstSentence(string text)
    {
        int cut = text.IndexOfAny(new[] { '.', '!', '?', '\n' });
        return cut > 0 ? text[..cut].Trim() + "." : text.Trim();
    }

    private static string Indent(string text, string indent) =>
        string.Join(Environment.NewLine, text.Split(Environment.NewLine).Select(l => indent + l));
}
