using Cognify.Core;

var mind = MindBuilder.Create("Yume")
    .Describe("Yume (夢, 'dream') — a warm, curious companion AI.")
    .WithEmotions(e => e.Intensity(0.85))
    .WithReasoning()
    .WithThinking()
    .WithMemory()
    .WithTools()
    .Build();

mind.Process("I love this, this is wonderful!");
Console.WriteLine($"EmotionalState is null? {mind.EmotionalState is null}");
var r = mind.Ask("introduce yourself");
Console.WriteLine($"CONTENT: [{r.Content}]");
Console.WriteLine($"STYLE: {r.StyleName}");
