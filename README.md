# Cognify

**A cognitive architecture library for .NET 8** — build AI models that can *reason, think, learn by themselves, feel emotions, remember, pursue goals, have a personality* — and that can also **see, hear, understand languages, embed meaning, use tools, write and analyze code, and generate media**.

Cognify gives you a `Mind`: a central cognitive kernel surrounded by **pluggable cognitive modules**. You decide which capabilities your AI gets, configure them fluently, watch them work through a fully explainable chain of thought, and persist the entire mind (knowledge, memories, emotions, personality, semantic vectors, even the trained neural network) to SQLite.

```csharp
using Cognify.Core;

using var mind = MindBuilder.Create("Aria")
    .Describe("A curious, warm assistant with senses and hands")
    .WithPersonality(p => p.Traits(openness: 0.9, extraversion: 0.7))
    .WithEmotions(e => e.Intensity(0.9))
    .WithReasoning()
    .WithThinking()
    .WithSelfLearning(l => l.Rate(0.1))
    .WithMemory()
    .WithGoals(g => g.AddGoal("learn", priority: 0.8, "curiosity"))
    .WithVision()                 // see images & scenes
    .WithAudio()                  // hear sounds & speech
    .WithEmbeddings()             // semantic vectors & similarity
    .WithMultilingual()           // language detection & translation
    .WithTools()                  // tool use / function calling
    .WithCode()                   // code generation & analysis
    .WithMediaGeneration()        // image / video / speech / audio synthesis
    .WithPersistence("aria.db")
    .Build();

var answer = mind.Ask("what is a bird?");
mind.Teach("birds are animals");
mind.See("a red apple on a wooden table");
mind.Hear("what time is it?");
mind.Reinforce(0.9);          // reward the last response -> the mind learns
await mind.SaveAsync();       // persist the entire mind state
```

---

## Why Cognify?

| | |
|---|---|
| **Explainable by design** | Every response comes with a step-by-step `ChainOfThought` and a live inner monologue (`ThoughtStream`). No black boxes. |
| **Hybrid engine** | Cognitive simulation (appraisal rules, symbolic inference) *plus* real machine learning (a micro neural network with true backpropagation, hashed n-gram embeddings, a real recursive-descent calculator parser, real WAV synthesis). |
| **Modular** | Fourteen built-in modules + bring your own via `ICognitiveModule`. Enable only what you need. |
| **Persistent** | One call saves facts, rules, memories, semantic vectors, goals, emotions, personality and trained network weights to SQLite. Restore anytime. |
| **Zero external AI services** | Runs fully offline in-process. No API keys, no GPU. Pluggable backends let you wire in real models when you want them. |

---

## The cognitive cycle

Each `mind.Process(stimulus)` runs one structured cycle. Modules run in order and share a `CognitiveContext` workspace:

```
stimulus (text | image | audio | video)
   |
   v
 personality (5)     traits shape exploration, deliberation, sensitivity
   |
 memory (10)         recall related memories, store the stimulus
   |
 vision (11)         scenes -> objects, colors, brightness, captions        [WithVision]
 audio (12)          sounds -> transcription, tempo, loudness               [WithAudio]
 multilingual (13)   detect language, translate into preferred language    [WithMultilingual]
 embeddings (14)     embed text, search semantic neighbors                 [WithEmbeddings]
   |
 emotions (20)       OCC-style appraisal -> emotions, mood update
   |
 reasoning (30)      answer questions, forward-chain rules, induce patterns
   |
 goals (40)          pick focus goal, suggest action, track progress
   |
 tools (45)          invoke registered tools, turn results into answers    [WithTools]
 code (46)           generate scaffolds, analyze snippets                  [WithCode]
 generation (47)     create image/video specs, synthesize real WAV audio   [WithMediaGeneration]
   |
 thinking (50)       chain of thought; defers to perception/action modules
   |
 learning (60)       encode features, self-supervised valence training
   |
 response            MindResponse (content + confidence + full trace)
```

Perception and action modules may set the response first; the thinking module then keeps it (honoring the "first module wins" contract) while still crediting the specialized response style in the learning system.

**The mind never parrots you.** Recalled memories are usually the user's own words, so responses reference them obliquely ("it connects with what you told me earlier about apples") instead of quoting them back. When the input teaches a fact ("2 + 2 is 4"), the reply acknowledges the learning ("Got it - I just learned 1 new thing(s) from that") rather than echoing the sentence, and deliberation penalizes recently-used styles, word-for-word repeats of the previous reply AND repeats of the previous reply's sentence *template* (numbers normalized — "learned 1 new thing" and "learned 3 new things" are the same line, so the phrase pool rotates instead of drilling the same shape). Topic extraction ignores time fillers, so "...how many do i currently have?" yields the topic "apples", never "currently".

---

## Turning capabilities on and off

The builder is **opt-in**: a mind gets only the modules you ask for. The minimal build below has *exactly* three modules — no audio, no tools, no vision, nothing else:

```csharp
var mind = MindBuilder.Create("ExampleAI")
    .WithThinking()
    .WithSelfLearning()
    .WithMemory()
    .Build();                    // 3 modules attached, everything else excluded
```

If you start from the full set, prune it with `Without(...)` (case-insensitive, chainable):

```csharp
var mind = MindBuilder.Create("ExampleAI")
    .WithAllModules()
    .Without("audio", "tools", "code", "generation")
    .Build();                    // full mind minus the four stripped capabilities
```

Capability names (see `MindBuilder.KnownCapabilities`): `reasoning`, `thinking`, `learning`, `emotions`, `memory`, `goals`, `personality`, `vision`, `audio`, `embeddings`, `multilingual`, `tools`, `code`, `generation`. Unknown names throw `MindConfigurationException`. You can also detach a module at runtime with `mind.DetachModule<T>()` (or attach new ones with `mind.AttachModule(...)`), and every remaining module keeps working — the cognitive cycle simply skips what isn't there.

## Lenient mode — don't crash on disabled capabilities

By default the mind is **strict**: calling a capability that wasn't enabled (`See` without vision, `Hear` without audio, `InvokeTool` without tools, ...) throws `MindConfigurationException` so misconfigurations stay loud. If you'd rather not need *every* module you touch, add `.Lenient()` and the gated calls degrade gracefully instead:

```csharp
var mind = MindBuilder.Create("Yume")
    .WithThinking()
    .WithMemory()
    .Lenient()                    // <- no MindConfigurationException anymore
    .Build();

mind.See("a sunset over the mountains"); // works: full cognitive cycle, no vision module needed
mind.Hear("soft rain on a window");      // works too
mind.InvokeTool("clock");                // returns a failed ToolResult instead of throwing
mind.Reinforce(0.8);                     // no learning module? still stirs the mind's emotions
```

What lenient mode does for each gated call:

| Call | Without module, in lenient mode |
|------|-------------------------------|
| `See(desc)` / `Hear(content)` | Still runs the **full cognitive cycle** on the description (the mind registers it as text, minus scene/audio analysis). A note lands in `mind.Stream`. |
| `Teach(sentence)` | Keeps the sentence as a plain memory and returns `0` facts learned. |
| `Reinforce(reward)` | Trains nothing but still makes the mind feel the reward. |
| `InvokeTool(name)` | Returns `ToolResult.Fail(...)` — no exception. |
| `GenerateMedia(...)` | Returns `null` — nothing was generated. |
| `DetectLanguage(text)` | Returns `null`. |
| `SemanticSimilarity(a, b)` | Returns `0`. |

Check `mind.Lenient` at runtime to see which mode a mind is in. Strict mode remains the default so that typos and missing modules surface immediately during development.

## Core capabilities

### Reasoning — give your model logic
A symbolic knowledge base of `Fact`s (subject-predicate-object) and variable-carrying `LogicRule`s, with three inference modes:

- **Deduction** — forward chaining (modus ponens). Teach `birds are animals`, `animals are living things` and the mind derives `bird is-a living-thing`.
- **Induction** — concepts observed together repeatedly become `related-to` facts automatically.
- **Abduction** — find the rule + antecedents that best explain an observation.

Facts are also learned from ordinary sentences during every cognitive cycle (not just via `Teach()`) — say "my dog is a mammal" in conversation and it becomes knowledge automatically, matching CognifyPy behavior.

**Facts are spoken in natural words, never slugs.** Multi-word knowledge is stored and recalled with spaces — teach "Before is the affect before the cause" and asking "what is before?" yields *"Before is affect before cause."*, never the hyphenated `before-is-affect-before-cause` form. A display polish (`ReasoningModule.Polish`) capitalizes the standalone "I" and sentence starts when knowledge is spoken. Leading filler words ("now what is after?", "so what is X?") are skipped before question matching, so preambles never hide the actual question. Questions like "is a bird a living thing?" match taught facts through a letter-skeleton comparison, so the spaced form and any legacy `living-thing` slug compare equal.

```csharp
mind.Teach("a bird can fly");
var response = mind.Ask("what is a bird?");   // "bird is a animal. bird can fly. ..."
mind.Knowledge.AddRule(new LogicRule(          // custom rule: birds in cages cannot fly
    new[] { new FactPattern("?s", "is-a", "bird"), new FactPattern("?s", "is-in", "cage") },
    new FactPattern("?s", "can-not", "fly")));
```

### Thinking — give your model an inner voice
The thinking module builds an inspectable `ChainOfThought` (Observe -> Recall -> Feel -> Hypothesize -> Evaluate -> Generate -> Deliberate -> Conclude) and streams an inner monologue you can subscribe to:

```csharp
mind.ThoughtProduced += (_, e) => Console.WriteLine($"  thinking: {e.Step.Text}");
var response = mind.Process("hello");          // response.Chain renders the full trace
Console.WriteLine(response.Chain.Render());
```

Deliberation generates candidate responses in seven styles (informative, empathetic, curious, concise, reflective, encouraging, trained) and picks one using the hybrid learned value model (symbolic Q-table + personality prior + neural style-conditioned prediction).

**Thinking levels — a token budget for the inner voice.** Just like thinking budgets in modern language models, `WithThinking(t => t.At(ThinkingLevel.Low | Medium | High | VeryHigh | Max))` sizes how many thoughts the **thought generator** produces BEFORE any reply is composed: Low = 1, Medium = 2 (default), High = 4, VeryHigh = 6, Max = 8 — or set an explicit token budget with `Budget(300)` (≈ budget/50 thoughts). Every level keeps its promise even on a bare "hello". The generated thoughts are grounded in what the cycle actually gathered — the topic, the current feeling, recalled memories, inferred facts, the private goal, new vocabulary — so the reply ARRIVES from thought instead of appearing out of nowhere (read them any time via `mind.Thinking.LastThoughts` or `response.Chain`). The `tone` flavors how the thoughts read: `"balanced"`, `"focused"`, `"playful"`, `"poetic"` or `"plain"`. And thinking has its own boolean switch, independent of the module: `WithThinking(enabled: false)` — or `mind.Config.Thinking.Enable(false)` at runtime — makes this model answer directly, with no thoughts generated at all.

```csharp
.WithThinking(t => t.At(ThinkingLevel.High)   // 4 thoughts per cycle
                     .WithTone("poetic"))     // ...in a soft, imagistic voice
.WithThinking(enabled: false)                 // or no thinking at all — your call
```

**Thinking ceiling — cap how far a model may ever think.** The level says where the mind thinks NOW; the ceiling says the highest it may EVER go. Five builder presets match the ladder:

| Preset | Allowed levels |
| --- | --- |
| `ThinkingCeiling.Only1` | Low |
| `ThinkingCeiling.Only2` | Low, Medium |
| `ThinkingCeiling.Only3` | Low, Medium, High |
| `ThinkingCeiling.Only4` | Low, Medium, High, Very High |
| `ThinkingCeiling.Only5` | Low, Medium, High, Very High, Max (default) |

Any level requested above the ceiling — at build time or later at runtime via `mind.Config.Thinking.At(...)` — is silently clamped down to the ceiling's top level, so a model capped at Only3 that is asked for Max still thinks, just at High. The cap travels with the model into exported files (`cognify.thinking.ceiling` GGUF metadata next to `cognify.thinking.enabled` / `cognify.thinking.level`), so serving stacks (llama.cpp / Unsloth) can read the model's thinking guarantee from the weights.

```csharp
.WithThinking(ceiling: ThinkingCeiling.Only1)                 // this model ONLY thinks at Low
.WithThinking(ceiling: ThinkingCeiling.Only2)                 // Low or Medium, never higher
.WithThinking(t => t.At(ThinkingLevel.Max)
                     .UpTo(ThinkingCeiling.Only3))            // Max requested → clamped to High
// runtime: mind.Config.Thinking.UpTo(ThinkingCeiling.Only4)
```

In the Yume demo the same controls exist as `--thinking-ceiling <only1..only5|1..5>` on the command line and `/ceiling <n>` in chat (`/level` reports when the ceiling clamped its request).

### Self-learning — give your model growth
A genuine **hybrid engine** — symbolic and neural halves blended adaptively, so every decision is a learned function rather than a fixed rule:

1. **MicroNet** — a real feed-forward neural network (input hashing -> tanh hidden layer -> sigmoid outputs) trained with backpropagation + gradient descent. It learns to predict reward and valence from stimulus features, generalizing across similar inputs.
2. **NeuralStyleEngine** — a second, *style-conditioned* network. Its input is `text features ⊕ style one-hot ⊕ mood (valence, arousal)` and it learns how rewarding each response STYLE is in the current context and mood — the closest a tiny net gets to model-like behavior.
3. **Reinforcement** — call `mind.Reinforce(reward)` (-1..1); a Q-table per response style is the symbolic half, with personality-driven epsilon-greedy exploration.
4. **Adaptive blending** — the final style score is `(1 − w)·symbolic + w·neural`. The weight `w` (0.25–0.80) drifts with the style net's prediction accuracy: learned behavior is trusted only as fast as it earns it. The chain of thought states the blend explicitly ("hybrid blend: 52% neural / 48% symbolic").
5. **Self-supervised** — every cycle trains valence prediction from its own emotional appraisal, so learning never stops even without explicit feedback.
6. **Experience replay** — `learning.Replay()` re-trains on stored experiences.

```csharp
mind.Process("tell me a joke");
mind.Reinforce(0.85);                        // good response
var stats = mind.Learning.Stats;             // training steps, avg reward, loss
double w = mind.Learning.NeuralWeight;       // how neural the mind has become
```

### Vocabulary growth — the mind adds words as it talks
Alongside the hybrid engine, a `Lexicon` listens to every message and **picks up words it has never heard before**. Fresh words are woven back into conversation — "I just picked up the word \"axolotl\" from you — what does it mean in your world?" — each new word spawning a brand-new sentence the mind has never said. While talking it also coins its own topic-flavored sayings ("The more I think about photosynthesis, the more I want to know.") into its phrase pool. Every conversation therefore leaves the mind with MORE things it can say — the structural opposite of wearing the same few sentences out. Words already woven into a reply are never asked about again, and everything (lexicon + spoken words + style net) survives save/restore.

```csharp
int words = mind.Lexicon.Size;               // words picked up so far
var recent = mind.Lexicon.RecentNewWords();  // newest picks
var picks  = mind.Lexicon.LastPicks;         // what THIS turn taught her
```

### Next-token prediction — a language model in the mind
The mind owns a real **trigram language model with stupid backoff** — its statistical next-token predictor. It observes EVERYTHING the mind reads and says: user turns, its own replies, taught facts, trained corpora (`Train`/`Teach`), even web search results. Two things come out of it:

1. **Fluency-aware deliberation** — each candidate reply gets a small fluency score (mean per-token backoff log-probability). It breaks close ties toward wordings that read like the language the mind has actually absorbed, without ever overpowering knowledge or the learned style preference.
2. **Literal predictions** — `mind.PredictNext("birds are")` returns ranked next tokens with probabilities, and the demo's `/predict` command shows them live.

Since the deep-learning engine, `PredictNext` is a **hybrid**: the n-gram model blends with the mind's transformer (below) — exact local statistics plus generalization to contexts the mind never saw verbatim. The model persists with the mind's snapshot (SQLite `langmodel` table) and merges on restore, so predictions keep improving across activations. Untrained minds score 0 — no opinion — so the fluency term is inert until the mind has absorbed something.

### Deep learning — a real neural stack inside the library
The library now ships a from-scratch deep-learning stack — no external ML runtime, no native dependencies, pure C# — in `Cognify.Learning`:

1. **`DeepNet`** — a genuinely DEEP feed-forward network: an arbitrary number of hidden layers with ReLU/GELU/Tanh activations, He initialization, inverted dropout, L2 weight decay, mini-batch **Adam** training and full backpropagation. Softmax + cross-entropy for classification, sigmoid/linear + MSE for regression. Where the older `MicroNet` is one hidden layer deep, `DeepNet` composes features through real depth — and it learns XOR, the classic non-linearly-separable sanity check, straight through its public API.
2. **`TransformerLM`** — a decoder-only **transformer language model**, the same architecture family as GPT: token + learned positional embeddings, pre-LayerNorm blocks of multi-head causal self-attention and GELU feed-forwards, a residual stream, a weight-tied output head, gradient clipping and temperature sampling — every piece hand-derived, including the backward pass.
3. **`HybridNextToken`** — the blend of both language models. The n-gram is exact about what it has literally seen; the transformer generalizes to unseen contexts; an untrained transformer degrades gracefully to pure n-gram, so the mind is never worse off for having a neural half.

The mind wires it in for you: `mind.NeuralLanguageModel` observes the same text stream as the n-gram model and **trains lazily** — automatically once ~600 new tokens have accumulated, or explicitly via `mind.TrainDeepModel(epochs)` (the demo's `/trainbrain`). Predictions and fluency scoring blend both models automatically.

```csharp
// her neural half, explicitly:
var report = mind.TrainDeepModel(epochs: 20);
Console.WriteLine(report);   // "20 epoch(s) on 512 window(s): loss 4.1 -> 1.9 (21,856 parameters, 96 words, 812 ms)"

var next = mind.PredictNext("birds are", 3);   // hybrid: n-gram + transformer
string completion = mind.NeuralLanguageModel.Generate("birds are", maxTokens: 8);
Console.WriteLine($"{mind.NeuralLanguageModel.ParameterCount:N0} parameters, loss {mind.NeuralLanguageModel.LastLoss:F3}");

// or use the pieces directly:
var net = new DeepNet(16, [32, 32], 4, DeepActivation.GELU, DeepOutput.Softmax, dropout: 0.1);
net.Train(X, classLabels, epochs: 100, learningRate: 0.01);
var lm = new TransformerLM(new TransformerConfig { Dim = 48, Layers = 3, ContextLen = 24 });
lm.Observe("some training text");
lm.Train(epochs: 30);
```

The transformer persists with the mind's snapshot (SQLite `deeplangmodel` table) — **weights, vocabulary, training buffer and architecture**, so a restored mind keeps her exact neural configuration. When trained, her transformer weights also export into GGUF/safetensors as `cognify.transformer.*` tensors. Deep-learning specifics are up to you: everything above is configurable through `TransformerConfig` (dim, heads, layers, context length, dropout, learning rate, batch size, gradient clipping).

### Self-improvement — she learns from how you react
A dedicated module that studies the conversation and gets better from it — no teacher required:

1. **Implied feedback** — the user's next message often grades the previous reply. "good job / thanks / that's right" and "no, that's wrong / stop saying that / you're repeating" are classified into reward, credited to the style and wording that was ACTUALLY spoken one turn earlier, and fed straight into the hybrid learning engine (Q-table + neural style net). Questions are never grades — "are you sure that's right?" does not count as approval.
2. **Wording reputations** — every reply template keeps an approval EMA. Disliked wordings are demoted (deliberation steers away from them), approved ones promoted.
3. **Conscious review** — `mind.SelfImprove()` (the demo's `/improve`) runs an explainable pass over the ledger and RETIRES trained sayings whose score stays clearly negative — she literally stops saying what never lands. Everything is reported: signals seen, wordings tracked, sayings retired.

```csharp
mind.Ask("what is a bird?");
mind.Ask("no, that is wrong");               // implied disapproval, graded on the last reply
var report = mind.SelfImprove();
Console.WriteLine(report.Summary);           // "Reviewed 2 turn(s) — 0 approval(s), 1 disapproval(s)..."
mind.SelfImprovement.RecordFeedback(reply, -1.0);  // explicit grading when the host knows better
```

The ledger persists with the mind (SQLite `selfimprovement` table). The module rides on the learning capability — `Without("learning")` removes it too.

### Emotions — give your model feelings
**37 emotions in total**: Plutchik's eight primaries (joy, trust, fear, surprise, sadness, disgust, anger, anticipation) PLUS the finer-grained feelings from Darwin's taxonomy (*The Expression of the Emotions in Man and Animals*) — love, devotion, tenderness, suffering, weeping, high spirits, low spirits, anxiety, grief, dejection, despair, hatred, disdain, contempt, guilt, pride, helplessness, patience, affirmation, negation, self-attention, shyness, modesty, blushing, reflection, meditation, ill-temper, sulkiness and determination. ("Joy, anger, disgust, surprise and fear were already in the set; the other 29 are new, and new members are appended so persisted minds restore cleanly.")

An OCC-inspired appraisal engine maps events to emotions via keyword desirability, expectation violation and reward signals — each Darwin emotion has its own trigger vocabulary ("despair/hopeless", "weeping/sobbing", "determined/persist", "blush/flushed", yes/no for affirmation/negation…). Emotions blend, decay over `Tick()`, and aggregate into slow-moving **mood** (valence + arousal). PascalCase names are always spoken humanized: `EmotionalState.Humanize(EmotionType.SelfAttention)` → "self attention".

**Social appraisal triggers (research-based).** Appraisal psychology (OCC) says ANGER comes from *blame* and *unfairness* — not from negativity alone — so insults aimed at the mind ("you are stupid", "i hate you") and unfairness raise anger far more than observed negativity, and dismissal ("shut up") stings as ill temper. Startle markers ("guess what", "did you know") prime surprise, and the user's shown interest ("tell me more", "interesting") is socially contagious — it feeds anticipation. A statement that **negates something the mind knows** ("birds are not animals" after "birds are animals") is a violated expectation — the textbook surprise trigger — and negations are never stored as garbled facts.

```csharp
var r = mind.Process("I love this wonderful day!");
Console.WriteLine(mind.EmotionalState.Describe());   // love (0.96), joy (0.62) | mood positive (+0.18)
mind.EmotionalState.Feel(EmotionType.Surprise, 0.5); // direct control too
```

### The feeling API — read the state, keep the chat natural
Feelings are FELT, not announced. By default a reply simply SOUNDS like the mood — an epistemic feeling (surprise, anticipation) comes out as natural curiosity ("Hm - that one caught me off guard", "Now I am curious about this one"), an unknown word gets an honest, natural reaction ("I just picked up the word \"quokka\" from you - what does it mean in your world?") instead of a bracketed introspection. The old mechanical style ("I am picking up a quiet, neutral tone here (my current read: anticipation)") is opt-in: `.WithExpression(e => e.Feelings(true))` brings it back.

The developer reads the live state programmatically instead — the `yume.Feeling.CurrentStatus` path:

```csharp
Console.WriteLine(yume.Feeling?.CurrentStatus);
// "feeling anticipation (intensity 0.15) - mood positive (+0.00)"
```

`FeelingSnapshot` exposes `Name`, `Intensity`, `Valence`, `Arousal`, `Mood` and `CurrentStatus` (null when the emotion module is off).

**Internals are invisible in chat until the developer shows them.** Everything internal still runs and stays reachable on the response object (`response.Chain`, `mind.Thinking.LastThoughts`, `response.Confidence`, style, cycle) — chat only shows the words. The switches:

```csharp
.WithExpression(e => e.Feelings(true)   // announce feelings mechanically again
                     .FollowUps(false)  // she only answers, never asks her own questions
                     .Thinking(true)    // hosts may surface the thought-generator trace
                     .Meta(true))       // hosts may surface the confidence/style/cycle line
```

**Follow-up questions are optional.** The `Pick(...)` asks ("Who are they to you? I do not know them yet - what are they like?", "Tell me more about them - I am curious who they are.") are on by default; `FollowUps(false)` turns every one of them off so the mind only answers. Her reaction to a brand-new word is NOT a follow-up — it is her own curiosity about language — and stays available even when the questions are off.

### Memory — give your model experience
Three stores with human-like dynamics:
- **Short-term** — fixed-capacity recency buffer.
- **Long-term** — important or repeatedly-recalled memories; strength decays unless rehearsed ("use it or lose it").
- **Episodic** — events tagged with the emotion felt at the time. Intense ones become **flashbulb memories** that never decay.

Recall combines keyword overlap, a spreading-activation **associative network** (co-occurring concepts get linked) and salience — judged by **content words**, so sharing only "about"/"the" doesn't make two sentences related. Three anti-repetition guards are built in: restating near-identical content **rehearses** the existing memory instead of stacking a copy, the current stimulus is never echoed back as a "memory" (the present is not the past), and only genuinely related memories surface — a fresh-but-unrelated message can't hijack recall.

```csharp
mind.Memory.Remember("the vault code is 4471", Cognify.Memory.MemoryKind.LongTerm, importance: 0.95);
var hits = mind.Memory.Recall("what was the vault code?");   // finds it
```

### Goals — give your model motivation
Four internal drives (**curiosity, social, achievement, calm**) grow over time. When a drive becomes pressing, the mind **autonomously spawns a goal** to satisfy it — intrinsic motivation, not just scripted tasks. Goals have priority, progress, deadlines and urgency-based focus selection.

**Goals are private motivation — part of the inner prompt, never spoken.** A goal like "make the user smile" is recorded in the chain of thought and biases HOW the mind talks (a prosocial goal makes warm styles more likely) — but the mind never recites it. You will never hear "every exchange like this also moves my goal '…' forward"; motivation is felt, not announced.

```csharp
mind.GoalSystem.AddGoal("finish-report", priority: 0.9, tags: new[] { "achievement" },
                         deadline: DateTimeOffset.UtcNow.AddDays(2));
for (int i = 0; i < 40; i++) mind.Tick();     // time passes -> drives grow -> goals spawn
```

### Dual-process thinking — think before speaking, unless emotion surges
The mind always *thinks before it speaks*: the chain of thought (observe → recall → feel → hypothesize → evaluate → deliberate → conclude) runs first. But per Zajonc's **affect primacy**, strong emotion can answer before thinking finishes — a dual-process impulse gate. When a fresh emotional surge crosses `EmotionOptions.ImpulseThreshold` (default 0.55), or blame/dismissal/contradiction trips the reflex, the mind blurts a short first-person reaction ("Hey - that actually stings.", "Wait - WHAT? I did not see that coming at all!") instead of a deliberated reply — and returns to normal conversation as the surge decays. Confident knowledge (arithmetic, taught facts, identity) is never blurted over: knowing the answer wins; feeling only colors how it is said.

**Epistemic questions (information-gap theory).** Surprise and interest open a knowledge gap, and the natural human move is to ASK: "What is X, exactly? You have me curious now." When a turn's fresh appraisal carries a surprise or interest spike — and the user is not already asking a question — the mind asks about the topic.

```csharp
mind.Process("tell me more about volcanoes, they seem really interesting");
// -> "Oh? Tell me more about volcanoes - I am genuinely interested."
mind.Process("Birds are animals.");
mind.Process("Birds are not animals.");
// -> "Wait - WHAT? I did not see that coming at all!"  (impulse: violated expectation)
```

### Personality — give your model character
Big Five (OCEAN) traits shape everything: **neuroticism** amplifies negative emotions, **extraversion** amplifies joy, **openness** drives exploration and curiosity, **conscientiousness** deepens deliberation and goal focus, **agreeableness** warms responses and softens anger. The same input *feels* different to different minds.

```csharp
.WithPersonality(p => p.Traits(openness: 0.9, conscientiousness: 0.7,
                               extraversion: 0.3, agreeableness: 0.8, neuroticism: 0.6))
```

### Gender — give your model an identity
A mind can know *who it is*: an identity word it uses for itself plus the pronoun set others use for it. Exactly two identity presets exist — `PronounSet.Female` (she/her) and `PronounSet.Male` (he/him) — and a mind can also be **genderless** (`PronounSet.Unspecified`, the default): it then speaks first-person "I" only and never volunteers pronouns. **they/them is not a setting for a mind's own pronouns** — that set (`PronounSet.UnknownPerson`) is reserved for one thing only: describing *other people whose gender the mind does not know*. When you talk about "a friend" or "someone", the mind naturally asks about *them* with they/them.

```csharp
var mind = MindBuilder.Create("Yume")
    .WithGender("girl", PronounSet.Female)          // identity + pronouns (she/her)
    .Build();

mind.DescribeSelf();     // "I am Yume (夢, 'dream') — ... I am a girl, and I go by she/her pronouns."
mind.Ask("who are you?"); // the identity answer includes the gender clause
mind.SetGender("boy", PronounSet.Male);     // grow a new sense of self at runtime
mind.SetPronouns(PronounSet.Female);        // or just the pronouns

WithGender("he/him");                       // preset label → pronouns only

// they/them is rejected for the mind itself — with an explanation:
mind.SetPronouns(PronounSet.UnknownPerson); // throws MindConfigurationException

// ...but used automatically for unknown people:
mind.Process("someone new joined my class today");
// → "Who are they to you? I do not know them yet - what are they like?"
```

Gender is part of "who the mind is", so it **survives save/restore** like personality traits (older snapshots that still carry a they/them self-pronoun set are restored genderless). The Yume demo speaks it out of the box (`gender <identity>` / `pronouns <set>` chat commands included — `they/them` explains itself when typed).

---

## Perception & action capabilities

### Vision — give your model eyes
`mind.See(...)` (or any `StimulusKind.Image`/`Video` stimulus) is analyzed by a pluggable `IVisionBackend`. The built-in `SimulatedVisionBackend` turns scene descriptions into structured `VisualScene`s: detected objects with confidences and frame positions, dominant colors, brightness, scene type (indoor/outdoor/portrait/document) and a natural-language caption. What the mind sees is remembered and felt.

```csharp
mind.See("a red apple on a wooden table");
var scene = mind.Vision.LastScene;
Console.WriteLine(scene.Caption);            // "I see 2 things, including a apple with red tones..."
Console.WriteLine(scene.Objects[0]);         // apple (91%)
Console.WriteLine(scene.SceneType);          // indoor
```

Bring your own real vision model by implementing `IVisionBackend` (it can decode actual pixels) and assigning `WithVision(v => v.Backend = myBackend)`.

### Audio — give your model ears
`mind.Hear(...)` (or any `StimulusKind.Audio` stimulus) is analyzed by a pluggable `IAudioBackend`. The built-in backend classifies clips as **speech / music / noise / tone**, estimates duration, loudness and tempo, and — for speech — produces a **transcription** that becomes the effective content of the cycle, so reasoning and memory operate on the spoken words.

```csharp
mind.Hear("please tell me about dogs");      // mind answers the question
mind.Hear("thunder crash boom");             // "A sound: thunder, crash, boom."
var features = mind.Hearing.LastHeard;       // category, tempo, loudness...
```

Loud sounds trigger a startle reaction (surprise). Swap in a real recognizer via `WithAudio(a => a.Backend = myWhisperBackend)`.

### Embeddings — give your model a sense of meaning
Every stimulus is embedded into a dense vector by `HashedNGramEmbedder` — a **real** embedding model built on signed feature hashing of word unigrams, bigrams and character trigrams, L2-normalized. The `EmbeddingStore` keeps a capacity-bounded semantic memory with cosine-similarity search, so the mind can later notice "this new thing resembles something I've seen" even with zero keyword overlap.

```csharp
double s = mind.SemanticSimilarity("the cat sat on the mat", "a cat sat on a mat");  // ~0.47
double d = mind.SemanticSimilarity("the cat sat on the mat", "quarterly revenue grew"); // ~0.0
var neighbors = mind.SemanticMemory.FindSimilar("cats and fish", topK: 3);
```

Vectors are persisted with the mind (semantic memory survives save/restore), and `IEmbeddingModel` is pluggable if you'd rather use a neural embedder.

### Multilingual — give your model languages
A two-stage, dependency-free `LanguageDetector`: **Unicode script analysis** (Han, Kana, Hangul, Cyrillic, Arabic, Devanagari — including Ukrainian vs Russian and Japanese-with-Kanji disambiguation) followed by **stopword + signature-word profiling** for Latin languages (English, Spanish, French, German, Portuguese, Italian, Dutch).

The `Phrasebook` translates common phrases between any pair of the supported languages (greetings, thanks, yes/no, politeness formulas) and accepts custom phrases. With `AutoTranslate`, foreign input is translated into the mind's preferred language before deeper processing — so a Spanish question still gets answered by the reasoning engine.

```csharp
var guess = mind.DetectLanguage("gracias");       // Spanish (latin, 95%)
mind.Ask("gracias");                              // "\"gracias\" means \"thank you\" in Spanish — got it!"
mind.Languages.Translator = myApiTranslator;      // optional real translation backend
```

### Tool use — give your model hands
The classic agent loop: a `ToolRegistry` of `ITool` instances (name, description, parameters, trigger phrases), natural-language argument extraction, invocation inside the cognitive cycle, and a tool result turned into the response. Three built-in tools ship with the mind:

- **clock** — current UTC date/time.
- **calculator** — a *real* tokenizer + recursive-descent expression parser (`+ - * / %`, parentheses, unary minus, `1,000`-style grouping). No `eval`, no dependencies.
- **text-stats** — word/sentence/character counts and reading time.

```csharp
mind.Ask("calculate (21 + 21) / 2");     // "(21 + 21) / 2 = 21"
mind.Ask("what time is it?");            // "It is currently 23:38:56 UTC on ..."
mind.Ask("2+2*3");                       // bare arithmetic works too: "2+2*3 = 8"
mind.Ask("Hello Yume, what is 2 + 2?");  // math found anywhere in the sentence: "2 + 2 = 4"
mind.Ask("what is the answer to 12 * 3?"); // "12 * 3 = 36"

mind.Tools.Registry.Register(new MyWeatherTool());   // your own tools
var result = mind.InvokeTool("weather", new Dictionary<string, string> { ["city"] = "Oslo" });
mind.ToolInvoked += (_, e) => Console.WriteLine($"used {e.ToolName}: {e.Result.Output}");
```

Arithmetic questions route to the calculator **only as a fallback** — see *Taught arithmetic* below. Plain questions like "what is a bird?" are never hijacked. Questions in general are never parsed into knowledge-base facts, so the mind won't claim to have "learned" from a question it was asked. Identity questions ("introduce yourself", "who are you") are answered from the mind's own name, description and current mood.

The module only fires when arguments can genuinely be extracted — "what is a bird?" is a question for reasoning, not a math request. And when the calculator *is* used because taught reasoning came up empty, the mind says so: "I have not been taught that one yet, so I used the calculator: 2 + 3 = 5."

### Web search — explicit and manual-only
The `web-search` tool gives the mind the internet, with a hard guard: **it is manual-only**. The tool carries no trigger phrases and is excluded from auto-invocation entirely — the mind NEVER goes online unprompted, no matter what the conversation says. The only paths are explicit ones:

```csharp
var mind = MindBuilder.Create("Yume")
    .WithWebSearch()                       // registers web-search (multi-engine HTTP chain)
    .Build();
var result = mind.SearchWeb("what is a quokka");   // the ONLY way she touches the internet
```

When the search succeeds she LEARNS from it: snippets become recallable long-term memories, clean "X is Y" sentences become knowledge-base facts (source "web"), the text feeds the next-token model, and the query's words count as known — she looked them up instead of asking what they mean. When it fails (offline, blocked, engine layout change), she says so cleanly: "I could not search the web for ...: web search could not complete (...). Web search needs an internet connection — try again, or teach me directly." The search retries once and enforces a timeout, so a flaky network turns into a message instead of a hang. Plug in your own backend (Bing, SerpAPI, a local index) via `Tools.WebSearchBackend = myIWebSearchBackend`.

### Web exploration — she reads INTO the pages
A search result is only a doorway: a snippet is 140 characters, and a brand-dominated SERP ("wikipedia flowers") can push the actual topic right out of view. So after a successful search she **opens the most query-relevant result and reads the page itself** (`Cognify.Tools.WebExplorer`) — scripts, styles and site chrome stripped, thousands of characters of readable text — and learns from what she read: memories, knowledge-base facts, both language models. If the first page turns out not to mention the topic at all, the next best result gets its chance; and when the page she lands on is a site's front door, she may **follow the in-page link that matches the query best** — the front door becomes the actual article. The chat report tells you exactly what she read:

```
Here is what I found on the web for "flowers":
 1. ... (www.ftd.com)
 ...
I also read into the most relevant page to learn more:
 • Flower - Wikipedia (en.wikipedia.org)
   A flower is the reproductive structure of flowering plants...
```

Reading one page further is part of the same consent — the user asked for the lookup, the page IS the lookup. It stays bounded (`WebExplorationOptions`), it never fires on its own, and you can shape it fluently:

```csharp
.WithWebSearch(t => t.Exploration(e => e.Explore(true)    // dig into results (default on)
                                     .Follow(true)        // follow a better in-page link (default on)
                                     .Pages(2)            // max pages per exploration (1..4)
                                     .Budget(4000)))      // readable-text budget per page
```

For "go read THAT site", there is a second manual-only tool: `mind.ExploreWeb(url)` opens any page directly (scheme-less addresses get `https://` for free), reads it, and learns the same way. Swap the fetcher (fixtures in tests, a caching proxy) via `Tools.WebPageFetcher = myIWebPageFetcher`; both the exploration settings and the fetcher ride along in the persisted config.

The demo exposes both as `/searchweb <query>` and `/explore <url>` slash commands — see below.

### Taught arithmetic — reasoning instead of a calculator
The mind learns arithmetic the way a child does: from what it is **taught**, and it reasons with that knowledge. The calculator is a crutch it does not need for anything you taught it — it only steps in when reasoning comes up empty (or when you explicitly ask for it).

1. **Teach** — any correct arithmetic statement becomes structured knowledge:

```csharp
mind.Teach("1 + 1 = 2");          // stored as the fact ("1+1", equals, "2")
mind.Teach("two plus two is four"); // number words are understood too
mind.Process("2 + 2 is 5");        // wrong results are refused — never learned
```

2. **Teach the number line** — a counting sequence becomes the mind's counting ladder. Counting along it is reasoning from teaching, so sums that stay on the line never need the calculator:

```csharp
mind.Process("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");
mind.Ask("what is 3 + 4?");
// "3 + 4 is 7 - I worked it out from what you taught me:
//  3 + 1 = 4 (counting on), then 4 + 1 = 5 (counting on), ..."
```

3. **Ask** — taught facts are answered directly, unknown ones are *derived*:

```csharp
mind.Teach("2 + 1 = 3");
mind.Teach("3 + 1 = 4");
mind.Ask("what is 2 + 2?");
// "2 + 2 is 4 - I worked it out from what you taught me:
//  2 + 1 = 3 (counting on), then 3 + 1 = 4 (counting on)."
```

Derivation strategies, all grounded in taught knowledge only: exact lookup, counting on via "n + 1" facts, larger taught jumps ("5 + 2 = 7"), counting along the taught number line, commutativity ("3 + 2" from a taught "2 + 3"), subtraction by working backwards, multiplication as repeated addition and division as repeated subtraction. Anything that leaves the taught line ("8 + 5" with only 0–9 taught) is admitted honestly — or, with tools attached, transparently peeked at the calculator.

4. **Word problems** — quantities plus transfer verbs are evaluated step by step. Natural quantities work: "no apples" is 0, "an apple"/"a apple"/"one apple" is 1, and the common typo "how may" is still read as a question. A setup that *starts from nothing* needs no teaching at all — "no apples and i buy an apple" reasons out to 1 through the zero identity (0 + b = b), because the problem itself states the zero:

```csharp
mind.Process("the system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. if i have no apples and i buy an apple. how many do i currently have?");
// "You start with 0 apples, and all of that works out to 1 apple
//  - buy an apple: 0 + 1 = 1 apple. I also picked up 1 new fact(s) from this."
```

If something was never taught and cannot be derived, the mind says so honestly (or, with tools attached, transparently peeks at the calculator — word problems included: "I have not been taught that one yet, so I used the calculator: You start with 4 apples, and all of that works out to 7 apples..."). Unlabeled questions ("yume what is 2 + 2") never leak into the fact parser, conditionals ("if i have no apples..."), existentials ("there is a 10 based number system") and "can be" copulas never become garbage facts — so "learned N new thing(s)" counts stay honest — and teaching confirmations are rotated at the TEMPLATE level: "learned 1 new thing" and "learned 3 new things" count as the same line, so the same sentence shape never comes out twice in a row.

## Training your mind — the train option

Activation gives a mind its personality; **training** gives it knowledge *and* vocabulary. `Train` is a bulk teacher that sorts every line it is given into four buckets:

| Line kind | Example | Becomes |
|---|---|---|
| Fact | `Birds are animals.` | knowledge-base fact (reasoning) |
| Dictionary line | `Joy means a bright feeling of gladness.` | `(joy, means, ...)` fact — recallable via "what does joy mean?" |
| Arithmetic / number line | `1 + 1 = 2.` / `0, 1, 2, ..., 9` | taught math (no calculator needed) |
| Plain conversational line | `I find the stars endlessly fascinating.` | **saying** — a sentence shape the mind can speak in its own voice |

Questions and conditional setups ("if i have no apples...") are requests, not knowledge or speech — they are skipped. Training never duplicates: re-feeding the same corpus adds nothing the second time.

Dictionary lines deserve a note: "X means Y." is parsed as a `(X, means, Y)` fact, so a trained word's meaning is *recallable knowledge*, not a forgotten string. After training the shipped dictionary, the mind answers all four phrasings:

```text
what does joy mean?              -> Joy means bright feeling of gladness deep inside.
what is the meaning of grief?    -> Grief means heavy sorrow felt after someone dear has gone.
define patience                  -> Patience means calm strength of being able to wait.
what is a fox?                   -> Fox means clever red rustler of night. ...
```

Words ending in "-ness" (kindness, happiness) keep their tail — the light singularizer no longer mangles them into "kindnes". And "what do you mean?" stays conversation: only real words hit the dictionary route.

```csharp
// One line, a whole block of text, a corpus file, or a stream of lines:
var report = mind.Train("I enjoy quiet mornings.");        // single utterance
var report = mind.TrainAll(File.ReadLines("corpus.txt"));  // bulk
var report = mind.TrainFile("corpus/yume-starter.txt");    // file (skips # comments)
// report: "43 line(s): 23 fact(s), 19 saying(s), 1 number line(s)."
```

Sayings feed straight into response composition as a dedicated `trained` style: whenever the mind has nothing stronger to answer with, it speaks a trained saying that is relevant to the conversation (word-overlap scoring), rotating through its vocabulary and skipping anything it said recently. Sayings are persisted in snapshots, so training survives save/restore — and the mind tells you its size:

```csharp
mind.Phrases.Count;   // 19 sayings it can speak in its own voice
mind.Phrases.All();   // the full vocabulary
```

The **Yume demo** ships with two corpora and training commands:

```bash
dotnet run --project demo/Cognify.Yume -- --train corpus/yume-starter.txt          # train at startup
dotnet run --project demo/Cognify.Yume -- --train corpus/words-and-meanings.txt   # + her dictionary (1,127 words)
# inside the chat:
#   train corpus/yume-starter.txt       # train from any corpus file
#   train corpus/words-and-meanings.txt # 1,127 word meanings she can recall
#   teach <sentence>                    # train one line mid-conversation
#   what does joy mean?                 # ask her dictionary (after training)
#   what is the meaning of grief?       # same lookup, other phrasing
#   define patience                     # dictionary, dictionary phrasing
#   sayings                             # show the mind's spoken vocabulary
#   vocab                               # words she picked up from talking with you
#   gender <identity>                   # who she is: girl, boy, she/her, he/him
#   pronouns <set>                      # change pronouns: she/her, he/him
#   who are you                         # her name, purpose, gender and mood
#
# SLASH COMMANDS (control, not conversation — she never runs these on her own):
#   /teach <sentence>                   # teach facts line-by-line
#   /train <file>                       # bulk-train a corpus file
#   /searchweb <query>                  # EXPLICIT web search — she only goes
#                                       # online when YOU type this, never on her own;
#                                       # she then READS INTO the best result page
#   /explore <url>                      # open any web page and read it (manual-only)
#   /improve                            # run a self-improvement review pass
#   /predict <text>                     # her next-token predictions after your
#                                       # words (hybrid: n-gram + transformer)
#   /trainbrain [epochs]                # deep-train her transformer language model
#   /brain                              # status of her transformer language model
#   /feeling                            # how she feels right now
#                                       # (yume.Feeling.CurrentStatus)
#   /feelings on|off                    # mechanical feeling announcements vs. natural
#   /followups on|off                   # let her ask her own follow-up questions
#   /thinking on|off                    # can this model think before speaking at all
#   /showthinking on|off                # show her thought trace in chat (hidden by default)
#   /level <low|medium|high|veryhigh|max>   thinking level (thoughts per cycle)
#   /ceiling <only1..only5|1..5>        hard cap: 1=Low only … 5=everything
#   /tone <balanced|focused|playful|poetic|plain>   thinking tone
#   /meta on|off                        # the confidence/style/cycle meta line
#   /vocab /sayings /gender <x> /pronouns <set> /who /help /exit
```

Startup switches for the demo (all optional — she never flips her own):
`--thinking <level>`, `--thinking-ceiling <only1..only5|1..5>`, `--thinking-tone <tone>`, `--no-thinking`,
`--show-feelings`, `--show-internals`, `--no-followups`.

`corpus/words-and-meanings.txt` is a ready-made training dictionary: **1,127 words with their meanings** across feelings & emotions, people & character, nature & weather, animals, food & drink, home & everyday things, mind & ideas, doing words, describing words, time & places and big ideas. Every line trains as a fact (zero sayings leak), so afterwards she can define any of those words back to you. The CognifyPy demo ships the identical file in `corpus/` — both minds learn exactly the same definitions.

### Why does my mind repeat the same sentences — and what can I do?

Partly, yes: it is a vocabulary problem. A mind's spoken variety comes from the sentence shapes it owns. Built-in templates are a fixed starter set, so an untrained mind circles the same few phrases — especially when it has nothing to answer with. Things that help, in order of impact:

1. **Train it.** Every saying in a corpus is one more sentence the mind can say. A mind trained on the starter corpus (19 sayings) has a whole extra response style fed by your lines; 50 sayings make repetition statistically rare. This is exactly what the train option is for.
1b. **Just talk.** Since the vocabulary-growth engine, the mind also grows ON ITS OWN while chatting: every new word you use gets picked up and woven back into a fresh question, and it coins its own topic-flavored sayings into its phrase pool as it goes. Training just makes it grow faster.
2. **Teach facts.** "learned N new things" confirmations appear when a line taught nothing. A knowledge-rich mind answers from knowledge instead of falling back to stock phrases — and DescribeEntity-style answers grow with every fact.
3. **Teach arithmetic and the number line.** Every taught sum is one less "I have not been taught that one yet, so I used the calculator" and one more reasoned answer.
4. **Keep chatting.** Memories give the composer recall-based styles ("that stirs a memory about ...") that vary naturally with the conversation.

The deliberation layer does its part on its own: styles used in the last few cycles lose favor, replies are penalized when they would repeat the previous reply **or its template** ("learned 1 new thing" and "learned 3 new things" are the same line), and trained sayings are filtered through the same template memory. And since the vocabulary-growth engine, the mind is no longer limited to what was trained in — it picks up words and coins new sentence shapes while you talk to it.

### Code — give your model a compiler's eye
Two skills in one module:

1. **Code generation** — `TemplateCodeBackend` emits clean scaffolds (class, function, loop, SQL query, endpoint) in C#, Python, JavaScript or SQL, with derived identifiers and doc comments. Requests like *"write a python class for a robot"* are recognized automatically; `ICodeBackend` is pluggable for real LLM generation.
2. **Code analysis** — `CodeAnalyzer` performs real static analysis: line/comment counts, bracket balance, nesting depth, a complexity score, frequent tokens and language detection (C#/Python/JS/SQL). Fenced markdown blocks and code-looking input are detected automatically.

```csharp
var response = mind.Ask("write a csharp class for a robot");
var analysis = mind.Ask("check this: ```python\ndef f(x):\n    return x\n```");
var metrics = mind.CodeEngine.Analyze("public class A { /* ... */ }");
Console.WriteLine(metrics.Summarize());   // csharp: 1 lines (1 code, 0 comments)...
```

### Video & audio generation — give your model a studio
The `MediaGenerationModule` recognizes generation intents ("generate an image of...", "create a video of... for 10 seconds", "say hello", "make a sound of...") and produces `MediaArtifact`s through a pluggable `IMediaBackend`:

- **Images & video** — deterministic specifications derived from a prompt hash: dimensions, fps, frame counts, sizes, seeds (same prompt = same artifact).
- **Speech & audio** — a *real, playable WAV file*: the prompt is synthesized into 16-bit PCM (RIFF header, tone bursts per syllable with pitch variation and envelopes, click-free fades). Save the bytes and any audio player can play them.

```csharp
var image = mind.GenerateMedia("a sunset over mountains", MediaType.Image);
var speech = mind.GenerateMedia("hello world", MediaType.Speech);
File.WriteAllBytes("hello.wav", speech.Data!);      // genuinely playable audio
Console.WriteLine(mind.MediaGeneration.Gallery.Count);
mind.MediaGenerated += (_, e) => Console.WriteLine($"created: {e.Artifact.Describe()}");
```

For real image/video diffusion or TTS, implement `IMediaBackend` and assign `WithMediaGeneration(g => g.Backend = myBackend)`.

### And more — extend it yourself
Implement `ICognitiveModule` (or derive from `CognitiveModuleBase`) and plug in anything: sensors, LLM backends, custom planners. Custom modules can produce responses too — earlier modules win; thinking defers to them.

```csharp
public class ParanoiaModule : CognitiveModuleBase
{
    public override string Name => "paranoia";
    public override int Order => 55;
    public override void Process(CognitiveContext ctx)
    {
        if (ctx.Stimulus.Content.Contains("who are you"))
            ctx.Response = new ResponseDraft("I could tell you, but then...", 0.9, 42, "paranoia");
    }
}

var mind = MindBuilder.Create("Secret").WithModule(new ParanoiaModule()).Build();
```

---

## Persistence

```csharp
// Configure at build time (also enables auto-save every 25 cycles)
.WithPersistence("aria.db", autoSaveEveryCycles: 25)

// Save / load the full mind state
await mind.SaveAsync();
await mind.LoadFromAsync("aria.db");

// Or restore a previously saved mind in one line
using var revived = MindBuilder.Restore("aria.db");
revived.Ask("what is a bird?");   // knowledge, memories, learning: all intact
```

**The database is created the moment the mind is built.** With `WithPersistence(path)` configured, `Build()` immediately writes the SQLite file — schema plus the mind's initial state — so the AI model uses its database from its very first cycle, and `MindBuilder.Restore(path)` works even before the first explicit `SaveAsync()`. Every subsequent cycle keeps using the same file (and auto-saves when configured).

The SQLite store persists: profile, config, emotion intensities, mood, all memories (with strength/recall counts), facts, rules, goals, drives, the trained neural network, the Q-table, the associative network **and the semantic (embedding) memory**.

## Export — GGUF, safetensors, tokenizer.json, llama.cpp configs

Export a trained mind into standard ML interchange formats with `Cognify.Export.MindExporter`. One call writes the whole bundle:

```csharp
using Cognify.Export;

var files = MindExporter.ExportBundle(mind, "export/yume");
// yume.gguf, yume.safetensors, tokenizer/tokenizer.json, train_config.json,
// finetune.sh, quantize.sh, training_data.txt, val_data.txt, config.json, README.md
```

Individual exports are available too:

```csharp
MindExporter.ExportGguf(mind, "yume.gguf");                 // GGUF v3 container
MindExporter.ExportSafetensors(mind, "yume.safetensors");   // Hugging Face tensors
MindExporter.ExportTokenizer(mind, "tokenizer/");           // HF WordPiece tokenizer
```

**Sharded safetensors checkpoints** — split the weights with the standard Hugging Face checkpoint naming (`model-00001-of-00062.safetensors`):

```csharp
var result = MindExporter.ExportSafetensors(mind, "yume.safetensors", 62);
// yume-00001-of-00006.safetensors ... yume-00006-of-00006.safetensors
// + yume.safetensors.index.json   (metadata.total_size + weight_map)

result.WrittenShards;   // shards actually written
result.WeightMap;       // tensor name -> shard file, exactly what transformers expects
```

Shards always hold whole tensors and are balanced by payload size, so the split is clamped to the tensor count — a Cognify mind has 6 tensors (plus the transformer's when her neural language model is trained), so asking for 62 writes all of them (`result.RequestedShards` / `WrittenShards` report both numbers). Split amount 1 writes one plain file with no index. `ExportOptions.SafetensorsSplit` applies the same sharding inside `ExportBundle`. Reassembly via the index is byte-for-byte identical to the unsplit file.

What each format carries:

| File | Contents |
|------|----------|
| `*.gguf` | **Spec-compliant GGUF v3**: llama.cpp-style metadata (`general.architecture`, `general.name`, personality traits, net dims, training stats) + F32 tensors `cognify.net.w1/b1/w2/b2` + RL `cognify.rl.q_values/q_counts` +, once the neural language model is trained, the transformer's `cognify.transformer.*` tensors (embeddings, attention projections, LayerNorms, FFN weights), all 32-byte aligned. Inspect with `gguf-dump`, requantize with `llama-quantize`. |
| `*.safetensors` | The same tensors in the Hugging Face format (padded JSON header, contiguous offsets, `__metadata__`). Load with `safetensors.numpy.load_file`. |
| `tokenizer/tokenizer.json` | HF WordPiece tokenizer built from the mind's **own corpus** (facts + memories), with `[PAD] [UNK] [BOS] [EOS] [MASK]` specials; loads via `tokenizers.Tokenizer.from_file`. Plus `vocab.json` and `special_tokens_map.json`. |
| `train_config.json` | llama.cpp finetune-style hyperparameters (context, batch, lr, epochs, Adam, LoRA, quantization plan). |
| `finetune.sh` / `quantize.sh` | Ready-to-run `llama-finetune` (LoRA) and `llama-quantize` → `Q4_K_M` / `Q5_K_M` / `Q8_0` pipelines. |
| `training_data.txt` / `val_data.txt` | The mind's text corpus, ~90/10 split, for fine-tuning. |

Tune the bundle with `ExportOptions` (model name, vocab size, quantization type, context/batch/epochs/lr/threads). Tune the mind itself via `MindBuilder` before exporting.

> **Honest scope note**: a Cognify mind is a cognitive architecture (micro-net + deep nets + a tiny internal transformer + reinforcement learning + knowledge graph), not an LLM-scale model. The GGUF/safetensors payloads are real, spec-compliant files containing the mind's learned weights and metadata — usable as training artifacts, LoRA seeds, weight snapshots or archives — but llama.cpp's shipped LLM graphs cannot run a Cognify mind directly.

## Events — react to the mind's life

```csharp
mind.CycleCompleted  += (_, e) => Console.WriteLine($"AI: {e.Response.Content}");
mind.EmotionChanged  += (_, e) => Console.WriteLine($"feels {e.Emotion} ({e.Intensity:0.00})");
mind.ThoughtProduced += (_, e) => Console.WriteLine($"  [{e.Step.Phase}] {e.Step.Text}");
mind.MemoryRecalled  += (_, e) => Console.WriteLine($"recalled {e.Memories.Count} memories");
mind.GoalUpdated     += (_, e) => Console.WriteLine($"goal {e.Goal.Name}: {e.Change}");
mind.Learned         += (_, e) => Console.WriteLine($"learned (avg reward {e.AverageReward:0.00})");
mind.ToolInvoked     += (_, e) => Console.WriteLine($"tool {e.ToolName} -> {e.Result.Output}");
mind.MediaGenerated  += (_, e) => Console.WriteLine($"created {e.Artifact.Type}: {e.Artifact.Describe()}");
```

---

## Project structure

```
Cognify/
├── Cognify.sln
├── src/Cognify/               # the library
│   ├── Core/                  # Mind kernel, MindBuilder, cognitive cycle
│   ├── Reasoning/             # facts, rules, inference engine
│   ├── Thinking/              # chain of thought, deliberation, inner monologue
│   ├── Learning/              # MicroNet (backprop), reinforcement, experiences
│   ├── Emotions/              # appraisal engine, emotional state, mood
│   ├── Memory/                # STM/LTM/episodic, associative network
│   ├── Goals/                 # drives, goals, planner
│   ├── Personality/           # Big Five traits
│   ├── Perception/            # vision: scenes, objects, colors, backends
│   ├── Audio/                 # hearing: classification, transcription, backends
│   ├── Embeddings/            # hashed n-gram embedder, vector store
│   ├── Multilingual/          # language detection, phrasebook, translation
│   ├── Reasoning/             # facts, inference engine, taught-arithmetic reasoner
│   ├── Tools/                 # tool registry, calculator parser, built-in tools
│   ├── Code/                  # code analyzer, template generation
│   ├── Generation/            # media artifacts, WAV synthesis, backends
│   ├── Export/                # GGUF v3 + safetensors + tokenizer.json + llama.cpp configs
│   └── Persistence/           # SQLite mind store (incl. semantic vectors)
├── demo/Cognify.Demo/         # interactive console demo (try this first!)
├── demo/Cognify.Yume/         # minimal "activate an AI named Yume" walkthrough
└── tests/Cognify.Tests/       # 345 xUnit tests covering all modules
```

## Getting started

```bash
# Requires the .NET 8 SDK
dotnet build Cognify.sln
dotnet test  tests/Cognify.Tests/Cognify.Tests.csproj     # 388 tests

# Interactive demo: teach, ask, show, speak, reward, watch it feel and learn
dotnet run --project demo/Cognify.Demo

# Or run the minimal "activation" walkthrough (activates an AI named Yume)
dotnet run --project demo/Cognify.Yume
```

In the demo chat you can use: `/teach <fact>`, `/feel <emotion> <0-1>`, `/reward <number>`, `/see <scene>`, `/hear <words>`, `/similar <text>`, `/generate <Image|Video|Speech|Audio> <prompt>`, `/state`, `/goals`, `/memories`, `/save`, `/exit`.

## Design notes

- **Determinism** — the network, deliberation and media generation are seeded, so identical builds behave identically; emotion and memory dynamics use real time.
- **Thread safety** — `Process` serializes access per mind; `ProcessAsync` offloads to the thread pool. Events are raised synchronously inside the cycle.
- **Pluggable backends, honest defaults** — every perception/generation capability ships with a dependency-free simulated backend that implements *real algorithms* (feature-hashed embeddings, a real expression parser, real WAV synthesis, script-based language detection). When you need production quality, implement the corresponding backend interface (`IVisionBackend`, `IAudioBackend`, `IEmbeddingModel`, `ITranslationBackend`, `ICodeBackend`, `IMediaBackend`) — nothing else changes.
- **No dependencies beyond `Microsoft.Data.Sqlite`** — the neural network, inference engine, embeddings, parsers and synthesizers are all implemented in the library itself.

## License

MIT.
