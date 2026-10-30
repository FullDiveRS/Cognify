using System.Text;
using System.Text.Json;

namespace Cognify.Export;

/// <summary>
/// Emits the usual llama.cpp training / fine-tuning / quantization artifact
/// set alongside the exported model:
/// <list type="bullet">
/// <item><c>train_config.json</c> — hyperparameter config in the llama.cpp
/// finetune style (context, lr, epochs, batch, LoRA, quantization target).</item>
/// <item><c>finetune.sh</c> — ready-to-run <c>llama-finetune</c> invocation (LoRA).</item>
/// <item><c>quantize.sh</c> — <c>llama-quantize</c> pipeline to Q4_K_M / Q5_K_M / Q8_0.</item>
/// <item><c>training_data.txt</c> / <c>val_data.txt</c> — text corpus harvested
/// from the mind (facts + memories), split ~90/10.</item>
/// </list>
/// </summary>
public static class LlamaCppConfigWriter
{
    /// <summary>Writes the full llama.cpp config set. Returns files written.</summary>
    public static List<string> Write(string directory, IReadOnlyList<string> corpus,
        string modelName, ExportOptions options)
    {
        Directory.CreateDirectory(directory);
        var written = new List<string>();
        var jsonOptions = new JsonSerializerOptions { WriteIndented = true };

        string modelGguf = $"{modelName}.gguf";
        string loraOut = $"{modelName}-lora.gguf";
        string quantized = $"{modelName}.{options.QuantizationType}.gguf";

        // ---- train_config.json (llama.cpp finetune-style keys) ------------
        var config = new Dictionary<string, object?>
        {
            // model
            ["model"] = modelGguf,
            ["lora_out"] = loraOut,
            ["lora_init"] = true,
            // data
            ["train_data"] = "training_data.txt",
            ["val_data"] = "val_data.txt",
            ["sample_start"] = "<|im_start|>user\n",
            ["sample_end"] = "<|im_end|>",
            // context / batching
            ["context_length"] = options.ContextLength,
            ["batch_size"] = options.BatchSize,
            ["gradient_clipping"] = 1.0,
            // optimization
            ["learning_rate"] = options.LearningRate,
            ["epochs"] = options.Epochs,
            ["use_adam"] = true,
            ["weight_decay"] = 0.0,
            ["warmup_batches"] = 20,
            // runtime
            ["threads"] = options.Threads,
            ["save_every"] = options.SaveEvery,
            ["checkpoint_out"] = $"{modelName}-checkpoint.gguf",
            // quantization plan
            ["quantization"] = new Dictionary<string, object?>
            {
                ["source"] = modelGguf,
                ["tool"] = "llama-quantize",
                ["types"] = new[] { "Q4_K_M", "Q5_K_M", "Q8_0" },
                ["target"] = quantized
            }
        };

        string configPath = Path.Combine(directory, "train_config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(config, jsonOptions));
        written.Add(configPath);

        // ---- finetune.sh --------------------------------------------------
        string finetune = $@"#!/usr/bin/env bash
# Fine-tune the exported {modelName} weights with llama.cpp (LoRA adapter).
# Prereq: llama.cpp built from source (https://github.com/ggml-org/llama.cpp).
set -euo pipefail

MODEL=""{modelGguf}""
TRAIN_DATA=""training_data.txt""
VAL_DATA=""val_data.txt""
LORA_OUT=""{loraOut}""

./llama-finetune \\
  --model ""$MODEL"" \\
  --train-data ""$TRAIN_DATA"" \\
  --val-data ""$VAL_DATA"" \\
  --lora-out ""$LORA_OUT"" \\
  --save-every {options.SaveEvery} \\
  --threads {options.Threads} \\
  --ctx {options.ContextLength} \\
  --batch {options.BatchSize} \\
  --lr {options.LearningRate.ToString(System.Globalization.CultureInfo.InvariantCulture)} \\
  --epochs {options.Epochs} \\
  --use-adam \\
  --warmup-batches 20

echo ""LoRA adapter written to $LORA_OUT""
echo ""Merge it into the base model with: llama-gguf --merge or keep it as an adapter.""
";
        string finetunePath = Path.Combine(directory, "finetune.sh");
        File.WriteAllText(finetunePath, finetune, new UTF8Encoding(false));
        written.Add(finetunePath);

        // ---- quantize.sh --------------------------------------------------
        string quantize = $@"#!/usr/bin/env bash
# Quantize {modelGguf} (F32) with llama.cpp.
set -euo pipefail

SRC=""{modelGguf}""

./llama-quantize ""$SRC"" ""{modelName}.Q4_K_M.gguf"" Q4_K_M
./llama-quantize ""$SRC"" ""{modelName}.Q5_K_M.gguf"" Q5_K_M
./llama-quantize ""$SRC"" ""{modelName}.Q8_0.gguf"" Q8_0

echo ""Quantized variants written. Typical choices: Q4_K_M (fast, small), Q8_0 (near-lossless).""
";
        string quantizePath = Path.Combine(directory, "quantize.sh");
        File.WriteAllText(quantizePath, quantize, new UTF8Encoding(false));
        written.Add(quantizePath);

        // ---- training / validation data (90/10 split) ---------------------
        var lines = corpus
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .Select(l => l.Replace('\n', ' ').Replace('\r', ' ').Trim())
            .Where(l => l.Length > 0)
            .Distinct()
            .ToList();

        int split = (int)(lines.Count * 0.9);
        string trainPath = Path.Combine(directory, "training_data.txt");
        File.WriteAllLines(trainPath, lines.Take(Math.Max(split, lines.Count)));
        written.Add(trainPath);

        string valPath = Path.Combine(directory, "val_data.txt");
        File.WriteAllLines(valPath, lines.Skip(split));
        written.Add(valPath);

        return written;
    }
}
