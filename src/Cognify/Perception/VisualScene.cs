namespace Cognify.Perception;

/// <summary>
/// One object the vision system believes it sees: a label, a confidence and
/// where it sits in the frame (relative coordinates 0..1).
/// </summary>
public sealed class DetectedObject
{
    public DetectedObject(string label, double confidence, double x, double y, double size)
    {
        Label = label;
        Confidence = Math.Clamp(confidence, 0, 1);
        X = Math.Clamp(x, 0, 1);
        Y = Math.Clamp(y, 0, 1);
        Size = Math.Clamp(size, 0, 1);
    }

    /// <summary>What the object appears to be (e.g. "apple").</summary>
    public string Label { get; }

    /// <summary>Detection confidence (0..1).</summary>
    public double Confidence { get; }

    /// <summary>Relative horizontal position in the frame (0 = left, 1 = right).</summary>
    public double X { get; }

    /// <summary>Relative vertical position in the frame (0 = top, 1 = bottom).</summary>
    public double Y { get; }

    /// <summary>Relative apparent size (0..1).</summary>
    public double Size { get; }

    public override string ToString() => $"{Label} ({Confidence:P0})";
}

/// <summary>
/// The result of analyzing one visual input: what is in the scene, where,
/// which colors dominate, how bright it is, and a natural-language caption.
/// </summary>
public sealed class VisualScene
{
    /// <summary>Natural-language caption of the scene.</summary>
    public string Caption { get; set; } = string.Empty;

    /// <summary>Objects detected in the scene, most confident first.</summary>
    public IReadOnlyList<DetectedObject> Objects { get; set; } = Array.Empty<DetectedObject>();

    /// <summary>Dominant colors as lowercase names (e.g. "red", "green").</summary>
    public IReadOnlyList<string> Colors { get; set; } = Array.Empty<string>();

    /// <summary>Average brightness of the scene (0 = pitch black, 1 = pure white).</summary>
    public double Brightness { get; set; } = 0.5;

    /// <summary>Coarse scene category (e.g. "indoor", "outdoor", "portrait", "document", "unknown").</summary>
    public string SceneType { get; set; } = "unknown";

    /// <summary>Declared or assumed frame width in pixels.</summary>
    public int Width { get; set; } = 640;

    /// <summary>Declared or assumed frame height in pixels.</summary>
    public int Height { get; set; } = 480;

    /// <summary>How visually rich the scene is (0..1) — many distinct objects and colors score higher.</summary>
    public double Richness => Math.Clamp(Objects.Count / 8.0 * 0.6 + Colors.Count / 4.0 * 0.4, 0, 1);

    /// <summary>One-line summary: caption plus object list.</summary>
    public string Summarize()
    {
        var objects = Objects.Count > 0 ? string.Join(", ", Objects.Take(5).Select(o => o.Label)) : "nothing distinctive";
        return $"{Caption} [objects: {objects}; colors: {string.Join("/", Colors.DefaultIfEmpty("n/a"))}; {SceneType}]";
    }
}

/// <summary>
/// The input to a vision backend. Because Cognify is dependency-free, the
/// default pipeline works on textual scene descriptions; a real backend
/// (ONNX model, cloud vision API...) can decode actual pixels instead.
/// </summary>
/// <param name="Description">Caption or textual description of what is visible.</param>
/// <param name="Metadata">Optional facts: width, height, format, brightness...</param>
public sealed record VisualInput(string Description, IReadOnlyDictionary<string, string>? Metadata = null)
{
    /// <summary>Builds a visual input from a generic stimulus.</summary>
    public static VisualInput From(Core.Stimulus stimulus) =>
        new(stimulus.Content, stimulus.Metadata);
}

/// <summary>
/// Pluggable vision backend. Swap the built-in heuristic analyzer for a real
/// model by implementing this interface and assigning it via
/// <c>WithVision(v => v.Backend = myBackend)</c>.
/// </summary>
public interface IVisionBackend : Core.IModuleBackend
{
    /// <summary>Analyzes one visual input and returns the perceived scene.</summary>
    VisualScene Analyze(VisualInput input);
}

/// <summary>
/// Deterministic, dependency-free vision backend. It "sees" by parsing the
/// textual description of the scene: known color words, known object nouns
/// and spatial prepositions produce a structured <see cref="VisualScene"/>.
/// Brightness and size estimates derive from the words used ("dark", "bright",
/// "huge", "tiny") and any metadata provided with the input.
/// </summary>
public sealed class SimulatedVisionBackend : IVisionBackend
{
    /// <summary>Object nouns the backend can recognize, with base detection confidence.</summary>
    public static readonly IReadOnlyDictionary<string, double> KnownObjects = new Dictionary<string, double>
    {
        ["person"] = 0.92, ["face"] = 0.88, ["man"] = 0.85, ["woman"] = 0.85, ["child"] = 0.82,
        ["dog"] = 0.90, ["cat"] = 0.90, ["bird"] = 0.84, ["horse"] = 0.83, ["fish"] = 0.78,
        ["car"] = 0.93, ["truck"] = 0.85, ["bicycle"] = 0.82, ["bus"] = 0.84, ["boat"] = 0.78,
        ["apple"] = 0.91, ["banana"] = 0.89, ["orange"] = 0.84, ["fruit"] = 0.80, ["flower"] = 0.86,
        ["tree"] = 0.90, ["plant"] = 0.83, ["leaf"] = 0.74, ["forest"] = 0.80, ["mountain"] = 0.86,
        ["table"] = 0.88, ["chair"] = 0.87, ["sofa"] = 0.84, ["bed"] = 0.85, ["desk"] = 0.82,
        ["book"] = 0.85, ["bottle"] = 0.83, ["cup"] = 0.81, ["phone"] = 0.86, ["laptop"] = 0.88,
        ["computer"] = 0.86, ["screen"] = 0.80, ["keyboard"] = 0.81, ["door"] = 0.80, ["window"] = 0.82,
        ["ball"] = 0.84, ["toy"] = 0.78, ["clock"] = 0.79, ["lamp"] = 0.80, ["key"] = 0.72,
        ["house"] = 0.87, ["building"] = 0.85, ["bridge"] = 0.80, ["road"] = 0.78, ["sky"] = 0.75,
        ["sun"] = 0.83, ["moon"] = 0.80, ["cloud"] = 0.76, ["rain"] = 0.72, ["snow"] = 0.74
    };

    private static readonly Dictionary<string, string> ColorWords = new()
    {
        ["red"] = "red", ["crimson"] = "red", ["scarlet"] = "red", ["blue"] = "blue", ["navy"] = "blue",
        ["cyan"] = "blue", ["azure"] = "blue", ["green"] = "green", ["emerald"] = "green", ["lime"] = "green",
        ["yellow"] = "yellow", ["gold"] = "yellow", ["golden"] = "yellow", ["orange"] = "orange",
        ["purple"] = "purple", ["violet"] = "purple", ["magenta"] = "purple", ["pink"] = "pink",
        ["brown"] = "brown", ["black"] = "black", ["white"] = "white",
        ["gray"] = "gray", ["grey"] = "gray", ["silver"] = "gray"
    };

    private static readonly HashSet<string> ColorOnly = new(ColorWords.Keys.Where(k => !KnownObjects.ContainsKey(k)));

    private static readonly string[] IndoorWords =
        { "table", "chair", "sofa", "bed", "desk", "laptop", "screen", "keyboard", "door", "window", "room", "kitchen", "office", "lamp" };

    private static readonly string[] OutdoorWords =
        { "sky", "tree", "forest", "mountain", "road", "car", "truck", "bus", "house", "building", "bridge", "beach", "garden", "field" };

    /// <inheritdoc />
    public VisualScene Analyze(VisualInput input)
    {
        var metadata = input.Metadata ?? new Dictionary<string, string>();
        var text = input.Description ?? string.Empty;
        var tokens = Tokenize(text);
        var tokenSet = new HashSet<string>(tokens);

        var scene = new VisualScene
        {
            Width = ReadInt(metadata, "width", 640),
            Height = ReadInt(metadata, "height", 480)
        };

        // ---- objects -------------------------------------------------
        var clauseHints = ExtractClauseHints(text);
        var objects = new List<DetectedObject>();
        foreach (var token in tokens)
        {
            if (!KnownObjects.TryGetValue(token, out var baseConfidence)) continue;
            double confidence = Math.Min(0.99, baseConfidence + (metadata.ContainsKey("format") ? 0.02 : 0));

            // Spatial hints give the object a place in the frame (per clause:
            // "a cat on the left, a dog on the right" places each correctly).
            var (hintX, hintY) = clauseHints.GetValueOrDefault(token, (0.5, 0.5));
            double x = hintX != 0.5 ? hintX
                : SpatialHint(tokens, token, "left", -1) ?? SpatialHint(tokens, token, "right", 1) ?? 0.5;
            double y = hintY != 0.5 ? hintY
                : SpatialHint(tokens, token, "top", -1) ?? SpatialHint(tokens, token, "bottom", 1) ?? 0.5;
            double size = tokenSet.Contains("huge") || tokenSet.Contains("giant") ? 0.9
                        : tokenSet.Contains("tiny") || tokenSet.Contains("small") ? 0.25
                        : 0.5;
            objects.Add(new DetectedObject(token, confidence, x, y, size));
        }

        objects = objects.GroupBy(o => o.Label).Select(g => g.First())
            .OrderByDescending(o => o.Confidence).ToList();

        // ---- colors --------------------------------------------------
        var colors = tokens.Where(t => ColorOnly.Contains(t))
            .Select(t => ColorWords[t]).Distinct().ToList();
        // "orange" is both a color and a fruit — treat it as a color only when no fruit reading exists.
        if (tokenSet.Contains("orange") && !colors.Contains("orange") && !objects.Any(o => o.Label == "orange"))
            colors.Add("orange");

        // ---- brightness ----------------------------------------------
        double brightness = 0.5;
        if (metadata.TryGetValue("brightness", out var raw) &&
            double.TryParse(raw, System.Globalization.CultureInfo.InvariantCulture, out var parsed))
            brightness = Math.Clamp(parsed, 0, 1);
        else if (tokenSet.Contains("dark") || tokenSet.Contains("night") || tokenSet.Contains("shadow"))
            brightness = 0.18;
        else if (tokenSet.Contains("bright") || tokenSet.Contains("sunny") || tokenSet.Contains("daylight"))
            brightness = 0.85;

        // ---- scene type ----------------------------------------------
        string sceneType = "unknown";
        bool indoor = tokens.Any(t => IndoorWords.Contains(t));
        bool outdoor = tokens.Any(t => OutdoorWords.Contains(t));
        if (tokenSet.Contains("face") || tokenSet.Contains("portrait") || (tokenSet.Contains("person") && objects.Count <= 2))
            sceneType = "portrait";
        else if (tokenSet.Contains("document") || tokenSet.Contains("text") || tokenSet.Contains("page"))
            sceneType = "document";
        else if (indoor && !outdoor) sceneType = "indoor";
        else if (outdoor && !indoor) sceneType = "outdoor";
        else if (indoor && outdoor) sceneType = "mixed";

        scene.Objects = objects.Take(12).ToList();
        scene.Colors = colors;
        scene.Brightness = brightness;
        scene.SceneType = sceneType;
        scene.Caption = Caption(objects, colors, brightness, sceneType);
        return scene;
    }

    private static string Caption(List<DetectedObject> objects, List<string> colors, double brightness, string sceneType)
    {
        if (objects.Count == 0 && colors.Count == 0)
            return brightness < 0.3 ? "A dark scene with nothing clearly visible."
                 : brightness > 0.75 ? "A bright scene with nothing clearly visible."
                 : "A scene without clearly recognizable objects.";

        var parts = new List<string>();
        if (objects.Count > 0)
            parts.Add(objects.Count == 1
                ? $"I see a {objects[0].Label}"
                : $"I see {objects.Count} things, including a {objects[0].Label}");
        if (colors.Count > 0)
            parts.Add($"with {string.Join(" and ", colors.Take(3))} tones");
        if (sceneType is "indoor" or "outdoor")
            parts.Add(sceneType == "indoor" ? "in an indoor setting" : "in an outdoor setting");

        var caption = string.Join(" ", parts);
        return caption.Length == 0 ? "A scene." : caption + ".";
    }

    private static double? SpatialHint(IReadOnlyList<string> tokens, string label, string hint, double value)
    {
        // Matches both "left of the cat" (hint before label) and "cat on the left"
        // (hint shortly after label) by scanning a window around the label.
        for (int i = 0; i < tokens.Count; i++)
        {
            if (tokens[i] != label) continue;
            for (int j = Math.Max(0, i - 3); j <= Math.Min(tokens.Count - 1, i + 3); j++)
            {
                if (tokens[j] == hint) return value < 0 ? 0.15 : 0.85;
            }
        }
        return null;
    }

    /// <summary>
    /// Splits the description into clauses (by comma/semicolon/period) and
    /// maps each object mentioned in a clause to that clause's spatial hints —
    /// so "a cat on the left, a dog on the right" resolves per clause rather
    /// than by global proximity.
    /// </summary>
    private static Dictionary<string, (double X, double Y)> ExtractClauseHints(string text)
    {
        var result = new Dictionary<string, (double, double)>();
        if (string.IsNullOrWhiteSpace(text)) return result;

        var clauses = text.Split(new[] { ',', ';', '.' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var clause in clauses)
        {
            var tokens = Tokenize(clause);
            var labels = tokens.Where(KnownObjects.ContainsKey).Distinct().ToList();
            if (labels.Count == 0) continue;

            double x = tokens.Contains("left") ? 0.15 : tokens.Contains("right") ? 0.85 : 0.5;
            double y = tokens.Contains("top") ? 0.15 : tokens.Contains("bottom") ? 0.85 : 0.5;
            foreach (var label in labels)
                result.TryAdd(label, (x, y));
        }
        return result;
    }

    private static string[] Tokenize(string text) =>
        text.ToLowerInvariant()
            .Split(new[] { ' ', ',', '.', ';', ':', '!', '?', '"', '\n', '\t', '(', ')' },
                   StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static int ReadInt(IReadOnlyDictionary<string, string> metadata, string key, int fallback) =>
        metadata.TryGetValue(key, out var raw) &&
        int.TryParse(raw, out var value) && value > 0 ? value : fallback;
}
