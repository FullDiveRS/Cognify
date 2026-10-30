namespace Cognify.Imaging;

using System.Text;

/// <summary>
/// The combined report of reading text from an image and training on it.
/// Pure data (no dependency on the mind), so the imaging layer stays
/// standalone.
/// </summary>
public sealed record ImageTrainingReport
{
    /// <summary>Creates the report.</summary>
    public ImageTrainingReport(string sourceFile, int width, int height,
        string ocrText, double ocrConfidence, int glyphCount,
        int linesLearned, int factsLearned, int sayingsAdded)
    {
        SourceFile = sourceFile;
        Width = width;
        Height = height;
        OcrText = ocrText;
        OcrConfidence = ocrConfidence;
        GlyphCount = glyphCount;
        LinesLearned = linesLearned;
        FactsLearned = factsLearned;
        SayingsAdded = sayingsAdded;
    }

    /// <summary>The image file that was read.</summary>
    public string SourceFile { get; }

    /// <summary>Image width in pixels.</summary>
    public int Width { get; }

    /// <summary>Image height in pixels.</summary>
    public int Height { get; }

    /// <summary>The text the mind saw in the image.</summary>
    public string OcrText { get; }

    /// <summary>Average glyph-match confidence 0..1.</summary>
    public double OcrConfidence { get; }

    /// <summary>How many glyphs were recognized.</summary>
    public int GlyphCount { get; }

    /// <summary>Training lines processed from the recognized text.</summary>
    public int LinesLearned { get; }

    /// <summary>Facts added by training on the recognized text.</summary>
    public int FactsLearned { get; }

    /// <summary>Sayings added by training on the recognized text.</summary>
    public int SayingsAdded { get; }

    /// <summary>True when nothing readable was found in the image.</summary>
    public bool IsEmpty => OcrText.Trim().Length == 0;

    /// <summary>One-line chat summary.</summary>
    public string Summarize()
    {
        if (IsEmpty)
            return $"I looked at {Path.GetFileName(SourceFile)} ({Width}x{Height}) but could not read any text in it.";
        var firstLine = OcrText.Split('\n')[0];
        return $"I read \"{Truncate(firstLine, 60)}\" from {Path.GetFileName(SourceFile)} " +
               $"({GlyphCount} characters at {OcrConfidence:P0} confidence) and learned " +
               $"{FactsLearned} fact(s) with {SayingsAdded} new saying(s).";
    }

    private static string Truncate(string text, int max) =>
        text.Length <= max ? text : text[..(max - 3)] + "...";
}

/// <summary>The result of reading text out of an image.</summary>
public sealed record OcrResult
{
    /// <summary>Creates the result.</summary>
    public OcrResult(string text, double meanConfidence, int glyphCount, int lineCount,
        int width, int height)
    {
        Text = text;
        MeanConfidence = meanConfidence;
        GlyphCount = glyphCount;
        LineCount = lineCount;
        Width = width;
        Height = height;
    }

    /// <summary>The recognized text (spaces between words, newlines between lines).</summary>
    public string Text { get; }

    /// <summary>Average per-glyph match confidence, 0..1.</summary>
    public double MeanConfidence { get; }

    /// <summary>How many glyphs were classified (including unknowns).</summary>
    public int GlyphCount { get; }

    /// <summary>How many text lines were detected.</summary>
    public int LineCount { get; }

    /// <summary>Source image width.</summary>
    public int Width { get; }

    /// <summary>Source image height.</summary>
    public int Height { get; }

    /// <summary>True when nothing readable was found.</summary>
    public bool IsEmpty => Text.Trim().Length == 0;

    /// <summary>One-line description used in chat.</summary>
    public string Describe() =>
        $"read {GlyphCount} character(s) over {LineCount} line(s) at {MeanConfidence:P0} confidence";
}

/// <summary>
/// A small dependency-free OCR engine for clean, machine-rendered text
/// images (screenshots of text, rendered cards, book pages with clear
/// type). Pipeline: grayscale → Otsu binarization (auto polarity for light
/// text on dark backgrounds) → projection-profile segmentation into
/// lines/words/glyphs → each glyph is scaled to the 5x7 template grid and
/// matched against the <see cref="GlyphAtlas"/> by ink overlap.
///
/// It is honest about its limits: handwriting, photos at an angle, exotic
/// fonts and unsupported characters come back as "?" with a low confidence
/// score rather than confident nonsense.
/// </summary>
public sealed class OcrEngine
{
    private const double UnknownThreshold = 0.55;
    private const double WordGapFraction = 0.5;   // of line height
    private const int MinGlyphHeight = 3;

    /// <summary>The shared engine instance (stateless — reuse freely).</summary>
    public static OcrEngine Shared { get; } = new();

    /// <summary>Reads text from an image buffer.</summary>
    public OcrResult Recognize(PixelBuffer image)
    {
        var binary = Binarize(image.ToGrayscale(), image.Width, image.Height, out int inkTotal);
        if (inkTotal == 0)
            return new OcrResult("", 0, 0, 0, image.Width, image.Height);

        // 1. Line bands from the row projection.
        var lineBands = FindBands(RowProfile(binary, image.Width, image.Height), minHeight: 2);
        var text = new StringBuilder();
        var confidences = new List<double>();
        int glyphCount = 0;

        for (int li = 0; li < lineBands.Count; li++)
        {
            var (top, bottom) = lineBands[li];
            int lineHeight = bottom - top + 1;

            // 2. Glyph column runs within the band.
            var runs = ColumnRuns(binary, image.Width, top, bottom);
            if (runs.Count == 0) continue;

            var lineText = new StringBuilder();
            int previousEnd = -1;
            foreach (var (left, right) in runs)
            {
                // Word gap: horizontal distance from the previous glyph.
                if (previousEnd >= 0 && left - previousEnd >= lineHeight * WordGapFraction)
                    lineText.Append(' ');
                previousEnd = right;

                // Tight bounding box within the run.
                var box = InkBox(binary, image.Width, image.Height, left, right, top, bottom);
                if (box is null) continue;
                var (x0, y0, x1, y1) = box.Value;
                int gw = x1 - x0 + 1, gh = y1 - y0 + 1;
                if (gh < MinGlyphHeight) continue; // speck

                var (ch, score) = Classify(binary, image.Width, x0, y0, gw, gh);
                if (ch == '\0') continue;           // ruled out as decoration
                lineText.Append(ch);
                confidences.Add(score);
                glyphCount++;
            }

            if (lineText.Length > 0)
            {
                if (text.Length > 0) text.Append('\n');
                text.Append(lineText.ToString().TrimEnd());
            }
        }

        var mean = confidences.Count > 0 ? confidences.Average() : 0.0;
        return new OcrResult(text.ToString(), mean, glyphCount, lineBands.Count,
            image.Width, image.Height);
    }

    // ------------------------------------------------------------------
    // Binarization
    // ------------------------------------------------------------------

    /// <summary>Otsu threshold; returns 1 for ink. Auto-flips polarity so
    /// light-on-dark text still reads as ink-minority.</summary>
    private static byte[] Binarize(byte[] gray, int width, int height, out int inkTotal)
    {
        var histogram = new int[256];
        foreach (var value in gray) histogram[value]++;
        int total = gray.Length;

        double sum = 0;
        for (int i = 0; i < 256; i++) sum += i * histogram[i];

        double sumB = 0;
        int wB = 0;
        double bestVariance = -1;
        int threshold = 127;
        for (int t = 0; t < 256; t++)
        {
            wB += histogram[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break;
            sumB += t * histogram[t];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double between = wB * (double)wF * (mB - mF) * (mB - mF);
            if (between > bestVariance)
            {
                bestVariance = between;
                threshold = t;
            }
        }

        var binary = new byte[total];
        int dark = 0;
        for (int i = 0; i < total; i++)
            if (gray[i] <= threshold) { binary[i] = 1; dark++; }

        // Text is normally the minority; if "dark" covers most of the image
        // the text is probably light — flip so ink stays the minority.
        if (dark > total * 0.45)
        {
            for (int i = 0; i < total; i++) binary[i] ^= 1;
            dark = total - dark;
        }
        inkTotal = dark;
        return binary;
    }

    // ------------------------------------------------------------------
    // Segmentation
    // ------------------------------------------------------------------

    private static int[] RowProfile(byte[] binary, int width, int height)
    {
        var profile = new int[height];
        for (int y = 0; y < height; y++)
        {
            int count = 0;
            var row = y * width;
            for (int x = 0; x < width; x++) count += binary[row + x];
            profile[y] = count;
        }
        return profile;
    }

    private static List<(int Top, int Bottom)> FindBands(int[] profile, int minHeight)
    {
        var bands = new List<(int, int)>();
        int start = -1;
        for (int y = 0; y <= profile.Length; y++)
        {
            bool ink = y < profile.Length && profile[y] > 0;
            if (ink && start < 0) start = y;
            if (!ink && start >= 0)
            {
                if (y - start >= minHeight) bands.Add((start, y - 1));
                start = -1;
            }
        }
        return bands;
    }

    private static List<(int Left, int Right)> ColumnRuns(byte[] binary, int width, int top, int bottom)
    {
        var runs = new List<(int, int)>();
        int start = -1;
        for (int x = 0; x <= width; x++)
        {
            bool ink = false;
            if (x < width)
            {
                for (int y = top; y <= bottom; y++)
                    if (binary[y * width + x] != 0) { ink = true; break; }
            }
            if (ink && start < 0) start = x;
            if (!ink && start >= 0)
            {
                runs.Add((start, x - 1));
                start = -1;
            }
        }
        return runs;
    }

    private static (int X0, int Y0, int X1, int Y1)? InkBox(byte[] binary, int width, int height,
        int left, int right, int top, int bottom)
    {
        int x0 = right + 1, x1 = left - 1, y0 = bottom + 1, y1 = top - 1;
        for (int y = top; y <= bottom; y++)
        {
            for (int x = left; x <= right; x++)
            {
                if (binary[y * width + x] == 0) continue;
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        }
        if (x1 < x0 || y1 < y0) return null;
        return (x0, y0, x1, y1);
    }

    // ------------------------------------------------------------------
    // Classification
    // ------------------------------------------------------------------

    private (char Character, double Score) Classify(byte[] binary, int width,
        int x0, int y0, int gw, int gh)
    {
        // Long thin horizontal strokes are rules/underlines, not glyphs;
        // anything wider than the atlas can express is decoration.
        if (gw > gh * 4 && gh <= MinGlyphHeight) return ('\0', 0);
        if (gh > gw * 12) return ('\0', 0);

        char best = '?';
        double bestScore = 0;
        foreach (var (character, pattern) in TemplateCache)
        {
            double score = MatchScore(binary, width, x0, y0, gw, gh, pattern);
            if (score > bestScore)
            {
                bestScore = score;
                best = character;
            }
        }
        return bestScore < UnknownThreshold ? ('?', bestScore) : (best, bestScore);
    }

    /// <summary>Nearest-neighbor scales the template (pre-cropped to its own
    /// ink box) to the glyph box and scores pixel accuracy (1.0 = perfect).</summary>
    private static double MatchScore(byte[] binary, int width,
        int x0, int y0, int gw, int gh, Template template)
    {
        int total = gw * gh, agree = 0;
        for (int y = 0; y < gh; y++)
        {
            int templateRow = y * template.Height / gh * template.Width;
            for (int x = 0; x < gw; x++)
            {
                int templateCol = x * template.Width / gw;
                int glyphInk = binary[(y0 + y) * width + x0 + x];
                int templateInk = template.Bits[templateRow + templateCol];
                if ((glyphInk != 0) == (templateInk != 0)) agree++;
            }
        }
        return total == 0 ? 0 : agree / (double)total;
    }

    /// <summary>A glyph template cropped to its own ink bounding box.</summary>
    private sealed record Template(byte[] Bits, int Width, int Height);

    private static readonly Dictionary<char, Template> TemplateCache = BuildTemplates();

    private static Dictionary<char, Template> BuildTemplates()
    {
        var templates = new Dictionary<char, Template>();
        foreach (var character in GlyphAtlas.KnownCharacters)
        {
            var rows = GlyphAtlas.Pattern(character)!;
            // Tight crop: find the ink bounding box inside the 5x7 grid.
            int top = 7, bottom = -1, left = 5, right = -1;
            for (int r = 0; r < 7; r++)
                for (int c = 0; c < 5; c++)
                    if (rows[r][c] == 'X')
                    {
                        top = Math.Min(top, r);
                        bottom = Math.Max(bottom, r);
                        left = Math.Min(left, c);
                        right = Math.Max(right, c);
                    }
            int height = bottom - top + 1;
            int glyphWidth = right - left + 1;
            var bits = new byte[height * glyphWidth];
            int i = 0;
            for (int r = top; r <= bottom; r++)
                for (int c = left; c <= right; c++)
                    bits[i++] = rows[r][c] == 'X' ? (byte)1 : (byte)0;
            templates[character] = new Template(bits, glyphWidth, height);
        }
        return templates;
    }
}
