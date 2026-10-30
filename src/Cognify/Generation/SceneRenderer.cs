using Cognify.Imaging;
using System.Security.Cryptography;
using System.Text;

namespace Cognify.Generation;

/// <summary>
/// Procedurally renders tiny animated scenes from a prompt's mood and
/// motion type (bounce, wave, spiral, pulse, stars, drift). Pure C#, fully
/// deterministic per prompt — the pixel source for images, GIFs and AVI
/// videos the mind generates.
/// </summary>
public sealed class SceneRenderer
{
    /// <summary>The parsed mood ("happy" by default).</summary>
    public string Mood { get; }

    /// <summary>The parsed motion type.</summary>
    public string Motion { get; }

    private readonly (double X, double Y, double Radius, int Color)[] _orbs;
    private readonly Random _random;
    private readonly (byte R, byte G, byte B)[] _palette;
    private readonly (byte R, byte G, byte B) _background;

    /// <summary>Creates a renderer for a prompt (deterministic per prompt).</summary>
    public SceneRenderer(string prompt, int seed = 7)
    {
        var lower = prompt.ToLowerInvariant();
        Mood = MusicSynthesizer.ParseMood(lower);
        Motion =
            System.Text.RegularExpressions.Regex.IsMatch(lower, "bounc|ball|jump") ? "bounce" :
            System.Text.RegularExpressions.Regex.IsMatch(lower, "wave|ocean|sea|water") ? "wave" :
            System.Text.RegularExpressions.Regex.IsMatch(lower, "spiral|galaxy|vortex") ? "spiral" :
            System.Text.RegularExpressions.Regex.IsMatch(lower, "pulse|heartbeat|breath|ring") ? "pulse" :
            System.Text.RegularExpressions.Regex.IsMatch(lower, "star|night|space|sky") ? "stars" :
            "drift";
        _random = new Random(StableSeed(prompt));
        (_palette, _background) = MoodPalette(Mood);
        _orbs = Enumerable.Range(0, 3)
            .Select(i => (_random.NextDouble() * 0.6 + 0.2, _random.NextDouble() * 0.6 + 0.2,
                          _random.NextDouble() * 0.10 + 0.08, i % _palette.Length))
            .ToArray();
    }

    /// <summary>Renders one frame into an RGB buffer.</summary>
    public PixelBuffer RenderFrame(double t, int width = 96, int height = 72)
    {
        var frame = new PixelBuffer(width, height);
        for (int y = 0; y < height; y++)
        {
            double vy = y / (double)height;
            for (int x = 0; x < width; x++)
            {
                double vx = x / (double)width;
                var (r, g, b) = Pixel(vx, vy, t);
                frame.SetPixel(x, y, r, g, b);
            }
        }
        return frame;
    }

    /// <summary>Renders a larger still frame with the prompt stamped as text
    /// (a "text image" the mind's own OCR can read back).</summary>
    public PixelBuffer RenderStill(string caption, int width = 320, int height = 240)
    {
        var frame = RenderFrame(0.35, width, height);
        if (string.IsNullOrWhiteSpace(caption)) return frame;

        // Caption strip along the bottom with crisp atlas text.
        string text = string.Join("", caption.Take(28));
        var (textWidth, textHeight) = GlyphAtlas.Measure(text, 2);
        int stripHeight = textHeight + 10;
        for (int y = height - stripHeight; y < height; y++)
            for (int x = 0; x < Math.Min(width, textWidth + 12); x++)
                frame.SetPixel(x, y, 12, 12, 16);
        var stamp = GlyphAtlas.RenderText(text, 2, margin: 2, ink: 240, paper: 12);
        for (int y = 0; y < stamp.Height; y++)
            for (int x = 0; x < stamp.Width; x++)
            {
                var (r, g, b) = stamp.GetPixel(x, y);
                if (r > 128) // only the glyphs, not the stamp's paper margin
                {
                    int targetY = height - stripHeight + 4 + y;
                    int targetX = 4 + x;
                    if (targetX < width && targetY < height)
                        frame.SetPixel(targetX, targetY, r, g, b);
                }
            }
        return frame;
    }

    private (byte R, byte G, byte B) Pixel(double vx, double vy, double t)
    {
        double r = _background.R, g = _background.G, b = _background.B;
        switch (Motion)
        {
            case "bounce":
            {
                double bx = 0.15 + 0.7 * (0.5 + 0.5 * Math.Sin(2 * Math.PI * t));
                double by = 0.75 - 0.55 * Math.Abs(Math.Sin(2 * Math.PI * t * 2));
                double d = Math.Sqrt((vx - bx) * (vx - bx) + (vy - by) * (vy - by) * 0.75 * 0.75);
                if (d < 0.14) (r, g, b) = _palette[0];
                else if (d < 0.16) (r, g, b) = Lift(_background, 40);
                break;
            }
            case "wave":
            {
                double phase = Math.Sin(6 * Math.PI * vx + 2 * Math.PI * t);
                double surface = 0.5 + 0.12 * phase;
                if (vy > surface)
                {
                    double depth = (vy - surface) * 2;
                    var color = _palette[depth < 0.5 ? 1 % _palette.Length : 2 % _palette.Length];
                    r = color.R * (1 - 0.4 * depth);
                    g = color.G * (1 - 0.4 * depth);
                    b = Math.Min(255, color.B * (1 - 0.2 * depth));
                }
                break;
            }
            case "spiral":
            {
                double dx = vx - 0.5, dy = vy - 0.5;
                double angle = Math.Atan2(dy, dx);
                double dist = Math.Sqrt(dx * dx + dy * dy);
                double arm = Math.Sin(6 * angle + 10 * dist - 4 * Math.PI * t);
                if (arm > 0.4 && dist < 0.48)
                    (r, g, b) = arm > 0.75 ? _palette[0] : _palette[1 % _palette.Length];
                break;
            }
            case "pulse":
            {
                double d = Math.Sqrt((vx - 0.5) * (vx - 0.5) + (vy - 0.5) * (vy - 0.5));
                double ring = (d - t % 1.0 * 0.5) % 0.25;
                if (ring < 0.03) (r, g, b) = _palette[0];
                else if (d < 0.08) (r, g, b) = _palette[2 % _palette.Length];
                break;
            }
            case "stars":
            {
                int h = StableHash((int)(vx * 24), (int)(vy * 18)) & 0xFF;
                if (h > 244)
                {
                    double twinkle = 0.5 + 0.5 * Math.Sin(4 * Math.PI * t + h);
                    byte value = (byte)(120 + 135 * twinkle);
                    r = g = b = value;
                }
                else if (vy < 0.22) (r, g, b) = _palette[1 % _palette.Length];
                break;
            }
            default: // drift
                foreach (var (ox, oy, radius, colorIndex) in _orbs)
                {
                    double px = ox + 0.06 * Math.Sin(2 * Math.PI * t + ox * 6);
                    double py = oy + 0.05 * Math.Cos(2 * Math.PI * t + oy * 6);
                    if (Math.Sqrt((vx - px) * (vx - px) + (vy - py) * (vy - py) * 0.8 * 0.8) < radius)
                        (r, g, b) = _palette[colorIndex];
                }
                break;
        }
        return ((byte)Math.Clamp((int)r, 0, 255),
                (byte)Math.Clamp((int)g, 0, 255),
                (byte)Math.Clamp((int)b, 0, 255));
    }

    private static (byte, byte, byte) Lift((byte R, byte G, byte B) color, int amount) =>
        ((byte)Math.Min(255, color.R + amount),
         (byte)Math.Min(255, color.G + amount),
         (byte)Math.Min(255, color.B + amount));

    private static ((byte, byte, byte)[], (byte, byte, byte)) MoodPalette(string mood) => mood switch
    {
        "happy" => ([(255, 200, 60), (90, 200, 250), (255, 240, 200), (30, 144, 255)], (30, 60, 120)),
        "sad" => ([(70, 90, 160), (50, 60, 110), (120, 130, 180), (30, 40, 80)], (12, 14, 30)),
        "calm" => ([(120, 220, 180), (80, 160, 200), (200, 240, 220), (40, 100, 140)], (16, 40, 52)),
        "energetic" => ([(255, 90, 60), (255, 200, 40), (240, 60, 120), (60, 40, 160)], (40, 10, 40)),
        _ => ([(150, 60, 60), (90, 40, 90), (60, 60, 70), (200, 80, 40)], (10, 8, 14)),
    };

    private static int StableHash(int a, int b)
    {
        unchecked
        {
            uint hash = 2166136261;
            hash ^= (uint)a; hash *= 16777619;
            hash ^= (uint)b; hash *= 16777619;
            return (int)(hash & 0x7FFFFFFF);
        }
    }

    private static int StableSeed(string text)
    {
        var hash = MD5.HashData(Encoding.UTF8.GetBytes(text.Trim().ToLowerInvariant()));
        return BitConverter.ToInt32(hash, 0);
    }
}
