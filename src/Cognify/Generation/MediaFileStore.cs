using Cognify.Imaging;

namespace Cognify.Generation;

/// <summary>
/// The mind's gallery on disk: saves media artifacts under a media
/// directory (default <c>cognify_media</c>) with human-friendly names
/// (<c>image-0007-sunset.png</c>), and lists what is there so she can talk
/// about her own creations.
/// </summary>
public sealed class MediaFileStore
{
    private readonly string _directory;
    private readonly object _gate = new();
    private int _counter;

    /// <summary>Creates a store rooted at the given directory.</summary>
    public MediaFileStore(string directory)
    {
        _directory = string.IsNullOrWhiteSpace(directory) ? "cognify_media" : directory;
    }

    /// <summary>The root directory of the store.</summary>
    public string Directory => _directory;

    /// <summary>Saves artifact bytes with the right extension; returns the full path.</summary>
    public string Save(string kind, string extension, byte[] data)
    {
        if (data is null || data.Length == 0)
            throw new ArgumentException("no bytes to save", nameof(data));
        lock (_gate)
        {
            _counter++;
            System.IO.Directory.CreateDirectory(_directory);
            var fileName = $"{kind}-{_counter:D4}{extension}";
            var path = Path.Combine(_directory, fileName);
            File.WriteAllBytes(path, data);
            return path;
        }
    }

    /// <summary>Writes an image buffer in the given format.</summary>
    public string SaveImage(string kind, PixelBuffer image, ImageFormat format)
    {
        var path = Path.Combine(_directory, $"{kind}-{Interlocked.Increment(ref _counter):D4}");
        var finalPath = path + ImageIO.ExtensionOf(format);
        System.IO.Directory.CreateDirectory(_directory);
        File.WriteAllBytes(finalPath, ImageIO.Encode(image, format));
        return finalPath;
    }

    /// <summary>Lists saved media files (name + size), newest first.</summary>
    public IReadOnlyList<string> ListFiles()
    {
        if (!System.IO.Directory.Exists(_directory)) return [];
        return System.IO.Directory.EnumerateFiles(_directory)
            .Select(f => (Name: Path.GetFileName(f), Size: new FileInfo(f).Length))
            .OrderByDescending(f => f.Name)
            .Select(f => $"{f.Name} ({f.Size / 1024.0:0} KB)")
            .ToList();
    }

    /// <summary>Resolves a user-supplied file name or path within the store.</summary>
    public string Resolve(string nameOrPath)
    {
        if (File.Exists(nameOrPath)) return nameOrPath;
        var candidate = Path.Combine(_directory, nameOrPath);
        if (File.Exists(candidate)) return candidate;
        throw new FileNotFoundException($"no media file named '{nameOrPath}' in '{System.IO.Path.GetFullPath(_directory)}'");
    }
}
