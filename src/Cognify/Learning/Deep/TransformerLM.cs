using System.Diagnostics;
using System.Text.Json;

namespace Cognify.Learning;

/// <summary>
/// Architecture + training knobs for <see cref="TransformerLM"/>. Defaults
/// describe a deliberately tiny GPT-style model (a few thousand parameters)
/// that trains on a CPU inside a cognitive cycle — deep enough to show real
/// attention behavior, small enough to never get in the user's way.
/// </summary>
public sealed record TransformerConfig
{
    /// <summary>Embedding / residual width (must be divisible by <see cref="Heads"/>).</summary>
    public int Dim { get; init; } = 32;

    /// <summary>Attention heads per block.</summary>
    public int Heads { get; init; } = 4;

    /// <summary>Transformer blocks stacked vertically (the model's depth).</summary>
    public int Layers { get; init; } = 2;

    /// <summary>Maximum context window in tokens (also the positional table size).</summary>
    public int ContextLen { get; init; } = 16;

    /// <summary>FFN hidden width = Dim × FfnMult (the classic 4).</summary>
    public int FfnMult { get; init; } = 4;

    /// <summary>Inverted dropout on attention probabilities and FFN activations.</summary>
    public double Dropout { get; init; } = 0.1;

    /// <summary>Adam learning rate.</summary>
    public double LearningRate { get; init; } = 3e-3;

    /// <summary>Random seed — the whole model is deterministic under a fixed seed.</summary>
    public int Seed { get; init; } = 42;

    /// <summary>Hard cap on vocabulary size (most words a mind meets in a niche
    /// conversation stay far below this; extra words fall back to &lt;unk&gt;).</summary>
    public int MaxVocab { get; init; } = 2048;

    /// <summary>Adam batch size in windows.</summary>
    public int BatchSize { get; init; } = 16;

    /// <summary>Maximum windows sampled per training epoch (bounds CPU time on
    /// large corpora; each epoch sees a fresh random subset).</summary>
    public int MaxWindowsPerEpoch { get; init; } = 512;

    /// <summary>Gradient clipping (global L2 norm) — keeps tiny-model training stable.</summary>
    public double GradClip { get; init; } = 1.0;
}

/// <summary>Everything one <see cref="TransformerLM.Train"/> pass produced.</summary>
public sealed record TransformerTrainingReport
{
    /// <summary>Epochs actually run.</summary>
    public int Epochs { get; init; }

    /// <summary>Training windows consumed in the final epoch.</summary>
    public int Windows { get; init; }

    /// <summary>Mean cross-entropy (nats/token) before the first update.</summary>
    public double InitialLoss { get; init; }

    /// <summary>Mean cross-entropy after the last update.</summary>
    public double FinalLoss { get; init; }

    /// <summary>Total trainable parameters in the model.</summary>
    public int Parameters { get; init; }

    /// <summary>Vocabulary size the model was trained with.</summary>
    public int Vocabulary { get; init; }

    /// <summary>Wall-clock duration of the whole pass.</summary>
    public TimeSpan Duration { get; init; }

    public override string ToString() =>
        $"{Epochs} epoch(s) on {Windows} window(s): loss {InitialLoss:F3} -> {FinalLoss:F3} " +
        $"({Parameters:N0} parameters, {Vocabulary} words, {Duration.TotalMilliseconds:F0} ms)";
}

/// <summary>
/// A decoder-only Transformer language model — the same architecture family
/// as GPT (token + learned positional embeddings, pre-LayerNorm blocks of
/// multi-head causal self-attention and GELU feed-forwards, residual stream,
/// weight-tied output head), implemented from scratch in pure C# with full
/// backpropagation and the <see cref="AdamOptimizer"/>. No external ML
/// runtime: the mind's "deep learning" runs anywhere .NET does.
///
/// Word-level vocabulary (same normalization as <see cref="NextTokenModel"/>)
/// so both models see the same language. The mind feeds it everything she
/// reads and says via <see cref="Observe"/>; training is lazy — either
/// automatic once enough new text has accumulated (<see cref="ShouldAutoTrain"/>)
/// or explicit (<see cref="Train"/>).
///
/// Inside the cognitive cycle the transformer blends with the n-gram model
/// (see <see cref="HybridNextToken"/>): the n-gram contributes exact local
/// statistics, the transformer contributes generalization — it can predict a
/// word it saw in a similar context even when the exact trigram is unseen.
/// </summary>
public sealed class TransformerLM
{
    private readonly TransformerConfig _cfg;
    private readonly object _gate = new();
    private readonly Random _rng;
    private readonly AdamOptimizer _adam = new();

    // Special tokens always occupy ids 0..3.
    private const string PadTok = "<pad>";
    private const string UnkTok = "<unk>";
    private const string BosTok = "<bos>";
    private const string EosTok = "<eos>";
    private const int PadId = 0, UnkId = 1, BosId = 2, EosId = 3;
    private const int SpecialCount = 4;

    // Vocabulary ---------------------------------------------------------
    private readonly List<string> _vocab = [PadTok, UnkTok, BosTok, EosTok];
    private readonly Dictionary<string, int> _index = new(StringComparer.Ordinal)
    {
        [PadTok] = PadId, [UnkTok] = UnkId, [BosTok] = BosId, [EosTok] = EosId,
    };

    // Parameters ----------------------------------------------------------
    private double[][] _wte = [];   // [V][D]  token embeddings (also the output head, tied)
    private double[][] _wpe = [];   // [T][D]  learned positional embeddings
    private BlockParams[] _blocks = [];

    private sealed class BlockParams
    {
        public double[] Ln1G = [], Ln1B = [];
        public double[][] Wq = [], Wk = [], Wv = [], Wo = [];
        public double[] Bq = [], Bk = [], Bv = [], Bo = [];
        public double[] Ln2G = [], Ln2B = [];
        public double[][] F1 = [], F2 = [];
        public double[] Bf1 = [], Bf2 = [];
    }

    private double[] _lnFG = [], _lnFB = [];   // final LayerNorm

    // Rolling training corpus ---------------------------------------------
    private readonly Queue<string> _buffer = new();
    private long _tokensObserved;
    private long _tokensSinceTrain;
    private int _epochsTrained;
    private double _lastLoss;

    /// <summary>Creates an untrained model with the given configuration.</summary>
    public TransformerLM(TransformerConfig? config = null)
    {
        _cfg = config ?? new TransformerConfig();
        if (_cfg.Dim <= 0 || _cfg.Heads <= 0 || _cfg.Layers <= 0 || _cfg.ContextLen <= 2)
            throw new ArgumentException("TransformerConfig dimensions must be positive (ContextLen > 2).");
        if (_cfg.Dim % _cfg.Heads != 0)
            throw new ArgumentException($"Dim ({_cfg.Dim}) must be divisible by Heads ({_cfg.Heads}).");
        _rng = new Random(_cfg.Seed);
        Allocate();
    }

    private void Allocate()
    {
        int d = _cfg.Dim, t = _cfg.ContextLen, f = d * _cfg.FfnMult;
        _wte = Empty(_vocab.Count, d);
        _wpe = Empty(t, d);
        _blocks = new BlockParams[_cfg.Layers];
        for (int l = 0; l < _cfg.Layers; l++)
        {
            var b = new BlockParams
            {
                Ln1G = Ones(d), Ln1B = Zeros(d),
                Wq = Empty(d, d), Bq = Zeros(d),
                Wk = Empty(d, d), Bk = Zeros(d),
                Wv = Empty(d, d), Bv = Zeros(d),
                Wo = Empty(d, d), Bo = Zeros(d),
                Ln2G = Ones(d), Ln2B = Zeros(d),
                F1 = Empty(f, d), Bf1 = Zeros(f),
                F2 = Empty(d, f), Bf2 = Zeros(d),
            };
            _blocks[l] = b;
        }
        _lnFG = Ones(d);
        _lnFB = Zeros(d);
        InitGaussianAll(std: 0.02);
    }

    // GPT-style init: N(0, 0.02) on learned matrices, ones/zeros on LayerNorm.
    private void InitGaussianAll(double std)
    {
        Fill(_wte, std);
        Fill(_wpe, std);
        foreach (var b in _blocks)
        {
            Fill(b.Wq, std); Fill(b.Wk, std); Fill(b.Wv, std); Fill(b.Wo, std);
            Fill(b.F1, std); Fill(b.F2, std);
        }
    }

    private void Fill(double[][] m, double std)
    {
        for (int r = 0; r < m.Length; r++)
            for (int c = 0; c < m[r].Length; c++)
                m[r][c] = Gaussian(std);
    }

    private double Gaussian(double std)
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * std;
    }

    private static double[][] Empty(int rows, int cols) =>
        Enumerable.Range(0, rows).Select(_ => new double[cols]).ToArray();

    private static double[] Zeros(int n) => new double[n];
    private static double[] Ones(int n) => Enumerable.Repeat(1.0, n).ToArray();

    // ------------------------------------------------------------------
    // Public surface
    // ------------------------------------------------------------------

    /// <summary>The model's architecture configuration.</summary>
    public TransformerConfig Config { get { lock (_gate) return _cfg; } }

    /// <summary>True once the model has completed at least one training pass.</summary>
    public bool IsTrained { get { lock (_gate) return _epochsTrained > 0; } }

    /// <summary>Distinct words the model knows (including 4 special tokens).</summary>
    public int VocabularySize { get { lock (_gate) return _vocab.Count; } }

    /// <summary>Total tokens the mind has fed the model so far.</summary>
    public long TokensObserved { get { lock (_gate) return _tokensObserved; } }

    /// <summary>How many sentences are waiting in the training buffer.</summary>
    public int BufferedSentences { get { lock (_gate) return _buffer.Count; } }

    /// <summary>Training epochs completed.</summary>
    public int EpochsTrained { get { lock (_gate) return _epochsTrained; } }

    /// <summary>Mean cross-entropy of the most recent training pass (nats/token).</summary>
    public double LastLoss { get { lock (_gate) return _lastLoss; } }

    /// <summary>Total trainable parameters (embeddings + blocks + head).</summary>
    public int ParameterCount
    {
        get
        {
            lock (_gate)
            {
                int total = _wte.Length * _cfg.Dim + _wpe.Length * _cfg.Dim;
                foreach (var b in _blocks)
                {
                    total += b.Ln1G.Length * 2 + b.Ln2G.Length * 2;
                    total += b.Wq.Length * _cfg.Dim * 4;   // wq, wk, wv, wo
                    total += b.Bq.Length * 4;              // bq, bk, bv, bo
                    total += b.F1.Length * _cfg.FfnMult * _cfg.Dim + b.F2.Length * _cfg.Dim;
                    total += b.Bf1.Length + b.Bf2.Length;
                }
                total += _lnFG.Length * 2;
                return total;
            }
        }
    }

    /// <summary>
    /// True when enough NEW language has accumulated since the last training
    /// pass to justify an automatic refresh (>= <c>AutoTrainTokens</c> new
    /// tokens and at least a handful of buffered sentences).
    /// </summary>
    public bool ShouldAutoTrain => _tokensSinceTrain >= AutoTrainTokens && _buffer.Count >= 8;

    /// <summary>New tokens that trigger a lazy automatic training pass.</summary>
    public const int AutoTrainTokens = 600;

    /// <summary>Epochs used by the automatic (lazy) training pass.</summary>
    public const int AutoTrainEpochs = 6;

    // ------------------------------------------------------------------
    // Observation — the buffer that training draws from
    // ------------------------------------------------------------------

    /// <summary>
    /// Feeds a piece of text to the model: sentences go into the rolling
    /// training buffer and their words join the vocabulary (new words get
    /// fresh embedding rows). Call for anything the mind reads or says.
    /// </summary>
    public void Observe(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        lock (_gate)
        {
            int added = 0;
            foreach (var sentence in SplitSentences(text))
            {
                _buffer.Enqueue(sentence);
                added++;
                var words = Tokenize(sentence);
                _tokensObserved += words.Length;
                _tokensSinceTrain += words.Length;
                foreach (var w in words) EnsureWord(w);
            }
            while (_buffer.Count > MaxBufferedSentences) _buffer.Dequeue();
        }
    }

    /// <summary>Maximum sentences kept in the rolling training buffer.</summary>
    public const int MaxBufferedSentences = 1000;

    private void EnsureWord(string word)
    {
        if (_index.ContainsKey(word)) return;
        if (_vocab.Count >= _cfg.MaxVocab) return; // beyond the cap: <unk> covers it

        // Grow the token embedding table and reset Adam moments (shapes changed).
        int id = _vocab.Count;
        _vocab.Add(word);
        _index[word] = id;
        var row = new double[_cfg.Dim];
        for (int c = 0; c < _cfg.Dim; c++) row[c] = Gaussian(0.02);
        _wte = [.. _wte, row];
        _adam.Reset();
    }

    // ------------------------------------------------------------------
    // Tokenization — identical normalization to NextTokenModel
    // ------------------------------------------------------------------

    private static IEnumerable<string> SplitSentences(string text)
    {
        foreach (var part in text.Split(new[] { '.', '!', '?', ';', '\n', '\r' },
                     StringSplitOptions.RemoveEmptyEntries))
        {
            var t = part.Trim();
            if (t.Length > 0) yield return t;
        }
    }

    private static string[] Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];
        var clean = new string(text.ToLowerInvariant()
            .Select(c => char.IsLetterOrDigit(c) ? c : ' ')
            .ToArray());
        return clean.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private int EncodeWord(string word) => _index.TryGetValue(word, out var id) ? id : UnkId;

    private int[] EncodeContext(string context)
    {
        var words = Tokenize(context ?? "");
        var ids = new List<int>();
        foreach (var w in words) ids.Add(EncodeWord(w));
        if (ids.Count == 0) ids.Add(BosId);
        if (ids.Count > _cfg.ContextLen) ids = ids.TakeLast(_cfg.ContextLen).ToList();
        return [.. ids];
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /// <summary>
    /// Runs an automatic (lazy) training pass when <see cref="ShouldAutoTrain"/>
    /// allows it; otherwise a no-op returning null. Called by the mind after
    /// observation bursts.
    /// </summary>
    public TransformerTrainingReport? AutoTrain()
    {
        lock (_gate)
            return ShouldAutoTrain ? TrainLocked(AutoTrainEpochs) : null;
    }

    /// <summary>
    /// Trains the transformer on everything in the observation buffer for
    /// <paramref name="epochs"/> epochs. Returns the training report.
    /// </summary>
    public TransformerTrainingReport Train(int epochs = 12)
    {
        lock (_gate)
            return TrainLocked(Math.Max(1, epochs));
    }

    private TransformerTrainingReport TrainLocked(int epochs)
    {
        var clock = Stopwatch.StartNew();

        // Build the token stream: sentences separated by explicit boundaries,
        // so n-grams (here: attention patterns) never cross sentence ends.
        var stream = new List<int>();
        foreach (var sentence in _buffer)
        {
            stream.Add(BosId);
            foreach (var w in Tokenize(sentence)) stream.Add(EncodeWord(w));
            stream.Add(EosId);
        }
        _tokensSinceTrain = 0;

        if (stream.Count < 3)
        {
            // Not enough language yet — do not perturb the weights on 2 tokens.
            return new TransformerTrainingReport
            {
                Epochs = 0, Windows = 0, InitialLoss = _lastLoss, FinalLoss = _lastLoss,
                Parameters = ParameterCount, Vocabulary = _vocab.Count, Duration = clock.Elapsed,
            };
        }

        // Training windows: EVERY contiguous subsequence of every sentence,
        // left-padded with <bos> when shorter than the context. Covering all
        // (start, end) pairs means every layout inference can produce —
        // sentence-initial prefixes, mid-sentence fragments like
        // "sat on the", long full windows, and the empty all-<bos> prompt —
        // appears in training exactly as <see cref="PadTokens"/> builds it.
        var windowsX = new List<int[]>();
        var windowsY = new List<int[]>();
        foreach (var sentence in _buffer)
        {
            var ids = new List<int> { BosId };
            foreach (var w in Tokenize(sentence)) ids.Add(EncodeWord(w));
            ids.Add(EosId);

            int len = ids.Count;
            for (int st = 0; st < len - 1; st++)
            {
                int maxE = Math.Min(len, st + 1 + _cfg.ContextLen);
                for (int e = st + 1; e < maxE; e++)
                {
                    // Context = ids[st..e-1] left-padded with <bos>; the
                    // target row is the context shifted left with the true
                    // next token appended at the last position.
                    var x = new int[_cfg.ContextLen];
                    int windowLen = e - st;
                    int pad = _cfg.ContextLen - windowLen;
                    for (int j = 0; j < pad; j++) x[j] = BosId;
                    for (int j = 0; j < windowLen; j++) x[pad + j] = ids[st + j];
                    windowsX.Add(x);

                    var y = new int[_cfg.ContextLen];
                    for (int j = 0; j < _cfg.ContextLen - 1; j++) y[j] = x[j + 1];
                    y[_cfg.ContextLen - 1] = ids[e];
                    windowsY.Add(y);
                }
            }
        }

        int epochWindowCount = Math.Min(windowsX.Count, _cfg.MaxWindowsPerEpoch);
        var order = Enumerable.Range(0, windowsX.Count).ToArray();
        var rng = new Random(_cfg.Seed + _epochsTrained * 7919);
        double initialLoss = double.NaN, finalLoss = double.NaN;

        for (int e = 0; e < epochs; e++)
        {
            // Fisher-Yates with the epoch rng, then take a bounded sample.
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            double lossSum = 0;
            int lossN = 0;

            for (int start = 0; start < epochWindowCount; start += _cfg.BatchSize)
            {
                int end = Math.Min(epochWindowCount, start + _cfg.BatchSize);
                double loss = TrainBatch(windowsX, windowsY, order, start, end);
                if (double.IsNaN(initialLoss)) initialLoss = loss;
                lossSum += loss * (end - start);
                lossN += end - start;
            }
            finalLoss = lossN > 0 ? lossSum / lossN : finalLoss;
            _lastLoss = finalLoss;
        }

        _epochsTrained += epochs;
        clock.Stop();
        return new TransformerTrainingReport
        {
            Epochs = epochs,
            Windows = epochWindowCount,
            InitialLoss = double.IsNaN(initialLoss) ? finalLoss : initialLoss,
            FinalLoss = finalLoss,
            Parameters = ParameterCount,
            Vocabulary = _vocab.Count,
            Duration = clock.Elapsed,
        };
    }

    // Trains one Adam batch of windows; returns the batch's mean loss.
    private double TrainBatch(List<int[]> xs, List<int[]> ys, int[] order, int start, int end)
    {
        int t = _cfg.ContextLen, v = _vocab.Count;
        var grads = NewGrads();
        double lossSum = 0;

        for (int s = start; s < end; s++)
        {
            var cache = ForwardWindow(xs[order[s]], training: true);
            var targets = ys[order[s]];

            // Cross-entropy + softmax over the whole vocabulary, at every position.
            var dLogits = new double[t][];
            for (int p = 0; p < t; p++)
            {
                var row = new double[v];
                double lossRow = 0;
                for (int i = 0; i < v; i++)
                {
                    row[i] = cache.Probs[p][i];
                    if (i == targets[p])
                    {
                        lossRow -= Math.Log(Math.Max(cache.Probs[p][i], 1e-12));
                        row[i] -= 1.0;
                    }
                }
                for (int i = 0; i < v; i++) row[i] /= t;   // mean over positions
                dLogits[p] = row;
                lossSum += lossRow;
            }
            lossSum /= t; // per-position mean for this window

            BackwardWindow(cache, targets, dLogits, grads);
        }

        int batch = end - start;
        ApplyGrads(grads, batch);
        return lossSum / batch;
    }

    // ------------------------------------------------------------------
    // Forward pass (one window)
    // ------------------------------------------------------------------

    private sealed class WindowCache
    {
        public int[] Tokens = [];
        public double[][] X0 = [];
        public List<LayerCache> Layers = [];
        public double[] LnFMean = [], LnFInvStd = [];
        public double[][] LnFHat = [], LnFOut = [];
        public double[][] Logits = [], Probs = [];
    }

    private sealed class LayerCache
    {
        public double[][] XIn = [];
        public double[] Mean1 = [], InvStd1 = [];
        public double[][] Ln1Hat = [], Ln1Out = [];
        public double[][] Q = [], K = [], V = [];
        public double[][][] Attn = [];        // [H][T][T] post-dropout probabilities (what the forward used)
        public double[][][] AttnRaw = [];     // [H][T][T] pre-dropout softmax output (for backward)
        public double[][][]? AttnMask;        // dropout mask (scaled), null when dropout is off
        public double[][] AttnConcat = [], AttnProj = [];
        public double[][] X2 = [];
        public double[] Mean2 = [], InvStd2 = [];
        public double[][] Ln2Hat = [], Ln2Out = [];
        public double[][] FPre = [], FGel = [];
        public double[][]? FMask;
        public double[][] FOut = [], XOut = [];
    }

    private WindowCache ForwardWindow(int[] tokens, bool training)
    {
        int d = _cfg.Dim, t = _cfg.ContextLen;
        var cache = new WindowCache { Tokens = tokens };

        // Embeddings: token + learned position.
        var x0 = new double[t][];
        for (int p = 0; p < t; p++)
        {
            var row = new double[d];
            int tok = tokens[p];
            for (int c = 0; c < d; c++) row[c] = _wte[tok][c] + _wpe[p][c];
            x0[p] = row;
        }
        cache.X0 = x0;

        double[][] a = x0;
        foreach (var block in _blocks)
        {
            var lc = new LayerCache { XIn = a };
            ForwardBlock(block, lc, training);
            a = lc.XOut;
            cache.Layers.Add(lc);
        }

        // Final LayerNorm.
        var (lnOut, lnHat, mean, invStd) = LayerNormForward(a, _lnFG, _lnFB);
        cache.LnFOut = lnOut; cache.LnFHat = lnHat; cache.LnFMean = mean; cache.LnFInvStd = invStd;

        // Weight-tied head: logits = hf @ Wte^T  →  [T][V]
        var logits = new double[t][];
        for (int p = 0; p < t; p++)
        {
            var row = new double[_vocab.Count];
            for (int i = 0; i < _vocab.Count; i++)
            {
                double sum = 0;
                var emb = _wte[i];
                for (int c = 0; c < d; c++) sum += lnOut[p][c] * emb[c];
                row[i] = sum;
            }
            logits[p] = row;
        }
        cache.Logits = logits;

        // Softmax (for scoring/sampling and as the cached training probability).
        var probs = new double[t][];
        for (int p = 0; p < t; p++) probs[p] = Softmax(logits[p]);
        cache.Probs = probs;

        return cache;
    }

    private void ForwardBlock(BlockParams b, LayerCache lc, bool training)
    {
        int d = _cfg.Dim, t = _cfg.ContextLen, h = _cfg.Heads, dh = d / h, f = d * _cfg.FfnMult;

        // ---- sub-layer 1: attention over LayerNorm(residual) ----
        var (ln1Out, ln1Hat, mean1, invStd1) = LayerNormForward(lc.XIn, b.Ln1G, b.Ln1B);
        lc.Ln1Out = ln1Out; lc.Ln1Hat = ln1Hat; lc.Mean1 = mean1; lc.InvStd1 = invStd1;

        var q = MatMulAdd(ln1Out, b.Wq, b.Bq);   // [T][D]
        var k = MatMulAdd(ln1Out, b.Wk, b.Bk);
        var v = MatMulAdd(ln1Out, b.Wv, b.Bv);
        lc.Q = q; lc.K = k; lc.V = v;

        // Causal multi-head attention.
        var attnRaw = new double[h][][];
        var attnUsed = new double[h][][];
        double[][][]? attnMask = null;
        if (training && _cfg.Dropout > 0) attnMask = new double[h][][];
        double scale = 1.0 / Math.Sqrt(dh);

        for (int head = 0; head < h; head++)
        {
            var scores = new double[t][];
            for (int i = 0; i < t; i++)
            {
                var row = new double[t];
                for (int j = 0; j <= i; j++)          // causal: only j <= i
                {
                    double sum = 0;
                    for (int c = 0; c < dh; c++)
                        sum += q[i][head * dh + c] * k[j][head * dh + c];
                    row[j] = sum * scale;
                }
                for (int j = i + 1; j < t; j++) row[j] = -1e9;  // masked future
                scores[i] = row;
            }
            var rawRows = new double[t][];
            for (int i = 0; i < t; i++) rawRows[i] = Softmax(scores[i]);
            attnRaw[head] = rawRows;

            if (attnMask is not null)
            {
                var maskRows = new double[t][];
                var usedRows = new double[t][];
                double keep = 1.0 - _cfg.Dropout;
                for (int i = 0; i < t; i++)
                {
                    var mrow = new double[t];
                    var urow = new double[t];
                    for (int j = 0; j < t; j++)
                    {
                        mrow[j] = _rng.NextDouble() < _cfg.Dropout ? 0.0 : 1.0 / keep;
                        urow[j] = rawRows[i][j] * mrow[j];
                    }
                    maskRows[i] = mrow;
                    usedRows[i] = urow;
                }
                attnMask[head] = maskRows;
                attnUsed[head] = usedRows;
            }
            else
            {
                attnUsed[head] = rawRows;
            }
        }
        lc.Attn = attnUsed; lc.AttnRaw = attnRaw; lc.AttnMask = attnMask;

        // Weighted sum of values → concat heads → [T][D].
        var concat = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[d];
            for (int head = 0; head < h; head++)
                for (int c = 0; c < dh; c++)
                {
                    double sum = 0;
                    for (int j = 0; j < t; j++) sum += attnUsed[head][i][j] * v[j][head * dh + c];
                    row[head * dh + c] = sum;
                }
            concat[i] = row;
        }
        lc.AttnConcat = concat;

        var proj = MatMulAdd(concat, b.Wo, b.Bo);
        lc.AttnProj = proj;

        // Residual 1.
        var x2 = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[d];
            for (int c = 0; c < d; c++) row[c] = lc.XIn[i][c] + proj[i][c];
            x2[i] = row;
        }
        lc.X2 = x2;

        // ---- sub-layer 2: feed-forward over LayerNorm(residual) ----
        var (ln2Out, ln2Hat, mean2, invStd2) = LayerNormForward(x2, b.Ln2G, b.Ln2B);
        lc.Ln2Out = ln2Out; lc.Ln2Hat = ln2Hat; lc.Mean2 = mean2; lc.InvStd2 = invStd2;

        var fPre = MatMulAdd(ln2Out, b.F1, b.Bf1);   // [T][F]
        lc.FPre = fPre;
        var fGel = new double[t][];
        double[][]? fMask = null;
        if (training && _cfg.Dropout > 0)
        {
            fMask = new double[t][];
            double keep = 1.0 - _cfg.Dropout;
            for (int i = 0; i < t; i++)
            {
                var grow = new double[f];
                var mrow = new double[f];
                for (int c = 0; c < f; c++)
                {
                    mrow[c] = _rng.NextDouble() < _cfg.Dropout ? 0.0 : 1.0 / keep;
                    grow[c] = DeepNet.Gelu(fPre[i][c]) * mrow[c];
                }
                fGel[i] = grow;
                fMask[i] = mrow;
            }
        }
        else
        {
            for (int i = 0; i < t; i++)
            {
                var grow = new double[f];
                for (int c = 0; c < f; c++) grow[c] = DeepNet.Gelu(fPre[i][c]);
                fGel[i] = grow;
            }
        }
        lc.FGel = fGel; lc.FMask = fMask;

        var fOut = MatMulAdd(fGel, b.F2, b.Bf2);
        lc.FOut = fOut;

        // Residual 2.
        var xOut = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[d];
            for (int c = 0; c < d; c++) row[c] = x2[i][c] + fOut[i][c];
            xOut[i] = row;
        }
        lc.XOut = xOut;
    }

    // ------------------------------------------------------------------
    // Backward pass (one window)
    // ------------------------------------------------------------------

    private sealed class GradSet
    {
        public double[][] GWte = [], GWpe = [];
        public double[] GLnFG = [], GLnFB = [];
        public BlockGrads[] Blocks = [];
    }

    private sealed class BlockGrads
    {
        public double[] GLn1G = [], GLn1B = [];
        public double[][] GWq = [], GWk = [], GWv = [], GWo = [];
        public double[] GBq = [], GBk = [], GBv = [], GBo = [];
        public double[] GLn2G = [], GLn2B = [];
        public double[][] GF1 = [], GF2 = [];
        public double[] GBf1 = [], GBf2 = [];
    }

    private GradSet NewGrads()
    {
        int d = _cfg.Dim, t = _cfg.ContextLen, f = d * _cfg.FfnMult;
        var gs = new GradSet
        {
            GWte = Empty(_wte.Length, d),
            GWpe = Empty(t, d),
            GLnFG = Zeros(d),
            GLnFB = Zeros(d),
            Blocks = new BlockGrads[_blocks.Length],
        };
        for (int l = 0; l < _blocks.Length; l++)
        {
            gs.Blocks[l] = new BlockGrads
            {
                GLn1G = Zeros(d), GLn1B = Zeros(d),
                GWq = Empty(d, d), GWk = Empty(d, d), GWv = Empty(d, d), GWo = Empty(d, d),
                GBq = Zeros(d), GBk = Zeros(d), GBv = Zeros(d), GBo = Zeros(d),
                GLn2G = Zeros(d), GLn2B = Zeros(d),
                GF1 = Empty(f, d), GF2 = Empty(d, f),
                GBf1 = Zeros(f), GBf2 = Zeros(d),
            };
        }
        return gs;
    }

    private void BackwardWindow(WindowCache cache, int[] targets, double[][] dLogits, GradSet g)
    {
        int d = _cfg.Dim, t = _cfg.ContextLen;

        // Weight-tied head: logits = LnFOut @ Wte^T.
        var dLnF = new double[t][];
        for (int p = 0; p < t; p++)
        {
            var row = new double[d];
            var drow = dLogits[p];
            for (int i = 0; i < _vocab.Count; i++)
            {
                if (drow[i] == 0) continue;
                var emb = _wte[i];
                for (int c = 0; c < d; c++)
                {
                    row[c] += drow[i] * emb[c];
                    g.GWte[i][c] += drow[i] * cache.LnFOut[p][c];
                }
            }
            dLnF[p] = row;
        }

        // Final LayerNorm backward.
        var dFinal = LayerNormBackward(dLnF, cache.LnFHat, _lnFG, cache.LnFInvStd, g.GLnFG, g.GLnFB);

        // Blocks, top-down; the gradient entering block l+1 is block l's output grad.
        var dUp = dFinal;
        for (int l = _blocks.Length - 1; l >= 0; l--)
            dUp = BackwardBlock(_blocks[l], cache.Layers[l], dUp, g.Blocks[l], g);

        // Embedding gradients: X0[p] = Wte[tok] + Wpe[p].
        for (int p = 0; p < t; p++)
        {
            int tok = cache.Tokens[p];
            for (int c = 0; c < d; c++)
            {
                g.GWte[tok][c] += dUp[p][c];
                g.GWpe[p][c] += dUp[p][c];
            }
        }
    }

    private double[][] BackwardBlock(BlockParams b, LayerCache lc, double[][] dOut, BlockGrads gb, GradSet g)
    {
        int d = _cfg.Dim, t = _cfg.ContextLen, h = _cfg.Heads, dh = d / h, f = d * _cfg.FfnMult;

        // Residual 2: XOut = X2 + FOut.
        var dX2 = new double[t][];
        var dFOut = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var r1 = new double[d];
            var r2 = new double[d];
            for (int c = 0; c < d; c++) { r1[c] = dOut[i][c]; r2[c] = dOut[i][c]; }
            dX2[i] = r1; dFOut[i] = r2;
        }

        // FFN backward: FOut = FGel @ F2^T + Bf2.
        var dFGel = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[f];
            for (int r = 0; r < f; r++)
            {
                double sum = 0;
                for (int c = 0; c < d; c++)
                {
                    sum += dFOut[i][c] * b.F2[c][r];
                    gb.GF2[c][r] += dFOut[i][c] * lc.FGel[i][r];
                }
                row[r] = sum;
            }
            dFGel[i] = row;
        }
        for (int i = 0; i < t; i++)
            for (int c = 0; c < d; c++)
                gb.GBf2[c] += dFOut[i][c];

        // Dropout mask, then GELU derivative.
        for (int i = 0; i < t; i++)
        {
            for (int r = 0; r < f; r++)
            {
                if (lc.FMask is not null) dFGel[i][r] *= lc.FMask[i][r];
                dFGel[i][r] *= GeluGrad(lc.FPre[i][r]);
            }
        }

        // F1 backward: FPre = Ln2Out @ F1^T + Bf1.
        var dLn2 = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[d];
            for (int c = 0; c < d; c++)
            {
                double sum = 0;
                for (int r = 0; r < f; r++)
                {
                    sum += dFGel[i][r] * b.F1[r][c];
                    gb.GF1[r][c] += dFGel[i][r] * lc.Ln2Out[i][c];
                }
                row[c] = sum;
            }
            dLn2[i] = row;
        }
        for (int i = 0; i < t; i++)
            for (int r = 0; r < f; r++)
                gb.GBf1[r] += dFGel[i][r];

        // LayerNorm 2 backward, plus residual 1.
        var dX2FromLn = LayerNormBackward(dLn2, lc.Ln2Hat, b.Ln2G, lc.InvStd2, gb.GLn2G, gb.GLn2B);
        for (int i = 0; i < t; i++)
            for (int c = 0; c < d; c++)
                dX2[i][c] += dX2FromLn[i][c];

        // Residual 1: X2 = XIn + AttnProj.
        var dXIn = dX2;              // gradient into the block input (plus attention path below)
        var dProj = dX2;             // gradient into Wo output — same array, adds are separable

        // Wo backward: AttnProj = Concat @ Wo^T + Bo.
        var dConcat = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[d];
            for (int c = 0; c < d; c++)
            {
                double sum = 0;
                for (int r = 0; r < d; r++)
                {
                    sum += dProj[i][r] * b.Wo[r][c];
                    gb.GWo[r][c] += dProj[i][r] * lc.AttnConcat[i][c];
                }
                row[c] = sum;
            }
            dConcat[i] = row;
        }
        for (int i = 0; i < t; i++)
            for (int c = 0; c < d; c++)
                gb.GBo[c] += dProj[i][c];

        // Attention backward, per head.
        var dQ = new double[t][];
        var dK = new double[t][];
        var dV = new double[t][];
        for (int i = 0; i < t; i++) { dQ[i] = new double[d]; dK[i] = new double[d]; dV[i] = new double[d]; }

        for (int head = 0; head < h; head++)
        {
            var probsUsed = lc.Attn[head];   // what the forward weighted V with
            var probsRaw = lc.AttnRaw[head]; // what softmax actually produced
            var mask = lc.AttnMask?[head];
            var dOh = new double[t][];   // gradient into this head's output slice
            for (int i = 0; i < t; i++)
            {
                var row = new double[dh];
                for (int c = 0; c < dh; c++) row[c] = dConcat[i][head * dh + c];
                dOh[i] = row;
            }

            // dVh = Aused^T dOh ; dAh = dOh Vh^T
            var dVh = new double[t][];
            var dAh = new double[t][];
            for (int j = 0; j < t; j++) { dVh[j] = new double[dh]; dAh[j] = new double[t]; }

            for (int i = 0; i < t; i++)
                for (int j = 0; j < t; j++)
                    for (int c = 0; c < dh; c++)
                        dAh[i][j] += dOh[j][c] * lc.V[j][head * dh + c];

            for (int j = 0; j < t; j++)
                for (int c = 0; c < dh; c++)
                {
                    double s = 0;
                    for (int i = 0; i < t; i++) s += probsUsed[i][j] * dOh[i][c];
                    dVh[j][c] = s;
                }

            // Backward through attention dropout (identity scaled by the mask),
            // THEN through softmax: dS = Araw ⊙ (dAraw − rowsum(dAraw ⊙ Araw)).
            if (mask is not null)
                for (int i = 0; i < t; i++)
                    for (int j = 0; j < t; j++)
                        dAh[i][j] *= mask[i][j];

            for (int i = 0; i < t; i++)
            {
                double dot = 0;
                for (int j = 0; j < t; j++) dot += dAh[i][j] * probsRaw[i][j];
                for (int j = 0; j < t; j++)
                    dAh[i][j] = probsRaw[i][j] * (dAh[i][j] - dot);
            }

            // dQh = dS Kh / sqrt(dh); dKh = dS^T Qh / sqrt(dh).
            double scale = 1.0 / Math.Sqrt(dh);
            for (int i = 0; i < t; i++)
                for (int j = 0; j < t; j++)
                {
                    double s = dAh[i][j] * scale;
                    if (s == 0) continue;
                    for (int c = 0; c < dh; c++)
                    {
                        dQ[i][head * dh + c] += s * lc.K[j][head * dh + c];
                        dK[j][head * dh + c] += s * lc.Q[i][head * dh + c];
                    }
                }

            for (int j = 0; j < t; j++)
                for (int c = 0; c < dh; c++)
                    dV[j][head * dh + c] += dVh[j][c];
        }

        // QKV projection backward: Q = Ln1Out @ Wq^T + Bq (same for K, V).
        var dLn1 = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[d];
            for (int c = 0; c < d; c++)
            {
                double sum = 0;
                for (int r = 0; r < d; r++)
                {
                    double dq = dQ[i][r], dk = dK[i][r], dv = dV[i][r];
                    if (dq != 0) { sum += dq * b.Wq[r][c]; gb.GWq[r][c] += dq * lc.Ln1Out[i][c]; }
                    if (dk != 0) { sum += dk * b.Wk[r][c]; gb.GWk[r][c] += dk * lc.Ln1Out[i][c]; }
                    if (dv != 0) { sum += dv * b.Wv[r][c]; gb.GWv[r][c] += dv * lc.Ln1Out[i][c]; }
                }
                row[c] = sum;
            }
            dLn1[i] = row;
        }
        for (int i = 0; i < t; i++)
            for (int c = 0; c < d; c++)
            {
                gb.GBq[c] += dQ[i][c];
                gb.GBk[c] += dK[i][c];
                gb.GBv[c] += dV[i][c];
            }

        // LayerNorm 1 backward adds into the block-input gradient.
        var dXInFromLn = LayerNormBackward(dLn1, lc.Ln1Hat, b.Ln1G, lc.InvStd1, gb.GLn1G, gb.GLn1B);
        for (int i = 0; i < t; i++)
            for (int c = 0; c < d; c++)
                dXIn[i][c] += dXInFromLn[i][c];

        return dXIn;
    }

    // Adam step on every tensor: divide accumulated grads by batch size,
    // clip the global norm, then let the optimizer do its thing.
    private void ApplyGrads(GradSet g, int batch)
    {
        double inv = 1.0 / batch;

        void Scale(double[][] m) { for (int r = 0; r < m.Length; r++) for (int c = 0; c < m[r].Length; c++) m[r][c] *= inv; }
        void ScaleV(double[] v) { for (int i = 0; i < v.Length; i++) v[i] *= inv; }

        Scale(g.GWte); Scale(g.GWpe); ScaleV(g.GLnFG); ScaleV(g.GLnFB);
        foreach (var gb in g.Blocks)
        {
            Scale(gb.GWq); Scale(gb.GWk); Scale(gb.GWv); Scale(gb.GWo);
            Scale(gb.GF1); Scale(gb.GF2);
            ScaleV(gb.GLn1G); ScaleV(gb.GLn1B); ScaleV(gb.GBq); ScaleV(gb.GBk);
            ScaleV(gb.GBv); ScaleV(gb.GBo); ScaleV(gb.GLn2G); ScaleV(gb.GLn2B);
            ScaleV(gb.GBf1); ScaleV(gb.GBf2);
        }

        // Global gradient clipping.
        double sq = 0;
        void Accum(double[][] m) { foreach (var row in m) foreach (var x in row) sq += x * x; }
        void AccumV(double[] v) { foreach (var x in v) sq += x * x; }
        Accum(g.GWte); Accum(g.GWpe); AccumV(g.GLnFG); AccumV(g.GLnFB);
        foreach (var gb in g.Blocks)
        {
            Accum(gb.GWq); Accum(gb.GWk); Accum(gb.GWv); Accum(gb.GWo);
            Accum(gb.GF1); Accum(gb.GF2);
            AccumV(gb.GLn1G); AccumV(gb.GLn1B); AccumV(gb.GBq); AccumV(gb.GBk);
            AccumV(gb.GBv); AccumV(gb.GBo); AccumV(gb.GLn2G); AccumV(gb.GLn2B);
            AccumV(gb.GBf1); AccumV(gb.GBf2);
        }
        double norm = Math.Sqrt(sq);
        double clip = _cfg.GradClip;
        if (norm > clip && norm > 0)
        {
            double s = clip / norm;
            void Mul(double[][] m) { for (int r = 0; r < m.Length; r++) for (int c = 0; c < m[r].Length; c++) m[r][c] *= s; }
            void MulV(double[] v) { for (int i = 0; i < v.Length; i++) v[i] *= s; }
            Mul(g.GWte); Mul(g.GWpe); MulV(g.GLnFG); MulV(g.GLnFB);
            foreach (var gb in g.Blocks)
            {
                Mul(gb.GWq); Mul(gb.GWk); Mul(gb.GWv); Mul(gb.GWo);
                Mul(gb.GF1); Mul(gb.GF2);
                MulV(gb.GLn1G); MulV(gb.GLn1B); MulV(gb.GBq); MulV(gb.GBk);
                MulV(gb.GBv); MulV(gb.GBo); MulV(gb.GLn2G); MulV(gb.GLn2B);
                MulV(gb.GBf1); MulV(gb.GBf2);
            }
        }

        double lr = _cfg.LearningRate;
        _adam.Step("wte", _wte, g.GWte, lr);
        _adam.Step("wpe", _wpe, g.GWpe, lr);
        _adam.StepVec("lnf.g", _lnFG, g.GLnFG, lr);
        _adam.StepVec("lnf.b", _lnFB, g.GLnFB, lr);
        for (int l = 0; l < _blocks.Length; l++)
        {
            var b = _blocks[l];
            var gb = g.Blocks[l];
            string p = $"l{l}.";
            _adam.StepVec(p + "ln1.g", b.Ln1G, gb.GLn1G, lr);
            _adam.StepVec(p + "ln1.b", b.Ln1B, gb.GLn1B, lr);
            _adam.Step(p + "wq", b.Wq, gb.GWq, lr);
            _adam.StepVec(p + "bq", b.Bq, gb.GBq, lr);
            _adam.Step(p + "wk", b.Wk, gb.GWk, lr);
            _adam.StepVec(p + "bk", b.Bk, gb.GBk, lr);
            _adam.Step(p + "wv", b.Wv, gb.GWv, lr);
            _adam.StepVec(p + "bv", b.Bv, gb.GBv, lr);
            _adam.Step(p + "wo", b.Wo, gb.GWo, lr);
            _adam.StepVec(p + "bo", b.Bo, gb.GBo, lr);
            _adam.StepVec(p + "ln2.g", b.Ln2G, gb.GLn2G, lr);
            _adam.StepVec(p + "ln2.b", b.Ln2B, gb.GLn2B, lr);
            _adam.Step(p + "f1", b.F1, gb.GF1, lr);
            _adam.StepVec(p + "bf1", b.Bf1, gb.GBf1, lr);
            _adam.Step(p + "f2", b.F2, gb.GF2, lr);
            _adam.StepVec(p + "bf2", b.Bf2, gb.GBf2, lr);
        }
    }

    // ------------------------------------------------------------------
    // Small numeric helpers
    // ------------------------------------------------------------------

    private static double[] Softmax(double[] x)
    {
        double max = x.Length > 0 ? x.Max() : 0;
        double sum = 0;
        var outP = new double[x.Length];
        for (int i = 0; i < x.Length; i++) { outP[i] = Math.Exp(x[i] - max); sum += outP[i]; }
        if (sum <= 0) sum = 1;
        for (int i = 0; i < x.Length; i++) outP[i] /= sum;
        return outP;
    }

    private static (double[][] Out, double[][] Hat, double[] Mean, double[] InvStd) LayerNormForward(
        double[][] x, double[] g, double[] bta)
    {
        int t = x.Length, d = x[0].Length;
        var outY = new double[t][];
        var hat = new double[t][];
        var mean = new double[t];
        var invStd = new double[t];
        for (int i = 0; i < t; i++)
        {
            double m = 0;
            for (int c = 0; c < d; c++) m += x[i][c];
            m /= d;
            double var = 0;
            for (int c = 0; c < d; c++) var += (x[i][c] - m) * (x[i][c] - m);
            var /= d;
            double inv = 1.0 / Math.Sqrt(var + 1e-5);
            mean[i] = m; invStd[i] = inv;
            var yrow = new double[d];
            var hrow = new double[d];
            for (int c = 0; c < d; c++)
            {
                hrow[c] = (x[i][c] - m) * inv;
                yrow[c] = hrow[c] * g[c] + bta[c];
            }
            hat[i] = hrow; outY[i] = yrow;
        }
        return (outY, hat, mean, invStd);
    }

    // LayerNorm backward: accumulates dG/dB and returns the input gradient.
    private static double[][] LayerNormBackward(double[][] dy, double[][] hat, double[] g,
        double[] invStd, double[] dG, double[] dB)
    {
        int t = dy.Length, d = dy[0].Length;
        var dx = new double[t][];
        for (int i = 0; i < t; i++)
        {
            double meanDxhat = 0, meanDxhatHat = 0;
            var dxhat = new double[d];
            for (int c = 0; c < d; c++)
            {
                dxhat[c] = dy[i][c] * g[c];
                dG[c] += dy[i][c] * hat[i][c];
                dB[c] += dy[i][c];
                meanDxhat += dxhat[c];
                meanDxhatHat += dxhat[c] * hat[i][c];
            }
            meanDxhat /= d;
            meanDxhatHat /= d;
            var row = new double[d];
            for (int c = 0; c < d; c++)
                row[c] = invStd[i] * (dxhat[c] - meanDxhat - hat[i][c] * meanDxhatHat);
            dx[i] = row;
        }
        return dx;
    }

    private static double[][] MatMulAdd(double[][] a, double[][] w, double[] bias)
    {
        // out[i][r] = bias[r] + sum_c a[i][c] * w[r][c]
        int t = a.Length, outDim = w.Length, inDim = a[0].Length;
        var outM = new double[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new double[outDim];
            for (int r = 0; r < outDim; r++)
            {
                double sum = bias[r];
                var wrow = w[r];
                for (int c = 0; c < inDim; c++) sum += a[i][c] * wrow[c];
                row[r] = sum;
            }
            outM[i] = row;
        }
        return outM;
    }

    private static double GeluGrad(double x)
    {
        const double c = 0.7978845608028654;
        double inner = c * (x + 0.044715 * x * x * x);
        double tanh = Math.Tanh(inner);
        double dt = (1 - tanh * tanh) * c * (1 + 3 * 0.044715 * x * x);
        return 0.5 * (1 + tanh) + 0.5 * x * dt;
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /// <summary>
    /// Next-token predictions after the given context, best first — the
    /// neural half of the mind's next-token prediction. Untrained models
    /// return an empty list (the n-gram model still has an answer).
    /// </summary>
    public IReadOnlyList<TokenPrediction> PredictNext(string context, int count = 5)
    {
        if (count < 1) count = 1;
        lock (_gate)
        {
            if (_epochsTrained == 0) return [];

            var tokens = EncodeContext(context);
            var cache = ForwardWindow(PadTokens(tokens), training: false);
            var probs = Softmax(cache.Logits[t - 1]);

            var candidates = new List<TokenPrediction>();
            for (int i = SpecialCount; i < _vocab.Count; i++)
                if (probs[i] > 1e-9)
                    candidates.Add(new TokenPrediction(_vocab[i], probs[i]));

            var top = candidates.OrderByDescending(p => p.Probability).Take(count).ToList();
            double raw = top.Sum(p => p.Probability);
            if (raw > 0)
                top = top.Select(p => new TokenPrediction(p.Token, p.Probability / raw)).ToList();
            return top;
        }
    }

    /// <summary>
    /// Mean per-token log-probability of a text under the model — the neural
    /// fluency score (same convention as <see cref="NextTokenModel.Score"/>:
    /// 0 means "no opinion", closer to 0 from below means more natural).
    /// </summary>
    public double Score(string text)
    {
        lock (_gate)
        {
            if (_epochsTrained == 0) return 0;
            var words = Tokenize(text ?? "");
            if (words.Length == 0) return 0;

            var ids = new List<int> { BosId };
            foreach (var w in words) ids.Add(EncodeWord(w));
            ids.Add(EosId);

            double sum = 0;
            int n = 0;
            for (int pos = 1; pos < ids.Count; pos++)
            {
                int start = Math.Max(0, pos - _cfg.ContextLen);
                var window = ids.GetRange(start, pos - start);
                var cache = ForwardWindow(PadTokens([.. window]), training: false);
                var probs = Softmax(cache.Logits[t - 1]);
                sum += Math.Log(Math.Max(probs[ids[pos]], 1e-12));
                n++;
            }
            return n == 0 ? 0 : sum / n;
        }
    }

    /// <summary>
    /// Free-running generation: continues the prompt token by token
    /// (temperature sampling) until <paramref name="maxTokens"/> or the end
    /// of a sentence. This is the model speaking in its own learned voice.
    /// </summary>
    public string Generate(string prompt, int maxTokens = 12, double temperature = 0.8)
    {
        lock (_gate)
        {
            if (_epochsTrained == 0) return "";
            var ids = EncodeContext(prompt).ToList();
            var produced = new List<string>();

            for (int step = 0; step < Math.Max(1, maxTokens); step++)
            {
                var window = ids.Count > _cfg.ContextLen ? ids.TakeLast(_cfg.ContextLen).ToArray() : [.. ids];
                var cache = ForwardWindow(PadTokens([.. window]), training: false);
                var logits = cache.Logits[t - 1];

                // Temperature-scaled sampling over non-special tokens.
                double max = logits.Skip(SpecialCount).Max();
                var probs = new double[_vocab.Count];
                double sum = 0;
                for (int i = SpecialCount; i < _vocab.Count; i++)
                {
                    probs[i] = Math.Exp((logits[i] - max) / Math.Max(0.05, temperature));
                    sum += probs[i];
                }
                double r = _rng.NextDouble() * sum;
                int chosen = SpecialCount;
                double acc = 0;
                for (int i = SpecialCount; i < _vocab.Count; i++)
                {
                    acc += probs[i];
                    if (r <= acc) { chosen = i; break; }
                }
                if (chosen == EosId) break;
                var word = chosen < _vocab.Count ? _vocab[chosen] : UnkTok;
                produced.Add(word);
                ids.Add(chosen);
            }
            return string.Join(" ", produced);
        }
    }

    private int t => _cfg.ContextLen;

    // Pads short contexts on the left with <bos> so the window is always
    // exactly ContextLen long (positional embeddings are position-bound).
    private int[] PadTokens(int[] tokens)
    {
        int tlen = _cfg.ContextLen;
        if (tokens.Length >= tlen) return tokens[^tlen..];
        var padded = new int[tlen];
        int padCount = tlen - tokens.Length;
        // Left-padded positions use the BOS embedding rather than PAD so the
        // first real token keeps a sensible "beginning" signal.
        Array.Fill(padded, BosId);
        for (int i = 0; i < tokens.Length; i++)
            padded[padCount + i] = tokens[i];
        return padded;
    }

    // ------------------------------------------------------------------
    // Persistence + tensor export
    // ------------------------------------------------------------------

    private sealed class LmDto
    {
        public int Dim { get; set; }
        public int Heads { get; set; }
        public int Layers { get; set; }
        public int ContextLen { get; set; }
        public int FfnMult { get; set; }
        public double Dropout { get; set; }
        public double LearningRate { get; set; }
        public int Seed { get; set; }
        public int MaxVocab { get; set; }
        public List<string>? Vocab { get; set; }
        public List<double[]>? Wte { get; set; }
        public List<double[]>? Wpe { get; set; }
        public List<BlockDto>? Blocks { get; set; }
        public double[]? LnFG { get; set; }
        public double[]? LnFB { get; set; }
        public List<string>? Buffer { get; set; }
        public long TokensObserved { get; set; }
        public int EpochsTrained { get; set; }
        public double LastLoss { get; set; }
    }

    private sealed class BlockDto
    {
        public double[] Ln1G { get; set; } = [];
        public double[] Ln1B { get; set; } = [];
        public List<double[]> Wq { get; set; } = [];
        public List<double[]> Wk { get; set; } = [];
        public List<double[]> Wv { get; set; } = [];
        public List<double[]> Wo { get; set; } = [];
        public double[] Bq { get; set; } = [];
        public double[] Bk { get; set; } = [];
        public double[] Bv { get; set; } = [];
        public double[] Bo { get; set; } = [];
        public double[] Ln2G { get; set; } = [];
        public double[] Ln2B { get; set; } = [];
        public List<double[]> F1 { get; set; } = [];
        public List<double[]> F2 { get; set; } = [];
        public double[] Bf1 { get; set; } = [];
        public double[] Bf2 { get; set; } = [];
    }

    private static double Round6(double x) => Math.Round(x, 6);

    private static double[][] R(double[][] m) => m.Select(r => r.Select(Round6).ToArray()).ToArray();
    private static double[] RV(double[] v) => v.Select(Round6).ToArray();
    private static double[][] U(double[][] m) => m.Select(r => r.ToArray()).ToArray();
    private static double[] UV(double[] v) => (double[])v.Clone();

    /// <summary>
    /// Reads just the architecture configuration from a serialized model —
    /// used by persistence so a restored mind gets back the exact
    /// architecture she trained (not the current defaults).
    /// Returns null when the payload is not a valid model JSON.
    /// </summary>
    public static TransformerConfig? ConfigFromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            var dto = JsonSerializer.Deserialize<LmDto>(json);
            if (dto is null || dto.Vocab is null) return null;
            return new TransformerConfig
            {
                Dim = dto.Dim, Heads = dto.Heads, Layers = dto.Layers,
                ContextLen = dto.ContextLen, FfnMult = dto.FfnMult,
                Dropout = dto.Dropout, LearningRate = dto.LearningRate,
                Seed = dto.Seed, MaxVocab = dto.MaxVocab,
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Serializes the full model (config, vocabulary, weights, buffer) to JSON.</summary>
    public string Serialize()
    {
        lock (_gate)
        {
            var dto = new LmDto
            {
                Dim = _cfg.Dim, Heads = _cfg.Heads, Layers = _cfg.Layers,
                ContextLen = _cfg.ContextLen, FfnMult = _cfg.FfnMult,
                Dropout = _cfg.Dropout, LearningRate = _cfg.LearningRate,
                Seed = _cfg.Seed, MaxVocab = _cfg.MaxVocab,
                Vocab = [.. _vocab],
                Wte = R(_wte).ToList(),
                Wpe = R(_wpe).ToList(),
                Blocks = _blocks.Select(b => new BlockDto
                {
                    Ln1G = RV(b.Ln1G), Ln1B = RV(b.Ln1B),
                    Wq = R(b.Wq).ToList(), Wk = R(b.Wk).ToList(),
                    Wv = R(b.Wv).ToList(), Wo = R(b.Wo).ToList(),
                    Bq = RV(b.Bq), Bk = RV(b.Bk), Bv = RV(b.Bv), Bo = RV(b.Bo),
                    Ln2G = RV(b.Ln2G), Ln2B = RV(b.Ln2B),
                    F1 = R(b.F1).ToList(), F2 = R(b.F2).ToList(),
                    Bf1 = RV(b.Bf1), Bf2 = RV(b.Bf2),
                }).ToList(),
                LnFG = RV(_lnFG),
                LnFB = RV(_lnFB),
                Buffer = [.. _buffer],
                TokensObserved = _tokensObserved,
                EpochsTrained = _epochsTrained,
                LastLoss = _lastLoss,
            };
            return JsonSerializer.Serialize(dto);
        }
    }

    /// <summary>
    /// Restores a serialized model into this instance (kept when the payload
    /// is valid and matches the architecture). Returns true on success.
    /// Older snapshots without a neural model simply never call this.
    /// </summary>
    public bool TryRestore(string json)
    {
        if (string.IsNullOrWhiteSpace(json)) return false;
        try
        {
            var dto = JsonSerializer.Deserialize<LmDto>(json);
            if (dto?.Vocab is null || dto.Wte is null || dto.Wpe is null || dto.Blocks is null)
                return false;
            if (dto.Dim != _cfg.Dim || dto.Heads != _cfg.Heads || dto.Layers != _cfg.Layers ||
                dto.ContextLen != _cfg.ContextLen)
                return false;

            lock (_gate)
            {
                _vocab.Clear();
                _vocab.AddRange(dto.Vocab);
                _index.Clear();
                for (int i = 0; i < _vocab.Count; i++) _index[_vocab[i]] = i;

                _wte = dto.Wte.Select(r => r.ToArray()).ToArray();
                _wpe = dto.Wpe.Select(r => r.ToArray()).ToArray();
                _blocks = dto.Blocks.Select(b => new BlockParams
                {
                    Ln1G = UV(b.Ln1G), Ln1B = UV(b.Ln1B),
                    Wq = U(b.Wq.ToArray()), Wk = U(b.Wk.ToArray()),
                    Wv = U(b.Wv.ToArray()), Wo = U(b.Wo.ToArray()),
                    Bq = UV(b.Bq), Bk = UV(b.Bk), Bv = UV(b.Bv), Bo = UV(b.Bo),
                    Ln2G = UV(b.Ln2G), Ln2B = UV(b.Ln2B),
                    F1 = U(b.F1.ToArray()), F2 = U(b.F2.ToArray()),
                    Bf1 = UV(b.Bf1), Bf2 = UV(b.Bf2),
                }).ToArray();
                _lnFG = UV(dto.LnFG ?? Ones(_cfg.Dim));
                _lnFB = UV(dto.LnFB ?? Zeros(_cfg.Dim));

                _buffer.Clear();
                if (dto.Buffer is not null)
                    foreach (var s in dto.Buffer) _buffer.Enqueue(s);
                _tokensObserved = dto.TokensObserved;
                _epochsTrained = dto.EpochsTrained;
                _lastLoss = dto.LastLoss;
                _adam.Reset();
            }
            return true;
        }
        catch (JsonException)
        {
            return false; // corrupt payload: keep the fresh model
        }
    }

    /// <summary>
    /// Named weight tensors for the export pipeline (safetensors/GGUF):
    /// the token/positional embeddings, every block's projections and LayerNorm
    /// parameters, and the final LayerNorm — all F32-ready.
    /// </summary>
    public IReadOnlyList<(string Name, double[][] Matrix, double[] Vector)> ExportTensors()
    {
        lock (_gate)
        {
            var list = new List<(string, double[][], double[])>
            {
                ("cognify.transformer.wte", _wte, []),
                ("cognify.transformer.wpe", _wpe, []),
            };
            for (int l = 0; l < _blocks.Length; l++)
            {
                var b = _blocks[l];
                string p = $"cognify.transformer.l{l}.";
                list.Add((p + "attn.wq", b.Wq, []));
                list.Add((p + "attn.wk", b.Wk, []));
                list.Add((p + "attn.wv", b.Wv, []));
                list.Add((p + "attn.wo", b.Wo, []));
                list.Add((p + "attn.bq", [], b.Bq));
                list.Add((p + "attn.bk", [], b.Bk));
                list.Add((p + "attn.bv", [], b.Bv));
                list.Add((p + "attn.bo", [], b.Bo));
                list.Add((p + "ln1.g", [], b.Ln1G));
                list.Add((p + "ln1.b", [], b.Ln1B));
                list.Add((p + "ln2.g", [], b.Ln2G));
                list.Add((p + "ln2.b", [], b.Ln2B));
                list.Add((p + "ffn.w1", b.F1, []));
                list.Add((p + "ffn.w2", b.F2, []));
                list.Add((p + "ffn.b1", [], b.Bf1));
                list.Add((p + "ffn.b2", [], b.Bf2));
            }
            list.Add(("cognify.transformer.lnf.g", [], _lnFG));
            list.Add(("cognify.transformer.lnf.b", [], _lnFB));
            return list;
        }
    }
}
