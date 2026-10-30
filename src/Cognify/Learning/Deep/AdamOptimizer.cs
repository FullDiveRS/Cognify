namespace Cognify.Learning;

/// <summary>
/// Adam optimizer (Kingma &amp; Ba, 2015) over the library's named parameter
/// tensors — the update rule behind both <see cref="DeepNet"/> and
/// <see cref="TransformerLM"/>. Adam keeps exponential moving averages of the
/// gradient (first moment) and the squared gradient (second moment) per
/// parameter, bias-corrects them, and steps each parameter by
/// <c>lr * m̂ / (√v̂ + ε)</c> — so a single learning rate works across layers
/// whose gradients have very different scales.
///
/// Parameters are addressed by string keys ("wte", "l0.attn.wq", "w0", …),
/// which is how the models reference their tensors. State lives in dense
/// arrays shaped exactly like the parameter tensors.
/// </summary>
public sealed class AdamOptimizer
{
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private long _t;
    private readonly Dictionary<string, (double[][] M, double[][] V)> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (double[] M, double[] V)> _vectors = new(StringComparer.Ordinal);

    /// <summary>Creates the optimizer. Defaults are the paper's constants.</summary>
    public AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    /// <summary>How many optimization steps have been taken (shared across tensors).</summary>
    public long Steps => _t;

    /// <summary>Clears all moment state — used when a parameter tensor is resized
    /// (e.g. the transformer's vocabulary grew) so shapes stay consistent.</summary>
    public void Reset()
    {
        _t = 0;
        _matrices.Clear();
        _vectors.Clear();
    }

    /// <summary>
    /// One Adam step on a matrix parameter. <paramref name="grad"/> must have
    /// the same shape as <paramref name="param"/>; the update is applied in place.
    /// </summary>
    public void Step(string key, double[][] param, double[][] grad, double learningRate)
    {
        if (!_matrices.TryGetValue(key, out var state))
        {
            state.M = new double[param.Length][];
            state.V = new double[param.Length][];
            for (int r = 0; r < param.Length; r++)
            {
                state.M[r] = new double[param[r].Length];
                state.V[r] = new double[param[r].Length];
            }
            _matrices[key] = state;
        }

        _t++;
        double bc1 = 1.0 - Math.Pow(_beta1, _t);
        double bc2 = 1.0 - Math.Pow(_beta2, _t);

        for (int r = 0; r < param.Length; r++)
        {
            var p = param[r];
            var g = grad[r];
            var m = state.M[r];
            var v = state.V[r];
            for (int c = 0; c < p.Length; c++)
            {
                m[c] = _beta1 * m[c] + (1.0 - _beta1) * g[c];
                v[c] = _beta2 * v[c] + (1.0 - _beta2) * g[c] * g[c];
                double mHat = m[c] / bc1;
                double vHat = v[c] / bc2;
                p[c] -= learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
            }
        }
    }

    /// <summary>One Adam step on a vector parameter (bias, LayerNorm gain/shift).</summary>
    public void StepVec(string key, double[] param, double[] grad, double learningRate)
    {
        if (!_vectors.TryGetValue(key, out var state))
        {
            state.M = new double[param.Length];
            state.V = new double[param.Length];
            _vectors[key] = state;
        }

        _t++;
        double bc1 = 1.0 - Math.Pow(_beta1, _t);
        double bc2 = 1.0 - Math.Pow(_beta2, _t);

        for (int i = 0; i < param.Length; i++)
        {
            state.M[i] = _beta1 * state.M[i] + (1.0 - _beta1) * grad[i];
            state.V[i] = _beta2 * state.V[i] + (1.0 - _beta2) * grad[i] * grad[i];
            double mHat = state.M[i] / bc1;
            double vHat = state.V[i] / bc2;
            param[i] -= learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
        }
    }
}
