"""
Full-reference image quality metrics (PSNR / SSIM / NRMSE / complex RMSE / edge &
bias) for spiral-MRI deblurring evaluation.

All functions operate on foreground-masked, scaled images and return plain NumPy
scalars, so they can be used from a notebook, a .py script, or a research
pipeline without importing torch.

Scaling / units
---------------
Images are complex (magnitude is what is typically reported) and live on unknown
absolute scales. To compare methods (GIRF, MFI, IR, network output) against one
reference (the iterative/MFI "gold standard"), every image must be on a SHARED
scale. The notebook's convention is used here:

  * Non-AI reference ``img_IR`` is brought onto the input scale:
        reference = img_IR / sf_IR,    sf_IR = sqrt(Ns) / sqrt(Nx*Ny) / mean(dcf)
  * Network output is denormalised by the training scale factor ``sf``:
        prediction = net_out * sf
  * The GIR input ``img_girf`` is already on the input physical scale.

Callers pass already-rescaled, on-same-scale arrays; these helpers only compute
metrics and therefore do not care how the scaling was done as long as it is
consistent between the reference and candidate.

Masking
-------
Metrics are computed only on voxels inside a binary foreground mask (the
convex-hull mask built in the notebook). Masked-out voxels are excluded from
numerator and denominator, so the background does not bias them.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def _mag(volume):
    a = np.asarray(volume)
    return np.abs(a) if np.iscomplexobj(a) else np.abs(a)


def _mask(mask):
    m = np.asarray(mask)
    return m.astype(bool) if m.dtype != bool else m


# --------------------------------------------------------------------------- #
# Magnitude-based metrics (inside an optional mask)
# --------------------------------------------------------------------------- #
def psnr_masked(target, candidate, mask=None, data_range=None):
    """PSNR (dB) on magnitude inside mask. data_range defaults to max |target| in mask."""
    t = _mag(target)
    c = _mag(candidate)
    if mask is not None:
        m = _mask(mask)
        t, c = t[m], c[m]
    if data_range is None:
        data_range = float(t.max())
    mse = float(np.mean((t - c) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(float(data_range) ** 2 / mse))


def ssim_masked(target, candidate, mask=None, data_range=None, win_size=None, gaussian_weights=False):
    """
    SSIM on magnitude inside the mask.

    skimage.metrics.structural_similarity matches 3-D volumes when both inputs
    have the same ndim and channel_axis=None. Masked voxels are gathered into a
    dense volume first; the trailing spatial dims remain contiguous so that the
    Gaussian-window neighbourhood stays physically meaningful.
    """
    t = _mag(target)
    c = _mag(candidate)
    if mask is not None:
        m = _mask(mask)
        t, c = t[m], c[m]
    if data_range is None:
        data_range = float(np.max(t))

    # skimage SSIM needs a neighbourhood; refuse degenerate (empty / single-voxel) sets.
    if t.size == 0:
        raise ValueError("mask selects no voxels")
    return float(_ssim(t, c, data_range=data_range, win_size=win_size,
                       gaussian_weights=gaussian_weights))


def nrmse_masked(target, candidate, mask=None):
    """Normalized RMSE of magnitude: || |t|-|c| || / (|| |t| || + eps). 0 = perfect."""
    t = _mag(target)
    c = _mag(candidate)
    if mask is not None:
        m = _mask(mask)
        t, c = t[m], c[m]
    return float(np.linalg.norm(t - c) / (np.linalg.norm(t) + 1e-12))


def complex_rmse_masked(target, candidate, mask=None):
    """RMSE of the complex difference (forgets phase/magnitude separately)."""
    a = np.asarray(target, dtype=np.complex128)
    b = np.asarray(candidate, dtype=np.complex128)
    if mask is not None:
        m = _mask(mask)
        a, b = a[m], b[m]
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


def bias_magnitude(target, candidate, mask=None):
    """
    Output bias (percent) relative to reference magnitude mean inside mask.
    Positive => candidate over-estimates intensity on average.
    """
    t = _mag(target)
    c = _mag(candidate)
    if mask is not None:
        m = _mask(mask)
        t, c = t[m], c[m]
    mu = float(np.mean(t))
    return 100.0 * (float(np.mean(c)) - mu) / (mu + 1e-12)


def edge_grad_energy(img, mask=None):
    """
    Gradient (sharpness) energy of an image inside mask:
    mean over masked voxels of the L2 norm of the (central-forward) gradient.
    Works for a 2-D slice (H, W) or a 3-D volume (D, H, W). Larger value =
    sharper / more edge detail (report it for both the reference and the
    candidate to judge edge preservation relatively).
    """
    a = _mag(img)

    # Each central-forward diff (np.gradient) shrinks its own axis by 1, so a
    # gradient taken along axis k has length n_k - 1 on that axis and n_j on the
    # others. To combine them we crop every gradient back to the common interior,
    # dropping the outermost voxel along every axis (a 1-voxel margin).
    grads = [np.abs(np.gradient(a, axis=k)) for k in range(a.ndim)]
    crop = (slice(1, -1),) * a.ndim
    grads = [g[crop] for g in grads]
    g = np.sqrt(sum(grad ** 2 for grad in grads))

    if mask is not None:
        m = _mask(mask)[crop]  # interior mask aligned with the trimmed gradients
        g = g[m]
    if g.size == 0:
        return float("nan")
    return float(np.mean(g))


def evaluate(target, candidate, mask=None, data_range=None, win_size=7, gaussian_weights=False):
    """
    Convenience: compute all magnitude metrics for one (reference, candidate)
    pair and return them in a dict for easy tabulation.
    """
    out = {}
    out["PSNR(dB)"] = psnr_masked(target, candidate, mask, data_range)
    out["SSIM"] = ssim_masked(target, candidate, mask, data_range, win_size, gaussian_weights)
    out["NRMSE"] = nrmse_masked(target, candidate, mask)
    out["complex_RMSE"] = complex_rmse_masked(target, candidate, mask)
    out["bias(%)"] = bias_magnitude(target, candidate, mask)
    out["edge_ref"] = edge_grad_energy(target, mask)
    out["edge_cand"] = edge_grad_energy(candidate, mask)
    return out