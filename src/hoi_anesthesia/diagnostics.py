"""Helpers for the preprocessing diagnostics (01_A) and O-information sensitivity (01_B) notebooks.

Time series follow the THOI convention: one scan is an array of shape (T, N)
(time points x regions); a state is an array of shape (n_scans, T, N).
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import signal, stats
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from thoi.commons import gaussian_copula_covmat
from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
from src.hoi_anesthesia.states import NONRESPONSIVE_STATES, RESPONSIVE_STATES

TR_SECONDS = {"MA": 2.4, "DBS": 1.25}

# State name used in the analysis -> variable name inside the MATLAB files.
# Dictionary order is the plotting order used in the figure notebooks.
MAT_KEYS = {
    "MA": {
        "MA_awake": "ts_aw",
        "ts_selv2": "ts_selv2",
        "ts_selv4": "ts_selv4",
        "moderate_propofol": "ts_lpp",
        "deep_propofol": "ts_dpp",
        "ketamine": "ts_keta",
    },
    "DBS": {
        "DBS_awake": "ts_aw",
        "ts_on_5V": "ts_on_5V",
        "ts_on_3V": "ts_on_3V",
        "ts_on_5V_control": "ts_on_5V_control",
        "ts_on_3V_control": "ts_on_3V_control",
        "ts_off": "ts_off",
    },
}

PRETTY_LABELS = {
    "MA_awake": "Awake",
    "ts_selv2": "2% Sevo",
    "ts_selv4": "4% Sevo",
    "deep_propofol": "Deep Ppfl",
    "moderate_propofol": "Moderate Ppfl",
    "ketamine": "Ketamine",
    "DBS_awake": "Awake",
    "ts_on_5V": "CT high",
    "ts_off": "off",
    "ts_on_3V_control": "VT low",
    "ts_on_5V_control": "VT high",
    "ts_on_3V": "CT low",
}

CLASS_COLORS = {"C": "#FF9900", "NR": "#9C7AD8", "descriptive": "#9E9E9E"}

MATCHED_LENGTH = 250


# --------------------------------------------------------------------------- #
# Loading and bookkeeping
# --------------------------------------------------------------------------- #
def load_timeseries(mat_path, dataset):
    """Return {state: array (n_scans, T, N)} in the analysis state order."""
    mat = loadmat(str(mat_path))
    return {
        state: np.stack([np.asarray(cell, dtype=np.float64).T for cell in mat[key].flat])
        for state, key in MAT_KEYS[dataset].items()
    }


def state_class(dataset, state):
    if state in RESPONSIVE_STATES[dataset]:
        return "C"
    if state in NONRESPONSIVE_STATES[dataset]:
        return "NR"
    return "descriptive"


def classification_arrays(covs_by_state, dataset):
    """Stack covariances as [responsive..., non-responsive...]; y = 1 for responsive."""
    c = [covs_by_state[s] for s in RESPONSIVE_STATES[dataset]]
    nr = [covs_by_state[s] for s in NONRESPONSIVE_STATES[dataset]]
    X = np.concatenate(c + nr, axis=0)
    n_c = sum(len(a) for a in c)
    y = np.zeros(len(X), dtype=int)
    y[:n_c] = 1
    return X, y


def to_builtin(obj):
    """Recursively convert NumPy scalars/arrays so the object is JSON serialisable."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# Covariance / correlation utilities
# --------------------------------------------------------------------------- #
def gc_covariances(ts):
    """Gaussian-copula covariance (THOI) for each scan: (n, T, N) -> (n, N, N)."""
    return np.stack([gaussian_copula_covmat(np.ascontiguousarray(x)) for x in ts])


def cov_to_corr(cov):
    d = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    return cov / (d[..., :, None] * d[..., None, :])


def upper(mat):
    iu = np.triu_indices(mat.shape[-1], 1)
    return mat[..., iu[0], iu[1]]


def save_covariance_h5(path, covs):
    """Save {dataset: {state: (n, N, N)}} with the schema of data/covariance_matrices_gc.h5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        for dataset, states in covs.items():
            grp = h5f.create_group(dataset)
            for state, arr in states.items():
                grp.create_dataset(state, data=arr, compression="gzip", chunks=True)


# --------------------------------------------------------------------------- #
# Global signal
# --------------------------------------------------------------------------- #
def zscore_regions(x):
    x = x - x.mean(axis=0)
    return x / x.std(axis=0)


def global_signal(x, standardize=True):
    """Mean regional signal. Regions are z-scored first so that arbitrary
    per-region scaling (which the Gaussian copula ignores) does not set the weights."""
    z = zscore_regions(x) if standardize else x - x.mean(axis=0)
    return z.mean(axis=1)


def regress_global_signal(x, standardize=True):
    """Conventional GSR: per-region OLS on [1, g(t)], returning the residuals (T, N)."""
    g = global_signal(x, standardize=standardize)
    design = np.column_stack([np.ones_like(g), g])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ beta


def global_signal_metrics(x):
    """Strength of the shared (global) component in one scan."""
    z = zscore_regions(x)
    n_regions = z.shape[1]
    corr = np.corrcoef(z.T)
    off = upper(corr)
    g = z.mean(axis=1)
    g_std = (g - g.mean()) / g.std()
    r_with_g = (z * g_std[:, None]).mean(axis=0)
    eigvals = np.linalg.eigvalsh(corr)
    g_raw = (x - x.mean(axis=0)).mean(axis=1)
    return {
        "gs_r2_median": float(np.median(r_with_g**2)),
        "mean_r": float(off.mean()),
        "frac_neg_r": float((off < 0).mean()),
        "pc1_frac": float(eigvals[-1] / n_regions),
        "gs_definition_corr": float(np.corrcoef(g, g_raw)[0, 1]),
    }


def gsr_covariance_change(x):
    """How much regional GSR changes the Gaussian-copula correlation structure."""
    before = upper(cov_to_corr(gaussian_copula_covmat(x)))
    after = upper(cov_to_corr(gaussian_copula_covmat(regress_global_signal(x))))
    return {
        "mean_r_gc_before": float(before.mean()),
        "mean_r_gc_after": float(after.mean()),
        "mean_abs_delta_r_gc": float(np.abs(after - before).mean()),
    }


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
def welch_psd(x, tr, segment_seconds=300.0):
    """Region-averaged Welch PSD; each region is normalised to unit total power."""
    nperseg = int(round(segment_seconds / tr))
    nfft = 4 * 2 ** int(np.ceil(np.log2(nperseg)))
    f, p = signal.welch(
        x, fs=1.0 / tr, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        nfft=nfft, detrend="constant", axis=0,
    )
    p = p / p.sum(axis=0, keepdims=True)
    return f, p.mean(axis=1)


def periodogram_psd(x, tr):
    """Region-averaged rectangular-window periodogram at native resolution.
    A notch implemented by zeroing FFT bins stays visible at this resolution."""
    f, p = signal.periodogram(x, fs=1.0 / tr, window="boxcar", detrend="constant", axis=0)
    p = p / p.sum(axis=0, keepdims=True)
    return f, p.mean(axis=1)


def band_fraction(f, p, lo, hi):
    mask = (f >= lo) & (f <= hi)
    return float(p[mask].sum() / p.sum())


def notch_depth(f, p, f0=0.03, half_width=0.005, flank=(0.007, 0.012)):
    """min(PSD within f0 +/- half_width) / median(PSD in the flanks).
    ~1 means no notch; ~0 a deep notch. The window is wide because a nominal
    '0.03 Hz' FFT notch zeroes the nearest Fourier bin (0.0333 Hz in MA)."""
    dist = np.abs(f - f0)
    centre = dist <= half_width
    flanks = (dist >= flank[0]) & (dist <= flank[1])
    return float(p[centre].min() / np.median(p[flanks]))


def notch_frequency(f, p, f0=0.03, half_width=0.005):
    """Frequency of the PSD minimum within f0 +/- half_width."""
    centre = np.abs(f - f0) <= half_width
    return float(f[centre][np.argmin(p[centre])])


def spectral_metrics(x, tr, bands):
    f, p = welch_psd(x, tr)
    fp, pp = periodogram_psd(x, tr)
    lo, hi = bands["pass_low"], bands["pass_high"]
    in_band = band_fraction(f, p, lo, hi)
    mask = (f >= lo) & (f <= hi)
    return {
        "frac_in_band": in_band,
        "frac_out_of_band": band_fraction(f, p, bands["out_of_band_from"], np.inf),
        "frac_above_ma_nyquist": (
            band_fraction(f, p, bands["ma_nyquist"], np.inf) if f.max() > bands["ma_nyquist"] else np.nan
        ),
        "low_frac_in_band": band_fraction(f, p, lo, bands["low_split"]) / in_band,
        "centroid_in_band_hz": float((f[mask] * p[mask]).sum() / p[mask].sum()),
        "notch_depth": notch_depth(fp, pp, f0=bands["notch"]),
        "notch_hz": notch_frequency(fp, pp, f0=bands["notch"]),
    }


# --------------------------------------------------------------------------- #
# Temporal properties and harmonisation
# --------------------------------------------------------------------------- #
def effective_sample_size(x):
    """Per-region effective sample size, T / (1 + 2 * sum of the initial positive
    autocorrelation sequence)."""
    T = x.shape[0]
    z = x - x.mean(axis=0)
    nfft = 2 ** int(np.ceil(np.log2(2 * T)))
    spec = np.fft.rfft(z, n=nfft, axis=0)
    acov = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=0)[:T]
    rho = acov / acov[0]
    nonpos = rho[1:] <= 0
    first = np.where(nonpos.any(axis=0), nonpos.argmax(axis=0) + 1, T)
    lags = np.arange(T)[:, None]
    mask = (lags >= 1) & (lags < first[None, :])
    tau = 1.0 + 2.0 * (rho * mask).sum(axis=0)
    return np.clip(T / tau, 1.0, T)


def lowpass(x, tr, cutoff_hz, order=8):
    sos = signal.butter(order, cutoff_hz, btype="low", fs=1.0 / tr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def bandpass(x, tr, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype="band", fs=1.0 / tr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def fft_notch(x, tr, f0=0.03, half_width=0.001):
    """Zero-phase FFT notch: zero the Fourier bins within f0 +/- half_width."""
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], d=tr)
    spec[np.abs(f - f0) <= max(half_width, f[1] / 2)] = 0
    return np.fft.irfft(spec, n=x.shape[0], axis=0)


def temporal_harmonise(x, dataset, segment="primary", antialias=False):
    """Bring a scan to ~250 samples at TR ~2.4-2.5 s (~600-625 s).

    MA  : primary = first 250 samples, replication = last 250 samples (TR 2.4 s, 600 s).
    DBS : primary = even samples, replication = odd samples (TR 2.5 s, 625 s).
    """
    if dataset == "MA":
        return x[:MATCHED_LENGTH] if segment == "primary" else x[-MATCHED_LENGTH:]
    if antialias:
        x = lowpass(x, TR_SECONDS["DBS"], cutoff_hz=0.18)
    phase = 0 if segment == "primary" else 1
    return x[phase::2][:MATCHED_LENGTH]


def apply_pipeline(x, dataset, gsr=False, temporal=None, antialias=False,
                   bandpass_hz=None, notch_hz=None):
    """Order: filtering (full length) -> temporal harmonisation -> GSR."""
    tr = TR_SECONDS[dataset]
    if bandpass_hz is not None:
        x = bandpass(x, tr, *bandpass_hz)
    if notch_hz is not None:
        x = fft_notch(x, tr, f0=notch_hz)
    if temporal is not None:
        x = temporal_harmonise(x, dataset, segment=temporal, antialias=antialias)
    if gsr:
        x = regress_global_signal(x)
    return x


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def cliffs_delta(a, b):
    """P(a > b) - P(a < b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def average_precision_columns(y, scores):
    """Column-wise average precision, including tied scores, as in scikit-learn."""
    order = np.argsort(-scores, axis=0, kind="stable")
    return _average_precision_sorted(np.asarray(y, dtype=float)[order], scores, order)


def _average_precision_sorted(ys, scores, order):
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    ranks = np.arange(len(ys))[:, None]
    ends = np.r_[sorted_scores[:-1] != sorted_scores[1:], np.ones((1, scores.shape[1]), bool)]
    group_end = np.minimum.accumulate(
        np.where(ends, ranks, len(ys))[::-1], axis=0,
    )[::-1]
    tp = np.cumsum(ys, axis=0)
    precision = np.take_along_axis(tp / (ranks + 1), group_end, axis=0)
    return (precision * ys).sum(axis=0) / ys.sum(axis=0)


def omega_separation_scores(omega, y):
    """Class summaries and two APs: higher Omega identifies C or identifies NR."""
    result = omega_class_summary(omega, y)
    included = y >= 0
    scores, labels = omega[included], y[included]
    order = np.argsort(-scores, axis=0, kind="stable")
    sorted_labels = labels[order].astype(float)
    result["pr_auc_C"] = _average_precision_sorted(sorted_labels, scores, order)
    result["pr_auc_NR"] = _average_precision_sorted(1 - sorted_labels, scores, order)
    return result


# --------------------------------------------------------------------------- #
# O-information
# --------------------------------------------------------------------------- #
def sample_nplets(n_regions, order, n_samples, rng):
    """Unique sorted n-plets sampled uniformly at random, shape (n_samples, order)."""
    total = 1
    for i in range(order):
        total = total * (n_regions - i) // (i + 1)
    if n_samples >= total:
        return np.array(list(itertools.combinations(range(n_regions), order)))
    seen, out = set(), []
    while len(out) < n_samples:
        cand = tuple(sorted(rng.choice(n_regions, size=order, replace=False).tolist()))
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return np.array(out)


@torch.no_grad()
def omega_matrix(covs, nplets, device="cpu", batch_size=2048):
    """O-information for every scan and n-plet: covs (D, N, N), nplets (B, k) -> (D, B)."""
    X = torch.as_tensor(np.asarray(covs), dtype=torch.float64, device=device)
    nplets_t = torch.as_tensor(np.asarray(nplets), dtype=torch.long, device=device)
    measures = nplets_measures(
        X, nplets=nplets_t, covmat_precomputed=True, device=device, verbose=logging.WARNING,
        batch_size=batch_size,
    )
    return measures[..., 2].detach().cpu().numpy().T


def omega_class_summary(omega, y):
    """Reduce scan-by-n-plet Omega to class medians and the mean C-minus-NR effect."""
    if not np.isfinite(omega).all():
        raise FloatingPointError("Non-finite Omega values; inspect the input covariances.")
    return {
        "C": np.median(omega[y == 1], axis=0),
        "NR": np.median(omega[y == 0], axis=0),
        "delta": omega[y == 1].mean(axis=0) - omega[y == 0].mean(axis=0),
    }


def exhaustive_omega_summary(
    covs, y, order, batch_size=2048, device="cpu", output_group: h5py.Group | None = None,
):
    """Enumerate n-plets in batches, optionally persisting every scan-level value.

    Entries follow itertools.combinations order, shared across datasets and variants.
    Saved omega has axes (scan, n-plet); nplets contains zero-based region indices.
    """
    total = math.comb(covs.shape[-1], order)
    summary = {key: np.empty(total) for key in ("C", "NR", "delta")}
    offset = 0
    if output_group is not None:
        saved_omega = output_group.create_dataset(
            "omega", shape=(len(covs), total), dtype=np.float64,
            chunks=(1, min(batch_size, total)), compression="gzip", shuffle=True,
        )
        saved_nplets = output_group.create_dataset(
            "nplets", shape=(total, order), dtype=np.int32,
            chunks=(min(batch_size, total), order), compression="gzip", shuffle=True,
        )
        output_group.attrs["complete"] = False

    def collect(nplets, tc, dtc, omega, s, batch_number):
        nonlocal offset
        scan_values = omega.detach().cpu().numpy().T
        values = omega_class_summary(scan_values, y)
        end = offset + len(nplets)
        if output_group is not None:
            saved_omega[:, offset:end] = scan_values
            saved_nplets[offset:end] = nplets.detach().cpu().numpy()
        for key in summary:
            summary[key][offset:end] = values[key]
        offset = end

    multi_order_measures(
        covs, min_order=order, max_order=order, covmat_precomputed=True,
        T=None, device=device, batch_size=batch_size * len(covs),
        batch_data_collector=collect, batch_aggregation=lambda batches: None,
    )
    if offset != total:
        raise RuntimeError(f"Exhaustive enumeration returned {offset} of {total} n-plets.")
    if output_group is not None:
        output_group.attrs["complete"] = True
    return summary


def save_exhaustive_comparison(covs_by_dataset, labels_by_dataset, order, variant_group,
                              batch_size=512, device="cpu"):
    """Stream every Omega and MA/DBS/pooled separation score in matching n-plet order."""
    names = list(covs_by_dataset)
    covs = np.concatenate([covs_by_dataset[ds] for ds in names])
    labels = np.concatenate([labels_by_dataset[ds] for ds in names])
    total = math.comb(covs.shape[-1], order)
    width = min(batch_size, total)
    shared = variant_group.file.require_group("nplets")
    indices = shared.require_dataset(
        str(order), shape=(total, order), dtype=np.int32,
        chunks=(width, order), compression="gzip", shuffle=True,
    )
    slices, start = {}, 0
    for ds in names:
        slices[ds] = slice(start, start + len(covs_by_dataset[ds]))
        start += len(covs_by_dataset[ds])
    slices["combined"] = slice(None)
    groups = {}
    for ds, selection in slices.items():
        group = variant_group.create_group(f"{ds}/{order}")
        group["nplets"] = indices
        group.attrs["exhaustive"] = True
        group.attrs["scan_metadata"] = f"/scans/{ds}"
        y = labels[selection]
        group.attrs["chance_pr_auc_C"] = np.mean(y[y >= 0] == 1)
        group.attrs["chance_pr_auc_NR"] = np.mean(y[y >= 0] == 0)
        if ds != "combined":
            group.create_dataset(
                "omega", shape=(len(y), total), dtype=np.float64,
                chunks=(1, width), compression="gzip", shuffle=True,
            )
        for key in ("C", "NR", "delta", "pr_auc_C", "pr_auc_NR"):
            group.create_dataset(
                key, shape=(total,), dtype=np.float64,
                chunks=(width,), compression="gzip", shuffle=True,
            )
        groups[ds] = group
    offset = 0

    def collect(nplets, tc, dtc, omega, s, batch_number):
        nonlocal offset
        scan_values = omega.detach().cpu().numpy().T
        end = offset + len(nplets)
        indices[offset:end] = nplets.detach().cpu().numpy()
        for ds, selection in slices.items():
            group = groups[ds]
            values = scan_values[selection]
            if ds != "combined":
                group["omega"][:, offset:end] = values
            for key, array in omega_separation_scores(values, labels[selection]).items():
                group[key][offset:end] = array
        offset = end

    multi_order_measures(
        covs, min_order=order, max_order=order, covmat_precomputed=True, T=None,
        batch_size=batch_size * len(covs), device=device,
        batch_data_collector=collect, batch_aggregation=lambda batches: None,
    )
    variant_group.file.flush()


def saved_omega_density(omega, scans, bins=1024, batch_size=4096):
    """Binned Gaussian KDE using every selected scan/n-plet value, in bounded memory."""
    count, total, square_total = 0, 0.0, 0.0
    lo, hi = np.inf, -np.inf
    for start in range(0, omega.shape[1], batch_size):
        values = omega[scans, start:start + batch_size]
        count += values.size
        total += values.sum()
        square_total += np.square(values).sum()
        lo, hi = min(lo, values.min()), max(hi, values.max())
    mean = total / count
    std = np.sqrt(max(0.0, square_total / count - mean**2))
    bandwidth = max(
        1.06 * std * count**(-0.2),
        32 * np.finfo(float).eps * max(abs(lo), abs(hi), 1),
    )
    edges = np.linspace(lo - 4 * bandwidth, hi + 4 * bandwidth, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    for start in range(0, omega.shape[1], batch_size):
        counts += np.histogram(omega[scans, start:start + batch_size], bins=edges)[0]
    step = edges[1] - edges[0]
    density = gaussian_filter1d(
        counts.astype(float), bandwidth / step, mode="constant",
    ) / (count * step)
    return {
        "coords": (edges[:-1] + edges[1:]) / 2,
        "density": density, "count": count, "histogram_count": int(counts.sum()),
    }
