"""B0: summary-statistic baseline.

Plan reference: sections 25, 34.

Section 25 states the requirement plainly: deep learning must beat something
simple. Without this, a strong CNN number means nothing, because the reader
cannot tell whether spatial representation learning contributed anything at all
over a handful of scalars.

Three feature sets are provided so the comparison also answers a mechanistic
question rather than only a gatekeeping one:

  moments    per-map statistics of the log field: mean, sd, skew, kurtosis,
             quantiles, histogram. No spatial information whatsoever -- these
             are invariant to any permutation of the pixels.
  spectrum   binned 2D power spectrum. Pure two-point spatial information.
  both       the concatenation.

The gap between ``moments`` and ``spectrum`` says how much of the recoverable
signal is spatial, and the gap between ``both`` and the CNN says how much lies
beyond two-point statistics -- which is the part a network could plausibly be
adding.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FEATURE_SETS", "moment_features", "spectrum_features", "extract_features"]

FEATURE_SETS = ("moments", "spectrum", "both")

_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
# Fixed histogram range in log10 space. Fixed rather than per-map or per-suite:
# a range derived from the data would encode suite-specific information into the
# feature definition itself, which is the same leak the quantised cache avoids.
_HIST_RANGE = (9.0, 16.0)
_HIST_BINS = 16


def moment_features(log_map: np.ndarray) -> np.ndarray:
    """Permutation-invariant statistics of one map's log field."""
    x = np.asarray(log_map, dtype=np.float64).ravel()
    mean = x.mean()
    sd = x.std()
    if sd > 0:
        z = (x - mean) / sd
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4))
    else:
        skew = kurt = 0.0

    q = np.quantile(x, _QUANTILES)
    hist, _ = np.histogram(x, bins=_HIST_BINS, range=_HIST_RANGE, density=True)

    return np.concatenate([[mean, sd, skew, kurt], q, hist]).astype(np.float64)


def spectrum_features(linear_map: np.ndarray, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
    """log10 of the binned power spectrum of one map.

    Taken in log because P(k) spans orders of magnitude, and a linear-regression
    probe on raw P(k) would be dominated entirely by the largest scales.
    """
    from ..evaluation.spectra import power_spectrum_2d

    pk = power_spectrum_2d(linear_map, bin_index, n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(pk)
    # Empty k bins are NaN by design; fill so the feature vector stays finite.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(
    maps,
    indices: np.ndarray,
    quant_spec=None,
    feature_set: str = "both",
    n_spectrum_bins: int = 16,
    progress=None,
) -> np.ndarray:
    """Build the feature matrix for the given map indices.

    ``maps`` may hold raw float32 densities or uint16 quantised log codes; when
    ``quant_spec`` is given the codes are decoded to log10 and exponentiated back
    for the spectrum, which needs the linear field.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")

    want_spectrum = feature_set in ("spectrum", "both")
    want_moments = feature_set in ("moments", "both")

    bin_index = None
    if want_spectrum:
        from ..evaluation.spectra import radial_bins

        npix = int(maps.shape[1])
        bin_index, _, _ = radial_bins(npix, n_bins=n_spectrum_bins)

    rows: list[np.ndarray] = []
    for j, i in enumerate(indices):
        raw = np.asarray(maps[int(i)])
        if quant_spec is not None:
            log_map = quant_spec.decode(raw, dtype=np.float64)
        else:
            log_map = np.log10(raw.astype(np.float64))

        parts: list[np.ndarray] = []
        if want_moments:
            parts.append(moment_features(log_map))
        if want_spectrum:
            linear = np.power(10.0, log_map)
            parts.append(spectrum_features(linear, bin_index, n_spectrum_bins))
        rows.append(np.concatenate(parts))

        if progress is not None and (j % 200 == 0 or j == len(indices) - 1):
            progress(j + 1, len(indices))

    return np.vstack(rows)
