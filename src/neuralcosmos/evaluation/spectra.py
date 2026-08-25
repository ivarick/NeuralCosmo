"""2D power spectra of projected density maps.

Plan reference: sections 4 (Phase 4 diagnostics), 35, 73.

WHY THIS EXISTS
---------------
Phase 4 measured a strong directional asymmetry: a SIMBA-trained model transfers
to IllustrisTNG almost for free, while a TNG-trained model degrades sharply on
SIMBA. The hypothesis raised at the time was that SIMBA's aggressive AGN feedback
suppresses small-scale structure, so a TNG-trained model comes to rely on
fine-grained features that simply are not present in SIMBA, whereas a
SIMBA-trained model is forced onto coarser features that survive in both.

That hypothesis makes a concrete, falsifiable prediction about the data alone,
independent of any network: the ratio of SIMBA to TNG power should fall with
increasing k. This module measures it.

CONVENTIONS
-----------
The power spectrum is computed on the density CONTRAST of each map,

    delta = rho / mean(rho) - 1

not on the log field the network consumes. Dividing by each map's own mean makes
the measurement insensitive to the overall normalisation and isolates the shape
of the fluctuations, which is what the hypothesis is about.

CAMELS 2D maps cover a 25 h^-1 Mpc box at 256 pixels, so the fundamental mode is
2*pi/25 ~ 0.25 h/Mpc and the Nyquist mode is pi*256/25 ~ 32 h/Mpc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BOX_SIZE_MPC_H", "radial_bins", "power_spectrum_2d", "mean_power_spectrum"]

# CAMELS CMD 2D maps: 25 x 25 x 5 (h^-1 Mpc)^3 slices.
BOX_SIZE_MPC_H = 25.0


@dataclass
class SpectrumResult:
    k: np.ndarray            # bin centres, h/Mpc
    power: np.ndarray        # mean P(k) over maps
    std: np.ndarray          # standard deviation across maps
    n_modes: np.ndarray      # modes per bin
    n_maps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k.tolist(),
            "power": self.power.tolist(),
            "std": self.std.tolist(),
            "n_modes": self.n_modes.tolist(),
            "n_maps": self.n_maps,
        }


def radial_bins(
    npix: int,
    box_size: float = BOX_SIZE_MPC_H,
    n_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the radial |k| grid and its bin assignment.

    Returns ``(bin_index, k_centres, n_modes)``. ``bin_index`` is -1 for modes
    outside the usable range: the DC mode carries no fluctuation information,
    and modes beyond Nyquist are aliased.

    Some low-k bins can legitimately contain ZERO modes. On a discrete grid the
    available |k| are k_fundamental * sqrt(nx^2 + ny^2), whose values are sparse
    at small radius -- nothing lies between sqrt(2) and 2, for instance -- so a
    narrow log bin there can fall in a gap. Such bins are reported with
    ``n_modes == 0`` and ``k`` NaN, and callers must filter them rather than
    treating them as zero power.
    """
    kfreq = np.fft.fftfreq(npix, d=box_size / npix) * 2.0 * np.pi
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2)

    k_fundamental = 2.0 * np.pi / box_size
    k_nyquist = np.pi * npix / box_size

    edges = np.logspace(np.log10(k_fundamental), np.log10(k_nyquist), n_bins + 1)
    flat = kmag.ravel()
    idx = np.digitize(flat, edges) - 1
    # A value landing exactly on the top edge returns n_bins from digitize,
    # which is out of range and silently lengthens the bincount output. Fold it
    # into the last bin rather than discarding a legitimate mode.
    idx[idx == n_bins] = n_bins - 1
    idx[(flat < edges[0]) | (flat > edges[-1])] = -1

    valid = idx >= 0
    n_modes = np.bincount(idx[valid], minlength=n_bins).astype(np.int64)

    # Mode-weighted bin centre, which is more faithful than the geometric centre
    # when bins are wide and modes are unevenly distributed within them.
    k_sum = np.bincount(idx[valid], weights=kmag.ravel()[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        k_centres = np.where(n_modes > 0, k_sum / np.maximum(n_modes, 1), np.nan)

    return idx.reshape(kmag.shape), k_centres, n_modes


def power_spectrum_2d(
    density_map: np.ndarray,
    bin_index: np.ndarray,
    n_bins: int,
    box_size: float = BOX_SIZE_MPC_H,
) -> np.ndarray:
    """P(k) of one map's density contrast."""
    m = np.asarray(density_map, dtype=np.float64)
    mean = m.mean()
    if mean <= 0:
        raise ValueError("map has non-positive mean; cannot form a density contrast")
    delta = m / mean - 1.0

    npix = m.shape[0]
    fft = np.fft.fft2(delta)
    # Normalisation giving P(k) in (h^-1 Mpc)^2 for a 2D field.
    p2d = (np.abs(fft) ** 2) * (box_size / npix**2) ** 2

    flat_idx = bin_index.ravel()
    valid = flat_idx >= 0
    sums = np.bincount(flat_idx[valid], weights=p2d.ravel()[valid], minlength=n_bins)
    counts = np.bincount(flat_idx[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def mean_power_spectrum(
    maps: np.ndarray,
    indices: np.ndarray,
    n_bins: int = 24,
    box_size: float = BOX_SIZE_MPC_H,
    progress=None,
) -> SpectrumResult:
    """Average P(k) over the selected maps of an array."""
    npix = int(maps.shape[1])
    bin_index, k_centres, n_modes = radial_bins(npix, box_size, n_bins)

    acc = np.zeros(n_bins, dtype=np.float64)
    acc_sq = np.zeros(n_bins, dtype=np.float64)
    n = 0

    for j, i in enumerate(indices):
        pk = power_spectrum_2d(np.asarray(maps[int(i)]), bin_index, n_bins, box_size)
        acc += np.nan_to_num(pk)
        acc_sq += np.nan_to_num(pk) ** 2
        n += 1
        if progress is not None and (j % 25 == 0 or j == len(indices) - 1):
            progress(j + 1, len(indices))

    mean = acc / max(n, 1)
    var = np.maximum(acc_sq / max(n, 1) - mean**2, 0.0)
    return SpectrumResult(
        k=k_centres,
        power=mean,
        std=np.sqrt(var),
        n_modes=n_modes,
        n_maps=n,
    )
