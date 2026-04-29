import numpy as np


def score_spatial_patterns_physio(
    patterns,
    ch_names,
    roi_channels=("C3", "CP3", "FC3", "C1", "C5"),
    eps=1e-10,
):
    """
    Assess physiological plausibility of CSP spatial patterns.

    Parameters
    ----------
    patterns : array (n_components, n_channels)
    ch_names : list of str
    roi_channels : tuple

    Returns
    -------
    scores : dict
        roi_ratio, peak_in_roi, locality, contrast, total
    """
    roi_idx = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]

    if len(roi_idx) == 0:
        raise ValueError("ROI channels not found")

    roi_ratio = []
    peak_in_roi = []
    locality = []
    contrast = []

    for comp in patterns:
        comp_abs = np.abs(comp)

        roi_energy = np.sum(comp_abs[roi_idx])
        total_energy = np.sum(comp_abs)
        roi_ratio.append(roi_energy / (total_energy + eps))

        peak_idx = np.argmax(comp_abs)
        peak_in_roi.append(1 if peak_idx in roi_idx else 0)

        sorted_vals = np.sort(comp_abs)[::-1]
        cumulative = np.cumsum(sorted_vals)
        n_80 = np.searchsorted(cumulative, 0.8 * total_energy) + 1
        locality.append(1 - n_80 / len(comp_abs))

        non_roi_idx = list(set(range(len(comp))) - set(roi_idx))
        non_roi_energy = np.sum(comp_abs[non_roi_idx])
        contrast.append(roi_energy / (roi_energy + non_roi_energy + eps))

    roi_ratio = np.array(roi_ratio)
    peak_in_roi = np.array(peak_in_roi)
    locality = np.array(locality)
    contrast = np.array(contrast)

    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)

    roi_n = norm(roi_ratio)
    loc_n = norm(locality)
    con_n = norm(contrast)
    peak_n = peak_in_roi

    total = 0.4 * roi_n + 0.2 * peak_n + 0.2 * loc_n + 0.2 * con_n

    return {
        "roi_ratio": roi_ratio,
        "peak_in_roi": peak_in_roi,
        "locality": locality,
        "contrast": contrast,
        "total": total,
    }


def calculate_eigenscore(evals):
    eig_score = np.abs(np.log(evals / (1 - evals) + 1e-10))
    return np.round(eig_score, 3)
