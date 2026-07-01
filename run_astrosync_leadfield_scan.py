import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from h5py import File
from matplotlib.pyplot import close
from mne.transforms import Transform


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calculate_csp import process_records_csp
from src.analysis.preprocessing import read_good_epoch_mask
from src.utils.montage_processing import get_channel_names


PROJECT = "pr_AstroSync"
DEFAULT_STAGE = "exp"
MONTAGE_PATH = PROJECT_ROOT / "resources" / "mks64_standard.ced"
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
CSP_SEMANTICS = "projForward=spatial_patterns;projInverse=spatial_filters"

CONFIG = {
    "Fs": 1000,
    "do_filtering": True,
    "low_freq": 5,
    "high_freq": 35,
    "baseline_ms": 500,
    "trial_dur_ms": 6000,
    "start_shift_ms": 1000,
    "end_shift_ms": 0,
    "epoch_len_ms": None,
    "epochs_step_ms": None,
    "idxs_keys": "1-2",
}

CONFIG_CSP = {
    "bands": [[8, 12], [9, 13], [10, 14], [8, 15]],
    "robust": True,
    "concat": True,
    "regularization": False,
    "alpha": 0.1,
}

ROIS = {
    "left_M1_hand": {"center": [-36, -24, 56], "radius": 45, "weight": 1.0},
    "left_S1_hand": {"center": [-42, -30, 52], "radius": 45, "weight": 0.8},
    "left_PMd": {"center": [-28, -8, 56], "radius": 45, "weight": 0.7},
    "SMA": {"center": [0, -6, 58], "radius": 45, "weight": 0.7},
}


def component_indices(n_components):
    """Return the first and last five CSP component indices."""
    if n_components <= 10:
        return np.arange(n_components, dtype=int)
    return np.r_[np.arange(5, dtype=int), np.arange(n_components - 5, n_components, dtype=int)]


def patterns_as_components(projForward, n_channels):
    """Return projForward as (n_components, n_channels)."""
    projForward = np.asarray(projForward, dtype=float)
    if projForward.ndim != 2:
        raise ValueError("projForward must be a 2D matrix.")
    if projForward.shape[0] == n_channels:
        return projForward.T
    if projForward.shape[1] == n_channels:
        return projForward
    raise ValueError(
        f"projForward shape {projForward.shape} does not match channel count {n_channels}."
    )


def normalize_topomap(x):
    """Center a channel topography and normalize it to unit L2 norm."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm == 0:
        return np.zeros_like(x, dtype=float)
    return x / norm


def get_source_rr_mm(fwd):
    """Return source coordinates from an MNE forward solution in millimeters."""
    rr_parts = []
    for src in fwd["src"]:
        if "vertno" in src:
            rr = src["rr"][src["vertno"]]
        elif "inuse" in src:
            rr = src["rr"][np.asarray(src["inuse"], dtype=bool)]
        else:
            rr = src["rr"]
        rr_parts.append(np.asarray(rr, dtype=float))

    if not rr_parts:
        return np.empty((0, 3), dtype=float)

    rr_m = np.vstack(rr_parts)
    n_sources = int(fwd["sol"]["data"].shape[1] // 3)
    if rr_m.shape[0] != n_sources:
        raise ValueError(
            f"Source coordinate count ({rr_m.shape[0]}) does not match leadfield sources ({n_sources})."
        )
    return rr_m * 1000.0


def sources_in_spherical_rois(rr_mm, rois):
    """Return source indices that fall into at least one spherical ROI."""
    rr_mm = np.asarray(rr_mm, dtype=float)
    if rr_mm.ndim != 2 or rr_mm.shape[1] != 3:
        raise ValueError("rr_mm must have shape (n_sources, 3).")

    selected = np.zeros(rr_mm.shape[0], dtype=bool)
    for name, roi in rois.items():
        center = np.asarray(roi["center"], dtype=float)
        radius = float(roi["radius"])
        if center.shape != (3,):
            raise ValueError(f"ROI {name!r} center must contain three coordinates.")
        selected |= np.linalg.norm(rr_mm - center, axis=1) <= radius
    return np.flatnonzero(selected)


def _forward_channel_names(fwd):
    """Return leadfield row names in the same order as fwd['sol']['data']."""
    return list(fwd["sol"].get("row_names") or fwd["info"]["ch_names"])


def _resolve_picks(pattern, fwd, picks):
    """Resolve optional channel picks for pattern and forward leadfield rows."""
    row_names = _forward_channel_names(fwd)
    pattern = np.asarray(pattern, dtype=float)

    if picks is None:
        if pattern.shape[0] != len(row_names):
            raise ValueError(
                f"Pattern length ({pattern.shape[0]}) does not match forward channels ({len(row_names)})."
            )
        return pattern, np.arange(len(row_names), dtype=int)

    if all(isinstance(pick, str) for pick in picks):
        pick_names = list(picks)
        fwd_idx = [row_names.index(name) for name in pick_names]
    else:
        fwd_idx = [int(pick) for pick in picks]

    fwd_idx = np.asarray(fwd_idx, dtype=int)
    if pattern.shape[0] == len(row_names):
        return pattern[fwd_idx], fwd_idx
    if pattern.shape[0] == len(fwd_idx):
        return pattern, fwd_idx

    raise ValueError(
        f"Pattern length ({pattern.shape[0]}) must match either all forward channels "
        f"({len(row_names)}) or selected picks ({len(fwd_idx)})."
    )


def pattern_leadfield_scan(pattern, fwd, source_indices=None, picks=None):
    """
    Compare one CSP spatial pattern with free-orientation leadfield maps.

    The sign of a CSP pattern is arbitrary, so the score uses abs(correlation).
    CSP-pattern amplitude is not interpreted physiologically: both the pattern
    and leadfield columns are centered and L2-normalized before comparison.
    """
    pattern = np.asarray(pattern, dtype=float)
    if np.isnan(pattern).any():
        raise ValueError("Pattern contains NaN values.")

    leadfield = np.asarray(fwd["sol"]["data"], dtype=float)
    if leadfield.shape[1] % 3 != 0:
        raise ValueError(
            f"Expected free-orientation leadfield with columns divisible by 3, got {leadfield.shape[1]}."
        )

    pattern, fwd_idx = _resolve_picks(pattern, fwd, picks)
    leadfield = leadfield[fwd_idx, :]
    n_sources = leadfield.shape[1] // 3

    if source_indices is None:
        source_indices = np.arange(n_sources, dtype=int)
    else:
        source_indices = np.asarray(source_indices, dtype=int)
        if source_indices.size == 0:
            raise ValueError("source_indices is empty.")
        if source_indices.min() < 0 or source_indices.max() >= n_sources:
            raise ValueError("source_indices contains values outside the source space.")

    pattern_norm = normalize_topomap(pattern)
    scores_per_source = np.zeros(n_sources, dtype=float)

    columns = (source_indices[:, np.newaxis] * 3 + np.arange(3, dtype=int)).ravel()
    leadfield_subset = leadfield[:, columns]
    leadfield_subset = leadfield_subset - np.nanmean(leadfield_subset, axis=0, keepdims=True)
    norms = np.linalg.norm(leadfield_subset, axis=0)
    valid = np.isfinite(norms) & (norms > 0)
    leadfield_subset[:, valid] /= norms[valid]
    leadfield_subset[:, ~valid] = 0.0

    orientation_scores = np.abs(pattern_norm @ leadfield_subset).reshape(len(source_indices), 3)
    source_scores = orientation_scores.max(axis=1)
    source_orientations = orientation_scores.argmax(axis=1)
    scores_per_source[source_indices] = source_scores

    local_best = int(np.argmax(source_scores))
    best_score = float(source_scores[local_best])
    best_source_idx = int(source_indices[local_best])
    best_orientation = int(source_orientations[local_best])

    return best_score, best_source_idx, best_orientation, scores_per_source


def source_score_locality(scores_per_source, source_indices=None, threshold_fraction=0.9):
    """
    Estimate how sharp the best source-space maximum is after leadfield scan.

    locality is best_score - second_best_score. Higher values mean that the best
    source is more clearly separated from the next candidate. n90 counts sources
    with score >= threshold_fraction * best_score; smaller n90 means a sharper,
    more local solution.
    """
    scores = np.asarray(scores_per_source, dtype=float)
    if source_indices is None:
        selected_scores = scores[np.isfinite(scores)]
    else:
        source_indices = np.asarray(source_indices, dtype=int)
        if source_indices.size == 0:
            raise ValueError("source_indices is empty.")
        selected_scores = scores[source_indices]
        selected_scores = selected_scores[np.isfinite(selected_scores)]

    if selected_scores.size == 0:
        return {
            "best_score": np.nan,
            "second_best_score": np.nan,
            "locality": np.nan,
            "locality_ratio": np.nan,
            "n90": 0,
            "fraction90": np.nan,
        }

    sorted_scores = np.sort(selected_scores)[::-1]
    best_score = float(sorted_scores[0])
    second_best_score = float(sorted_scores[1]) if sorted_scores.size > 1 else 0.0
    locality = best_score - second_best_score
    locality_ratio = best_score / second_best_score if second_best_score > 0 else np.inf
    if best_score > 0:
        n90 = int(np.sum(selected_scores >= threshold_fraction * best_score))
    else:
        n90 = int(selected_scores.size)
    fraction90 = float(n90 / selected_scores.size)

    return {
        "best_score": best_score,
        "second_best_score": second_best_score,
        "locality": float(locality),
        "locality_ratio": float(locality_ratio),
        "n90": n90,
        "fraction90": fraction90,
    }


def score_csp_patterns_with_leadfield(projForward, fwd, motor_source_indices, eigenvalues=None):
    """
    Score all CSP spatial patterns by global and motor-ROI leadfield similarity.

    projForward is expected to contain spatial patterns. If it is stored as
    (n_channels, n_components), it is transposed to (n_components, n_channels).
    """
    leadfield = np.asarray(fwd["sol"]["data"], dtype=float)
    if leadfield.shape[1] % 3 != 0:
        raise ValueError(
            f"Expected free-orientation leadfield with columns divisible by 3, got {leadfield.shape[1]}."
        )

    motor_source_indices = np.asarray(motor_source_indices, dtype=int)
    if motor_source_indices.size == 0:
        raise ValueError("motor_source_indices is empty.")

    n_channels_fwd = leadfield.shape[0]
    patterns = patterns_as_components(projForward, n_channels_fwd)

    if eigenvalues is not None:
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        if eigenvalues.shape[0] < patterns.shape[0]:
            raise ValueError(
                f"eigenvalues length ({eigenvalues.shape[0]}) is smaller than component count ({patterns.shape[0]})."
            )

    rows = []
    for component, pattern in enumerate(patterns):
        global_score, global_source_idx, global_orientation, global_scores_per_source = pattern_leadfield_scan(
            pattern,
            fwd,
        )
        motor_score, motor_source_idx, motor_orientation, motor_scores_per_source = pattern_leadfield_scan(
            pattern,
            fwd,
            source_indices=motor_source_indices,
        )
        motority = float(motor_score / global_score) if global_score > 0 else 0.0
        global_locality = source_score_locality(global_scores_per_source)
        motor_locality = source_score_locality(
            motor_scores_per_source,
            source_indices=motor_source_indices,
        )

        row = {
            "component": int(component),
            "global_score": float(global_score),
            "motor_score": float(motor_score),
            "motority": motority,
            "global_second_score": global_locality["second_best_score"],
            "global_locality": global_locality["locality"],
            "global_locality_ratio": global_locality["locality_ratio"],
            "global_n90": global_locality["n90"],
            "global_fraction90": global_locality["fraction90"],
            "motor_second_score": motor_locality["second_best_score"],
            "motor_locality": motor_locality["locality"],
            "motor_locality_ratio": motor_locality["locality_ratio"],
            "motor_n90": motor_locality["n90"],
            "motor_fraction90": motor_locality["fraction90"],
            "global_source_idx": int(global_source_idx),
            "motor_source_idx": int(motor_source_idx),
            "global_orientation": int(global_orientation),
            "motor_orientation": int(motor_orientation),
        }

        if eigenvalues is None:
            row["final_score"] = 0.6 * motority + 0.4 * motor_score
        else:
            separability_score = float(np.clip(abs(eigenvalues[component] - 0.5) * 2.0, 0.0, 1.0))
            row["separability_score"] = separability_score
            row["final_score"] = (
                0.45 * motority + 0.35 * motor_score + 0.20 * separability_score
            )
        rows.append(row)

    return pd.DataFrame(rows)


def get_best_source_coordinates(results_df, rr_mm):
    """Add global and motor best-source coordinates to a results DataFrame."""
    rr_mm = np.asarray(rr_mm, dtype=float)
    results_df = results_df.copy()

    def add_coords(prefix, source_column):
        coords = np.full((len(results_df), 3), np.nan, dtype=float)
        for row_idx, source_idx in enumerate(results_df[source_column].to_numpy()):
            if pd.notna(source_idx):
                source_idx = int(source_idx)
                if 0 <= source_idx < len(rr_mm):
                    coords[row_idx] = rr_mm[source_idx]
        results_df[f"{prefix}_x_mm"] = coords[:, 0]
        results_df[f"{prefix}_y_mm"] = coords[:, 1]
        results_df[f"{prefix}_z_mm"] = coords[:, 2]

    add_coords("global", "global_source_idx")
    add_coords("motor", "motor_source_idx")
    return results_df


def plot_source_scores(scores_per_source, rr_mm):
    """Plot source-space scores as a simple 3D scatter and mark the maximum."""
    scores_per_source = np.asarray(scores_per_source, dtype=float)
    rr_mm = np.asarray(rr_mm, dtype=float)
    if len(scores_per_source) != len(rr_mm):
        raise ValueError("scores_per_source and rr_mm must have the same length.")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        rr_mm[:, 0],
        rr_mm[:, 1],
        rr_mm[:, 2],
        c=scores_per_source,
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    best_idx = int(np.nanargmax(scores_per_source))
    ax.scatter(
        rr_mm[best_idx, 0],
        rr_mm[best_idx, 1],
        rr_mm[best_idx, 2],
        c="red",
        s=85,
        edgecolors="white",
        linewidths=1.0,
        label="max",
    )
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")
    ax.set_zlabel("z, mm")
    ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08, label="leadfield score")
    return fig, ax


def example_usage(projForward, fwd, eigenvalues):
    """Minimal example matching the intended interactive usage."""
    rr_mm = get_source_rr_mm(fwd)
    motor_source_indices = sources_in_spherical_rois(rr_mm, ROIS)
    results = score_csp_patterns_with_leadfield(
        projForward,
        fwd,
        motor_source_indices,
        eigenvalues=eigenvalues,
    )
    results = get_best_source_coordinates(results, rr_mm)
    print(results.sort_values("final_score", ascending=False))
    return results


def good_channel_names():
    """Return AstroSync EEG channels after removing channels excluded in this project."""
    return [ch for ch in get_channel_names(str(MONTAGE_PATH)) if ch not in BAD_CHANNELS]


def matrix_filename(record_name, band):
    """Build the CSP matrix filename used by scripts.calculate_csp."""
    robust = "robust" if CONFIG_CSP["robust"] else "standard"
    concat = "concat" if CONFIG_CSP["concat"] else "mean"
    return f"MATRIX_{band}_{robust}_{concat}+reg{CONFIG_CSP['alpha']}_" + record_name[len("EPOCHS_") :]


def matrix_has_current_semantics(matrix_path):
    """Check whether a CSP matrix uses projForward as spatial patterns."""
    if not matrix_path.exists():
        return False
    try:
        with File(matrix_path, "r") as h5f:
            return (
                h5f.attrs.get("csp_matrix_semantics", "") == CSP_SEMANTICS
                and h5f.attrs.get("projForward_kind", "") == "spatial_patterns"
                and h5f.attrs.get("projInverse_kind", "") == "spatial_filters"
            )
    except OSError:
        return False


def ensure_current_csp(folder_epochs, folder_csp, records, force=False):
    """Calculate CSP matrices that are missing or still have old naming semantics."""
    folder_csp.mkdir(parents=True, exist_ok=True)
    records_to_recalculate = []
    for record in records:
        expected = [folder_csp / matrix_filename(record, band) for band in CONFIG_CSP["bands"]]
        if force or not all(matrix_has_current_semantics(path) for path in expected):
            records_to_recalculate.append(record)

    if records_to_recalculate:
        try:
            folder_input = str(folder_epochs.relative_to(PROJECT_ROOT))
        except ValueError:
            folder_input = str(folder_epochs)
        process_records_csp(
            folder_input=folder_input,
            records=records_to_recalculate,
            folder_output=str(folder_csp),
            config=CONFIG,
            config_csp=CONFIG_CSP,
        )
    return records_to_recalculate


def read_json_dataset(h5f, name):
    """Read a JSON string dataset from an HDF5 file."""
    if name not in h5f:
        return {}
    value = h5f[name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def load_epochs(epochs_path):
    """Load clean epochs and labels from an AstroSync EPOCHS HDF5 file."""
    with File(epochs_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
        good_mask = read_good_epoch_mask(h5f, len(epochs))
    return epochs[good_mask], labels[good_mask]


def make_mne_epochs(epochs, labels, channel_names):
    """Create MNE EpochsArray with the project montage and average reference."""
    ced = pd.read_csv(MONTAGE_PATH, sep="\t").set_index("labels")
    missing = [ch for ch in channel_names if ch not in ced.index]
    if missing:
        raise ValueError(f"Channels are missing in montage: {missing}")

    ch_pos = {
        ch: np.array([-ced.loc[ch, "Y"], ced.loc[ch, "X"], ced.loc[ch, "Z"]], dtype=float)
        for ch in channel_names
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(channel_names, sfreq=CONFIG["Fs"], ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    info["dev_head_t"] = Transform("meg", "head", np.eye(4))

    mne_epochs = mne.EpochsArray(
        np.transpose(epochs, (0, 2, 1)),
        info,
        tmin=-CONFIG["baseline_ms"] / 1000,
        events=np.c_[np.arange(len(labels)), np.zeros(len(labels), dtype=int), labels],
        event_id={str(label): int(label) for label in np.unique(labels)},
        verbose=False,
    )
    mne_epochs.set_eeg_reference("average", projection=True, verbose=False)
    mne_epochs.apply_proj(verbose=False)
    mne_epochs.apply_baseline((None, 0), verbose=False)
    return mne_epochs


def make_forward_solution(info, grid_mm=15.0, sphere_radius_mm=90.0):
    """Create a simple free-orientation EEG forward solution for leadfield scan."""
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.0),
        head_radius=sphere_radius_mm / 1000.0,
        verbose=False,
    )
    src = mne.setup_volume_source_space(
        subject=None,
        pos=float(grid_mm),
        sphere=(0.0, 0.0, 0.0, sphere_radius_mm / 1000.0),
        mindist=5.0,
        exclude=5.0,
        sphere_units="m",
        verbose=False,
    )
    return mne.make_forward_solution(
        info,
        trans=None,
        src=src,
        bem=sphere,
        meg=False,
        eeg=True,
        mindist=0.0,
        n_jobs=1,
        on_inside="ignore",
        verbose=False,
    )


def iter_epoch_records(folder_epochs):
    """Yield AstroSync EPOCHS HDF5 filenames in a subject folder."""
    return sorted(path.name for path in folder_epochs.glob("EPOCHS_*.hdf") if path.is_file())


def iter_subject_folders(stage):
    """Yield subject folders for a project stage."""
    root = PROJECT_ROOT / "data" / PROJECT / "trans" / stage
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def load_csp_matrix(matrix_path):
    """Load spatial patterns, eigenvalues, and metadata from a CSP matrix file."""
    with File(matrix_path, "r") as h5f:
        projForward = h5f["projForward"][:]
        eigenvalues = h5f["evals"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")
    return projForward, eigenvalues, metadata_csp


def write_tables(df, output_root):
    """Write CSV and XLSX result tables."""
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "leadfield_scan_component_table.csv"
    xlsx_path = output_root / "leadfield_scan_component_table.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"Could not write Excel table {xlsx_path}: {exc}")
    return csv_path, xlsx_path


def plot_component_metric_summary(matrix_path, output_path, projForward, eigenvalues, results, info):
    """Save CSP topomaps annotated with leadfield-scan component metrics."""
    patterns = patterns_as_components(projForward, len(info["ch_names"]))
    selected = component_indices(patterns.shape[0])
    results_by_component = results.set_index("component", drop=False)

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.0, 1, 1, 1, 1, 1], wspace=0.18, hspace=0.48)
    ax_eig = fig.add_subplot(gs[:, 0])
    x = np.arange(len(eigenvalues))
    ax_eig.plot(x, eigenvalues, color="black", linewidth=1.4)
    ax_eig.scatter(x, eigenvalues, s=18, color="black")
    ax_eig.scatter(selected, eigenvalues[selected], s=70, color="crimson", zorder=3)
    ax_eig.set_title("CSP eigenvalues")
    ax_eig.set_xlabel("component")
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.set_ylim(0, 1)
    ax_eig.grid(alpha=0.25)

    for plot_idx, comp_idx in enumerate(selected):
        row_idx = plot_idx // 5
        col_idx = plot_idx % 5 + 1
        ax = fig.add_subplot(gs[row_idx, col_idx])
        pattern = patterns[comp_idx]
        vmax = np.nanmax(np.abs(pattern))
        vlim = (-vmax, vmax) if np.isfinite(vmax) and vmax > 0 else (None, None)
        image, _ = mne.viz.plot_topomap(
            pattern,
            info,
            axes=ax,
            show=False,
            contours=0,
            sphere=0.1,
            image_interp="cubic",
            extrapolate="head",
            cmap="jet",
            vlim=vlim,
        )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

        metrics = results_by_component.loc[comp_idx]
        ax.set_title(
            f"CSP {comp_idx}\n"
            f"global_score={metrics['global_score']:.2f}\n"
            f"motor_score={metrics['motor_score']:.2f}\n"
            f"motority={metrics['motority']:.2f}, final_score={metrics['final_score']:.2f}\n"
            f"locality={metrics['global_locality']:.2f}, n90={int(metrics['global_n90'])}",
            fontsize=8.0,
        )

    fig.suptitle(matrix_path.name, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    close(fig)


def save_top_score_plots(projForward, fwd, rr_mm, results, figure_dir, matrix_stem, top_n):
    """Save 3D source-score plots for the best components of one CSP matrix."""
    if top_n <= 0:
        return

    leadfield_n_channels = fwd["sol"]["data"].shape[0]
    patterns = patterns_as_components(projForward, leadfield_n_channels)
    figure_dir.mkdir(parents=True, exist_ok=True)

    top = results.sort_values("final_score", ascending=False).head(top_n)
    for _, row in top.iterrows():
        component = int(row["component"])
        _, _, _, scores_per_source = pattern_leadfield_scan(patterns[component], fwd)
        fig, ax = plot_source_scores(scores_per_source, rr_mm)
        ax.set_title(
            f"{matrix_stem}\ncomponent {component}, final={row['final_score']:.3f}, "
            f"motor={row['motor_score']:.3f}"
        )
        fig_path = figure_dir / f"leadfield_scores_{matrix_stem}_component_{component:02d}.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        close(fig)


def process_subject(stage, subject_folder, force_recalculate=False, grid_mm=15.0, plot_top_n=3):
    """Run leadfield scan for one AstroSync subject."""
    subject = subject_folder.name
    folder_epochs = PROJECT_ROOT / "data" / PROJECT / "trans" / stage / subject
    folder_csp = PROJECT_ROOT / "data" / PROJECT / "features" / "csp" / stage / subject
    output_root = PROJECT_ROOT / "results" / PROJECT / stage / subject / "leadfield_scan"
    figure_root = output_root / "figures"

    records = iter_epoch_records(folder_epochs)
    if not records:
        print(f"Skip {stage}/{subject}: no EPOCHS files.")
        return pd.DataFrame()

    recalculated = ensure_current_csp(
        folder_epochs=folder_epochs,
        folder_csp=folder_csp,
        records=records,
        force=force_recalculate,
    )
    if recalculated:
        print(f"Recalculated CSP for {stage}/{subject}: {', '.join(recalculated)}")

    channel_names = good_channel_names()
    first_epochs, first_labels = load_epochs(folder_epochs / records[0])
    mne_epochs = make_mne_epochs(first_epochs, first_labels, channel_names)
    fwd = make_forward_solution(mne_epochs.info, grid_mm=grid_mm)
    rr_mm = get_source_rr_mm(fwd)
    motor_source_indices = sources_in_spherical_rois(rr_mm, ROIS)
    if motor_source_indices.size == 0:
        raise ValueError(
            f"No sources from {stage}/{subject} fall inside motor ROIs. "
            "Try increasing ROI radius or using a denser/larger source grid."
        )

    subject_rows = []
    for record in records:
        print(f"Leadfield scan {stage}/{subject}/{record}")
        for band in CONFIG_CSP["bands"]:
            matrix_path = folder_csp / matrix_filename(record, band)
            if not matrix_path.exists():
                print(f"  Missing CSP matrix: {matrix_path}")
                continue

            projForward, eigenvalues, metadata_csp = load_csp_matrix(matrix_path)
            results = score_csp_patterns_with_leadfield(
                projForward,
                fwd,
                motor_source_indices,
                eigenvalues=eigenvalues,
            )
            results = get_best_source_coordinates(results, rr_mm)
            band_value = metadata_csp.get("band", band)
            results.insert(0, "project", PROJECT)
            results.insert(1, "stage", stage)
            results.insert(2, "subject", subject)
            results.insert(3, "record", record)
            results.insert(4, "matrix", matrix_path.name)
            results.insert(5, "band", json.dumps(band_value))
            results["matrix_path"] = str(matrix_path)
            subject_rows.append(results)

            record_stem = Path(record).stem
            if record_stem.startswith("EPOCHS_"):
                record_stem = record_stem[len("EPOCHS_") :]
            figure_dir = figure_root / record_stem
            summary_path = figure_dir / f"leadfield_components_{matrix_path.stem}.png"
            plot_component_metric_summary(
                matrix_path=matrix_path,
                output_path=summary_path,
                projForward=projForward,
                eigenvalues=eigenvalues,
                results=results,
                info=mne_epochs.info,
            )
            save_top_score_plots(
                projForward,
                fwd,
                rr_mm,
                results,
                figure_dir,
                matrix_path.stem,
                plot_top_n,
            )

    if not subject_rows:
        return pd.DataFrame()
    df_subject = pd.concat(subject_rows, ignore_index=True)
    write_tables(df_subject, output_root)
    return df_subject


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple leadfield scan for AstroSync CSP spatial patterns."
    )
    parser.add_argument("--stage", default=DEFAULT_STAGE, help="Stage to process, or 'all'. Default: exp.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject names to process.")
    parser.add_argument("--force-recalculate-csp", action="store_true", help="Recalculate CSP matrices.")
    parser.add_argument("--grid-mm", type=float, default=15.0, help="Volume source grid spacing in mm.")
    parser.add_argument(
        "--plot-top-n",
        type=int,
        default=3,
        help="Number of best components per matrix to visualize as 3D source-score plots.",
    )
    return parser.parse_args()


def run():
    args = parse_args()
    stages = [args.stage]
    if args.stage == "all":
        trans_root = PROJECT_ROOT / "data" / PROJECT / "trans"
        stages = [path.name for path in sorted(trans_root.iterdir()) if path.is_dir()]

    all_rows = []
    for stage in stages:
        for subject_folder in iter_subject_folders(stage):
            if args.subjects and subject_folder.name not in args.subjects:
                continue
            df_subject = process_subject(
                stage=stage,
                subject_folder=subject_folder,
                force_recalculate=args.force_recalculate_csp,
                grid_mm=args.grid_mm,
                plot_top_n=args.plot_top_n,
            )
            if not df_subject.empty:
                all_rows.append(df_subject)

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        output_root = PROJECT_ROOT / "results" / PROJECT / "leadfield_scan"
        if len(stages) == 1:
            output_root = PROJECT_ROOT / "results" / PROJECT / stages[0] / "leadfield_scan"
        csv_path, xlsx_path = write_tables(df_all, output_root)
        print(f"Saved combined table -> {csv_path}")
        print(f"Saved combined table -> {xlsx_path}")
    else:
        print("No leadfield-scan rows were produced.")


if __name__ == "__main__":
    run()
