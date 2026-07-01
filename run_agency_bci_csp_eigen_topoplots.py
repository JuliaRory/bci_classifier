import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.montage_processing import find_ch_idx, get_channel_names, get_topo_positions
from src.analysis.CSP import compute_csp
from src.analysis.preprocessing import bandpass_filter, read_good_epoch_mask


PROJECT = "pr_Agency_BCI"
STAGE = "test"
OUTPUT_FOLDER = "CSP_components_eigenvalues"

CONFIG = {
    "Fs": 1000,
    "do_filtering": True,
    "low_freq": 5,
    "high_freq": 35,
    "baseline_ms": 500,
    "trial_dur_ms": 4000,
    "start_shift_ms": 1000,
    "end_shift_ms": 0,
    "epoch_len_ms": None,
    "epochs_step_ms": None,
    "idxs_keys": "2-3",
}

CONFIG_CSP = {
    "bands": [[8, 12], [9, 13], [10, 14], [8, 15]],
    "robust": True,
    "concat": True,
    "regularization": False,
    "alpha": 0.1,
}

BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
MONTAGE_PATH = r"resources/mks64_standard.ced"


def build_xy_positions():
    labels = get_channel_names(MONTAGE_PATH)
    good_channel_indices = [
        find_ch_idx(ch, MONTAGE_PATH)
        for ch in labels
        if ch not in BAD_CHANNELS
    ]
    return get_topo_positions(MONTAGE_PATH)[good_channel_indices]


def component_indices(n_components):
    first = np.arange(min(5, n_components), dtype=int)
    if n_components <= 5:
        return first

    last_start = max(len(first), n_components - 5)
    last = np.arange(n_components - 1, last_start - 1, -1, dtype=int)
    return np.r_[first, last]


def read_json_dataset(h5f, name):
    if name not in h5f:
        return {}

    value = h5f[name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def sanitize_filename_part(value):
    return re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip()


def band_from_matrix_name(matrix_path):
    name = matrix_path.name
    start = name.find("[")
    end = name.find("]", start)
    if start < 0 or end < 0:
        return None
    return name[start : end + 1]


def record_from_matrix_name(matrix_path):
    stem = matrix_path.stem
    if stem.startswith("MATRIX_"):
        stem = stem[len("MATRIX_") :]

    if stem.startswith("["):
        end = stem.find("]")
        if end >= 0:
            stem = stem[end + 1 :].lstrip("_")

    match = re.match(r"^(?:robust|standard)_(?:concat|mean)(?:\+reg[^_]+)?_?(.*)$", stem)
    return match.group(1) if match else stem


def output_filename(subject_name, matrix_path, metadata_csp):
    band = metadata_csp.get("band")
    band_text = str(band) if band is not None else band_from_matrix_name(matrix_path)
    robust = "robust" if metadata_csp.get("robust", True) else "standard"
    concat = "concat" if metadata_csp.get("concat", True) else "mean"

    reg = ""
    if metadata_csp.get("regularization", False):
        reg = f"+reg{metadata_csp.get('alpha')}"

    record = record_from_matrix_name(matrix_path)
    parts = [
        sanitize_filename_part(subject_name),
        sanitize_filename_part(band_text),
        f"{robust}_{concat}{reg}",
    ]
    if record:
        parts.append(sanitize_filename_part(record))
    return "_".join(parts) + ".png"


def csp_config_name_part(config_csp):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    reg = f"+reg{config_csp['alpha']}" if config_csp.get("regularization", False) else ""
    return f"{robust}_{concat}{reg}"


def matrix_filename(record_name, band, config_csp=CONFIG_CSP):
    return (
        f"MATRIX_{band}_{csp_config_name_part(config_csp)}_"
        + record_name[len("EPOCHS_") :]
    )


def iter_epoch_paths(project=PROJECT, stage=STAGE):
    trans_root = Path("data") / project / "trans" / stage
    if not trans_root.exists():
        raise FileNotFoundError(f"Epoch folder not found: {trans_root}")

    for subject_folder in sorted(path for path in trans_root.iterdir() if path.is_dir()):
        for epoch_path in sorted(subject_folder.glob("EPOCHS_*.hdf")):
            yield subject_folder.name, epoch_path


def load_epochs(epoch_path):
    with File(epoch_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze()
        good_epoch_mask = read_good_epoch_mask(h5f, len(epochs))

    if not good_epoch_mask.all():
        print(f"Rejected bad epochs: {(~good_epoch_mask).sum()} / {len(good_epoch_mask)}")
    return epochs[good_epoch_mask], labels[good_epoch_mask]


def csp_time_window(epochs, config):
    ms_to_samples = lambda x: int(x * config["Fs"] / 1000)
    baseline = ms_to_samples(config["baseline_ms"])
    start_shift = ms_to_samples(config["start_shift_ms"])
    end_shift = ms_to_samples(config["end_shift_ms"])
    end = epochs.shape[1] - end_shift
    return baseline + start_shift, end


def calculate_and_save_csp(epoch_path, output_folder, config=CONFIG, config_csp=CONFIG_CSP):
    epochs, labels = load_epochs(epoch_path)
    epochs_1 = epochs[np.where(labels == 0)]
    epochs_2 = epochs[np.where(labels == 1)]
    if len(epochs_1) == 0 or len(epochs_2) == 0:
        raise ValueError(
            f"Need both labels 0 and 1 to calculate CSP for {epoch_path}; "
            f"got {len(epochs_1)} and {len(epochs_2)} epochs."
        )

    start, end = csp_time_window(epochs, config)
    output_folder.mkdir(parents=True, exist_ok=True)
    matrix_paths = []

    for band in config_csp["bands"]:
        epochs_1_band = np.array(
            [bandpass_filter(ep, fs=config["Fs"], low=band[0], high=band[1])[0] for ep in epochs_1]
        )
        epochs_2_band = np.array(
            [bandpass_filter(ep, fs=config["Fs"], low=band[0], high=band[1])[0] for ep in epochs_2]
        )
        epochs_1_clean = epochs_1_band[:, start:end, :]
        epochs_2_clean = epochs_2_band[:, start:end, :]
        print("Clean epochs shape: ", epochs_1_clean.shape, epochs_2_clean.shape)

        spatial_filters, spatial_patterns, eigenvalues = compute_csp(
            epochs_1_clean,
            epochs_2_clean,
            config_csp,
        )

        metadata_csp = {**config_csp, "band": band}
        matrix_path = output_folder / matrix_filename(epoch_path.name, band, config_csp)
        with File(matrix_path, "w") as h5f:
            h5f.create_dataset("projForward", data=spatial_patterns)
            h5f.create_dataset("projInverse", data=spatial_filters)
            h5f.create_dataset("evals", data=eigenvalues)
            h5f.attrs["csp_matrix_semantics"] = "projForward=spatial_patterns;projInverse=spatial_filters"
            h5f.attrs["projForward_kind"] = "spatial_patterns"
            h5f.attrs["projInverse_kind"] = "spatial_filters"
            h5f.create_dataset("metadata", data=json.dumps(config))
            h5f.create_dataset("metadata_csp", data=json.dumps(metadata_csp))

        print("output file ->", matrix_path)
        matrix_paths.append(matrix_path)

    return matrix_paths


def plot_selected_components(spatial_patterns, eigenvalues, xy):
    indices = component_indices(spatial_patterns.shape[1])
    n_cols = 5 if len(indices) > 5 else len(indices)
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig = plt.figure(figsize=(3.1 * n_cols + 4.0, 3.9 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.35] + [1] * n_cols,
        wspace=0.42,
        hspace=0.38,
    )

    ax_eigen = fig.add_subplot(gs[:, 0])
    x = np.arange(1, len(eigenvalues) + 1)
    ax_eigen.plot(x, eigenvalues, color="black", linewidth=1.2)
    ax_eigen.scatter(x, eigenvalues, color="black", s=12)
    ax_eigen.scatter(indices + 1, eigenvalues[indices], color="crimson", s=36, zorder=3)
    ax_eigen.set_title("Eigenvalues", fontsize=11)
    ax_eigen.set_xlabel("CSP")
    ax_eigen.set_ylabel("lambda")

    axes = []
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols + 1
        axes.append(fig.add_subplot(gs[row, col]))

    for ax, component_idx in zip(axes, indices):
        pattern = spatial_patterns[:, component_idx]
        limit = float(np.nanmax(np.abs(pattern)))
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0

        im, _ = plot_topomap(
            pattern,
            xy,
            axes=ax,
            show=False,
            contours=0,
            sphere=0.5,
            cmap="jet",
            extrapolate="head",
            vlim=(-limit, limit),
        )
        ax.set_title(
            f"CSP {component_idx + 1}\n"
            f"lambda={eigenvalues[component_idx]:.4g}",
            fontsize=11,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([-limit, 0, limit])
        cbar.ax.tick_params(labelsize=8)

    for ax in axes[len(indices) :]:
        ax.axis("off")

    return fig


def save_matrix_plot(subject_name, matrix_path, output_root, xy):
    with File(matrix_path, "r") as h5f:
        spatial_patterns = h5f["projForward"][:]
        eigenvalues = h5f["evals"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")

    if spatial_patterns.shape[1] > len(eigenvalues):
        raise ValueError(
            f"Eigenvalue count is smaller than component count in {matrix_path}: "
            f"{len(eigenvalues)} < {spatial_patterns.shape[1]}"
        )

    fig = plot_selected_components(spatial_patterns, eigenvalues, xy)

    output_path = output_root / subject_name / output_filename(subject_name, matrix_path, metadata_csp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run(project=PROJECT, stage=STAGE):
    xy = build_xy_positions()
    output_root = Path("results") / project / stage / OUTPUT_FOLDER
    total = 0

    for subject_name, epoch_path in iter_epoch_paths(project=project, stage=stage):
        print(f"\nSubject {subject_name}")
        print(f"Record {epoch_path.name}")
        csp_folder = Path("data") / project / "features" / "csp" / stage / subject_name
        matrix_paths = calculate_and_save_csp(epoch_path, csp_folder)
        for matrix_path in matrix_paths:
            output_path = save_matrix_plot(subject_name, matrix_path, output_root, xy)
            print("output file ->", output_path)
            total += 1

    print(f"Saved {total} CSP component plots.")


if __name__ == "__main__":
    run()
