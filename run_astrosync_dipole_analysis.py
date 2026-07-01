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
from src.analysis.csp_component_scores import get_selected_component_indices
from src.utils.montage_processing import get_channel_names


PROJECT = "pr_AstroSync"
DEFAULT_STAGE = "exp"
MONTAGE_PATH = PROJECT_ROOT / "resources" / "mks64_standard.ced"
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]




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
    "left_M1": {"center": [-36, -24, 56], "radius": 35, "weight": 1.0},
    "left_S1": {"center": [-42, -30, 52], "radius": 35, "weight": 0.8},
    "SMA": {"center": [0, -6, 58], "radius": 35, "weight": 0.7},
    "left_PMd": {"center": [-28, -8, 56], "radius": 35, "weight": 0.6},
    "right_M1": {"center": [36, -24, 56], "radius": 35, "weight": 1.0},
    "right_S1": {"center": [42, -30, 52], "radius": 35, "weight": 0.8},
    "right_PMd": {"center": [28, -8, 56], "radius": 35, "weight": 0.6},
}

CSP_SEMANTICS = "projForward=spatial_patterns;projInverse=spatial_filters"


def good_channel_names():
    return [ch for ch in get_channel_names(str(MONTAGE_PATH)) if ch not in BAD_CHANNELS]


def matrix_filename(record_name, band):
    robust = "robust" if CONFIG_CSP["robust"] else "standard"
    concat = "concat" if CONFIG_CSP["concat"] else "mean"
    return f"MATRIX_{band}_{robust}_{concat}+reg{CONFIG_CSP['alpha']}_" + record_name[len("EPOCHS_") :]


def matrix_has_current_semantics(matrix_path):
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
    with File(epochs_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
        good_mask = read_good_epoch_mask(h5f, len(epochs))
    return epochs[good_mask], labels[good_mask]


def make_mne_epochs(epochs, labels, channel_names):
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


def baseline_covariance(mne_epochs):
    return mne.compute_covariance(
        mne_epochs,
        tmin=None,
        tmax=0,
        method="empirical",
        rank=None,
        verbose=False,
    )


def sphere_model(mne_epochs):
    return mne.make_sphere_model(
        r0="auto",
        head_radius="auto",
        info=mne_epochs.info,
        verbose=False,
    )


def fit_component_dipole(pattern, mne_epochs, cov, sphere):
    if len(pattern) != len(mne_epochs.info["ch_names"]):
        raise ValueError(
            f"Pattern length {len(pattern)} does not match info channels {len(mne_epochs.info['ch_names'])}"
        )

    pattern = np.asarray(pattern, dtype=float)
    pattern = pattern - np.nanmean(pattern)
    evoked = mne.EvokedArray(
        pattern[:, np.newaxis],
        info=mne_epochs.info.copy(),
        tmin=0.0,
        nave=len(mne_epochs),
    )
    evoked.apply_proj(verbose=False)

    with mne.use_log_level("WARNING"):
        dipole, residual = mne.fit_dipole(
            evoked,
            cov=cov,
            bem=sphere,
            trans=None,
            n_jobs=1,
            verbose=False,
        )
    return dipole, residual


def roi_membership(pos_mm):
    pos_mm = np.asarray(pos_mm, dtype=float)
    result = {}
    matched = []
    weights = []
    distances = {}
    for name, roi in ROIS.items():
        center = np.asarray(roi["center"], dtype=float)
        distance = float(np.linalg.norm(pos_mm - center))
        inside = distance <= float(roi["radius"])
        distances[name] = distance
        result[f"dist_{name}_mm"] = distance
        result[f"in_{name}"] = bool(inside)
        if inside:
            matched.append(name)
            weights.append(float(roi["weight"]))

    result["roi_names"] = ";".join(matched)
    result["roi_count"] = len(matched)
    result["roi_weights"] = ";".join(f"{weight:g}" for weight in weights)
    result["best_roi"] = matched[int(np.argmax(weights))] if weights else ""
    result["best_roi_weight"] = float(max(weights)) if weights else 0.0
    result["nearest_roi"] = min(distances, key=distances.get) if distances else ""
    result["nearest_roi_dist_mm"] = min(distances.values()) if distances else np.nan
    return result


def component_indices(n_components):
    return get_selected_component_indices(n_components)


def fit_matrix_dipoles(matrix_path, epochs_path, subject, stage, record_name, mne_epochs, cov, sphere):
    with File(matrix_path, "r") as h5f:
        spatial_patterns = h5f["projForward"][:]
        eigvals = h5f["evals"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")

    selected = component_indices(spatial_patterns.shape[1])
    band = metadata_csp.get("band")
    rows = []
    fit_results = {}
    for selected_order, comp_idx in enumerate(selected):
        edge_group = "first5" if selected_order < 5 else "last5"
        row = {
            "project": PROJECT,
            "stage": stage,
            "subject": subject,
            "record": record_name,
            "matrix": matrix_path.name,
            "band": json.dumps(band),
            "component": int(comp_idx),
            "component_1based": int(comp_idx + 1),
            "selected_order": int(selected_order),
            "edge_group": edge_group,
            "eigenvalue": float(eigvals[comp_idx]),
            "matrix_path": str(matrix_path),
            "epochs_path": str(epochs_path),
        }
        try:
            dipole, residual = fit_component_dipole(
                spatial_patterns[:, comp_idx],
                mne_epochs=mne_epochs,
                cov=cov,
                sphere=sphere,
            )
            pos_m = dipole.pos[0]
            pos_mm = pos_m * 1000
            ori = dipole.ori[0]
            row.update(
                {
                    "gof": float(dipole.gof[0]),
                    "rv": float(100 - dipole.gof[0]),
                    "x_mm": float(pos_mm[0]),
                    "y_mm": float(pos_mm[1]),
                    "z_mm": float(pos_mm[2]),
                    "ori_x": float(ori[0]),
                    "ori_y": float(ori[1]),
                    "ori_z": float(ori[2]),
                    "amplitude": float(dipole.amplitude[0]),
                    "fit_error": "",
                }
            )
            row.update(roi_membership(pos_mm))
            fit_results[comp_idx] = row.copy()
        except Exception as exc:
            row.update(
                {
                    "gof": np.nan,
                    "rv": np.nan,
                    "x_mm": np.nan,
                    "y_mm": np.nan,
                    "z_mm": np.nan,
                    "ori_x": np.nan,
                    "ori_y": np.nan,
                    "ori_z": np.nan,
                    "amplitude": np.nan,
                    "fit_error": str(exc),
                    **roi_membership([np.nan, np.nan, np.nan]),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows), spatial_patterns, eigvals, fit_results


def short_roi_text(row):
    if row is None or not row.get("roi_names"):
        return "ROI: none"
    return "ROI: " + row["roi_names"].replace(";", ", ")


def plot_matrix_components(matrix_path, output_path, spatial_patterns, eigvals, fit_results, info):
    selected = component_indices(spatial_patterns.shape[1])
    finite_amplitudes = [
        abs(float(fit_results[comp_idx]["amplitude"]))
        for comp_idx in selected
        if comp_idx in fit_results and np.isfinite(fit_results[comp_idx].get("amplitude", np.nan))
    ]
    max_amplitude = max(finite_amplitudes) if finite_amplitudes else 0.0
    max_arrow_len_m = 0.05 #25

    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.0, 1, 1, 1, 1, 1], wspace=0.18, hspace=0.42)
    ax_eig = fig.add_subplot(gs[:, 0])
    x = np.arange(len(eigvals))
    ax_eig.plot(x, eigvals, color="black", linewidth=1.4)
    ax_eig.scatter(x, eigvals, s=18, color="black")
    ax_eig.scatter(selected, eigvals[selected], s=70, color="crimson", zorder=3)
    ax_eig.set_title("CSP eigenvalues")
    ax_eig.set_xlabel("component")
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.set_ylim(0, 1)
    ax_eig.grid(alpha=0.25)

    for plot_idx, comp_idx in enumerate(selected):
        row = plot_idx // 5
        col = plot_idx % 5 + 1
        ax = fig.add_subplot(gs[row, col])
        pattern = spatial_patterns[:, comp_idx]
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

        fit_row = fit_results.get(comp_idx)
        if fit_row and np.isfinite(fit_row.get("x_mm", np.nan)) and np.isfinite(fit_row.get("y_mm", np.nan)):
            x_m = fit_row["x_mm"] / 1000
            y_m = fit_row["y_mm"] / 1000
            ax.scatter(x_m, y_m, s=95, c="red", edgecolors="white", linewidths=1.2, zorder=10)
            if np.isfinite(fit_row.get("ori_x", np.nan)) and np.isfinite(fit_row.get("ori_y", np.nan)):
                amplitude = abs(float(fit_row.get("amplitude", 0.0)))
                arrow_len = max_arrow_len_m * amplitude / max_amplitude if max_amplitude > 0 else 0.0
                ax.quiver(
                    x_m,
                    y_m,
                    fit_row["ori_x"] * arrow_len,
                    fit_row["ori_y"] * arrow_len,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="black",
                    width=0.010,
                    zorder=11,
                )

        gof_text = "GOF: n/a"
        amp_text = "amp: n/a"
        title_weight = "normal"
        if fit_row and np.isfinite(fit_row.get("gof", np.nan)):
            gof_text = f"GOF: {fit_row['gof']:.1f}"
            if fit_row["gof"] > 50:
                title_weight = "bold"
        if fit_row and np.isfinite(fit_row.get("amplitude", np.nan)):
            amp_text = f"amp: {fit_row['amplitude'] * 1e9:.1f} nAm"
        ax.set_title(
            f"CSP {comp_idx}\neig={eigvals[comp_idx]:.3f}, {gof_text}\n{amp_text}\n{short_roi_text(fit_row)}",
            fontsize=10,
            fontweight=title_weight,
        )

    fig.suptitle(matrix_path.name, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    close(fig)


def write_tables(df, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "dipole_component_table.csv"
    xlsx_path = output_root / "dipole_component_table.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"Could not write Excel table {xlsx_path}: {exc}")
    return csv_path, xlsx_path


def iter_epoch_records(folder_epochs):
    return sorted(
        path.name
        for path in folder_epochs.glob("EPOCHS_*.hdf")
        if path.is_file()
    )


def iter_subject_folders(stage):
    root = PROJECT_ROOT / "data" / PROJECT / "trans" / stage
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def process_subject(stage, subject_folder, force_recalculate=False):
    subject = subject_folder.name
    folder_epochs = PROJECT_ROOT / "data" / PROJECT / "trans" / stage / subject
    folder_csp = PROJECT_ROOT / "data" / PROJECT / "features" / "csp" / stage / subject
    output_root = PROJECT_ROOT / "results" / PROJECT / stage / subject / "dipole_analysis"
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
    subject_rows = []
    for record in records:
        epochs_path = folder_epochs / record
        print(f"Dipoles {stage}/{subject}/{record}")
        epochs, labels = load_epochs(epochs_path)
        mne_epochs = make_mne_epochs(epochs, labels, channel_names)
        cov = baseline_covariance(mne_epochs)
        sphere = sphere_model(mne_epochs)

        for band in CONFIG_CSP["bands"]:
            mat_path = folder_csp / matrix_filename(record, band)
            if not mat_path.exists():
                print(f"  Missing CSP matrix: {mat_path}")
                continue
            df_matrix, patterns, eigvals, fit_results = fit_matrix_dipoles(
                matrix_path=mat_path,
                epochs_path=epochs_path,
                subject=subject,
                stage=stage,
                record_name=record,
                mne_epochs=mne_epochs,
                cov=cov,
                sphere=sphere,
            )
            subject_rows.append(df_matrix)
            record_stem = Path(record).stem
            if record_stem.startswith("EPOCHS_"):
                record_stem = record_stem[len("EPOCHS_") :]
            band_text = f"{band[0]}-{band[1]}"
            fig_path = figure_root / record_stem / f"dipoles_{band_text}_{mat_path.stem}.png"
            plot_matrix_components(
                matrix_path=mat_path,
                output_path=fig_path,
                spatial_patterns=patterns,
                eigvals=eigvals,
                fit_results=fit_results,
                info=mne_epochs.info,
            )

    if not subject_rows:
        return pd.DataFrame()
    df_subject = pd.concat(subject_rows, ignore_index=True)
    write_tables(df_subject, output_root)
    return df_subject


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit dipoles for AstroSync CSP spatial patterns and save ROI tables/figures."
    )
    parser.add_argument("--stage", default=DEFAULT_STAGE, help="Stage to process, or 'all'. Default: exp.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject names to process.")
    parser.add_argument("--force-recalculate-csp", action="store_true", help="Recalculate CSP matrices even if marked current.")
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
            )
            if not df_subject.empty:
                all_rows.append(df_subject)

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        output_root = PROJECT_ROOT / "results" / PROJECT / "dipole_analysis"
        if len(stages) == 1:
            output_root = PROJECT_ROOT / "results" / PROJECT / stages[0] / "dipole_analysis"
        csv_path, xlsx_path = write_tables(df_all, output_root)
        print(f"Saved combined table -> {csv_path}")
        print(f"Saved combined table -> {xlsx_path}")
    else:
        print("No dipole rows were produced.")


if __name__ == "__main__":
    run()
