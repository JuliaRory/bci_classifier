import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from h5py import File
from mne.transforms import Transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_astrosync_dipole_analysis import (  # noqa: E402
    baseline_covariance,
    fit_component_dipole,
    roi_membership,
    sphere_model,
)
from src.analysis.CSP import compute_csp  # noqa: E402
from src.analysis.csp_component_scores import (  # noqa: E402
    GOOD_CHANNEL_LABELS,
    PHYSIO_ROI_CHANNELS_CONTRA,
    PHYSIO_ROI_CHANNELS_IPSI,
    WEIGHTS_CONTRA,
    WEIGHTS_IPSI,
    calculate_weighted_score,
    get_selected_component_indices,
)
from src.analysis.evaluate_spatial_patterns import (  # noqa: E402
    calculate_eigenscore,
    score_spatial_patterns_physio,
)
from src.analysis.preprocessing import bandpass_filter, read_good_epoch_mask  # noqa: E402
from src.utils.montage_processing import get_channel_names  # noqa: E402


PROJECT = "pr_Agency_EBCI"
STAGE = "test"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "algorithm"
MONTAGE_PATH = PROJECT_ROOT / "resources" / "mks64_standard.ced"
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]

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


def write_table(df, output_root, basename):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{basename}.csv"
    xlsx_path = output_root / f"{basename}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def is_calibration_record(record):
    record = record.lower()
    return "_calib" in record or "_calibration" in record or "_calbration" in record


def record_from_epoch_path(epoch_path):
    stem = epoch_path.stem
    return stem[len("EPOCHS_") :] if stem.startswith("EPOCHS_") else stem


def discover_epoch_paths(project, stage, subject=None, record=None):
    trans_root = PROJECT_ROOT / "data" / project / "trans" / stage
    if not trans_root.exists():
        raise FileNotFoundError(f"Epoch root not found: {trans_root}")

    for subject_dir in sorted(path for path in trans_root.iterdir() if path.is_dir()):
        if subject and subject_dir.name != subject:
            continue
        for epoch_path in sorted(subject_dir.glob("EPOCHS_*.hdf")):
            record_name = record_from_epoch_path(epoch_path)
            if record and record_name != record:
                continue
            if is_calibration_record(record_name):
                yield subject_dir.name, record_name, epoch_path


def load_epochs_with_metadata(epoch_path):
    with File(epoch_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
        metadata = {**CONFIG, **read_json_dataset(h5f, "metadata")}
        good_mask = read_good_epoch_mask(h5f, len(epochs))

    if not good_mask.all():
        print(f"Rejected bad epochs for {epoch_path.name}: {(~good_mask).sum()} / {len(good_mask)}")
    return epochs[good_mask], labels[good_mask], metadata


def good_channel_names():
    return [ch for ch in get_channel_names(str(MONTAGE_PATH)) if ch not in BAD_CHANNELS]


def make_mne_epochs(epochs, labels, metadata, channel_names):
    ced = pd.read_csv(MONTAGE_PATH, sep="\t").set_index("labels")
    ch_pos = {
        ch: np.array([-ced.loc[ch, "Y"], ced.loc[ch, "X"], ced.loc[ch, "Z"]], dtype=float)
        for ch in channel_names
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(channel_names, sfreq=float(metadata.get("Fs", 1000)), ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    info["dev_head_t"] = Transform("meg", "head", np.eye(4))

    mne_epochs = mne.EpochsArray(
        np.transpose(epochs, (0, 2, 1)),
        info,
        tmin=-float(metadata.get("baseline_ms", 500)) / 1000,
        events=np.c_[np.arange(len(labels)), np.zeros(len(labels), dtype=int), labels],
        event_id={str(label): int(label) for label in np.unique(labels)},
        verbose=False,
    )
    mne_epochs.set_eeg_reference("average", projection=True, verbose=False)
    mne_epochs.apply_proj(verbose=False)
    mne_epochs.apply_baseline((None, 0), verbose=False)
    return mne_epochs


def csp_config_name_part(config_csp):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    reg = f"+reg{config_csp['alpha']}" if config_csp.get("regularization", False) else ""
    return f"{robust}_{concat}{reg}"


def matrix_filename(epoch_path, band, config_csp):
    return f"MATRIX_{band}_{csp_config_name_part(config_csp)}_{record_from_epoch_path(epoch_path)}.hdf"


def csp_time_window(epochs, metadata):
    fs = float(metadata.get("Fs", 1000))
    ms_to_samples = lambda value: int(float(value) * fs / 1000)
    baseline = ms_to_samples(metadata.get("baseline_ms", 500))
    start_shift = ms_to_samples(metadata.get("start_shift_ms", 1000))
    end_shift = ms_to_samples(metadata.get("end_shift_ms", 0))
    return baseline + start_shift, epochs.shape[1] - end_shift


def matrix_is_readable(matrix_path):
    try:
        with File(matrix_path, "r") as h5f:
            return all(name in h5f for name in ("projForward", "projInverse", "evals"))
    except OSError:
        return False


def normalize_band(band):
    return tuple(float(value) for value in band or [])


def discover_existing_matrices(folder_csp, record):
    if not folder_csp.exists():
        return []
    matrices = sorted(folder_csp.glob(f"MATRIX_*_{record}.hdf"))
    return [path for path in matrices if matrix_is_readable(path)]


def matrix_band(matrix_path):
    try:
        with File(matrix_path, "r") as h5f:
            metadata_csp = read_json_dataset(h5f, "metadata_csp")
    except OSError:
        return tuple()
    return normalize_band(metadata_csp.get("band"))


def discover_current_matrices(folder_csp, epoch_path, config_csp):
    record = record_from_epoch_path(epoch_path)
    existing = discover_existing_matrices(folder_csp, record)
    existing_by_band = {}
    for matrix_path in existing:
        existing_by_band.setdefault(matrix_band(matrix_path), []).append(matrix_path)

    selected = []
    missing_bands = []
    for band in config_csp["bands"]:
        expected = folder_csp / matrix_filename(epoch_path, band, config_csp)
        if matrix_is_readable(expected):
            selected.append(expected)
            continue

        candidates = existing_by_band.get(normalize_band(band), [])
        if candidates:
            selected.append(candidates[0])
        else:
            missing_bands.append(band)
    return selected, missing_bands


def calculate_and_save_csp(epoch_path, folder_csp, config_csp, bands=None):
    epochs, labels, metadata = load_epochs_with_metadata(epoch_path)
    epochs_1 = epochs[np.where(labels == 0)]
    epochs_2 = epochs[np.where(labels == 1)]
    if len(epochs_1) == 0 or len(epochs_2) == 0:
        raise ValueError(
            f"Need both labels 0 and 1 to calculate CSP for {epoch_path}; "
            f"got {len(epochs_1)} and {len(epochs_2)} epochs."
        )

    start, end = csp_time_window(epochs, metadata)
    folder_csp.mkdir(parents=True, exist_ok=True)
    matrix_paths = []

    for band in bands or config_csp["bands"]:
        matrix_path = folder_csp / matrix_filename(epoch_path, band, config_csp)
        if matrix_is_readable(matrix_path):
            matrix_paths.append(matrix_path)
            continue

        epochs_1_band = np.array(
            [bandpass_filter(epoch, fs=metadata["Fs"], low=band[0], high=band[1])[0] for epoch in epochs_1]
        )
        epochs_2_band = np.array(
            [bandpass_filter(epoch, fs=metadata["Fs"], low=band[0], high=band[1])[0] for epoch in epochs_2]
        )
        spatial_filters, spatial_patterns, eigenvalues = compute_csp(
            epochs_1_band[:, start:end, :],
            epochs_2_band[:, start:end, :],
            config_csp,
        )

        metadata_csp = {**config_csp, "band": band}
        with File(matrix_path, "w") as h5f:
            h5f.create_dataset("projForward", data=spatial_patterns)
            h5f.create_dataset("projInverse", data=spatial_filters)
            h5f.create_dataset("evals", data=eigenvalues)
            h5f.attrs["csp_matrix_semantics"] = "projForward=spatial_patterns;projInverse=spatial_filters"
            h5f.attrs["projForward_kind"] = "spatial_patterns"
            h5f.attrs["projInverse_kind"] = "spatial_filters"
            h5f.create_dataset("metadata", data=json.dumps(metadata))
            h5f.create_dataset("metadata_csp", data=json.dumps(metadata_csp))
        print(f"Calculated CSP matrix: {matrix_path}")
        matrix_paths.append(matrix_path)

    return matrix_paths


def ensure_matrices(epoch_path, project, stage, subject, config_csp, include_all_matrices=False):
    record = record_from_epoch_path(epoch_path)
    folder_csp = PROJECT_ROOT / "data" / project / "features" / "csp" / stage / subject
    if include_all_matrices:
        existing = discover_existing_matrices(folder_csp, record)
        if existing:
            return existing, False
        return calculate_and_save_csp(epoch_path, folder_csp, config_csp), True

    selected, missing_bands = discover_current_matrices(folder_csp, epoch_path, config_csp)
    if not missing_bands:
        return selected, False

    calculated = calculate_and_save_csp(epoch_path, folder_csp, config_csp, bands=missing_bands)
    selected, _ = discover_current_matrices(folder_csp, epoch_path, config_csp)
    if selected:
        return selected, True
    return calculated, True


def read_matrix(matrix_path):
    with File(matrix_path, "r") as h5f:
        spatial_patterns = h5f["projForward"][:]
        spatial_filters = h5f["projInverse"][:]
        eigenvalues = h5f["evals"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")
    return spatial_patterns, spatial_filters, eigenvalues, metadata_csp


def selected_component_indices(n_components):
    selected = get_selected_component_indices(n_components)
    return [idx for i, idx in enumerate(selected) if 0 <= idx < n_components and idx not in selected[:i]]


def component_relative_label(component, n_components):
    return int(component) if component < 5 else int(component - n_components)


def edge_group(component, n_components):
    return "first5" if component < 5 else "last5"


def eigengap(eigenvalues, component):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    value = float(eigenvalues[component])
    if value < 0.5 and component + 1 < len(eigenvalues):
        return abs(float(eigenvalues[component + 1]) - value)
    if value >= 0.5 and component > 0:
        return abs(value - float(eigenvalues[component - 1]))
    return 0.0


def gof_coef(gof, low=35, high=75, min_coef=0.2):
    """GOF-based coefficient for CSP one-dipole-likeness."""
    if not np.isfinite(gof):
        return 0.0
    if gof < low:
        return 0.0
    if gof >= high:
        return 1.0
    x = (gof - low) / (high - low)
    return min_coef + (1 - min_coef) * x


def normalize_gof_percent(gof):
    if not np.isfinite(gof):
        return np.nan
    return float(gof * 100.0) if abs(float(gof)) <= 1.0 else float(gof)


def fit_selected_dipoles(spatial_patterns, selected, mne_epochs, cov, sphere, enabled=True):
    rows = {}
    for component in selected:
        if not enabled:
            row = {
                "gof": np.nan,
                "rv": np.nan,
                "x_mm": np.nan,
                "y_mm": np.nan,
                "z_mm": np.nan,
                "ori_x": np.nan,
                "ori_y": np.nan,
                "ori_z": np.nan,
                "amplitude": np.nan,
                "fit_error": "dipole fitting disabled",
            }
            row.update(roi_membership([np.nan, np.nan, np.nan]))
            rows[component] = row
            continue

        try:
            dipole, _ = fit_component_dipole(
                spatial_patterns[:, component],
                mne_epochs=mne_epochs,
                cov=cov,
                sphere=sphere,
            )
            pos_mm = dipole.pos[0] * 1000
            ori = dipole.ori[0]
            row = {
                "gof": normalize_gof_percent(float(dipole.gof[0])),
                "rv": 100.0 - normalize_gof_percent(float(dipole.gof[0])),
                "x_mm": float(pos_mm[0]),
                "y_mm": float(pos_mm[1]),
                "z_mm": float(pos_mm[2]),
                "ori_x": float(ori[0]),
                "ori_y": float(ori[1]),
                "ori_z": float(ori[2]),
                "amplitude": float(dipole.amplitude[0]),
                "fit_error": "",
            }
            row.update(roi_membership(pos_mm))
        except Exception as exc:
            row = {
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
            }
            row.update(roi_membership([np.nan, np.nan, np.nan]))
        rows[component] = row
    return rows


def band_to_json(band):
    if band is None:
        return ""
    return json.dumps([int(x) if float(x).is_integer() else float(x) for x in band])


def score_matrix_components(
    matrix_path,
    project,
    stage,
    subject,
    record,
    epoch_path,
    spatial_patterns,
    eigenvalues,
    metadata_csp,
    dipoles,
):
    n_components = spatial_patterns.shape[1]
    selected = selected_component_indices(n_components)
    patterns = np.abs(spatial_patterns.T[selected, :])

    weighted_contra = calculate_weighted_score(patterns, WEIGHTS_CONTRA)
    weighted_ipsi = calculate_weighted_score(patterns, WEIGHTS_IPSI)
    eigscore_abs_diff = calculate_eigenscore(eigenvalues[selected], method="abs_diff")
    eigscore_logit = calculate_eigenscore(eigenvalues[selected], method="logit")

    physio_contra = score_spatial_patterns_physio(
        patterns=patterns,
        ch_names=GOOD_CHANNEL_LABELS,
        roi_channels=PHYSIO_ROI_CHANNELS_CONTRA,
    )
    physio_ipsi = score_spatial_patterns_physio(
        patterns=patterns,
        ch_names=GOOD_CHANNEL_LABELS,
        roi_channels=PHYSIO_ROI_CHANNELS_IPSI,
    )

    rows = []
    band = metadata_csp.get("band")
    for order, component in enumerate(selected):
        locality = float(physio_contra["locality"][order])
        contra_local_weighted = float((1 + locality) * weighted_contra[order])
        ipsi_local_weighted = float((1 + locality) * weighted_ipsi[order])
        weighted_local_sum = contra_local_weighted + ipsi_local_weighted
        gap = eigengap(eigenvalues, component)
        gap_eigen_multiplier = 1 + 2 * float(eigscore_abs_diff[order]) * (1 + 2 * gap)
        eigen_abs_boost = 1 + 2 * float(eigscore_abs_diff[order])
        eigengap_boost = 1 + 2 * gap

        fit_row = dipoles.get(component, {})
        gof = normalize_gof_percent(float(fit_row.get("gof", np.nan)))
        gof_score = gof_coef(gof)

        final_score_1 = weighted_local_sum * float(eigscore_abs_diff[order])
        final_score_2 = weighted_local_sum * float(eigscore_logit[order])
        final_score_3 = weighted_local_sum * gap_eigen_multiplier
        final_score_4 = weighted_local_sum * eigen_abs_boost * gof_score
        final_score_5 = weighted_local_sum * eigen_abs_boost * eigengap_boost * gof_score

        row = {
            "project": project,
            "stage": stage,
            "subject": subject,
            "record": record,
            "file": f"{stage}/{subject}/{record}",
            "epoch_path": str(epoch_path),
            "matrix": matrix_path.name,
            "matrix_path": str(matrix_path),
            "band": band_to_json(band),
            "component": int(component),
            "component_1based": int(component + 1),
            "component_relative": component_relative_label(component, n_components),
            "edge_group": edge_group(component, n_components),
            "eigenvalue": float(eigenvalues[component]),
            "eigen_score_abs_diff": float(eigscore_abs_diff[order]),
            "eigen_score_logit": float(eigscore_logit[order]),
            "eigengap": float(gap),
            "eigen_gap_multiplier": float(gap_eigen_multiplier),
            "eigen_abs_boost": float(eigen_abs_boost),
            "eigengap_boost": float(eigengap_boost),
            "weighted_contra": float(weighted_contra[order]),
            "weighted_ipsi": float(weighted_ipsi[order]),
            "locality": locality,
            "locality_ipsi": float(physio_ipsi["locality"][order]),
            "contra_local_weighted": contra_local_weighted,
            "ipsi_local_weighted": ipsi_local_weighted,
            "weighted_local_sum": weighted_local_sum,
            "contrast_contra": float(physio_contra["contrast"][order]),
            "contrast_ipsi": float(physio_ipsi["contrast"][order]),
            "gof": gof,
            "gof_coef": float(gof_score),
            "final_score_1": float(final_score_1),
            "final_score_2": float(final_score_2),
            "final_score_3": float(final_score_3),
            "final_score_4": float(final_score_4),
            "final_score_5": float(final_score_5),
        }
        row.update({key: value for key, value in fit_row.items() if key not in row})
        rows.append(row)

    return pd.DataFrame(rows)


def build_ranked_table(df_scores):
    rows = []
    id_columns = [
        "project",
        "stage",
        "subject",
        "record",
        "file",
        "matrix",
        "matrix_path",
        "band",
        "component",
        "component_1based",
        "component_relative",
        "edge_group",
    ]
    metric_columns = [
        "eigenvalue",
        "eigen_score_abs_diff",
        "eigen_score_logit",
        "eigengap",
        "weighted_contra",
        "weighted_ipsi",
        "locality",
        "gof",
        "gof_coef",
    ]
    for method_index in range(1, 6):
        score_column = f"final_score_{method_index}"
        method_df = df_scores[id_columns + metric_columns + [score_column]].copy()
        method_df["final_score_method"] = score_column
        method_df["final_score"] = method_df[score_column]
        method_df = method_df.drop(columns=[score_column])
        rows.append(method_df)

    df_ranked = pd.concat(rows, ignore_index=True)
    df_ranked = df_ranked.sort_values(
        ["file", "final_score_method", "final_score"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    df_ranked["rank_within_file"] = (
        df_ranked.groupby(["file", "final_score_method"]).cumcount() + 1
    )
    df_ranked["rank_within_matrix"] = (
        df_ranked.groupby(["file", "matrix", "final_score_method"])["final_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df_ranked


def process_epoch(subject, record, epoch_path, args):
    matrix_paths, calculated = ensure_matrices(
        epoch_path=epoch_path,
        project=args.project,
        stage=args.stage,
        subject=subject,
        config_csp=CONFIG_CSP,
        include_all_matrices=args.include_all_matrices,
    )

    epochs, labels, metadata = load_epochs_with_metadata(epoch_path)
    mne_epochs = None
    cov = None
    sphere = None
    if not args.skip_dipoles:
        mne_epochs = make_mne_epochs(epochs, labels, metadata, good_channel_names())
        cov = baseline_covariance(mne_epochs)
        sphere = sphere_model(mne_epochs)

    component_tables = []
    for matrix_path in matrix_paths:
        print(f"Scoring {args.stage}/{subject}/{record}: {matrix_path.name}")
        spatial_patterns, _, eigenvalues, metadata_csp = read_matrix(matrix_path)
        selected = selected_component_indices(spatial_patterns.shape[1])
        dipoles = fit_selected_dipoles(
            spatial_patterns=spatial_patterns,
            selected=selected,
            mne_epochs=mne_epochs,
            cov=cov,
            sphere=sphere,
            enabled=not args.skip_dipoles,
        )
        component_tables.append(
            score_matrix_components(
                matrix_path=matrix_path,
                project=args.project,
                stage=args.stage,
                subject=subject,
                record=record,
                epoch_path=epoch_path,
                spatial_patterns=spatial_patterns,
                eigenvalues=eigenvalues,
                metadata_csp=metadata_csp,
                dipoles=dipoles,
            )
        )

    summary = {
        "project": args.project,
        "stage": args.stage,
        "subject": subject,
        "record": record,
        "epoch_path": str(epoch_path),
        "matrices_count": len(matrix_paths),
        "matrices_calculated": bool(calculated),
    }
    return component_tables, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate missing CSP matrices and score first/last five CSP components "
            "for calibration records in pr_Agency_EBCI."
        )
    )
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--stage", default=STAGE)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--skip-dipoles",
        action="store_true",
        help="Skip dipole fitting; GOF fields will be empty and GOF-based final scores will be zero.",
    )
    parser.add_argument(
        "--include-all-matrices",
        action="store_true",
        help="Score every MATRIX_*_{record}.hdf variant instead of one current CSP matrix per band.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    epoch_rows = list(discover_epoch_paths(args.project, args.stage, args.subject, args.record))
    if not epoch_rows:
        print(f"No calibration epoch files found in data/{args.project}/trans/{args.stage}.")
        return

    print(f"Found {len(epoch_rows)} calibration epoch files.")
    component_tables = []
    summaries = []
    for subject, record, epoch_path in epoch_rows:
        try:
            tables, summary = process_epoch(subject, record, epoch_path, args)
            component_tables.extend(tables)
            summaries.append(summary)
        except Exception as exc:
            print(f"Failed {args.stage}/{subject}/{record}: {exc}")
            summaries.append(
                {
                    "project": args.project,
                    "stage": args.stage,
                    "subject": subject,
                    "record": record,
                    "epoch_path": str(epoch_path),
                    "matrices_count": 0,
                    "matrices_calculated": False,
                    "error": str(exc),
                }
            )

    if not component_tables:
        raise RuntimeError("No component score rows were produced.")

    df_scores = pd.concat(component_tables, ignore_index=True)
    df_ranked = build_ranked_table(df_scores)
    df_summary = pd.DataFrame(summaries)

    output_root = args.output_root
    scores_csv, scores_xlsx = write_table(df_scores, output_root, "agency_ebci_component_scores_all")
    ranked_csv, ranked_xlsx = write_table(df_ranked, output_root, "agency_ebci_component_scores_ranked_by_file")
    summary_csv, summary_xlsx = write_table(df_summary, output_root, "agency_ebci_component_scores_summary")

    print(f"Saved scores: {scores_xlsx}")
    print(f"Saved scores CSV: {scores_csv}")
    print(f"Saved ranked table: {ranked_xlsx}")
    print(f"Saved ranked CSV: {ranked_csv}")
    print(f"Saved summary: {summary_xlsx}")
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
