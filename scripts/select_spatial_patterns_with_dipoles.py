import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from h5py import File
from mne.transforms import Transform
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_astrosync_dipole_analysis import (  # noqa: E402
    baseline_covariance,
    fit_component_dipole,
    roi_membership,
    short_roi_text,
    sphere_model,
)
from src.analysis.csp_component_scores import (  # noqa: E402
    GOOD_CHANNEL_LABELS,
    PHYSIO_ROI_CHANNELS_CONTRA,
    PHYSIO_ROI_CHANNELS_IPSI,
    WEIGHTS_CONTRA,
    WEIGHTS_IPSI,
    calculate_weighted_score,
)
from src.analysis.evaluate_spatial_patterns import (  # noqa: E402
    calculate_eigenscore,
    score_spatial_patterns_physio,
)
from src.analysis.features import get_csp_features  # noqa: E402
from src.analysis.preprocessing import bandpass_filter, read_good_epoch_mask  # noqa: E402
from src.utils.montage_processing import get_channel_names  # noqa: E402

try:
    from scripts.calculate_csp import config_csp as CURRENT_CSP_CONFIG  # noqa: E402
except Exception:
    CURRENT_CSP_CONFIG = {
        "bands": [[8, 12], [9, 13], [10, 14], [8, 15]],
        "robust": True,
        "concat": True,
        "regularization": False,
        "alpha": 0.1,
    }

DEFAULT_PROJECT = "pr_Agency_EBCI"
DEFAULT_STAGE = "test"
DEFAULT_SUBJECT = "03_30 Artem"
DEFAULT_RECORD = "01_calib"
DEFAULT_OUTPUT_FOLDER = "spatial_patterns_selection"
DEFAULT_OUTPUT_ROOT = Path("results") / "algorihm_test"
MONTAGE_PATH = PROJECT_ROOT / "resources" / "mks64_standard.ced"
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
BRIER_SCORE_PENALTY_L = 5.0
TOP_COLUMN = "\u0442\u043e\u043f"


def read_json_dataset(h5f, name):
    if name not in h5f:
        return {}
    value = h5f[name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def write_table(df, output_root, basename):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{basename}.csv"
    xlsx_path = output_root / f"{basename}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def good_channel_names():
    return [ch for ch in get_channel_names(str(MONTAGE_PATH)) if ch not in BAD_CHANNELS]


def load_epochs_with_metadata(epochs_path):
    with File(epochs_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
        metadata = read_json_dataset(h5f, "metadata")
        good_mask = read_good_epoch_mask(h5f, len(epochs))

    return epochs[good_mask], labels[good_mask], metadata


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


def component_candidates(n_components, edge_count=3):
    first = list(range(min(edge_count, n_components)))
    last_start = max(len(first), n_components - edge_count)
    last = list(range(last_start, n_components))
    return first + [component for component in last if component not in first]


def component_relative_label(component, n_components, edge_count=3):
    if component < edge_count:
        return component
    return component - n_components


def band_to_text(band):
    values = [int(x) if float(x).is_integer() else float(x) for x in band]
    return f"{values[0]}-{values[1]}"


def band_to_json(band):
    values = [int(x) if float(x).is_integer() else float(x) for x in band]
    return json.dumps(values)


def gof_coef(gof, low=0.2, high=0.7, min_coef=0.2):
    if not np.isfinite(gof):
        return 0.0
    if gof < low:
        return 0.0
    if gof >= high:
        return 1.0
    x = (gof - low) / (high - low)
    return min_coef + (1 - min_coef) * x


def normalize_gof(gof):
    if not np.isfinite(gof):
        return np.nan
    return gof / 100.0 if gof > 1.0 else gof


def eigengap(eigvals, component):
    eigvals = np.asarray(eigvals, dtype=float)
    value = float(eigvals[component])
    if value < 0.5 and component + 1 < len(eigvals):
        return abs(float(eigvals[component + 1]) - value)
    if value >= 0.5 and component > 0:
        return abs(value - float(eigvals[component - 1]))
    return 0.0


def fit_selected_dipoles(spatial_patterns, selected, mne_epochs, cov, sphere):
    rows = {}
    for component in selected:
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


def score_matrix_components(
    matrix_path,
    project,
    stage,
    subject,
    record,
    spatial_patterns,
    eigvals,
    metadata_csp,
    dipoles,
    eigenscore_method="logit",
):
    n_components = spatial_patterns.shape[1]
    selected = component_candidates(n_components)
    selected_patterns = np.abs(spatial_patterns.T[selected, :])
    eigscore = calculate_eigenscore(eigvals[selected], method=eigenscore_method)
    eigscore_multiplier = 1 + eigscore if eigenscore_method == "abs_diff" else eigscore

    weighted_contra = calculate_weighted_score(selected_patterns, WEIGHTS_CONTRA)
    weighted_ipsi = calculate_weighted_score(selected_patterns, WEIGHTS_IPSI)
    physio_contra = score_spatial_patterns_physio(
        patterns=selected_patterns,
        ch_names=GOOD_CHANNEL_LABELS,
        roi_channels=PHYSIO_ROI_CHANNELS_CONTRA,
    )
    physio_ipsi = score_spatial_patterns_physio(
        patterns=selected_patterns,
        ch_names=GOOD_CHANNEL_LABELS,
        roi_channels=PHYSIO_ROI_CHANNELS_IPSI,
    )

    band = metadata_csp.get("band", [])
    rows = []
    for order, component in enumerate(selected):
        fit_row = dipoles.get(component, {})
        gof_norm = normalize_gof(float(fit_row.get("gof", np.nan)))
        gof_score = gof_coef(gof_norm)
        gap = eigengap(eigvals, component)
        locality = float(physio_contra["locality"][order])
        contra_score = float(weighted_contra[order] * (1 + locality))
        ipsi_score = float(weighted_ipsi[order] * (1 + locality))
        final_score = float(
            eigscore_multiplier[order]
            * (contra_score + ipsi_score)
            * (1 + gap)
            * (1 + gof_score)
        )

        row = {
            "project": project,
            "stage": stage,
            "subject": subject,
            "record": record,
            "matrix": matrix_path.name,
            "matrix_path": str(matrix_path),
            "band": band_to_json(band),
            "component": int(component),
            "component_relative": int(component_relative_label(component, n_components)),
            "component_1based": int(component + 1),
            "edge_group": "first3" if component < 3 else "last3",
            "eigenvalue": float(eigvals[component]),
            "eigscore_method": eigenscore_method,
            "eigscore": float(eigscore[order]),
            "eigscore_multiplier": float(eigscore_multiplier[order]),
            "eigengap": float(gap),
            "weighted_contra": float(weighted_contra[order]),
            "weighted_ipsi": float(weighted_ipsi[order]),
            "locality": locality,
            "contrast_contra": float(physio_contra["contrast"][order]),
            "contrast_ipsi": float(physio_ipsi["contrast"][order]),
            "contra_score": contra_score,
            "ipsi_score": ipsi_score,
            "gof_norm": float(gof_norm) if np.isfinite(gof_norm) else np.nan,
            "gof_coef": float(gof_score),
            "final_score": final_score,
        }
        row.update(fit_row)
        rows.append(row)

    return pd.DataFrame(rows)


def read_matrix(matrix_path):
    with File(matrix_path, "r") as h5f:
        spatial_patterns = h5f["projForward"][:]
        spatial_filters = h5f["projInverse"][:]
        eigvals = h5f["evals"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")
    return spatial_patterns, spatial_filters, eigvals, metadata_csp


def normalize_band(band):
    if band is None:
        return tuple()
    return tuple(float(value) for value in band)


def current_csp_config(args):
    config = dict(CURRENT_CSP_CONFIG)
    if args.csp_alpha is not None:
        config["alpha"] = args.csp_alpha
    if args.csp_regularization is not None:
        config["regularization"] = args.csp_regularization
    return config


def csp_metadata_matches_config(metadata_csp, config):
    for key in ("robust", "concat", "regularization"):
        if bool(metadata_csp.get(key)) != bool(config.get(key)):
            return False

    if float(metadata_csp.get("alpha", np.nan)) != float(config.get("alpha", np.nan)):
        return False

    expected_bands = {normalize_band(band) for band in config.get("bands", [])}
    matrix_band = metadata_csp.get("band")
    if expected_bands and normalize_band(matrix_band) not in expected_bands:
        return False

    return True


def matrix_matches_current_csp_config(matrix_path, config):
    try:
        with File(matrix_path, "r") as h5f:
            metadata_csp = read_json_dataset(h5f, "metadata_csp")
    except Exception as exc:
        print(f"  Skip {matrix_path.name}: cannot read metadata_csp ({exc}).")
        return False

    if not csp_metadata_matches_config(metadata_csp, config):
        band = metadata_csp.get("band")
        print(
            f"  Skip {matrix_path.name}: CSP config mismatch "
            f"(band={band}, regularization={metadata_csp.get('regularization')}, "
            f"alpha={metadata_csp.get('alpha')})."
        )
        return False

    return True


def discover_matrices(folder_csp, record, matrix_pattern, config_csp):
    pattern = matrix_pattern.format(record=record)
    candidates = sorted(path for path in folder_csp.glob(pattern) if path.is_file())
    return [path for path in candidates if matrix_matches_current_csp_config(path, config_csp)]


def record_stem_from_epoch_path(epoch_path):
    stem = epoch_path.stem
    return stem[len("EPOCHS_") :] if stem.startswith("EPOCHS_") else stem


def discover_records(project, stage, subjects=None, records=None):
    trans_root = PROJECT_ROOT / "data" / project / "trans" / stage
    if not trans_root.exists():
        raise FileNotFoundError(f"Epoch root not found: {trans_root}")

    subject_filter = set(subjects or [])
    record_filter = set(records or [])
    discovered = []
    for subject_folder in sorted(path for path in trans_root.iterdir() if path.is_dir()):
        if subject_filter and subject_folder.name not in subject_filter:
            continue
        for epoch_path in sorted(subject_folder.glob("EPOCHS_*.hdf")):
            record = record_stem_from_epoch_path(epoch_path)
            if record_filter and record not in record_filter:
                continue
            discovered.append((subject_folder.name, record))
    return discovered


def bandpass_epochs(epochs, fs, band):
    return np.array([bandpass_filter(epoch, fs=fs, low=band[0], high=band[1])[0] for epoch in epochs])


def features_for_components(epochs, spatial_filters, band, components, fs):
    epochs_band = bandpass_epochs(epochs, fs=fs, band=band)
    epochs_csp = np.array([epoch @ spatial_filters[:, components] for epoch in epochs_band])
    return get_csp_features(epochs_csp)


def classifier_predictions(features, labels, cv_splits, random_state):
    class_counts = np.unique(labels, return_counts=True)[1]
    n_splits = min(cv_splits, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two epochs per class for cross-validated probabilities.")

    classifier = LDA()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = cross_val_predict(classifier, features, labels, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    classifier.fit(features, labels)
    y_proba_insample = classifier.predict_proba(features)[:, 1]
    return y_pred, y_proba, y_proba_insample, n_splits


def safe_roc_auc(labels, y_proba):
    try:
        return float(roc_auc_score(labels, y_proba))
    except ValueError:
        return np.nan


def safe_log_loss(labels, y_proba):
    try:
        return float(log_loss(labels, np.column_stack([1 - y_proba, y_proba]), labels=[0, 1]))
    except ValueError:
        return np.nan


def evaluate_component_sets(
    matrix_path,
    spatial_filters,
    metadata_csp,
    df_components,
    epochs,
    labels,
    fs,
    min_components,
    max_components,
    cv_splits,
    random_state,
):
    band = metadata_csp.get("band")
    if band is None:
        raise ValueError(f"Matrix has no band metadata: {matrix_path}")

    n_components = spatial_filters.shape[1]
    candidates = component_candidates(n_components)
    max_components = min(max_components, len(candidates))
    score_by_component = df_components.set_index("component")["final_score"].to_dict()
    rows = []
    prediction_cache = {}

    for n_selected in range(min_components, max_components + 1):
        for components in combinations(candidates, n_selected):
            components = list(components)
            features = features_for_components(epochs, spatial_filters, band, components, fs)
            y_pred, y_proba, y_proba_insample, n_splits = classifier_predictions(
                features=features,
                labels=labels,
                cv_splits=cv_splits,
                random_state=random_state,
            )
            component_scores = [float(score_by_component[component]) for component in components]
            component_score_mean = float(np.mean(component_scores))
            brier = float(brier_score_loss(labels, y_proba))
            balanced_accuracy = float(balanced_accuracy_score(labels, y_pred))
            ranking_score = float(
                component_score_mean
                * (1 + balanced_accuracy)
                * (1 + 1 / (1 + BRIER_SCORE_PENALTY_L * brier))
            )
            relative_components = [
                int(component_relative_label(component, n_components)) for component in components
            ]

            cache_key = tuple(components)
            prediction_cache[cache_key] = {
                "features": features,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "y_proba_insample": y_proba_insample,
            }
            rows.append(
                {
                    "matrix": matrix_path.name,
                    "matrix_path": str(matrix_path),
                    "band": band_to_json(band),
                    "components": json.dumps(relative_components),
                    "absolute_components": json.dumps([int(component) for component in components]),
                    "n_components": int(n_selected),
                    "component_score_values": json.dumps(component_scores),
                    "component_score_mean": component_score_mean,
                    "component_score_sum": float(np.sum(component_scores)),
                    "component_score_min": float(np.min(component_scores)),
                    "classifier": "lda",
                    "cv_splits": int(n_splits),
                    "accuracy": float(accuracy_score(labels, y_pred)),
                    "balanced_accuracy": balanced_accuracy,
                    "roc_auc": safe_roc_auc(labels, y_proba),
                    "f1": float(f1_score(labels, y_pred, zero_division=0)),
                    "recall": float(recall_score(labels, y_pred, zero_division=0)),
                    "precision": float(precision_score(labels, y_pred, zero_division=0)),
                    "brier_score": brier,
                    "log_loss": safe_log_loss(labels, y_proba),
                    "ranking_score": ranking_score,
                }
            )

    return pd.DataFrame(rows), prediction_cache


def plot_components_with_eigenvalues(
    matrix_path,
    output_path,
    spatial_patterns,
    eigvals,
    df_components,
    selected_components,
    info,
):
    selected_set = set(selected_components)
    rows_by_component = df_components.set_index("component").to_dict("index")

    fig = plt.figure(figsize=(4.5 + 2.4 * len(selected_components), 4.8))
    gs = fig.add_gridspec(1, len(selected_components) + 1, width_ratios=[1.6] + [1] * len(selected_components))
    ax_eig = fig.add_subplot(gs[0, 0])
    x = np.arange(len(eigvals))
    ax_eig.plot(x, eigvals, color="black", linewidth=1.2)
    ax_eig.scatter(x, eigvals, s=12, color="black", alpha=0.6)
    ax_eig.scatter(selected_components, eigvals[selected_components], s=55, color="crimson", zorder=3)
    ax_eig.set_ylim(0, 1)
    ax_eig.set_title("Eigenvalues")
    ax_eig.set_xlabel("component")
    ax_eig.set_ylabel("lambda")
    ax_eig.grid(alpha=0.25)

    selected_patterns = spatial_patterns[:, selected_components]
    vmax = float(np.nanmax(np.abs(selected_patterns)))
    vlim = (-vmax, vmax) if np.isfinite(vmax) and vmax > 0 else (None, None)
    image = None
    for index, component in enumerate(selected_components, start=1):
        ax = fig.add_subplot(gs[0, index])
        image, _ = mne.viz.plot_topomap(
            spatial_patterns[:, component],
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
        score_row = rows_by_component.get(component, {})
        score = score_row.get("final_score", np.nan)
        gof = score_row.get("gof", np.nan)
        roi = short_roi_text(score_row)
        title = (
            f"CSP {component}\n"
            f"lambda={eigvals[component]:.3f}\n"
            f"score={score:.2f}, GOF={gof:.1f}\n"
            f"{roi}"
        )
        ax.set_title(title, fontsize=9, fontweight="bold" if component in selected_set else "normal")

    if image is not None:
        cbar = fig.colorbar(image, ax=fig.axes, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(matrix_path.name, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_probability(output_path, labels, y_proba, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.2))
    x = np.arange(len(labels))
    ax.step(x, labels, where="mid", label="class", linewidth=2.4, color="black", alpha=0.55)
    ax.plot(x, y_proba, label="predicted P(class 1)", linewidth=1.8, color="tab:blue")
    ax.axhline(0.5, linewidth=0.8, color="darkgrey")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_component_spectra(output_path, epochs, labels, spatial_filters, components, fs, band, title):
    csp_epochs = np.array([epoch @ spatial_filters[:, components] for epoch in epochs])
    nperseg = min(int(2 * fs), csp_epochs.shape[1])
    freqs, psd = welch(csp_epochs, fs=fs, nperseg=nperseg, axis=1)

    fig, axes = plt.subplots(len(components), 1, figsize=(8, max(2.3, 2.2 * len(components))), sharex=True)
    if len(components) == 1:
        axes = [axes]

    for ax, component_index, component in zip(axes, range(len(components)), components):
        for label, color in [(0, "tab:orange"), (1, "tab:blue")]:
            class_psd = psd[labels == label, :, component_index]
            if len(class_psd) == 0:
                continue
            mean_psd = class_psd.mean(axis=0)
            ax.plot(freqs, mean_psd, color=color, linewidth=1.5, label=f"class {label}")
        ax.axvspan(float(band[0]), float(band[1]), color="grey", alpha=0.15)
        ax.set_xlim(1, min(45, fs / 2))
        ax.set_ylabel(f"CSP {component}\nPSD")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Hz")
    fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def add_plot_paths(df_ranked, output_root):
    df_ranked = df_ranked.copy()
    df_ranked["component_plot"] = ""
    df_ranked["probability_plot"] = ""
    df_ranked["spectra_plot"] = ""
    for idx, row in df_ranked.head(2).iterrows():
        top = int(row[TOP_COLUMN])
        top_dir = output_root / "figures" / f"top_{top:02d}"
        df_ranked.at[idx, "component_plot"] = str(top_dir / "components_eigenvalues.png")
        df_ranked.at[idx, "probability_plot"] = str(top_dir / "predicted_probability.png")
        df_ranked.at[idx, "spectra_plot"] = str(top_dir / "component_spectra.png")
    return df_ranked


def process_record(args):
    project = args.project
    stage = args.stage
    subject = args.subject
    record = args.record

    folder_epochs = PROJECT_ROOT / "data" / project / "trans" / stage / subject
    folder_csp = PROJECT_ROOT / "data" / project / "features" / "csp" / stage / subject
    epochs_path = folder_epochs / f"EPOCHS_{record}.hdf"
    output_root = PROJECT_ROOT / Path(args.output_root) / project / stage / subject / record / args.output_folder
    table_root = output_root / "tables"

    if not epochs_path.exists():
        raise FileNotFoundError(f"Epoch file not found: {epochs_path}")
    if not folder_csp.exists():
        print(f"Skip {stage}/{subject}/{record}: CSP folder not found: {folder_csp}")
        return pd.DataFrame()

    config_csp = current_csp_config(args)
    matrices = discover_matrices(folder_csp, record, args.matrix_pattern, config_csp)
    if not matrices:
        print(
            f"Skip {stage}/{subject}/{record}: no CSP matrices matched "
            f"{args.matrix_pattern!r} and current CSP config."
        )
        return pd.DataFrame()

    epochs, labels, metadata = load_epochs_with_metadata(epochs_path)
    channel_names = good_channel_names()
    mne_epochs = make_mne_epochs(epochs, labels, metadata, channel_names)
    cov = baseline_covariance(mne_epochs)
    sphere = sphere_model(mne_epochs)
    fs = float(metadata.get("Fs", 1000))

    component_rows = []
    ranking_rows = []
    plot_context = {}

    for matrix_path in matrices:
        print(f"Matrix {matrix_path.name}")
        spatial_patterns, spatial_filters, eigvals, metadata_csp = read_matrix(matrix_path)
        selected = component_candidates(spatial_patterns.shape[1])
        dipoles = fit_selected_dipoles(
            spatial_patterns=spatial_patterns,
            selected=selected,
            mne_epochs=mne_epochs,
            cov=cov,
            sphere=sphere,
        )
        eigenscore_method = metadata_csp.get("eigenscore_method", args.eigenscore_method)
        df_components = score_matrix_components(
            matrix_path=matrix_path,
            project=project,
            stage=stage,
            subject=subject,
            record=record,
            spatial_patterns=spatial_patterns,
            eigvals=eigvals,
            metadata_csp=metadata_csp,
            dipoles=dipoles,
            eigenscore_method=eigenscore_method,
        )
        component_rows.append(df_components)

        df_rankings, prediction_cache = evaluate_component_sets(
            matrix_path=matrix_path,
            spatial_filters=spatial_filters,
            metadata_csp=metadata_csp,
            df_components=df_components,
            epochs=epochs,
            labels=labels,
            fs=fs,
            min_components=args.min_components,
            max_components=args.max_components,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
        )
        if not df_rankings.empty:
            df_rankings.insert(0, "project", project)
            df_rankings.insert(1, "stage", stage)
            df_rankings.insert(2, "subject", subject)
            df_rankings.insert(3, "record", record)
            ranking_rows.append(df_rankings)
            plot_context[matrix_path.name] = {
                "matrix_path": matrix_path,
                "spatial_patterns": spatial_patterns,
                "spatial_filters": spatial_filters,
                "eigvals": eigvals,
                "metadata_csp": metadata_csp,
                "df_components": df_components,
                "prediction_cache": prediction_cache,
            }

    if not component_rows or not ranking_rows:
        raise RuntimeError("No component or ranking rows were produced.")

    df_components_all = pd.concat(component_rows, ignore_index=True)
    df_rankings_all = pd.concat(ranking_rows, ignore_index=True)
    df_rankings_all = df_rankings_all.sort_values(
        ["ranking_score", "balanced_accuracy", "component_score_mean"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    df_rankings_all.insert(0, TOP_COLUMN, np.arange(1, len(df_rankings_all) + 1, dtype=int))
    df_rankings_all = add_plot_paths(df_rankings_all, output_root)

    component_csv, component_xlsx = write_table(df_components_all, table_root, "component_dipole_formula_scores")
    ranking_csv, ranking_xlsx = write_table(df_rankings_all, table_root, "component_set_rankings")

    for _, row in df_rankings_all.head(2).iterrows():
        context = plot_context[row["matrix"]]
        components = json.loads(row["absolute_components"])
        cache = context["prediction_cache"][tuple(components)]
        top = int(row[TOP_COLUMN])
        top_dir = output_root / "figures" / f"top_{top:02d}"
        band = context["metadata_csp"]["band"]
        title_suffix = (
            f"top {top}, band {band_to_text(band)}, components {row['components']}, "
            f"score {row['ranking_score']:.3f}"
        )
        plot_components_with_eigenvalues(
            matrix_path=context["matrix_path"],
            output_path=top_dir / "components_eigenvalues.png",
            spatial_patterns=context["spatial_patterns"],
            eigvals=context["eigvals"],
            df_components=context["df_components"],
            selected_components=components,
            info=mne_epochs.info,
        )
        plot_probability(
            output_path=top_dir / "predicted_probability.png",
            labels=labels,
            y_proba=cache["y_proba"],
            title=f"Cross-validated predicted probability, {title_suffix}",
        )
        plot_component_spectra(
            output_path=top_dir / "component_spectra.png",
            epochs=epochs,
            labels=labels,
            spatial_filters=context["spatial_filters"],
            components=components,
            fs=fs,
            band=band,
            title=f"Component spectra, {title_suffix}",
        )

    print(f"Saved component table: {component_xlsx}")
    print(f"Saved component CSV: {component_csv}")
    print(f"Saved ranking table: {ranking_xlsx}")
    print(f"Saved ranking CSV: {ranking_csv}")
    print(f"Saved top figures under: {output_root / 'figures'}")
    return df_rankings_all


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assess existing CSP matrices with dipoles, rank component sets from the first/last "
            "edge components, and save tables plus plots for the top two sets."
        )
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--subject", default=None, help="Optional subject filter. Default: all subjects.")
    parser.add_argument("--record", default=None, help="Optional record stem filter. Default: all records.")
    parser.add_argument(
        "--matrix-pattern",
        default="MATRIX_*_{record}.hdf",
        help="Glob pattern inside the CSP folder. Use {record} as a placeholder.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help="Root folder for outputs, relative to project root by default.",
    )
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--eigenscore-method", default="logit", choices=["logit", "abs_diff"])
    parser.add_argument(
        "--csp-alpha",
        type=float,
        default=None,
        help="Override current config_csp alpha for matrix metadata filtering.",
    )
    parser.add_argument(
        "--csp-regularization",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Override current config_csp regularization for matrix metadata filtering. "
            "Default: use scripts.calculate_csp.config_csp."
        ),
    )
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    records = discover_records(
        project=args.project,
        stage=args.stage,
        subjects=[args.subject] if args.subject else None,
        records=[args.record] if args.record else None,
    )
    if not records:
        print(f"No records found in data/{args.project}/trans/{args.stage}.")
        return

    print(f"Found {len(records)} records in data/{args.project}/trans/{args.stage}.")
    all_rankings = []
    for subject, record in records:
        print(f"\nRecord {args.stage}/{subject}/{record}")
        record_args = argparse.Namespace(**vars(args))
        record_args.subject = subject
        record_args.record = record
        try:
            df_record = process_record(record_args)
        except Exception as exc:
            print(f"Failed {args.stage}/{subject}/{record}: {exc}")
            continue
        if not df_record.empty:
            all_rankings.append(df_record)

    if not all_rankings:
        print("No ranking rows were produced.")
        return

    df_all = pd.concat(all_rankings, ignore_index=True)
    summary_root = PROJECT_ROOT / Path(args.output_root) / args.project / args.stage / args.output_folder
    write_table(df_all, summary_root, "component_set_rankings_all_records")
    print(f"Saved combined ranking table under: {summary_root}")


if __name__ == "__main__":
    main()
