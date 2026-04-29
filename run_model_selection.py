import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from h5py import File
from matplotlib.pyplot import close
from scipy.signal import butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autoselect_band_component_pairs import (
    COMPONENT_GROUP_TEMPLATES,
    autoselect_from_cv_scores,
    parse_tuple,
)
from scripts.calculate_csp import process_record as calculate_csp_record
from scripts.cross_validated_test import process_record as cross_validate_record
from src.analysis.features import get_csp_features
from src.analysis.preprocessing import bandpass_filter
from src.visualization.ROC_curve import plot_proba


config = {
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

config_csp = {
    "bands": [[8, 12], [9, 13], [10, 14], [8, 15]],
    "robust": True,
    "concat": True,
    "regularization": False,
    "alpha": 0.1,
}

config_cv = {
    "n_splits": 3,
    "test_size": 5,
    "feature_groups": COMPONENT_GROUP_TEMPLATES,
    "classifier": "lda",
}

project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]
record_contains = "04_calib"
top_n_models = 3
force_csp = False
force_cv = False


def csp_matrix_name(record, band):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    return f"MATRIX_{band}_{robust}_{concat}+reg{config_csp['alpha']}_" + record[len("EPOCHS_") :]


def csp_outputs_exist(folder_csp, record):
    expected = [Path(folder_csp) / csp_matrix_name(record, band) for band in config_csp["bands"]]
    return all(path.exists() for path in expected)


def ensure_csp_and_plots(full_path, folder_csp, record):
    if not force_csp and csp_outputs_exist(folder_csp, record):
        print("CSP matrices already exist; keeping existing component plots.")
        return

    os.makedirs(folder_csp, exist_ok=True)
    calculate_csp_record(full_path=full_path, folder_output=folder_csp, config=config, config_csp=config_csp)


def ensure_cv_scores(full_path, folder_cv):
    os.makedirs(folder_cv, exist_ok=True)
    cv_scores_path = Path(folder_cv) / f"{Path(full_path).stem[len('EPOCHS_'):]}.xlsx"
    if cv_scores_path.exists() and not force_cv:
        print("CV scores already exist; reusing them.")
        return str(cv_scores_path)

    cross_validate_record(
        full_path=full_path,
        folder_output=folder_cv,
        config=config,
        config_csp=config_csp,
        config_cv=config_cv,
    )
    return str(cv_scores_path)


def find_matrix_path(folder_csp, record_file, band):
    band_text = json.dumps(band)
    prefix = f"MATRIX_{band_text}_"
    matches = sorted(
        path
        for path in Path(folder_csp).iterdir()
        if path.name.startswith(prefix) and path.name.endswith(record_file)
    )
    if not matches:
        raise FileNotFoundError(f"No CSP matrix found for band {band_text} and record {record_file}")
    return matches[0]


def load_epochs(full_path):
    with File(full_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
    return epochs, labels


def build_model_features(epochs, spatial_filters, band, components):
    epochs_band = np.array([bandpass_filter(epoch, fs=config["Fs"], low=band[0], high=band[1])[0] for epoch in epochs])
    epochs_csp = np.array([epoch @ spatial_filters[:, components] for epoch in epochs_band])
    return get_csp_features(epochs_csp)


def train_final_classifier(epochs, labels, spatial_filters, band, components):
    features = build_model_features(epochs, spatial_filters, band, components)
    classifier = LDA()
    classifier.fit(features, labels)
    return classifier


def save_model(model, spatial_filters, band, components, output_path):
    sos_basic = butter(
        4,
        [config["low_freq"], config["high_freq"]],
        btype="bandpass",
        output="sos",
        fs=config["Fs"],
    )
    sos = butter(4, band, btype="bandpass", output="sos", fs=config["Fs"])

    model_data = {
        "spatialW": spatial_filters[:, components].tolist(),
        "sos_basic": sos_basic.tolist(),
        "sos": sos.tolist(),
        "features_type": "csp",
        "Cref": None,
        "inv_sqrt": None,
        "w_lda": model.coef_[0].tolist(),
        "b_lda": float(model.intercept_[0]),
        "sel_comp": components,
        "band": band,
        "fs": config["Fs"],
        "n_components": len(components),
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(model_data, file, indent=4)


def save_probability_plot(model, features, labels, output_path):
    y_proba = model.predict_proba(features)[:, 1]
    brier = brier_score_loss(labels, y_proba)
    fig = plot_proba(labels, y_proba)
    fig.suptitle(f"Brier score = {brier:.3f}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    close()


def save_best_models(full_path, folder_csp, folder_models, top_pairs):
    os.makedirs(folder_models, exist_ok=True)
    epochs, labels = load_epochs(full_path)
    record_file = Path(full_path).name[len("EPOCHS_") :]
    record_stem = Path(record_file).stem
    session = Path(full_path).parts[-2]
    folder_proba = Path("results") / project / stage / session / "PROBA_final"
    os.makedirs(folder_proba, exist_ok=True)

    saved_paths = []
    for rank, row in top_pairs.iterrows():
        band = json.loads(row["band"]) if isinstance(row["band"], str) else row["band"]
        components = list(parse_tuple(row["sel_comp"]))
        matrix_path = find_matrix_path(folder_csp, record_file, band)

        with File(matrix_path, "r") as h5f:
            spatial_filters = h5f["projForward"][:]

        features = build_model_features(
            epochs=epochs,
            spatial_filters=spatial_filters,
            band=band,
            components=components,
        )
        model = train_final_classifier(
            epochs=epochs,
            labels=labels,
            spatial_filters=spatial_filters,
            band=band,
            components=components,
        )

        band_text = json.dumps(band)
        output_name = f"rank{rank + 1}_feat{tuple(components)}_{band_text}_{record_stem}.json"
        output_path = Path(folder_models) / output_name
        save_model(model, spatial_filters, band, components, output_path)
        plot_output_path = folder_proba / f"{Path(output_name).stem}.png"
        save_probability_plot(model, features, labels, plot_output_path)
        saved_paths.append(str(output_path))

    return saved_paths


def print_top_pairs(top_pairs):
    print("\nTop selected [band]-[components]-[balanced accuracy]-[brier-score]")
    for _, row in top_pairs.iterrows():
        components = list(parse_tuple(row["sel_comp"]))
        print(
            f"{row['band']}-{components}-"
            f"{row['balanced accuracy']:.3f}-{row['brier score']:.3f}"
        )


def run_session(session):
    folder_epochs = Path("data") / project / "trans" / stage / session
    folder_csp = Path("data") / project / "features" / "csp" / stage / session
    folder_cv = Path("results") / project / stage / session / "cv_scores"
    folder_autoselection = Path("results") / project / stage / session / "autoselection"
    folder_models = Path("models") / project / stage / session

    records = [
        record
        for record in os.listdir(folder_epochs)
        if record.startswith("EPOCHS_") and record_contains in record
    ]

    for record in records:
        print(f"\nRecord {record}")
        full_path = str(folder_epochs / record)

        ensure_csp_and_plots(full_path, str(folder_csp), record)
        cv_scores_path = ensure_cv_scores(full_path, str(folder_cv))

        top_pairs = autoselect_from_cv_scores(
            folder_csp=str(folder_csp),
            cv_scores_path=cv_scores_path,
            folder_output=str(folder_autoselection),
            top_n=top_n_models,
        )
        print_top_pairs(top_pairs)

        model_paths = save_best_models(
            full_path=full_path,
            folder_csp=str(folder_csp),
            folder_models=str(folder_models),
            top_pairs=top_pairs,
        )

        print("\nSaved best models:")
        for path in model_paths:
            print(" ->", path)


if __name__ == "__main__":
    for session in sessions:
        run_session(session)
