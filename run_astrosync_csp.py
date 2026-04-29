import os
import sys
from pathlib import Path

import json
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

from scripts.calculate_csp import process_records_csp, plot_10_comp
from scripts.cross_validated_test import process_records_cross_validated
from src.analysis.features import get_csp_features
from src.analysis.preprocessing import bandpass_filter
from src.analysis.csp_component_scores import (
    build_component_assessment,
    build_component_assessment_table,
    save_component_assessment_table,
)
from src.visualization.ROC_curve import plot_proba


config = {
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
    "sel_comps": [0, 1, -1],
    "n_feat": [3],
    "classifier": "lda",
}

project = "pr_AstroSync"
stage = "exp"
fair_pipeline_name = "split_before_csp"
recalculate_csp = False
top_n_models = 3
component_2_score_threshold = 3.0


def get_epoch_records(folder_epochs):
    return sorted(
        path.name
        for path in Path(folder_epochs).iterdir()
        if path.is_file() and path.name.startswith("EPOCHS_") and path.suffix.lower() == ".hdf"
    )


def iter_subject_folders(folder_root):
    for path in sorted(Path(folder_root).iterdir()):
        if path.is_dir():
            yield path


def get_matrix_filename(record_name, band):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    return f"MATRIX_{band}_{robust}_{concat}+reg{config_csp['alpha']}_" + record_name[len("EPOCHS_") :]


def get_plot_filename(subject_name, record_name, band):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    reg = f"reg{config_csp['alpha']}_" if config_csp["regularization"] else ""
    folder = Path("results") / project / stage / subject_name / "CSP_components"
    os.makedirs(folder, exist_ok=True)
    filename = f"{band}_{robust}_{concat}+_{reg}" + record_name[len("EPOCHS_") : -4] + ".png"
    return folder / filename


def all_csp_matrices_exist(folder_csp, record_name):
    return all((Path(folder_csp) / get_matrix_filename(record_name, band)).exists() for band in config_csp["bands"])


def redraw_existing_csp_outputs(folder_csp, subject_name, record_name):
    source_name = record_name[len("EPOCHS_") :]
    for band in config_csp["bands"]:
        matrix_path = Path(folder_csp) / get_matrix_filename(record_name, band)
        with File(matrix_path, "r") as h5f:
            proj_inverse = h5f["projInverse"][:]
            evals = h5f["evals"][:]

        metadata_csp = {**config_csp, "band": band}
        assessment_df = build_component_assessment_table(
            proj_inverse=proj_inverse,
            evals=evals,
            metadata_csp=metadata_csp,
            session=subject_name,
            filename=matrix_path.name,
        )
        save_component_assessment_table(assessment_df, str(folder_csp), matrix_path.name)

        component_scores = build_component_assessment(proj_inverse, evals)
        plot_path = get_plot_filename(subject_name, record_name, band)
        plot_10_comp(evals, proj_inverse, band, str(plot_path), config_csp, component_scores)
        print("reused CSP matrix -> ", matrix_path)


def ensure_csp_outputs(folder_epochs, folder_csp, subject_name, records):
    records_to_calculate = []
    for record in records:
        if recalculate_csp or not all_csp_matrices_exist(folder_csp, record):
            records_to_calculate.append(record)
        else:
            print(f"Record {record}")
            redraw_existing_csp_outputs(folder_csp, subject_name, record)

    if records_to_calculate:
        process_records_csp(
            folder_input=str(folder_epochs),
            records=records_to_calculate,
            folder_output=str(folder_csp),
            config=config,
            config_csp=config_csp,
        )


def parse_component_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        inner = [part.strip() for part in text[1:-1].split(",") if part.strip()]
        return tuple(int(part) for part in inner)
    raise ValueError(f"Unsupported component tuple value: {value}")


def get_feature_groups_from_component_scores(folder_csp, record_name):
    record_stem = Path(record_name).stem[len("EPOCHS_") :]
    assessment_files = sorted(Path(folder_csp).glob(f"DATAFRAME_*_{record_stem}.xlsx"))
    groups = {
        (0, -1),
        (0, 1, -1),
        (0, 1, -2, -1),
        (0, -2, -1),
        (0, 1),
    }

    if not assessment_files:
        return sorted(groups, key=lambda group: (len(group), group))

    for assessment_file in assessment_files:
        df = pd.read_excel(assessment_file)
        score_column = "final_score" if "final_score" in df.columns else None
        if score_column is None:
            continue

        component_2_rows = df[df["n_comp"].astype(int) == 2]
        if not component_2_rows.empty:
            component_2_score = float(component_2_rows[score_column].max())
            if component_2_score > component_2_score_threshold:
                groups.add((0, 2, -1))
                groups.add((0, 1, 2, -1))

    return sorted(groups, key=lambda group: (len(group), group))


def save_fair_cv_summary(folder_cv):
    folder_cv = Path(folder_cv)
    cv_files = sorted(folder_cv.glob("*.xlsx"))
    if not cv_files:
        return

    rows = []
    for cv_file in cv_files:
        df = pd.read_excel(cv_file)
        required_columns = {"pipeline", "session", "record", "classifier", "band", "sel_comp"}
        if not required_columns.issubset(df.columns):
            continue
        df_fair = df[df["pipeline"] == fair_pipeline_name].copy()
        if df_fair.empty:
            continue

        summary = (
            df_fair.groupby(["session", "record", "classifier", "band", "sel_comp"], as_index=False)
            .agg(
                folds=("fold", "nunique"),
                balanced_accuracy_mean=("balanced accuracy", "mean"),
                balanced_accuracy_std=("balanced accuracy", "std"),
                accuracy_mean=("accuracy", "mean"),
                f1_mean=("f1", "mean"),
                recall_mean=("recall", "mean"),
                precision_mean=("precision", "mean"),
                roc_auc_mean=("roc-auc", "mean"),
                brier_score_mean=("brier score", "mean"),
                log_loss_mean=("log loss", "mean"),
            )
        )
        rows.append(summary)

    if not rows:
        return

    df_summary = pd.concat(rows, ignore_index=True)
    df_summary = df_summary.sort_values(
        by=["session", "record", "balanced_accuracy_mean", "brier_score_mean"],
        ascending=[True, True, False, True],
    )
    output_path = folder_cv / "fair_cv_summary.xlsx"
    df_summary.to_excel(output_path, index=False)
    print("output file -> ", output_path)


def load_epochs(full_path):
    with File(full_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
    return epochs, labels


def build_model_features(epochs, spatial_filters, band, components):
    epochs_band = np.array([bandpass_filter(epoch, fs=config["Fs"], low=band[0], high=band[1])[0] for epoch in epochs])
    epochs_csp = np.array([epoch @ spatial_filters[:, components] for epoch in epochs_band])
    return get_csp_features(epochs_csp)


def train_final_classifier(features, labels):
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
    close(fig)


def select_top_pairs_from_cv(cv_scores_path, record_name, top_n=3):
    df = pd.read_excel(cv_scores_path)
    df = df[(df["pipeline"] == fair_pipeline_name) & (df["record"] == record_name[len("EPOCHS_") :])].copy()
    if df.empty:
        raise ValueError(f"No fair CV rows found for record {record_name}")

    summary = (
        df.groupby(["session", "record", "classifier", "band", "sel_comp"], as_index=False)
        .agg(
            **{
                "balanced accuracy": ("balanced accuracy", "mean"),
                "brier score": ("brier score", "mean"),
            }
        )
    )
    summary = summary.sort_values(
        ["balanced accuracy", "brier score"],
        ascending=[False, True],
        ignore_index=True,
    )
    return summary.head(top_n).reset_index(drop=True)


def train_and_save_final_models(folder_epochs, folder_csp, folder_cv, subject_name, record_name):
    full_path = Path(folder_epochs) / record_name
    epochs, labels = load_epochs(full_path)
    top_pairs = select_top_pairs_from_cv(Path(folder_cv) / f"{Path(record_name).stem[len('EPOCHS_'):]}.xlsx", record_name, top_n_models)

    folder_models = Path("models") / project / stage / subject_name
    folder_proba = Path("results") / project / stage / subject_name / "PROBA_final"
    os.makedirs(folder_models, exist_ok=True)
    os.makedirs(folder_proba, exist_ok=True)

    saved_models = []
    for rank, row in top_pairs.iterrows():
        band = json.loads(row["band"]) if isinstance(row["band"], str) else row["band"]
        components = list(parse_component_tuple(row["sel_comp"]))
        matrix_path = Path(folder_csp) / get_matrix_filename(record_name, band)

        with File(matrix_path, "r") as h5f:
            spatial_filters = h5f["projForward"][:]

        features = build_model_features(epochs, spatial_filters, band, components)
        model = train_final_classifier(features, labels)

        record_stem = Path(record_name[len("EPOCHS_"):]).stem
        output_name = f"rank{rank + 1}_feat{tuple(components)}_{json.dumps(band)}_{record_stem}.json"
        model_path = folder_models / output_name
        save_model(model, spatial_filters, band, components, model_path)

        plot_path = folder_proba / f"{Path(output_name).stem}.png"
        save_probability_plot(model, features, labels, plot_path)
        saved_models.append(model_path)

    return saved_models, top_pairs


def run_cross_validated_for_subject(folder_epochs, records, folder_cv, folder_csp):
    for record in records:
        feature_groups = get_feature_groups_from_component_scores(folder_csp, record)
        config_cv_record = config_cv.copy()
        config_cv_record["feature_groups"] = feature_groups

        print(f"Record {record}")
        print(f"Feature groups: {feature_groups}")
        process_records_cross_validated(
            folder_input=str(folder_epochs),
            records=[record],
            folder_output=str(folder_cv),
            config=config,
            config_csp=config_csp,
            config_cv=config_cv_record,
        )


def run_subject(subject_folder):
    folder_epochs = Path("data") / project / "trans" / stage / subject_folder.name
    if not folder_epochs.exists():
        print(f"Skip {subject_folder.name}: dataset folder not found -> {folder_epochs}")
        return

    records = get_epoch_records(folder_epochs)
    if not records:
        print(f"Skip {subject_folder.name}: no EPOCHS_*.hdf files found.")
        return

    folder_csp = Path("data") / project / "features" / "csp" / stage / subject_folder.name
    folder_cv = Path("results") / project / stage / subject_folder.name / "cv_scores"
    os.makedirs(folder_csp, exist_ok=True)
    os.makedirs(folder_cv, exist_ok=True)

    print(f"\nSubject {subject_folder.name}")
    print("--- calculate or reuse CSP and component assessment ---")
    ensure_csp_outputs(
        folder_epochs=folder_epochs,
        folder_csp=folder_csp,
        subject_name=subject_folder.name,
        records=records,
    )

    print("--- run cross-validation ---")
    run_cross_validated_for_subject(
        folder_epochs=folder_epochs,
        records=records,
        folder_cv=folder_cv,
        folder_csp=folder_csp,
    )

    print("--- save fair CV summary ---")
    save_fair_cv_summary(folder_cv)

    print("--- train final models and save probability plots ---")
    for record in records:
        top_models, top_pairs = train_and_save_final_models(
            folder_epochs=folder_epochs,
            folder_csp=folder_csp,
            folder_cv=folder_cv,
            subject_name=subject_folder.name,
            record_name=record,
        )
        print(f"Record {record}")
        for _, row in top_pairs.iterrows():
            print(
                f"  selected {row['band']}-{row['sel_comp']}-"
                f"{row['balanced accuracy']:.3f}-{row['brier score']:.3f}"
            )
        for model_path in top_models:
            print("  model ->", model_path)


def run():
    folder_root = Path("data") / project / "trans" / stage
    if not folder_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {folder_root}")

    for subject_folder in iter_subject_folders(folder_root):
        if subject_folder.name in ["01TG", "02ES", "03AC", "04AB", "06KK", "07TS", "10AS", "11AK"]:
            continue
        run_subject(subject_folder)

if __name__ == "__main__":
    run()
