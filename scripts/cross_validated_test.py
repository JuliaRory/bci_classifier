import json
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from h5py import File
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.analysis.CSP import compute_csp
from src.analysis.features import get_csp_features
from src.analysis.preprocessing import bandpass_filter
from src.analysis.csp_component_scores import get_selected_component_indices


def generate_groups(nums=None, n_features=2):
    if nums is None:
        nums = [0, 1, -1, -2]
    return [group for group in combinations(nums, n_features)]


def get_feature_groups(config_cv):
    if "feature_groups" in config_cv:
        return [tuple(group) for group in config_cv["feature_groups"]]

    feature_groups = []
    for n_features in config_cv["n_feat"]:
        feature_groups.extend(generate_groups(config_cv["sel_comps"], n_features))
    return feature_groups


def get_classifier(config_cv):
    if config_cv["classifier"] == "lda":
        return LDA()
    raise ValueError(f'Unsupported classifier: {config_cv["classifier"]}')


def bandpass_epochs(epochs, fs, band):
    return np.array([bandpass_filter(epoch, fs=fs, low=band[0], high=band[1])[0] for epoch in epochs])


def crop_epochs_for_csp(epochs, config):
    ms_to_samples = lambda x: int(x * config["Fs"] / 1000)
    baseline = ms_to_samples(config["baseline_ms"])
    start_shift = ms_to_samples(config["start_shift_ms"])
    end_shift = ms_to_samples(config["end_shift_ms"])
    end_sample = epochs.shape[1] - end_shift if end_shift > 0 else epochs.shape[1]
    return epochs[:, baseline + start_shift : end_sample, :]


def project_epochs_to_features(epochs, spatial_filters):
    selected_components = get_selected_component_indices(spatial_filters.shape[1])
    epochs_csp = np.array([epoch @ spatial_filters[:, selected_components] for epoch in epochs])
    return get_csp_features(epochs_csp)


def build_time_series_folds(labels, n_splits, test_size):
    labels = np.asarray(labels).astype(int)
    class_values = np.sort(np.unique(labels))
    if len(class_values) != 2:
        raise ValueError("Exactly two classes are required for CSP cross-validation.")

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    class_indices = {value: np.where(labels == value)[0] for value in class_values}

    split_per_class = {}
    for value, indices in class_indices.items():
        dummy = np.zeros(len(indices))
        split_per_class[value] = list(splitter.split(dummy))

    folds = []
    for fold_idx in range(n_splits):
        train_parts = []
        test_parts = []
        for value in class_values:
            train_pos, test_pos = split_per_class[value][fold_idx]
            train_parts.append(class_indices[value][train_pos])
            test_parts.append(class_indices[value][test_pos])

        train_idx = np.sort(np.concatenate(train_parts))
        test_idx = np.sort(np.concatenate(test_parts))
        folds.append((train_idx, test_idx))

    return folds


def safe_roc_auc(y_true, y_proba):
    try:
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return np.nan


def calculate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc-auc": safe_roc_auc(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "brier score": brier_score_loss(y_true, y_proba),
        "log loss": log_loss(y_true, np.column_stack([1 - y_proba, y_proba]), labels=[0, 1]),
    }


def fit_and_score_classifier(X_train, y_train, X_test, y_test, config_cv):
    classifier = get_classifier(config_cv)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]
    return calculate_metrics(y_test, y_pred, y_proba)


def evaluate_pipeline_global_csp(epochs_band, epochs_band_cropped, labels, folds, config_csp, config_cv):
    spatial_filters, _, _ = compute_csp(
        epochs_band_cropped[labels == 0],
        epochs_band_cropped[labels == 1],
        config_csp,
    )
    all_features = project_epochs_to_features(epochs_band, spatial_filters)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        for sel_comp in get_feature_groups(config_cv):
            metrics = fit_and_score_classifier(
                all_features[train_idx][:, sel_comp],
                labels[train_idx],
                all_features[test_idx][:, sel_comp],
                labels[test_idx],
                config_cv,
            )
            rows.append(
                {
                    "pipeline": "csp_before_split",
                    "fold": fold_idx,
                    "sel_comp": str(tuple(sel_comp)),
                    **metrics,
                }
            )
    return rows


def evaluate_pipeline_split_first(epochs_band, epochs_band_cropped, labels, folds, config_csp, config_cv):
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        train_epochs_cropped = epochs_band_cropped[train_idx]
        train_labels = labels[train_idx]

        spatial_filters, _, _ = compute_csp(
            train_epochs_cropped[train_labels == 0],
            train_epochs_cropped[train_labels == 1],
            config_csp,
        )

        train_features = project_epochs_to_features(epochs_band[train_idx], spatial_filters)
        test_features = project_epochs_to_features(epochs_band[test_idx], spatial_filters)

        for sel_comp in get_feature_groups(config_cv):
            metrics = fit_and_score_classifier(
                train_features[:, sel_comp],
                train_labels,
                test_features[:, sel_comp],
                labels[test_idx],
                config_cv,
            )
            rows.append(
                {
                    "pipeline": "split_before_csp",
                    "fold": fold_idx,
                    "sel_comp": str(tuple(sel_comp)),
                    **metrics,
                }
            )
    return rows


def process_record(full_path, folder_output, config, config_csp, config_cv):
    with File(full_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
        metadata = json.loads(h5f["metadata"][()])

    folds = build_time_series_folds(
        labels=labels,
        n_splits=config_cv["n_splits"],
        test_size=config_cv["test_size"],
    )

    rows = []
    for band in config_csp["bands"]:
        epochs_band = bandpass_epochs(epochs, fs=config["Fs"], band=band)
        epochs_band_cropped = crop_epochs_for_csp(epochs_band, config)

        rows.extend(
            {
                "band": json.dumps(band),
                **row,
            }
            for row in evaluate_pipeline_global_csp(
                epochs_band=epochs_band,
                epochs_band_cropped=epochs_band_cropped,
                labels=labels,
                folds=folds,
                config_csp=config_csp,
                config_cv=config_cv,
            )
        )

        rows.extend(
            {
                "band": json.dumps(band),
                **row,
            }
            for row in evaluate_pipeline_split_first(
                epochs_band=epochs_band,
                epochs_band_cropped=epochs_band_cropped,
                labels=labels,
                folds=folds,
                config_csp=config_csp,
                config_cv=config_cv,
            )
        )

    df_scores = pd.DataFrame(rows)
    df_scores.insert(0, "session", Path(full_path).parts[-2])
    df_scores.insert(1, "record", Path(full_path).name[len("EPOCHS_") :])
    df_scores.insert(2, "classifier", config_cv["classifier"])
    df_scores["fs"] = metadata["Fs"]

    output_filename = os.path.join(folder_output, Path(full_path).stem[len("EPOCHS_") :] + ".xlsx")
    df_scores.to_excel(output_filename, index=False)
    print("output file -> ", output_filename)


def process_records_cross_validated(folder_input, records, folder_output, config, config_csp, config_cv):
    for record in records:
        print(f"Record {record}")
        process_record(
            full_path=os.path.join(folder_input, record),
            folder_output=folder_output,
            config=config,
            config_csp=config_csp,
            config_cv=config_cv,
        )


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
    "sel_comps": [0, 1, -1],
    "n_feat": [3],
    "classifier": "lda",
}

project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]


if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "trans", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("04_calib") != -1]
        records = [record for record in records if record.find("EPOCHS_") != -1]
        records = [record for record in records if record.find("calib") != -1]

        folder_output = os.path.join(r"results", project, stage, session, "cv_scores")
        os.makedirs(folder_output, exist_ok=True)
        process_records_cross_validated(folder_input, records, folder_output, config, config_csp, config_cv)
