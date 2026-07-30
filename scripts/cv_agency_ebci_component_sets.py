import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5py import File
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.features import get_csp_features  # noqa: E402
from src.analysis.preprocessing import bandpass_filter, read_good_epoch_mask  # noqa: E402


DEFAULT_PROJECT = "pr_Agency_EBCI"
DEFAULT_STAGE = "test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "algorithm"
DEFAULT_COMPONENT_SCORES = DEFAULT_OUTPUT_ROOT / "agency_ebci_component_scores_all.csv"
BRIER_SCORE_EXP_K = 10.0


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


def sanitize_filename_part(value):
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(value)).strip()
    return re.sub(r"\s+", "_", text)


def load_epochs_with_metadata(epoch_path):
    with File(epoch_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)
        metadata = read_json_dataset(h5f, "metadata")
        good_mask = read_good_epoch_mask(h5f, len(epochs))

    if not good_mask.all():
        print(f"Rejected bad epochs for {epoch_path.name}: {(~good_mask).sum()} / {len(good_mask)}")
    return epochs[good_mask], labels[good_mask], metadata


def read_matrix(matrix_path):
    with File(matrix_path, "r") as h5f:
        spatial_filters = h5f["projInverse"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")
    return spatial_filters, metadata_csp


def band_to_json(band):
    if band is None:
        return ""
    return json.dumps([int(x) if float(x).is_integer() else float(x) for x in band])


def bandpass_epochs(epochs, fs, band):
    return np.array([bandpass_filter(epoch, fs=fs, low=band[0], high=band[1])[0] for epoch in epochs])


def candidate_components(n_components):
    first = list(range(min(3, n_components)))
    last_start = max(len(first), n_components - 3)
    last = list(range(last_start, n_components))
    return first + [component for component in last if component not in first]


def component_relative_label(component, n_components):
    return int(component) if component < 3 else int(component - n_components)


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


def classifier_predictions(features, labels, cv_splits, cv_test_size):
    class_counts = np.unique(labels, return_counts=True)[1]
    max_splits = int((class_counts.min() - 1) // cv_test_size)
    n_splits = min(cv_splits, max_splits)
    if n_splits < 2:
        raise ValueError(
            "Need enough epochs per class for time-series cross-validation: "
            f"min class count={int(class_counts.min())}, test_size={cv_test_size}."
        )

    folds = build_time_series_folds(labels=labels, n_splits=n_splits, test_size=cv_test_size)
    y_proba = np.full(len(labels), np.nan, dtype=float)
    y_pred = np.full(len(labels), -1, dtype=int)

    for train_idx, test_idx in folds:
        classifier = LDA()
        classifier.fit(features[train_idx], labels[train_idx])
        fold_proba = classifier.predict_proba(features[test_idx])[:, 1]
        y_proba[test_idx] = fold_proba
        y_pred[test_idx] = (fold_proba >= 0.5).astype(int)

    tested_mask = np.isfinite(y_proba)
    return y_pred, y_proba, tested_mask, n_splits


def plot_probability(output_path, labels, y_proba, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.2))
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
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_component_scores(component_scores_path, project=None, stage=None, subject=None, record=None):
    df = pd.read_csv(component_scores_path)
    if project:
        df = df[df["project"].astype(str) == str(project)]
    if stage:
        df = df[df["stage"].astype(str) == str(stage)]
    if subject:
        df = df[df["subject"].astype(str) == str(subject)]
    if record:
        df = df[df["record"].astype(str) == str(record)]
    if df.empty:
        raise ValueError(f"No component score rows after filters in {component_scores_path}")
    return df


def component_score_lookup(df_matrix):
    by_component = df_matrix.set_index("component")
    lookup = {}
    for method_index in range(1, 6):
        column = f"final_score_{method_index}"
        lookup[column] = by_component[column].astype(float).to_dict()
    return lookup


def evaluate_matrix_sets(matrix_rows, epochs, labels, fs, args):
    matrix_path = Path(str(matrix_rows["matrix_path"].iloc[0]))
    spatial_filters, metadata_csp = read_matrix(matrix_path)
    band = metadata_csp.get("band")
    if band is None:
        band = json.loads(str(matrix_rows["band"].iloc[0]))

    candidates = candidate_components(spatial_filters.shape[1])
    epochs_band = bandpass_epochs(epochs, fs=fs, band=band)
    epochs_csp = np.array([epoch @ spatial_filters[:, candidates] for epoch in epochs_band])
    all_features = get_csp_features(epochs_csp)
    scores_by_method = component_score_lookup(matrix_rows)

    rows = []
    prediction_cache = {}
    for n_components in range(args.min_components, args.max_components + 1):
        for component_set in combinations(candidates, n_components):
            component_set = list(component_set)
            feature_columns = [candidates.index(component) for component in component_set]
            features = all_features[:, feature_columns]
            y_pred, y_proba, tested_mask, n_splits = classifier_predictions(
                features=features,
                labels=labels,
                cv_splits=args.cv_splits,
                cv_test_size=args.cv_test_size,
            )

            labels_test = labels[tested_mask]
            y_pred_test = y_pred[tested_mask]
            y_proba_test = y_proba[tested_mask]
            brier = float(brier_score_loss(labels_test, y_proba_test))
            balanced_accuracy = float(balanced_accuracy_score(labels_test, y_pred_test))
            accuracy = float(accuracy_score(labels_test, y_pred_test))

            absolute_components = [int(component) for component in component_set]
            relative_components = [
                component_relative_label(component, spatial_filters.shape[1])
                for component in component_set
            ]
            row = {
                "project": str(matrix_rows["project"].iloc[0]),
                "stage": str(matrix_rows["stage"].iloc[0]),
                "subject": str(matrix_rows["subject"].iloc[0]),
                "record": str(matrix_rows["record"].iloc[0]),
                "file": str(matrix_rows["file"].iloc[0]),
                "matrix": matrix_path.name,
                "matrix_path": str(matrix_path),
                "band": band_to_json(band),
                "components": json.dumps(relative_components),
                "absolute_components": json.dumps(absolute_components),
                "n_components": int(n_components),
                "classifier": "lda",
                "cv_method": "time_series_by_class",
                "cv_splits": int(n_splits),
                "cv_test_size": int(args.cv_test_size),
                "cv_tested_epochs": int(tested_mask.sum()),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "brier_score": brier,
            }
            for method_name, score_lookup in scores_by_method.items():
                score_values = [float(score_lookup[component]) for component in component_set]
                mean_score = float(np.mean(score_values))
                row[f"{method_name}_component_scores"] = json.dumps(score_values)
                row[f"{method_name}_mean_score"] = mean_score
                row[f"{method_name}_comp_set_score"] = float(
                    mean_score * np.exp(BRIER_SCORE_EXP_K * (0.20 - brier))
                )
            rows.append(row)

            cache_key = (str(matrix_path), tuple(absolute_components))
            prediction_cache[cache_key] = {
                "labels": labels.copy(),
                "y_proba": y_proba.copy(),
            }

    return rows, prediction_cache


def evaluate_all_sets(df_components, args):
    rows = []
    prediction_cache = {}
    epoch_cache = {}

    group_columns = ["project", "stage", "subject", "record", "matrix_path"]
    for group_values, matrix_rows in df_components.groupby(group_columns, sort=True):
        project, stage, subject, record, matrix_path = group_values
        epoch_path = Path(str(matrix_rows["epoch_path"].iloc[0]))
        if epoch_path not in epoch_cache:
            epochs, labels, metadata = load_epochs_with_metadata(epoch_path)
            epoch_cache[epoch_path] = (epochs, labels, float(metadata.get("Fs", 1000)))
        epochs, labels, fs = epoch_cache[epoch_path]

        print(f"CV {stage}/{subject}/{record}: {Path(matrix_path).name}")
        matrix_eval_rows, matrix_cache = evaluate_matrix_sets(matrix_rows, epochs, labels, fs, args)
        rows.extend(matrix_eval_rows)
        prediction_cache.update(matrix_cache)

    return pd.DataFrame(rows), prediction_cache


def build_ranked_table(df_cv, method_name):
    mean_column = f"{method_name}_mean_score"
    set_score_column = f"{method_name}_comp_set_score"
    component_scores_column = f"{method_name}_component_scores"

    columns = [
        "project",
        "stage",
        "subject",
        "record",
        "file",
        "matrix",
        "matrix_path",
        "band",
        "components",
        "absolute_components",
        "n_components",
        "classifier",
        "cv_method",
        "cv_splits",
        "cv_test_size",
        "cv_tested_epochs",
        "accuracy",
        "balanced_accuracy",
        "brier_score",
        component_scores_column,
        mean_column,
        set_score_column,
    ]
    df = df_cv[columns].copy()
    df = df.rename(
        columns={
            component_scores_column: "component_score_values",
            mean_column: "mean_score",
            set_score_column: "comp_set_score",
        }
    )
    df["final_score_method"] = method_name
    df = df.sort_values(
        ["file", "comp_set_score", "balanced_accuracy", "brier_score"],
        ascending=[True, False, False, True],
        ignore_index=True,
    )
    df["rank_within_file"] = df.groupby("file").cumcount() + 1
    df["rank_global"] = (
        df["comp_set_score"].rank(method="first", ascending=False).astype(int)
    )
    df["probability_plot"] = ""
    return df


def add_top_probability_plots(df_ranked, prediction_cache, output_root, method_name, top_n):
    df_ranked = df_ranked.copy()
    top_mask = df_ranked["rank_within_file"] <= top_n
    for idx, row in df_ranked[top_mask].iterrows():
        components = json.loads(row["absolute_components"])
        cache_key = (str(row["matrix_path"]), tuple(int(component) for component in components))
        cache = prediction_cache.get(cache_key)
        if cache is None:
            continue

        file_part = sanitize_filename_part(row["file"])
        top = int(row["rank_within_file"])
        figure_path = (
            output_root
            / "probability_plots"
            / method_name
            / file_part
            / f"top_{top:02d}_{sanitize_filename_part(row['band'])}_{sanitize_filename_part(row['components'])}.png"
        )
        title = (
            f"{row['file']} | {method_name} top {top} | "
            f"band {row['band']} | components {row['components']}"
        )
        plot_probability(figure_path, cache["labels"], cache["y_proba"], title)
        df_ranked.at[idx, "probability_plot"] = str(figure_path)
    return df_ranked


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run TimeSeries CV for CSP component sets built from first/last three components "
            "and rank sets with each final_score column."
        )
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-test-size", type=int, default=5)
    parser.add_argument("--top-plots", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    df_components = load_component_scores(
        component_scores_path=args.component_scores,
        project=args.project,
        stage=args.stage,
        subject=args.subject,
        record=args.record,
    )
    df_cv, prediction_cache = evaluate_all_sets(df_components, args)
    if df_cv.empty:
        raise RuntimeError("No CV rows were produced.")

    output_root = args.output_root
    all_csv, all_xlsx = write_table(
        df_cv,
        output_root,
        "agency_ebci_timeseries_component_set_cv_all",
    )
    print(f"Saved all CV rows: {all_xlsx}")
    print(f"Saved all CV CSV: {all_csv}")

    for method_index in range(1, 6):
        method_name = f"final_score_{method_index}"
        df_ranked = build_ranked_table(df_cv, method_name)
        df_ranked = add_top_probability_plots(
            df_ranked=df_ranked,
            prediction_cache=prediction_cache,
            output_root=output_root,
            method_name=method_name,
            top_n=args.top_plots,
        )
        csv_path, xlsx_path = write_table(
            df_ranked,
            output_root,
            f"agency_ebci_timeseries_component_set_rankings_{method_name}",
        )
        print(f"Saved {method_name} ranking: {xlsx_path}")
        print(f"Saved {method_name} ranking CSV: {csv_path}")


if __name__ == "__main__":
    main()
