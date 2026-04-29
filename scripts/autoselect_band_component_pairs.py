import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cross_validated_test import process_record


COMPONENT_GROUP_TEMPLATES = [
    (0, -1),
    (0, 1, -1),
    (0, -2, -1),
    (0, 1, -2, -1),
]

COMPONENT_SCORE_ALIASES = {
    "final": ["final_score"],
}


def parse_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return tuple(ast.literal_eval(value))


def normalize_series(series):
    series = pd.to_numeric(series, errors="coerce")
    span = series.max() - series.min()
    if pd.isna(span) or span == 0:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / span


def find_score_column(df, aliases):
    for column in aliases:
        if column in df.columns:
            return column
    raise KeyError(f"None of the score columns were found: {aliases}")


def read_component_assessment_tables(folder_csp, record_stem):
    pattern = f"DATAFRAME_*_{record_stem}.xlsx"
    files = sorted(Path(folder_csp).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No component assessment files found in {folder_csp} for {record_stem}")

    return pd.concat([pd.read_excel(file) for file in files], ignore_index=True)


def score_component_groups(df_components):
    final_score_col = find_score_column(df_components, COMPONENT_SCORE_ALIASES["final"])

    rows = []
    for band, df_band in df_components.groupby("band", sort=False):
        df_band = df_band.copy()
        df_band["component_score"] = pd.to_numeric(df_band[final_score_col], errors="coerce")
        component_scores = df_band["component_score"].to_numpy()

        for components in COMPONENT_GROUP_TEMPLATES:
            try:
                scores = [component_scores[component] for component in components]
            except IndexError:
                continue

            rows.append(
                {
                    "band": band,
                    "sel_comp": str(components),
                    "components": list(components),
                    "absolute_components": [int(df_band["n_comp"].iloc[component]) for component in components],
                    "component_assessment_score": float(np.sum(scores)),
                    "component_score_method": final_score_col,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["band", "component_assessment_score"],
        ascending=[True, False],
        ignore_index=True,
    )


def select_best_component_group_per_band(df_component_groups):
    return (
        df_component_groups.sort_values(
            ["band", "component_assessment_score"],
            ascending=[True, False],
        )
        .groupby("band", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def summarize_cv_scores(cv_scores_path, selected_pairs):
    df_cv = pd.read_excel(cv_scores_path)
    df_cv = df_cv[df_cv["pipeline"] == "split_before_csp"].copy()
    df_cv["sel_comp"] = df_cv["sel_comp"].apply(lambda value: str(parse_tuple(value)))

    selected = selected_pairs[["band", "sel_comp", "component_assessment_score"]]
    df_cv = df_cv.merge(selected, on=["band", "sel_comp"], how="inner")
    if df_cv.empty:
        raise ValueError("No matching split_before_csp CV rows found for the selected component-band pairs.")

    summary = (
        df_cv.groupby(["session", "record", "classifier", "band", "sel_comp"], as_index=False)
        .agg(
            **{
                "balanced accuracy": ("balanced accuracy", "mean"),
                "brier score": ("brier score", "mean"),
                "component_assessment_score": ("component_assessment_score", "first"),
            }
        )
    )
    summary["components"] = summary["sel_comp"].apply(lambda value: list(parse_tuple(value)))
    summary["ranking_score"] = summary["component_assessment_score"] * (2 - summary["brier score"])

    return summary.sort_values(
        ["ranking_score"],
        ascending=[False],
        ignore_index=True,
    )


def autoselect_from_cv_scores(folder_csp, cv_scores_path, folder_output, top_n=3):
    record_stem = Path(cv_scores_path).stem
    df_components = read_component_assessment_tables(folder_csp, record_stem)
    df_component_groups = score_component_groups(df_components)
    df_selected_pairs = select_best_component_group_per_band(df_component_groups)
    df_cv_summary = summarize_cv_scores(cv_scores_path, df_selected_pairs)
    save_outputs(df_component_groups, df_selected_pairs, df_cv_summary, folder_output)
    return df_cv_summary.head(top_n).reset_index(drop=True)


def save_outputs(df_component_groups, df_selected_pairs, df_cv_summary, folder_output):
    os.makedirs(folder_output, exist_ok=True)

    all_results_path = os.path.join(folder_output, "all_autoselection_results.xlsx")
    top3_path = os.path.join(folder_output, "top3_band_component_pairs.xlsx")
    top3_json_path = os.path.join(folder_output, "top3_band_component_pairs.json")

    top3 = df_cv_summary.head(3).copy()
    top3_view = top3[["band", "components", "component_assessment_score", "balanced accuracy", "brier score", "ranking_score"]]

    with pd.ExcelWriter(all_results_path) as writer:
        df_component_groups.to_excel(writer, sheet_name="component_group_scores", index=False)
        df_selected_pairs.to_excel(writer, sheet_name="selected_pairs_for_cv", index=False)
        df_cv_summary.to_excel(writer, sheet_name="cv_split_before_csp", index=False)
        top3_view.to_excel(writer, sheet_name="top3", index=False)

    top3_view.to_excel(top3_path, index=False)
    with open(top3_json_path, "w", encoding="utf-8") as file:
        json.dump(top3_view.to_dict(orient="records"), file, ensure_ascii=False, indent=2)

    return all_results_path, top3_path, top3_json_path


def autoselect_record(full_path, folder_csp, folder_output, config, config_csp, config_cv):
    os.makedirs(folder_output, exist_ok=True)

    record_stem = Path(full_path).stem[len("EPOCHS_") :]
    df_components = read_component_assessment_tables(folder_csp, record_stem)
    df_component_groups = score_component_groups(df_components)
    df_selected_pairs = select_best_component_group_per_band(df_component_groups)

    config_cv = config_cv.copy()
    config_cv["feature_groups"] = [tuple(row["components"]) for _, row in df_selected_pairs.iterrows()]

    process_record(
        full_path=full_path,
        folder_output=folder_output,
        config=config,
        config_csp=config_csp,
        config_cv=config_cv,
    )

    cv_scores_path = os.path.join(folder_output, f"{record_stem}.xlsx")
    df_cv_summary = summarize_cv_scores(cv_scores_path, df_selected_pairs)
    output_paths = save_outputs(df_component_groups, df_selected_pairs, df_cv_summary, folder_output)

    top3_view = df_cv_summary.head(3)[["band", "components", "component_assessment_score", "balanced accuracy", "brier score", "ranking_score"]]
    print("\nTop-3 [band]-[components]-[component-assessment]-[balanced accuracy]-[brier-score]-[ranking-score]")
    for _, row in top3_view.iterrows():
        print(
            f"{row['band']}-{row['components']}-"
            f"{row['component_assessment_score']:.3f}-"
            f"{row['balanced accuracy']:.3f}-{row['brier score']:.3f}-{row['ranking_score']:.3f}"
        )
    print("\noutput files:")
    for path in output_paths:
        print(" ->", path)


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
    "classifier": "lda",
}

project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]
record_contains = "04_calib"


if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join("data", project, "trans", stage, session)
        folder_csp = os.path.join("data", project, "features", "csp", stage, session)
        folder_output = os.path.join("results", project, stage, session, "autoselection")

        records = [
            record
            for record in os.listdir(folder_input)
            if record.startswith("EPOCHS_") and record_contains in record
        ]

        for record in records:
            print(f"Record {record}")
            autoselect_record(
                full_path=os.path.join(folder_input, record),
                folder_csp=folder_csp,
                folder_output=folder_output,
                config=config,
                config_csp=config_csp,
                config_cv=config_cv,
            )
