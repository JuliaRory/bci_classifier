import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from h5py import File
from matplotlib.figure import Figure
from matplotlib.pyplot import close

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from settings.settings import Settings
from src.analysis.csp_component_scores import get_selected_component_indices
from src.analysis.features import get_csp_features
from src.analysis.preprocessing import bandpass_filter
from src.utils.montage_processing import find_ch_idx, get_channel_names, get_topo_positions


BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
MONTAGE_PATH = r"resources/mks64_standard.ced"
CSP_COLORMAP = "jet"
DEFAULT_OUTPUT_FOLDER = "best_band_component_pairs"
PREFERRED_COMPONENT_SCORE_COLUMN = "final_score_5"
BRIER_SCORE_EXP_K = float(getattr(Settings(), "brier_score_exp_k", 10.0))
SUMMARY_COLUMNS = [
    "session",
    "record",
    "rank",
    "classifier",
    "band",
    "components",
    "absolute_components",
    "component_assessment_score",
    "balanced accuracy",
    "brier score",
    "probability_plot_brier_score",
    "ranking_score",
    "component_score_values",
    "accuracy",
    "roc-auc",
    "f1",
    "recall",
    "precision",
    "log loss",
    "component_plot",
    "probability_plot",
]


def parse_literal(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_components(value):
    parsed = parse_literal(value)
    if isinstance(parsed, tuple):
        return list(parsed)
    return list(parsed)


def coerce_band(value):
    parsed = parse_literal(value)
    return [float(item) for item in parsed]


def band_text(band):
    return str([int(x) if float(x).is_integer() else x for x in band])


def record_stem_from_epochs(path):
    stem = path.stem
    return stem[len("EPOCHS_") :] if stem.startswith("EPOCHS_") else stem


def average_cv_scores_across_folds(df_scores):
    if df_scores.empty or "fold" not in df_scores.columns:
        return df_scores

    group_columns = [
        column
        for column in ["session", "record", "classifier", "band", "pipeline", "sel_comp"]
        if column in df_scores.columns
    ]
    numeric_columns = [
        column
        for column in df_scores.select_dtypes(include="number").columns
        if column != "fold"
    ]
    if not group_columns or not numeric_columns:
        return df_scores

    return (
        df_scores.groupby(group_columns, as_index=False)[numeric_columns]
        .mean()
        .sort_values(group_columns, ignore_index=True)
    )


def read_component_tables(folder_csp, record_stem):
    files = sorted(folder_csp.glob(f"DATAFRAME_*_{record_stem}.xlsx"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_excel(file) for file in files], ignore_index=True)


def component_scores_by_band(df_components):
    scores_by_band = {}
    if df_components.empty or "band" not in df_components.columns:
        return scores_by_band

    for band, df_band in df_components.groupby("band", sort=False):
        if PREFERRED_COMPONENT_SCORE_COLUMN in df_band.columns:
            component_scores = pd.to_numeric(df_band[PREFERRED_COMPONENT_SCORE_COLUMN], errors="coerce").to_numpy()
            contra_score = pd.Series(np.nan, index=df_band.index, dtype=float)
            ipsi_score = pd.Series(np.nan, index=df_band.index, dtype=float)
        elif {"final_score_contra", "final_score_ipsi"}.issubset(df_band.columns):
            contra_score = pd.to_numeric(df_band["final_score_contra"], errors="coerce")
            ipsi_score = pd.to_numeric(df_band["final_score_ipsi"], errors="coerce")
            component_scores = contra_score.add(ipsi_score, fill_value=0).to_numpy()
        elif "final_score" in df_band.columns:
            component_scores = pd.to_numeric(df_band["final_score"], errors="coerce").to_numpy()
            contra_score = pd.Series(np.nan, index=df_band.index, dtype=float)
            ipsi_score = pd.Series(np.nan, index=df_band.index, dtype=float)
        else:
            continue

        band_scores = {
            "component": component_scores,
            "contra": contra_score.to_numpy(),
            "ipsi": ipsi_score.to_numpy(),
        }
        scores_by_band[band] = band_scores
        scores_by_band[str(band)] = band_scores
    return scores_by_band


def score_selected_components(row, scores_by_band):
    band_scores = scores_by_band.get(row["band"])
    if band_scores is None:
        band_scores = scores_by_band.get(str(row["band"]))
    if band_scores is None:
        return np.nan, [], []

    component_scores = band_scores["component"]
    components = parse_components(row["sel_comp"])
    try:
        selected_scores = [component_scores[component] for component in components]
        selected_details = [
            {
                "component": int(component),
                "contra": float(band_scores["contra"][component]),
                "ipsi": float(band_scores["ipsi"][component]),
                "component_score": float(component_scores[component]),
            }
            for component in components
        ]
    except (IndexError, TypeError):
        return np.nan, [], []
    return float(np.mean(selected_scores)), [float(score) for score in selected_scores], selected_details


def prepare_pair_scores(cv_scores_path, df_components):
    df_cv = pd.read_excel(cv_scores_path)
    df_cv = average_cv_scores_across_folds(df_cv)
    if "pipeline" in df_cv.columns:
        df_cv = df_cv[df_cv["pipeline"] == "split_before_csp"].copy()
    if df_cv.empty:
        return df_cv

    scores_by_band = component_scores_by_band(df_components)
    if not scores_by_band:
        return pd.DataFrame()

    scores = df_cv.apply(lambda row: score_selected_components(row, scores_by_band), axis=1)
    df_cv = df_cv.copy()
    df_cv["component_assessment_score"] = [score for score, _, _ in scores]
    df_cv["component_score_values"] = [values for _, values, _ in scores]
    df_cv["component_score_details"] = [details for _, _, details in scores]
    df_cv = df_cv.dropna(subset=["component_assessment_score"])
    if df_cv.empty:
        return df_cv

    df_cv["components"] = df_cv["sel_comp"].apply(parse_components)
    brier_score = pd.to_numeric(df_cv["brier score"], errors="coerce")
    mean_score = pd.to_numeric(df_cv["component_assessment_score"], errors="coerce")
    df_cv["ranking_score"] = mean_score * np.exp(BRIER_SCORE_EXP_K * (0.20 - brier_score))
    return df_cv.sort_values(["ranking_score"], ascending=[False], ignore_index=True)


def find_csp_matrix(folder_csp, band, record_stem):
    text = band_text(band)
    candidates = [
        path
        for path in folder_csp.iterdir()
        if path.name.startswith(f"MATRIX_{text}_")
        and path.suffix == ".hdf"
        and path.stem.endswith(f"_{record_stem}")
    ]
    return sorted(candidates)[0] if candidates else None


def topomap_positions():
    labels = get_channel_names(MONTAGE_PATH)
    good_channel_indices = np.array(
        [find_ch_idx(ch, MONTAGE_PATH) for ch in labels if ch not in BAD_CHANNELS]
    )
    return get_topo_positions(MONTAGE_PATH)[good_channel_indices]


def selected_component_score_details(row):
    values = row.get("component_score_details", [])
    if isinstance(values, str):
        return parse_literal(values)
    if isinstance(values, list):
        return values
    return []


def save_components_plot(matrix_path, band, components, output_path, xy, row):
    from mne.viz import plot_topomap

    with File(matrix_path, "r") as h5f:
        patterns = h5f["projForward"][:]
        evals = h5f["evals"][:]

    n_components = patterns.shape[1]
    selected_pool = get_selected_component_indices(n_components)
    absolute_components = [selected_pool[component] for component in components]
    score_details = selected_component_score_details(row)

    fig = Figure(figsize=(min(7.6, 2.4 + 1.35 * len(absolute_components)), 4.1), dpi=100)
    gs = fig.add_gridspec(
        1,
        len(absolute_components) + 1,
        width_ratios=[1.6] + [1.0] * len(absolute_components),
        wspace=0.35,
    )

    ax_eigs = fig.add_subplot(gs[0, 0])
    ax_eigs.plot(evals, color="black", linewidth=1.2)
    ax_eigs.scatter(np.arange(len(evals)), evals, s=12, color="black")
    ax_eigs.scatter(absolute_components, evals[absolute_components], s=38, color="crimson", zorder=3)
    ax_eigs.set_ylim(0, 1)
    ax_eigs.set_title("Eigenvalues", fontsize=9)
    ax_eigs.tick_params(labelsize=8)

    selected_patterns = patterns[:, absolute_components]
    vmax = np.nanmax(np.abs(selected_patterns))
    vlim = (-vmax, vmax) if vmax > 0 else (None, None)
    image = None
    for index, (absolute_component, relative_component) in enumerate(
        zip(absolute_components, components),
        start=1,
    ):
        ax_map = fig.add_subplot(gs[0, index])
        image, _ = plot_topomap(
            patterns[:, absolute_component],
            xy,
            axes=ax_map,
            show=False,
            contours=0,
            sphere=0.6,
            image_interp="cubic",
            extrapolate="head",
            cmap=CSP_COLORMAP,
            vlim=vlim,
        )
        title = f"CSP {relative_component}"
        if index - 1 < len(score_details):
            detail = score_details[index - 1]
            title = (
                f"{title}\n"
                f"contra {float(detail['contra']):.2f}\n"
                f"ipsi {float(detail['ipsi']):.2f}"
            )
        ax_map.set_title(title, fontsize=9)

    if image is not None:
        cbar = fig.colorbar(image, ax=fig.axes, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    component_assessment_text = f"{float(row['component_assessment_score']):.3f}"
    ranking_score_text = f"{float(row['ranking_score']):.3f}"
    fig.suptitle(f"Band {band}. Selected components", fontsize=10)
    fig.text(
        0.5,
        0.03,
        (
            f"Comps: {component_assessment_text}. "
            f"Bal acc: {float(row['balanced accuracy']):.3f}. "
            f"Brier score: {float(row['brier score']):.3f}. "
            f"FINAL: {ranking_score_text}. "
        ),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    close(fig)
    return absolute_components


def build_probability_features(epochs, spatial_filters, band, components, fs):
    epochs_band = np.array(
        [
            bandpass_filter(epoch, fs=fs, low=band[0], high=band[1])[0]
            for epoch in epochs
        ]
    )
    epochs_csp = np.array([epoch @ spatial_filters[:, components] for epoch in epochs_band])
    return get_csp_features(epochs_csp)


def save_probability_plot(epochs_path, matrix_path, row, band, components, output_path, fs):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import brier_score_loss
    from src.visualization.ROC_curve import plot_proba

    with File(epochs_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f["labels"][:].squeeze().astype(int)

    with File(matrix_path, "r") as h5f:
        spatial_filters = h5f["projInverse"][:]

    features = build_probability_features(epochs, spatial_filters, band, components, fs)
    classifier = LDA()
    classifier.fit(features, labels)
    y_proba = classifier.predict_proba(features)[:, 1]
    brier = brier_score_loss(labels, y_proba)

    fig = plot_proba(labels, y_proba)
    fig.suptitle(
        f"{row['session']}. {Path(str(row['record'])).stem}. Band {band}. "
        f"Components {tuple(components)}. "
        f"CV Brier = {float(row['brier score']):.3f}. "
        f"Plot in-sample Brier = {brier:.3f}"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    close(fig)
    return float(brier)


def process_record(project, stage, session, record_stem, top_n, output_root, xy, fs_default):
    folder_csp = Path("data") / project / "features" / "csp" / stage / session
    folder_epochs = Path("data") / project / "trans" / stage / session
    folder_cv = Path("results") / project / stage / session / "cv_scores"
    epochs_path = folder_epochs / f"EPOCHS_{record_stem}.hdf"
    cv_scores_path = folder_cv / f"{record_stem}.xlsx"

    if not epochs_path.exists() or not cv_scores_path.exists():
        return []

    df_components = read_component_tables(folder_csp, record_stem)
    if df_components.empty:
        return []

    df_pairs = prepare_pair_scores(cv_scores_path, df_components).head(top_n).copy()
    if df_pairs.empty:
        return []

    record_output = output_root / session / record_stem
    components_output = record_output / "components"
    probabilities_output = record_output / "probabilities"
    components_output.mkdir(parents=True, exist_ok=True)
    probabilities_output.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, (_, row) in enumerate(df_pairs.iterrows(), start=1):
        band = coerce_band(row["band"])
        components = parse_components(row["sel_comp"])
        matrix_path = find_csp_matrix(folder_csp, band, record_stem)
        if matrix_path is None:
            continue

        component_plot = components_output / f"rank_{rank}.png"
        probability_plot = probabilities_output / f"rank_{rank}.png"
        absolute_components = save_components_plot(matrix_path, band, components, component_plot, xy, row)
        fs = int(row["fs"]) if "fs" in row.index and pd.notna(row["fs"]) else fs_default
        probability_brier = save_probability_plot(
            epochs_path,
            matrix_path,
            row,
            band,
            components,
            probability_plot,
            fs,
        )

        output_row = row.to_dict()
        output_row["rank"] = rank
        output_row["band"] = band
        output_row["components"] = components
        output_row["absolute_components"] = absolute_components
        output_row["probability_plot_brier_score"] = probability_brier
        output_row["component_plot"] = str(component_plot)
        output_row["probability_plot"] = str(probability_plot)
        rows.append(output_row)

    return rows


def discover_sessions(project, stage):
    folder_csp = Path("data") / project / "features" / "csp" / stage
    if not folder_csp.exists():
        return []
    return sorted(path.name for path in folder_csp.iterdir() if path.is_dir())


def discover_record_stems(project, stage, session):
    folder_epochs = Path("data") / project / "trans" / stage / session
    if not folder_epochs.exists():
        return []
    return sorted(record_stem_from_epochs(path) for path in folder_epochs.glob("EPOCHS_*.hdf"))


def write_summary_tables(df_summary, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "best_band_component_pairs_summary.xlsx"
    csv_path = output_root / "best_band_component_pairs_summary.csv"
    columns = [column for column in SUMMARY_COLUMNS if column in df_summary.columns]
    extra_columns = [column for column in df_summary.columns if column not in columns]
    df_summary = df_summary[columns + extra_columns]
    df_summary.to_excel(table_path, index=False)
    df_summary.to_csv(csv_path, index=False)

    for session, df_session in df_summary.groupby("session", sort=False):
        session_output = output_root / str(session)
        session_output.mkdir(parents=True, exist_ok=True)
        df_session.to_excel(session_output / "best_band_component_pairs_summary.xlsx", index=False)
    return table_path, csv_path


def main():
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Find best band-selected components pairs and export top-ranked plots."
    )
    parser.add_argument("--project", default="pr_AstroSync")
    parser.add_argument("--stage", default=settings.stage)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder created under results/<project>/<stage>.",
    )
    args = parser.parse_args()

    sessions = args.sessions or discover_sessions(args.project, args.stage)
    output_root = Path("results") / args.project / args.stage / args.output_folder
    xy = topomap_positions()
    fs_default = settings.preprocess.Fs

    all_rows = []
    for session in sessions:
        record_stems = discover_record_stems(args.project, args.stage, session)
        print(f"{session}: {len(record_stems)} records")
        for record_stem in record_stems:
            rows = process_record(
                args.project,
                args.stage,
                session,
                record_stem,
                args.top_n,
                output_root,
                xy,
                fs_default,
            )
            all_rows.extend(rows)
            print(f"  {record_stem}: {len(rows)} ranked pairs")

    if not all_rows:
        print("No ranked pairs were exported.")
        return

    df_summary = pd.DataFrame(all_rows)
    table_path, csv_path = write_summary_tables(df_summary, output_root)
    print(f"Saved summary table: {table_path}")
    print(f"Saved summary csv: {csv_path}")


if __name__ == "__main__":
    main()
