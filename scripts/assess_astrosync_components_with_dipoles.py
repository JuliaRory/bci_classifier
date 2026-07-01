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
from matplotlib.pyplot import close


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from run_astrosync_dipole_analysis import (
    CONFIG_CSP,
    PROJECT,
    baseline_covariance,
    component_indices,
    ensure_current_csp,
    fit_matrix_dipoles,
    good_channel_names,
    iter_epoch_records,
    iter_subject_folders,
    load_epochs,
    make_mne_epochs,
    matrix_filename,
    short_roi_text,
    sphere_model,
)
from src.analysis.csp_component_scores import (
    GOOD_CHANNEL_LABELS,
    PHYSIO_ROI_CHANNELS_CONTRA,
    PHYSIO_ROI_CHANNELS_IPSI,
    WEIGHTS_CONTRA,
    WEIGHTS_IPSI,
    calculate_weighted_score,
)
from src.analysis.evaluate_spatial_patterns import (
    calculate_eigenscore,
    score_spatial_patterns_physio,
)


DEFAULT_STAGE = "all"
OUTPUT_NAME = "component_dipole_scores"


def gof_coef(gof, low=0.2, high=0.7, min_coef=0.2):
    """GOF-based coefficient for CSP one-dipole-likeness."""
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


def score_matrix_components(df_matrix, spatial_patterns, eigvals, eigenscore_method="logit"):
    selected = component_indices(spatial_patterns.shape[1])
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
    locality = physio_contra["locality"]

    additions = []
    for order, component in enumerate(selected):
        row = df_matrix.loc[df_matrix["component"] == component].iloc[0]
        gof_percent = float(row["gof"]) if pd.notna(row["gof"]) else np.nan
        gof_norm = normalize_gof(gof_percent)
        gof_score = gof_coef(gof_norm, low=0.2, high=0.7)
        gap = eigengap(eigvals, component)

        contra_score = float(weighted_contra[order] * (1 + locality[order]))
        ipsi_score = float(weighted_ipsi[order] * (1 + locality[order]))
        final_score = float(
            eigscore_multiplier[order]
            * (contra_score + ipsi_score)
            * (1 + gap)
            * (1 + gof_score)
        )

        additions.append(
            {
                "component": int(component),
                "eigscore": float(eigscore[order]),
                "eigengap": float(gap),
                "weighted_contra": float(weighted_contra[order]),
                "weighted_ipsi": float(weighted_ipsi[order]),
                "locality": float(locality[order]),
                "contrast_contra": float(physio_contra["contrast"][order]),
                "contrast_ipsi": float(physio_ipsi["contrast"][order]),
                "contra_score": contra_score,
                "ipsi_score": ipsi_score,
                "gof_norm": float(gof_norm) if np.isfinite(gof_norm) else np.nan,
                "gof_coef": float(gof_score),
                "final_score": final_score,
            }
        )

    df_scores = pd.DataFrame(additions)
    return df_matrix.merge(df_scores, on="component", how="left")


def plot_assessed_matrix_components(
    matrix_path,
    output_path,
    spatial_patterns,
    eigvals,
    df_scores,
    fit_results,
    info,
):
    selected = component_indices(spatial_patterns.shape[1])
    score_by_component = df_scores.set_index("component")["final_score"].to_dict()

    finite_amplitudes = [
        abs(float(fit_results[component]["amplitude"]))
        for component in selected
        if component in fit_results
        and np.isfinite(fit_results[component].get("amplitude", np.nan))
    ]
    max_amplitude = max(finite_amplitudes) if finite_amplitudes else 0.0
    max_arrow_len_m = 0.05

    fig = plt.figure(figsize=(23, 8.5))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.15, 1, 1, 1, 1, 1], wspace=0.2, hspace=0.5)
    ax_eig = fig.add_subplot(gs[:, 0])
    x = np.arange(len(eigvals))
    final_scores = np.array([score_by_component.get(component, np.nan) for component in selected])

    ax_eig.plot(x, eigvals, color="black", linewidth=1.3)
    ax_eig.scatter(x, eigvals, s=16, color="black", alpha=0.65)
    scatter = ax_eig.scatter(
        selected,
        eigvals[selected],
        c=final_scores,
        s=90,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    for component in selected:
        score = score_by_component.get(component, np.nan)
        label = "n/a" if not np.isfinite(score) else f"{score:.2f}"
        ax_eig.annotate(
            label,
            (component, eigvals[component]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax_eig.set_title("CSP eigenvalues")
    ax_eig.set_xlabel("component")
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.set_ylim(0, 1)
    ax_eig.grid(alpha=0.25)
    if np.isfinite(final_scores).any():
        cbar = fig.colorbar(scatter, ax=ax_eig, fraction=0.046, pad=0.03)
        cbar.set_label("final score", fontsize=9)

    rows_by_component = df_scores.set_index("component").to_dict("index")
    for plot_idx, component in enumerate(selected):
        row = plot_idx // 5
        col = plot_idx % 5 + 1
        ax = fig.add_subplot(gs[row, col])
        pattern = spatial_patterns[:, component]
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

        fit_row = fit_results.get(component)
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

        score_row = rows_by_component.get(component, {})
        final_score = score_row.get("final_score", np.nan)
        score_text = "score=n/a" if not np.isfinite(final_score) else f"score={final_score:.2f}"
        gof = score_row.get("gof", np.nan)
        gof_text = "GOF: n/a" if not np.isfinite(gof) else f"GOF: {gof:.1f}"
        gap = score_row.get("eigengap", np.nan)
        gap_text = "gap=n/a" if not np.isfinite(gap) else f"gap={gap:.3f}"
        ax.set_title(
            f"CSP {component} | {score_text}\neig={eigvals[component]:.3f}, {gap_text}, {gof_text}\n"
            f"{short_roi_text(fit_row)}",
            fontsize=9.5,
            fontweight="bold" if np.isfinite(final_score) and final_score > 0 else "normal",
        )

    fig.suptitle(matrix_path.name, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    close(fig)


def write_tables(df, output_root, basename=OUTPUT_NAME):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{basename}.csv"
    xlsx_path = output_root / f"{basename}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"Could not write Excel table {xlsx_path}: {exc}")
    return csv_path, xlsx_path


def record_stem(record):
    stem = Path(record).stem
    return stem[len("EPOCHS_") :] if stem.startswith("EPOCHS_") else stem


def matrix_filename_from_config(record, band, config_csp):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    return f"MATRIX_{band}_{robust}_{concat}+reg{config_csp['alpha']}_" + record[len("EPOCHS_") :]


def process_record_component_dipole_scores(
    folder_epochs,
    folder_csp,
    output_root,
    record,
    config_csp,
    project=PROJECT,
    stage="",
    subject="",
    channel_names=None,
):
    folder_epochs = Path(folder_epochs)
    folder_csp = Path(folder_csp)
    output_root = Path(output_root)
    figure_root = output_root / "figures"
    table_root = output_root / "tables"

    epochs_path = folder_epochs / record
    if not epochs_path.exists():
        raise FileNotFoundError(f"Epochs file not found: {epochs_path}")

    print(f"Component scores with dipoles {stage}/{subject}/{record}")
    epochs, labels = load_epochs(epochs_path)
    mne_epochs = make_mne_epochs(epochs, labels, channel_names or good_channel_names())
    cov = baseline_covariance(mne_epochs)
    sphere = sphere_model(mne_epochs)

    rows = []
    for band in config_csp["bands"]:
        matrix_path = folder_csp / matrix_filename_from_config(record, band, config_csp)
        if not matrix_path.exists():
            print(f"  Missing CSP matrix: {matrix_path}")
            continue

        df_matrix, patterns, eigvals, fit_results = fit_matrix_dipoles(
            matrix_path=matrix_path,
            epochs_path=epochs_path,
            subject=subject,
            stage=stage,
            record_name=record,
            mne_epochs=mne_epochs,
            cov=cov,
            sphere=sphere,
        )
        df_scores = score_matrix_components(
            df_matrix,
            patterns,
            eigvals,
            eigenscore_method=config_csp.get("eigenscore_method", "logit"),
        )
        df_scores["project"] = project
        df_scores["stage"] = stage
        df_scores["subject"] = subject
        if "n_comp" not in df_scores.columns:
            df_scores["n_comp"] = df_scores["component"]
        if "evals" not in df_scores.columns and "eigenvalue" in df_scores.columns:
            df_scores["evals"] = df_scores["eigenvalue"]
        rows.append(df_scores)

        band_text = f"{band[0]}-{band[1]}"
        base_name = f"{band_text}_{matrix_path.stem}"
        record_output = record_stem(record)
        write_tables(df_scores, table_root / record_output, basename=base_name)
        plot_assessed_matrix_components(
            matrix_path=matrix_path,
            output_path=figure_root / record_output / f"{base_name}.png",
            spatial_patterns=patterns,
            eigvals=eigvals,
            df_scores=df_scores,
            fit_results=fit_results,
            info=mne_epochs.info,
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def process_records_component_dipole_scores(
    folder_epochs,
    records,
    folder_csp,
    output_root,
    config_csp,
    project=PROJECT,
    stage="",
    subject="",
):
    all_rows = []
    channel_names = good_channel_names()
    for record in records:
        df_record = process_record_component_dipole_scores(
            folder_epochs=folder_epochs,
            folder_csp=folder_csp,
            output_root=output_root,
            record=record,
            config_csp=config_csp,
            project=project,
            stage=stage,
            subject=subject,
            channel_names=channel_names,
        )
        if not df_record.empty:
            all_rows.append(df_record)

    if not all_rows:
        return pd.DataFrame()
    df_all = pd.concat(all_rows, ignore_index=True)
    write_tables(df_all, output_root)
    return df_all


def process_subject(stage, subject_folder, force_recalculate=False):
    subject = subject_folder.name
    folder_epochs = PROJECT_ROOT / "data" / PROJECT / "trans" / stage / subject
    folder_csp = PROJECT_ROOT / "data" / PROJECT / "features" / "csp" / stage / subject
    output_root = PROJECT_ROOT / "results" / PROJECT / stage / subject / OUTPUT_NAME
    figure_root = output_root / "figures"
    table_root = output_root / "tables"

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
        print(f"Component scores {stage}/{subject}/{record}")
        epochs, labels = load_epochs(epochs_path)
        mne_epochs = make_mne_epochs(epochs, labels, channel_names)
        cov = baseline_covariance(mne_epochs)
        sphere = sphere_model(mne_epochs)

        for band in CONFIG_CSP["bands"]:
            matrix_path = folder_csp / matrix_filename(record, band)
            if not matrix_path.exists():
                print(f"  Missing CSP matrix: {matrix_path}")
                continue

            df_matrix, patterns, eigvals, fit_results = fit_matrix_dipoles(
                matrix_path=matrix_path,
                epochs_path=epochs_path,
                subject=subject,
                stage=stage,
                record_name=record,
                mne_epochs=mne_epochs,
                cov=cov,
                sphere=sphere,
            )
            df_scores = score_matrix_components(
                df_matrix,
                patterns,
                eigvals,
                eigenscore_method=CONFIG_CSP.get("eigenscore_method", "logit"),
            )
            subject_rows.append(df_scores)

            band_text = f"{band[0]}-{band[1]}"
            base_name = f"{band_text}_{matrix_path.stem}"
            record_output = record_stem(record)
            write_tables(df_scores, table_root / record_output, basename=base_name)
            plot_assessed_matrix_components(
                matrix_path=matrix_path,
                output_path=figure_root / record_output / f"{base_name}.png",
                spatial_patterns=patterns,
                eigvals=eigvals,
                df_scores=df_scores,
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
        description="Assess AstroSync CSP components with eigengap and dipole GOF."
    )
    parser.add_argument("--stage", default=DEFAULT_STAGE, help="Stage to process, or 'all'. Default: all.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject names to process.")
    parser.add_argument(
        "--force-recalculate-csp",
        action="store_true",
        help="Recalculate CSP matrices even if marked current.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "all":
        trans_root = PROJECT_ROOT / "data" / PROJECT / "trans"
        stages = [path.name for path in sorted(trans_root.iterdir()) if path.is_dir()]
    else:
        stages = [args.stage]

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
        output_root = PROJECT_ROOT / "results" / PROJECT / OUTPUT_NAME
        if len(stages) == 1:
            output_root = PROJECT_ROOT / "results" / PROJECT / stages[0] / OUTPUT_NAME
        csv_path, xlsx_path = write_tables(df_all, output_root)
        print(f"Saved combined table -> {csv_path}")
        print(f"Saved combined table -> {xlsx_path}")
    else:
        print("No component score rows were produced.")


if __name__ == "__main__":
    main()
