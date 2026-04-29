from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from h5py import File

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.csp_component_scores import build_component_assessment


PROJECT = "pr_AstroSync"
STAGE = "exp"


def load_component_metrics(folder_root):
    rows = []
    for matrix_path in sorted(Path(folder_root).rglob("MATRIX_*.hdf")):
        with File(matrix_path, "r") as h5f:
            proj_inverse = h5f["projInverse"][:]
            evals = h5f["evals"][:]

        scores = build_component_assessment(proj_inverse, evals)
        for i in range(len(scores["n_comp"])):
            rows.append(
                {
                    "subject": matrix_path.parent.name,
                    "matrix_file": matrix_path.name,
                    "locality": float(scores["locality"][i]),
                    "contrast_contra": float(scores["contrast_contra"][i]),
                    "contrast_ipsi": float(scores["contrast_ipsi"][i]),
                }
            )

    return pd.DataFrame(rows)


def plot_boxplot_with_points(df, output_path):
    metrics = ["locality", "contrast_contra", "contrast_ipsi"]
    labels = ["Locality", "Contra contrast", "Ipsi contrast"]
    values = [df[metric].dropna().to_numpy() for metric in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=labels, showfliers=False, widths=0.5)

    for idx, metric_values in enumerate(values, start=1):
        x = [idx] * len(metric_values)
        ax.scatter(x, metric_values, alpha=0.35, s=18)

    ax.set_title("AstroSync Component Metrics")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    folder_root = Path("data") / PROJECT / "features" / "csp" / STAGE
    output_path = Path("results") / PROJECT / STAGE / "component_metrics_boxplot.png"

    df = load_component_metrics(folder_root)
    if df.empty:
        raise FileNotFoundError(f"No MATRIX_*.hdf files found under {folder_root}")

    plot_boxplot_with_points(df, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
