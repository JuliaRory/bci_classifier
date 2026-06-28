import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from h5py import File
from matplotlib.pyplot import close

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.csp_component_scores import build_component_assessment
from src.utils.montage_processing import find_ch_idx, get_channel_names, get_topo_positions


BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
MONTAGE_PATH = r"resources/mks64_standard.ced"
DEFAULT_PROJECT = "pr_AstroSync"
DEFAULT_STAGE = "exp"


def build_xy_positions():
    labels = get_channel_names(MONTAGE_PATH)
    good_channel_indices = [
        find_ch_idx(ch, MONTAGE_PATH)
        for ch in labels
        if ch not in BAD_CHANNELS
    ]
    return get_topo_positions(MONTAGE_PATH)[good_channel_indices]


def read_json_dataset(h5f, name):
    if name not in h5f:
        return {}

    value = h5f[name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def fallback_band_from_matrix_name(matrix_path):
    name = matrix_path.name
    start = name.find("[")
    end = name.find("]", start)
    if start < 0 or end < 0:
        return None
    return json.loads(name[start : end + 1])


def plot_filename_from_matrix(matrix_path):
    name = matrix_path.name
    if name.startswith("MATRIX_"):
        name = name[len("MATRIX_") :]
    if name.endswith(".hdf"):
        name = name[:-4] + ".png"
    return name


def title_from_metadata(metadata_csp, band):
    robust = "robust" if metadata_csp.get("robust", True) else "standard"
    reg = "reg" + str(metadata_csp.get("alpha")) if metadata_csp.get("regularization", False) else ""
    concat = "concat" if metadata_csp.get("concat", True) else "mean"
    return f"CSP: {band} Hz, {robust}, {reg}, {concat}"


def save_component_plot(matrix_path, output_path, xy, same_vlim):
    from src.visualization.plot_csp_components import plot_10_csp_components

    with File(matrix_path, "r") as h5f:
        spatial_patterns = h5f["projForward"][:]
        evals = h5f["evals"][:]
        metadata_csp = read_json_dataset(h5f, "metadata_csp")

    band = metadata_csp.get("band") or fallback_band_from_matrix_name(matrix_path)
    component_scores = build_component_assessment(spatial_patterns, evals)
    fig = plot_10_csp_components(
        abs(evals),
        spatial_patterns,
        xy,
        component_scores=component_scores,
        same_vlim=same_vlim,
    )
    fig.suptitle(title_from_metadata(metadata_csp, band), fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    close(fig)


def iter_matrix_paths(project, stage, sessions=None):
    root = Path("data") / project / "features" / "csp" / stage
    if not root.exists():
        raise FileNotFoundError(f"CSP root not found: {root}")

    session_filter = set(sessions or [])
    for session_folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if session_filter and session_folder.name not in session_filter:
            continue
        for matrix_path in sorted(session_folder.glob("MATRIX_*.hdf")):
            yield session_folder.name, matrix_path


def redraw_all(project, stage, sessions=None, variants=("clear", "regular")):
    xy = build_xy_positions()
    total = 0
    for session, matrix_path in iter_matrix_paths(project, stage, sessions=sessions):
        filename = plot_filename_from_matrix(matrix_path)
        result_root = Path("results") / project / stage / session
        outputs = []
        if "clear" in variants:
            outputs.append((result_root / "CSP_components_clear" / filename, True))
        if "regular" in variants:
            outputs.append((result_root / "CSP_components" / filename, False))

        for output_path, same_vlim in outputs:
            save_component_plot(matrix_path, output_path, xy, same_vlim=same_vlim)
            print("output file ->", output_path)
            total += 1
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Redraw CSP component images from saved AstroSync CSP matrices."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument(
        "--variant",
        choices=["both", "clear", "regular"],
        default="both",
        help="clear writes equal-vlim plots to CSP_components_clear; regular writes per-map vlim plots to CSP_components.",
    )
    args = parser.parse_args()

    variants = ("clear", "regular") if args.variant == "both" else (args.variant,)
    total = redraw_all(
        project=args.project,
        stage=args.stage,
        sessions=args.sessions,
        variants=variants,
    )
    print(f"Saved {total} CSP component images.")


if __name__ == "__main__":
    main()
