import os
import sys
from pathlib import Path

from h5py import File
from matplotlib.pyplot import close


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.csp_component_scores import build_component_assessment
from src.visualization.plot_csp_components import plot_10_csp_components
from src.utils.montage_processing import get_topo_positions, get_channel_names, find_ch_idx


project = "pr_AstroSync"
stage = "exp"

config_csp = {
    "bands": [[8, 12], [9, 13], [10, 14], [8, 15]],
    "robust": True,
    "concat": True,
    "regularization": False,
    "alpha": 0.1,
}

SKIP_SUBJECTS = {"01TG", "02ES", "03AC", "04AB", "06KK", "07TS", "10AS", "11AK"}
SKIP_SUBJECTS = {"13AU", "14BE", "15AZ", "18KK", "19VB", "20EC", "21EC", "22ES", "23MM", "24EK", "25PP"}
BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
MONTAGE_PATH = r"resources/mks64_standard.ced"


def build_xy_positions():
    labels = get_channel_names(MONTAGE_PATH)
    good_channel_indices = [
        find_ch_idx(ch, MONTAGE_PATH)
        for ch in labels
        if ch not in BAD_CHANNELS
    ]
    return get_topo_positions(MONTAGE_PATH)[good_channel_indices]


XY = build_xy_positions()


def iter_subject_folders(folder_root):
    for path in sorted(Path(folder_root).iterdir()):
        if path.is_dir():
            yield path


def get_matrix_filename(record_name, band):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    return f"MATRIX_{band}_{robust}_{concat}+reg{config_csp['alpha']}_" + record_name[len("EPOCHS_") :]


def get_output_plot_path(subject_name, record_name, band):
    robust = "robust" if config_csp["robust"] else "standard"
    concat = "concat" if config_csp["concat"] else "mean"
    reg = f"reg{config_csp['alpha']}_" if config_csp["regularization"] else ""
    folder = Path("results") / project / stage / subject_name / "CSP_components_clear"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{band}_{robust}_{concat}+_{reg}" + record_name[len("EPOCHS_") : -4] + ".png"
    return folder / filename


def get_epoch_records(folder_epochs):
    return sorted(
        path.name
        for path in Path(folder_epochs).iterdir()
        if path.is_file() and path.name.startswith("EPOCHS_") and path.suffix.lower() == ".hdf"
    )


def redraw_record(folder_csp, subject_name, record_name):
    for band in config_csp["bands"]:
        matrix_path = Path(folder_csp) / get_matrix_filename(record_name, band)
        if not matrix_path.exists():
            print(f"skip missing matrix -> {matrix_path}")
            continue

        with File(matrix_path, "r") as h5f:
            proj_inverse = h5f["projInverse"][:]
            evals = h5f["evals"][:]

        component_scores = build_component_assessment(proj_inverse, evals)
        fig = plot_10_csp_components(abs(evals), proj_inverse, XY, component_scores=component_scores)

        robust = "robust" if config_csp["robust"] else "standard"
        reg = "reg" + str(config_csp["alpha"]) if config_csp["regularization"] else ""
        concat = "concat" if config_csp["concat"] else "mean"
        fig.suptitle(f"CSP: {band} Hz, {robust}, {reg}, {concat}", fontsize=16)

        output_path = get_output_plot_path(subject_name, record_name, band)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        close(fig)
        print("output file ->", output_path)


def run_subject(subject_folder):
    folder_epochs = Path("data") / project / "trans" / stage / subject_folder.name
    folder_csp = Path("data") / project / "features" / "csp" / stage / subject_folder.name

    if not folder_epochs.exists():
        print(f"Skip {subject_folder.name}: dataset folder not found -> {folder_epochs}")
        return
    if not folder_csp.exists():
        print(f"Skip {subject_folder.name}: CSP folder not found -> {folder_csp}")
        return

    records = get_epoch_records(folder_epochs)
    if not records:
        print(f"Skip {subject_folder.name}: no EPOCHS_*.hdf files found.")
        return

    print(f"\nSubject {subject_folder.name}")
    for record in records:
        print(f"Record {record}")
        redraw_record(folder_csp, subject_folder.name, record)


def run():
    folder_root = Path("data") / project / "trans" / stage
    if not folder_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {folder_root}")

    for subject_folder in iter_subject_folders(folder_root):
        if subject_folder.name in SKIP_SUBJECTS:
            continue
        run_subject(subject_folder)


if __name__ == "__main__":
    run()
