import os
import sys
from pathlib import Path
from numpy import asarray

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_dataset import process_records


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

project = "pr_AstroSync"
stage = "exp"
file_suffix = ".hdf"


def iter_subject_folders(folder_root):
    for path in sorted(Path(folder_root).iterdir()):
        if path.is_dir():
            yield path


def get_records(folder_input):
    return sorted(
        file.name
        for file in Path(folder_input).iterdir()
        if file.is_file() and file.suffix.lower() == file_suffix
    )

processed_files = ['01TG', '02ES', '03AC', '04AB', '06KK', '07TS', '10AS', '11AK', '13AU',  '14BE', '15AZ']

def run():
    folder_root = Path("data") / project / "raw" / stage
    if not folder_root.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_root}")
    
    for subject_folder in iter_subject_folders(folder_root):

        skip = sum(asarray([str(subject_folder).find(subj) != -1 for subj in processed_files]))
        if skip:
            continue
        records = get_records(subject_folder)
        if not records:
            print(f"Skip {subject_folder.name}: no {file_suffix} files found.")
            continue

        folder_output = Path("data") / project / "trans" / stage / subject_folder.name
        os.makedirs(folder_output, exist_ok=True)

        print(f"\nSubject {subject_folder.name}")
        process_records(
            folder_input=str(subject_folder),
            records=records,
            folder_output=str(folder_output),
            config=config,
        )


if __name__ == "__main__":
    run()
