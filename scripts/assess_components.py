
import os
import json
from pathlib import Path
from h5py import File
from numpy import arange

from src.analysis.csp_component_scores import (
    build_component_assessment_table,
    save_component_assessment_table,
)
from src.utils.montage_processing import get_channel_names


EEG_CHANNELS = arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
labels_good =  [ch for ch in labels if not(ch in bad_channels)]        # labels


def process_record(full_path, folder_output, config):
    with File(full_path, "r") as h5f:
        projInverse = h5f["projInverse"][:]     # [n_channels, n_components]
        evals = h5f["evals"][:]
        metadata_csp = h5f['metadata_csp'][()]

    metadata_csp = json.loads(metadata_csp)
    filename = Path(full_path).parts[-1]
    df_results = build_component_assessment_table(
        proj_inverse=projInverse,
        evals=evals,
        metadata_csp=metadata_csp,
        session=Path(full_path).parts[-2],
        filename=filename,
    )
    save_component_assessment_table(df_results, folder_output, filename)


def process_records_assessment(folder_input, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)


project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]


config = {
    "sth": "stt"
}

if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "features", "csp", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("MATRIX") != -1]
        # records = [record for record in records if record.find("04_calib") != -1]

        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records_assessment(folder_input, records, folder_output, config)
    
