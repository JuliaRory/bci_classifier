
import os
import json
from pathlib import Path
import copy
import pandas as pd
from h5py import File
from numpy import array, where, arange, argsort


from src.utils.montage_processing import get_channel_names

from src.analysis.features import get_csp_features


EEG_CHANNELS = arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS =  [ch for ch in labels if not(ch in bad_channels)]
# EEG_CHANNELS = array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])



def process_record(full_path, folder_output, config):
    with File(full_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f['labels'][:].squeeze()

    filename = Path(full_path).parts[-1]
    record = filename[filename.find("EPOCHS")+len("EPOCHS_"):]

    filenames = os.listdir(folder_output)
    filenames = [filename for filename in filenames if ((filename.find("MATRIX") != -1))]
    filenames = [filename for filename in filenames if (filename.find(record) != -1)]
    # print(filenames)
    
    for filename in filenames:
        path = os.path.join(folder_output, filename)
        with File(path, "r") as h5f:
            # projInverse = h5f["projInverse"][:]     # [n_channels, n_components]
            projForward = h5f["projForward"][:]
            # evals = h5f["evals"][:]
            metadata = h5f['metadata'][()]
            metadata_csp = h5f['metadata_csp'][()]
        ch_len = projForward.shape[0]
        sel_comp = [0, 1, 2, 3, 4, ch_len-5, ch_len-4, ch_len-3, ch_len-2, ch_len-1]

        epochs_csp = array([
            ep @ projForward[:, sel_comp] for ep in epochs
        ])

        features = get_csp_features(epochs_csp)

        output_filename = os.path.join(folder_output, "FEATURES_"+filename[filename.find("_")+1:])
        with File(output_filename, "w") as h5f:
            h5f.create_dataset("features", data=features)
            h5f.create_dataset("labels", data=labels)

            h5f.create_dataset("metadata", data=metadata)
            
            h5f.create_dataset("metadata_csp", data=metadata_csp)



def process_records_features(folder_input, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)


project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]

config = {
    "sel_comps": "std"
}

if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "trans", stage, session)
        records = os.listdir(folder_input)

        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records_features(folder_input, records, folder_output, config)
    