
import os

from scripts.create_dataset import process_records
from scripts.calculate_csp import process_records_csp
from scripts.calculate_features import process_records_features


stage = "test"
# project = "pr_Agency_EBCI"
# sessions = ["04_03 Artem"]


config = {
    "Fs": 1000,
    "do_filtering": True, 
    "low_freq": 5, 
    "high_freq": 35,
    "baseline_ms": 500,
    "trial_dur_ms": 4000,   # 4000  #8000
    "start_shift_ms": 1000,    # 1000   #0
    "end_shift_ms": 0,
    "epoch_len_ms": None,
    "epochs_step_ms": None, 
    "idxs_keys": "2-3" # "2-3" "1-2"
}

config_csp = {
    "bands": [[8, 12], [9, 13], [10, 14], [8, 15]], 
    "robust": True,
    "concat": True,
    "regularization": False,
    "alpha": 0.1,
}

project = "pr_Feedback_Quasi"
sessions = ["Evgeny"]

project = "pr_Agency_EBCI"
sessions = ["04_03 Artem"]

if __name__ == "__main__":

    for session in sessions:
        # CREATE DATASETS
        folder_input = os.path.join(r"data", project, "raw", stage, session)
        # folder_input = r"R:\projects_FEEDBACK_QUASI\data\tests\07 Evgeny 10.04.26"

        records = os.listdir(folder_input)
        # records = ["05_calib_QM.hdf"]   # if needed

        folder_datasets = os.path.join(r"data", project, "trans", stage, session)
        os.makedirs(folder_datasets, exist_ok=True)
        # process_records(folder_input, records, folder_datasets, config)
        print("--- dataset created ---")

        # CALCULATE CSP
        records = os.listdir(folder_datasets)
        records = [record for record in records if record.find("calib") != -1]

        folder_csp = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_csp, exist_ok=True)
        process_records_csp(folder_datasets, records, folder_csp, config, config_csp)
        print("--- csp calculated ---")

        # CALCULATE FEATURES

        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records_features(folder_datasets, records, folder_output, config)
        print("--- csp features calculated ---")

