
import os

from scripts.train_classifier import process_records_clf
from scripts.test_classifier import process_records_clf_test


stage = "test"
# project = "pr_Agency_EBCI"
# sessions = ["04_03 Artem"]


config = {
    "Fs": 1000,
    "do_filtering": True, 
    "low_freq": 5, 
    "high_freq": 35,
    "baseline_ms": 500,
    "trial_dur_ms": 8000,   # 4000
    "start_shift_ms": 0,    # 1000
    "end_shift_ms": 0,
    "epoch_len_ms": None,
    "epochs_step_ms": None, 
    "idxs_keys": "1-2" # "2-3"
}

config_csp = {
    "bands": [[8, 12]], #, [9, 13], [10, 14], [8, 15]
    "robust": True,
    "concat": True,
    "regularization": False,
    "alpha": 0.1,
}

config_clf = {
    "sel_comps": [0, 1, -1, -2],
    "n_feat": [2], 
    "classifier": "lda"
}

project = "pr_Feedback_Quasi"
sessions = ["Evgeny"]

if __name__ == "__main__":

    for session in sessions:
        # TRAIN MODELS
        folder_input = os.path.join(r"data", project, "features", "csp", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("FEATURES") != -1]
        records = [record for record in records if record.find("calib") != -1]

        folder_models = os.path.join(r"models", project, stage, session)
        os.makedirs(folder_models, exist_ok=True)
        process_records_clf(folder_input, records, folder_models, config_clf)

        # TEST MODELS

        folder_output = os.path.join(r"results", project, stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records_clf_test(folder_input, folder_models, records, folder_output, config_clf)


