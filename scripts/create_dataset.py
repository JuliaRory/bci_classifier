
import os
import json
from pathlib import Path

from h5py import File
from numpy import mean, diff, concatenate, ones, zeros

from src.analysis.preprocessing import bandpass_filter
from src.utils.parse_resonance_files import process_file_resonance, get_idxs
from src.utils.events import get_sliding_epochs, get_full_epochs


def process_record(full_path, folder_output, config):
    ms_to_samples = lambda x: int(x * config["Fs"] / 1000)
    baseline = ms_to_samples(config["baseline_ms"])
    start_shift = ms_to_samples(config["start_shift_ms"])
    end_shift = ms_to_samples(config["end_shift_ms"])

    eeg, idxs_1, idxs_2, idxs_3 = process_file_resonance(full_path, baseline=baseline, end_shift=end_shift, show_plot=True)
    
    if config["do_filtering"]:
        eeg, _ = bandpass_filter(eeg, config["Fs"], low=config["low_freq"], high=config["high_freq"])
    
    idxs1, idxs2 = get_idxs(config["idxs_keys"], idxs_1, idxs_2, idxs_3)
    if config["epoch_len_ms"] is None:  # get full epoch
        trial_dur = ms_to_samples(config["trial_dur_ms"])
        epochs_1 = get_full_epochs(eeg, idxs1, trial_dur=trial_dur+baseline)
        epochs_2 = get_full_epochs(eeg, idxs2, trial_dur=trial_dur+baseline)
    else:
        epoch_len = ms_to_samples(config["epoch_len_ms"])
        step = ms_to_samples(config["epochs_step_ms"])
        
        epoch_len = ms_to_samples(config["epoch_len_ms"])
        epochs_1 = get_sliding_epochs(eeg, idxs1, epoch_len, step, baseline, start_shift, end_shift)
        epochs_2 = get_sliding_epochs(eeg, idxs2, epoch_len, step, baseline, start_shift, end_shift)

    print(f"Epoch 1: n={len(idxs1)}, dur={epochs_1.shape[1]}. \nEpoch 2: n={len(idxs2)}, dur={epochs_2.shape[1]}.")
    
    n_samples = min(epochs_1.shape[1], epochs_2.shape[1])

    epochs_1 = epochs_1[:, :n_samples, :]
    epochs_2 = epochs_2[:, :n_samples, :]
    epochs = concatenate([epochs_1, epochs_2], axis=0)
    labels = concatenate([ones(len(epochs_1)), zeros(len(epochs_2))]).reshape((-1, 1))

    filename = Path(full_path).parts[-1]
    filename = os.path.join(folder_output, "EPOCHS_"+filename)
    with File(filename, "w") as h5f:
        h5f.create_dataset("epochs", data=epochs)
        h5f.create_dataset("labels", data=labels)

        config_str = json.dumps(config)
        h5f.create_dataset("metadata", data=config_str)
        print("output file -> ", filename)
    

def process_records(folder_input, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)


config = {
    "Fs": 1000,
    "do_filtering": True, 
    "low_freq": 5, 
    "high_freq": 35,
    "baseline_ms": 500,
    "trial_dur_ms": 4000,
    "start_shift_ms": 1000,
    "end_shift_ms": 0,
    "epoch_len_ms": None,
    "epochs_step_ms": None, 
    "idxs_keys": "2-3"
}

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

project = "pr_Feedback_Quasi"
sessions = ["Julia"]

if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "raw", stage, session)

        records = os.listdir(folder_input)
        records  =["02 calib QM fix.hdf"]

        folder_output = os.path.join(r"data", project, "trans", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, records, folder_output, config)
    
        
