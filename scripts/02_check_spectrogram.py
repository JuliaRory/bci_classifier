import os
import json
from pathlib import Path

from h5py import File
from numpy import where

from src.utils.montage_processing import get_ch_idxs
from src.visualization.spectrogram import plot_spectrogram

# def plot_spectr(eeg, idxs_1, idxs_2, Fs, mode, full_path):

#     fig = plot_psd_diff(epochs_1, epochs_2, Fs, ch_names=None, picks=[idx_C3, idx_C4])
#     fig.suptitle(f"PSD Difference: {mode}", fontsize=16)
#     output_folder = get_output_path(full_path)
#     output_filename = os.path.join(output_folder, f"PSD_diff_{mode}.png")
#     fig.savefig(output_filename, dpi=300, bbox_inches="tight") 

#     fig = plot_psd(epochs_1, epochs_2, Fs, ch_names=None, picks=[idx_C3, idx_C4])
#     fig.suptitle(f"PSD: {mode}", fontsize=16)
#     output_folder = get_output_path(full_path)
#     output_filename = os.path.join(output_folder, f"PSD_{mode}.png")
#     fig.savefig(output_filename, dpi=300, bbox_inches="tight") 


def process_record(full_path, folder_output, config):
    with File(full_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f['labels'][:].squeeze()

    epochs_1 =  epochs[where(labels == 0)]  # rest/left
    epochs_2 =  epochs[where(labels == 1)]  # right
    print(epochs_1.shape, epochs_2.shape)
    
    channels = get_ch_idxs(config["C3_ROI_labels"], r"resources/mks64_standard.ced")
    fig = plot_spectrogram(epochs_2, Fs=config["Fs"], fmin=5, fmax=30, baseline=(0, config["baseline_ms"]), start_shift=config["baseline_ms"], ch_roi=channels)

def process_records(folder_input, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)

        break

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
    "idxs_keys": "2-3", 
    "C3_ROI_labels": ["FC1", "FC3", "FC5", "C1", "C3", "C5", "CP1", "CP3", "CP5"]
}
project = "pr_Agency_EBCI"
stage = "test"
sessions = ["03_30 Artem"]

if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "trans", stage, session)
        records = os.listdir(folder_input)

        folder_output = os.path.join(r"data", project, "results", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, records, folder_output, config)
    
        