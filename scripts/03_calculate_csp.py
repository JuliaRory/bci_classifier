import os
import json
from pathlib import Path

from h5py import File
from numpy import array, where, arange

from src.analysis.preprocessing import bandpass_filter
from src.analysis.CSP import compute_csp
from src.visualization.plot_csp_components import plot_10_csp_components

from src.utils.montage_processing import get_topo_positions, get_channel_names, find_ch_idx

EEG_CHANNELS = arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS = array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])
xy = get_topo_positions("resources/mks64_standard.ced")[EEG_CHANNELS]

def plot_10_comp(evals, projForward,  band, output_filename, config):
    fig = plot_10_csp_components(abs(evals), projForward, xy)
    
    robust = "robust" if config["robust"] else "standard"
    reg = "reg"+str(config["alpha"]) if config["regularization"] else ""
    concat = "concat" if config["concat"] else "mean"
    fig.suptitle(f"CSP: {band} Hz, {robust}, {reg}, {concat}", fontsize=16)

    print(output_filename)
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")

def process_record(full_path, folder_output, config, config_csp):
    with File(full_path, "r") as h5f:
        epochs = h5f["epochs"][:]
        labels = h5f['labels'][:].squeeze()

    epochs_1 =  epochs[where(labels == 0)]  # rest/left
    epochs_2 =  epochs[where(labels == 1)]  # right

    ms_to_samples = lambda x: int(x * config["Fs"] / 1000)
    baseline = ms_to_samples(config["baseline_ms"])
    start_shift = ms_to_samples(config["start_shift_ms"])
    end_shift = ms_to_samples(config["end_shift_ms"])
    end_shift = epochs_1.shape[1] - end_shift

    for band in config_csp["bands"]:
        epochs_1 = array([bandpass_filter(ep, fs=config["Fs"], low=band[0], high=band[1])[0] for ep in epochs_1])
        epochs_2 = array([bandpass_filter(ep, fs=config["Fs"], low=band[0], high=band[1])[0] for ep in epochs_2])

        mask = lambda x: x[:, baseline+start_shift:end_shift, :]   # берём только главное для расчёта ковариаций
        epochs_1_clean = mask(epochs_1)
        epochs_2_clean = mask(epochs_2)
        print("Clean epochs shape: ", epochs_1_clean.shape, epochs_2_clean.shape)

        projForward, projInverse, evals = compute_csp(epochs_1_clean, epochs_2_clean, config_csp)
        filename = Path(full_path).parts[-1]
        rob = "robust" if config_csp["robust"] else "standard"
        con = "concat" if config_csp["concat"] else "mean"
        filename = os.path.join(folder_output, f"MATRIX_{band}_{rob}_{con}+reg{config_csp["alpha"]}_"+filename[len("EPOCHS")+1:])
        with File(filename, "w") as h5f:
            h5f.create_dataset("projInverse", data=projInverse)
            h5f.create_dataset("projForward", data=projForward)
            h5f.create_dataset("evals", data=evals)

            config_str = json.dumps(config)
            h5f.create_dataset("metadata", data=config_str)
            config_csp["band"] = band
            config_str = json.dumps(config_csp)
            h5f.create_dataset("metadata_csp", data=config_str)

            print("output file -> ", filename)

        parts = Path(full_path).parts
        folder = os.path.join("results", parts[1], parts[3], parts[4], "CSP_components")
        filename = parts[-1]
        os.makedirs(folder, exist_ok=True)
        output_filename = os.path.join(folder, f"{band}_{rob}_{con}+reg{config_csp["alpha"]}_"+filename[len("EPOCHS")+1:-4]+".png")
        plot_10_comp(evals, projInverse, band, output_filename, config_csp)


def process_records(folder_input, records, folder_output, config):
    for record in records[:1]:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config, config_csp=config_csp)


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

config_csp = {
    "bands": [[7, 10], [8, 12], [9, 13], [10, 14]],
    "robust": True,
    "concat": True,
    "regularization": True,
    "alpha": 0.05,
}

project = "pr_Agency_EBCI"
stage = "test"
sessions = ["03_30 Artem"]

if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "trans", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("calib") != -1]

        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, records, folder_output, config)
    