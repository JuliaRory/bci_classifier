
import numpy as np

from src.utils.parse_h5df import load_h5df, ttl2binary, reverse_trigger
from src.utils.montage_processing import  get_channel_names, find_ch_idx
from src.visualization.parsing import plot_events

# ==== unique for resonance hdf files ====

EEG_CHANNELS = np.arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS = np.array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])

Fs = 1000 # Hz

def get_idxs(idxs_keys, idxs_1, idxs_2, idxs_3):
    if idxs_keys == "2-3":
        idxs1 = idxs_2 
        idxs2 = idxs_3 
    elif idxs_keys == "1-2":
        idxs1 = idxs_1
        idxs2 = idxs_2
    elif idxs_keys == "1-3":
        idxs1 = idxs_1
        idxs2 = idxs_3
    return idxs1, idxs2

def process_file_resonance(filename, baseline=100, end_shift = 0, show_plot=False):
    """
    filename: str 
        absolute path
    """
    
    data, _ = load_h5df(filename)
    raw_eeg = data[:, EEG_CHANNELS] * 1E6 # uV

    trigger = reverse_trigger(ttl2binary(data[:, -1], bit_index=0))

    idxs_rest, idxs_right, idxs_left = parse_events(trigger, window_size=200, baseline=baseline, end_shift=end_shift)

    if show_plot:
        fig = plot_events(trigger, idxs_rest, idxs_right, idxs_left)
    
    return raw_eeg, idxs_rest, idxs_right, idxs_left

def define_label(dtrigger, idx, buff=600, labels={1: 0, 2: 1, 3: 2}):
        dtrig = dtrigger[idx-buff:idx-10] 
        n_shifts = np.where(dtrig == 1)[0].shape[0]
        for key in labels:
            if n_shifts == key:
                return labels[key]
        return np.nan

def parse_events(trigger, window_size=200, baseline=100, end_shift=100):
    strigger = np.convolve(trigger, np.ones(window_size, dtype=int), 'valid')   # sum of trigger in window  
    
    start_idx = np.where((strigger == window_size) & (np.diff(strigger, prepend=0) == 1))[0].reshape((-1, 1))
    end_idx = np.where((strigger == 0) & (np.diff(strigger, prepend=0) == -1))[0].reshape((-1, 1))

    dtrigger = np.diff(trigger)
    labels = np.array([define_label(dtrigger, idx[0]) for idx in start_idx])

    events = np.concatenate([start_idx-baseline, end_idx+end_shift], axis=1)

    idxs1 = events[labels == 0]
    idxs2 = events[labels == 1]
    idxs3 = events[labels == 2]
    
    # for quasi feedback
    # start_idx = (idxs2[:, 1] + 4000).reshape((-1, 1))
    # end_idx = (start_idx + 8000).reshape((-1, 1))
    # idxs1 = np.concatenate([start_idx, end_idx], axis=1)
    
    return idxs1, idxs2, idxs3

