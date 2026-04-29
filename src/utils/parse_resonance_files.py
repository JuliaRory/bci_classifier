
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

    idxs_rest, idxs_right, idxs_left = parse_events(trigger, baseline=baseline)

    if show_plot:
        fig = plot_events(trigger, idxs_rest, idxs_right, idxs_left)
    
    return raw_eeg, idxs_rest, idxs_right, idxs_left

def parse_events(
    photomark,
    sfreq=1000,
    black_value=0,
    white_value=1,
    long_white_min_sec=1.0,
    baseline=0,
):
    """
    Detect 1-, 2-, and 3-blink epochs in a photomark recording.

    An epoch of type N consists of N short white-black blinks followed by a
    long (sustain) white segment; it starts at the first white onset and
    ends the sample the sustain turns black.

    Parameters
    ----------
    photomark : array-like, 1-D
        Raw photomark samples (black = 65535.0, white = 65534.0).
    sfreq : float
        Sampling frequency in Hz.
    black_value, white_value : float
        Values coding black/white on the trace.
    long_white_min_sec : float
        Minimum duration for a white segment to count as the "sustain"
        (long) white. Anything between blink-length (~0.1 s) and
        sustain-length (≥ 4 s) works; 1.0 s is a safe default.

    baseline : int
        Number of samples to subtract from the start index in each interval.

    Returns
    -------
    idxs1, idxs2, idxs3 : np.ndarray, shape (n_epochs, 2), dtype int64
        Each row is [start, end] sample indices of an epoch (end exclusive).
    """
    pm = np.asarray(photomark)
    n = pm.size
    is_white = (pm == white_value)

    # Transitions (indices of the first sample AFTER the change)
    d = np.diff(is_white.astype(np.int8))
    bw = np.flatnonzero(d == 1) + 1   # black → white onsets
    wb = np.flatnonzero(d == -1) + 1  # white → black offsets

    # Assemble (start, end_exclusive) for every white segment,
    # handling recordings that begin or end in the white state.
    starts = bw.tolist()
    ends = wb.tolist()
    if is_white[0]:
        starts = [0] + starts
    if is_white[-1]:
        ends = ends + [n]
    segs = list(zip(starts, ends))

    long_min = int(round(long_white_min_sec * sfreq))

    buckets = {1: [], 2: [], 3: []}

    i = 0
    while i < len(segs):
        # Run of consecutive SHORT white segments
        j = i
        while j < len(segs) and (segs[j][1] - segs[j][0]) < long_min:
            j += 1
        short_count = j - i

        # j now points to a LONG white segment (or past the end)
        if j < len(segs) and short_count in (1, 2, 3):
            epoch_start = segs[i][0]
            epoch_end = segs[j][1]          # first black sample after sustain
            buckets[short_count].append((epoch_start, epoch_end))
            i = j + 1                       # continue after the sustain
        else:
            # Not a valid pattern — skip this segment (or the whole run)
            i = max(j, i + 1)

    def _to_arr(lst):
        arr = np.array(lst, dtype=np.int64) if lst else np.empty((0, 2), dtype=np.int64)
        if arr.size:
            arr[:, 0] -= baseline
        return arr

    return _to_arr(buckets[1]), _to_arr(buckets[2]), _to_arr(buckets[3])


# def define_label(dtrigger, idx, buff=600, labels={1: 0, 2: 1, 3: 2}):
#         dtrig = dtrigger[idx-buff:idx-10] 
#         n_shifts = np.where(dtrig == 1)[0].shape[0]
#         for key in labels:
#             if n_shifts == key:
#                 return labels[key]
#         return np.nan

# def parse_events(trigger, window_size=200, baseline=100, end_shift=100):
#     strigger = np.convolve(trigger, np.ones(window_size, dtype=int), 'valid')   # sum of trigger in window  
    
#     start_idx = np.where((strigger == window_size) & (np.diff(strigger, prepend=0) == 1))[0].reshape((-1, 1))
#     end_idx = np.where((strigger == 0) & (np.diff(strigger, prepend=0) == -1))[0].reshape((-1, 1))

#     dtrigger = np.diff(trigger)
#     labels = np.array([define_label(dtrigger, idx[0]) for idx in start_idx])

#     events = np.concatenate([start_idx-baseline, end_idx+end_shift], axis=1)

#     idxs1 = events[labels == 0]
#     idxs2 = events[labels == 1]
#     idxs3 = events[labels == 2]
    
#     # for quasi feedback
#     # start_idx = (idxs2[:, 1] + 4000).reshape((-1, 1))
#     # end_idx = (start_idx + 8000).reshape((-1, 1))
#     # idxs1 = np.concatenate([start_idx, end_idx], axis=1)
    
#     return idxs1, idxs2, idxs3

