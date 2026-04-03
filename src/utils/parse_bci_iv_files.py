from scipy.io import loadmat
from numpy import concatenate, float64, column_stack

# ==== unique for BCI Comp IV ====

def process_file_bci_comp(filename):
    """
    filename: str 
        absolute path
    """
    
    eeg, trigger, labels = load_data(filename)
    xy = get_electrode_positions(filename)

    def get_start_end(trigger, labels, label, dur=400):
        a = trigger[labels == label]
        return column_stack((a, a + dur))
    
    idxs_1 = get_start_end(trigger, labels, 1)
    idxs_2 = get_start_end(trigger, labels, -1)

    Fs = 100

    return eeg, idxs_1, idxs_2, xy, Fs

def load_data(filename):
    """
    filename: str 
        absolute path
    """
    data = loadmat(filename)
    eeg =  0.1* data["cnt"].astype(float64)
    trigger = data["mrk"]["pos"][0][0][0]
    labels = data["mrk"]["y"][0][0][0]
    return eeg, trigger, labels

def get_channel_labels(filename):
    """
    filename: str 
        absolute path
    """
    data = loadmat(filename)
    ch_labels = data["nfo"]["clab"][0][0][0]
    return concatenate(ch_labels)

def get_electrode_positions(filename):
    """
    filename: str 
        absolute path
    """
    data = loadmat(filename)
    x = data["nfo"]["xpos"][0][0]
    y = data["nfo"]["ypos"][0][0]
    xy = concatenate([y, x], axis=1)
    xy_rot = xy.copy()
    xy_rot[:, 0] = -xy[:, 1]
    xy_rot[:, 1] = xy[:, 0]
    return xy_rot