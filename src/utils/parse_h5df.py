from numpy import array, uint8
from h5py import File

def load_h5df(path):
    """
    Load EEG data and block metadata from an HDF5 (.h5f) file created by Resonance.

    Parameters
    ----------
    path : str
        Path to the HDF5 (.h5f) file.

    Returns
    -------
    data : array-like
        Signal data stored in the file.
    blocks : array-like of structured dtype [('created', '<u8'), ('received', '<u8'), ('samples', '<u4')]
        Metadata for recorded data blocks, including timestamps
        of block creation and reception, and the number of samples
        in each block.
    """
    with File(path, "r") as h5f:
        data = h5f["eeg"]["data"][:-1]
        blocks = h5f["eeg"]["blocks"][:]

    return data, blocks

def ttl2binary(ttl_signal, bit_index=0):
    """
    Decode a binary signal from a TTL channel by selecting a specific bit.

    Parameters
    ----------
    ttl_signal : array-like
        Integer TTL values.
    bit_index : int
        Bit position to decode.

    Returns
    -------
    binary_signal : array-like
        Binary (0 or 1) signal extracted from the TTL input.
    """
    ttl = array(ttl_signal, dtype=uint8)
    return ((ttl>>bit_index) & 0b1).astype(int)

def reverse_trigger(trigger):
    return 1 - trigger

