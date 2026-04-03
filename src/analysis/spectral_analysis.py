
from scipy.signal import ShortTimeFFT, windows
from numpy import abs, ones, float32, abs

def get_fft_fast(eeg, Fs=100, hop=10, window=100):
    """
    Compute short-time FFT of EEG data.

    Parameters
    ----------
    eeg : ndarray, shape (n_times, n_channels)
        EEG signal
    Fs : int
        Sampling frequency (Hz)
    hop : int
        Hop size for STFT
    window : int
        Window length for STFT

    Returns
    -------
    fft_res : ndarray, shape (n_freq, n_times, n_channels)
        Power spectral density
    fft_t : ndarray
        Time points of STFT
    fft_f : ndarray
        Frequency bins
    """

    eeg = eeg.astype(float32)

    SFT = ShortTimeFFT(
        win=windows.hann(window, sym=False),
        hop=hop,
        fs=Fs,
        fft_mode="onesided",
        scale_to="psd"
    )

    fft_res = abs(SFT.stft(eeg, axis=0))            
    fft_res = fft_res**2    # [n_freq, n_channels, n_times]

    fft_t = SFT.t(len(eeg))
    fft_f = SFT.f

    return fft_res, fft_t, fft_f