from numpy import asarray, eye, newaxis, ones, mean, integer, ndarray, ascontiguousarray
from scipy.signal import butter, sosfiltfilt

def subtract_baseline(data, baseline_samples=(0, 500)):
    """
    Вычитает бейзлайн для каждой эпохи отдельно.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Массив размерностью (n_epochs, n_samples, n_channels)
    baseline_samples : tuple
        Диапазон сэмплов для бейзлайна (start, end)
    
    Returns:
    --------
    numpy.ndarray
        Данные с вычтенным бейзлайном, той же размерности
    """
    start, end = baseline_samples
    # Среднее по временным точкам бейзлайна для каждой эпохи и канала
    baseline = mean(data[:, start:end, :], axis=1, keepdims=True)
    return data - baseline


def bandpass_filter(signal, fs, low=0.5, highpass=True, high=40.0, lowpass=True, order=4):
    """
    Apply a bandpass Butterworth filter to a signal.

    Parameters
    ----------
    signal : array-like
        Input signal. Can be 1D (n_samples,) or 2D (n_samples, n_channels).
    fs : float
        Sampling frequency in Hz.
    low : float, optional
        Low cutoff frequency in Hz. Default is 0.5 Hz.
    high : float, optional
        High cutoff frequency in Hz. Default is 40.0 Hz.
    order : int, optional
        Order of the Butterworth filter. Default is 4.
    lowpass: bool, optional. Default is True.
    highpass: bool, optional. Default is True.

    Returns
    -------
    filtered_signal : ndarray
        Bandpass-filtered signal with the same shape as input.
    """

    if not highpass and not lowpass:
        import warnings
        warnings.warn("No filter type is selected. No filtered signal will be returned.", UserWarning)
        return signal

    freqs = [low, high]
    ftype = "band"
    if not highpass:
        freqs = high
        ftype = 'lowpass'
    if not lowpass:
        freqs = low
        ftype = 'highpass'

    sos = butter(order, freqs, btype=ftype, output='sos', fs=fs)
    return sosfiltfilt(ascontiguousarray(sos), signal, axis=0), sos

def rereference_eeg(eeg_data, ref_idx):
    """
    Re-reference EEG data relative to one or several reference electrodes.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Original EEG signal.
    ref_idx : int or sequence of ints
        Index (or indices) of reference electrode(s) (0-based).

    Returns
    -------
    eeg_reref : ndarray, shape (n_samples, n_channels)
        EEG signal re-referenced to the given electrode(s).
    """
    eeg_data = asarray(eeg_data)
    n_channels = eeg_data.shape[1]

    # Приводим ref_idx к массиву индексов
    if isinstance(ref_idx, (int, integer)):
        ref_idx = [ref_idx]
    elif isinstance(ref_idx, (list, tuple, ndarray)):
        ref_idx = list(ref_idx)
    else:
        raise TypeError("ref_idx must be int or a sequence of ints.")

    # Проверка границ
    # for idx in ref_idx:
    #     if idx < 0 or idx >= n_channels:
    #         raise ValueError(
    #             f"ref_idx ({idx}) is out of bounds for {n_channels} channels."
    #         )

    # Вычисляем средний референсный сигнал
    ref_signal = eeg_data[:, ref_idx].mean(axis=1, keepdims=True)

    # Вычитаем его из всех каналов
    eeg_reref = eeg_data - ref_signal

    return eeg_reref

def rereference_eeg_matrix(eeg_data, ref_idx):
    """
    Re-reference EEG data relative to a specific reference electrode using a matrix.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Original EEG signal.
    ref_idx : int
        Index of the reference electrode (0-based).

    Returns
    -------
    eeg_reref : ndarray, shape (n_samples, n_channels)
        EEG signal re-referenced to the given electrode.
    """
    n_samples, n_channels = eeg_data.shape
    
    if ref_idx < 0 or ref_idx >= n_channels:
        raise ValueError(f"ref_idx ({ref_idx}) is out of bounds for {n_channels} channels.")
    
    R = eye(n_channels) - eye(n_channels)[:, ref_idx][:, newaxis] 
    
    # умножение по каналам
    eeg_reref = eeg_data @ R.T  # shape: (n_samples, n_channels)
    
    return eeg_reref

def rereference_eeg_simple(eeg_data, ref_idx):
    """
    Re-reference EEG data relative to a specific reference electrode.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Original EEG signal.
    ref_idx : int
        Index of the reference electrode (0-based).

    Returns
    -------
    eeg_reref : ndarray, shape (n_samples, n_channels)
        EEG signal re-referenced to the given electrode.
    """
    eeg_data = asarray(eeg_data)
    
    if ref_idx < 0 or ref_idx >= eeg_data.shape[1]:
        raise ValueError(f"ref_idx ({ref_idx}) is out of bounds for {eeg_data.shape[1]} channels.")
    
    # вычитаем сигнал опорного электрода из всех каналов
    eeg_reref = eeg_data - eeg_data[:, [ref_idx]]
    
    return eeg_reref

def apply_car(eeg_data, exclude_channels_idx=None):
    """
    Apply Common Average Reference (CAR) to EEG data.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_samples, n_channels)
        Raw EEG signal.
    exclude_channels_idx : list or ndarray, optional
        Indices of channels to exclude from CAR computation
        (e.g. bad channels or reference electrode).

    Returns
    -------
    eeg_car : ndarray, shape (n_samples, n_channels)
        EEG data after CAR re-referencing.
    """
    eeg_data = asarray(eeg_data)

    if exclude_channels_idx is None:
        exclude_channels_idx = []

    # каналы, участвующие в вычислении среднего
    include_mask = ones(eeg_data.shape[1], dtype=bool)
    include_mask[exclude_channels_idx] = False

    # общее среднее по каналам
    car = mean(eeg_data[:, include_mask], axis=1, keepdims=True)

    # вычитаем CAR
    eeg_car = eeg_data - car

    return eeg_car