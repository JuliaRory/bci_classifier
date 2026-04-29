from numpy import array, asarray, sum, diff

def get_sliding_epochs(
    eeg, idxs_1,
    window=1000, step=100,  # длина окна и шаг в сэмплах
    baseline=500, edge=250
):
    """
    Разрезает сигнал на эпохи с учетом скользящего окна.

    Параметры:
    eeg : np.ndarray, shape (n_samples, n_channels)
        Сырой сигнал
    idxs_1, idxs_2 : np.ndarray, shape (n_epochs, 2)
        Начало и конец каждой эпохи (в сэмплах)
    window : int
        Длина окна в сэмплах
    step : int
        Шаг скользящего окна в сэмплах

    Возвращает:
    epochs_1, epochs_2 : np.ndarray, shape (n_epochs, n_samples, n_channels)
    """

    def slice_epochs_sliding(eeg, idxs):
        windows_list = []
        for start_epoch, end_epoch in idxs:
            # убираем стартовый сдвиг и края
            s = start_epoch + baseline + edge
            e = end_epoch - edge
            for w_start in range(s, e - window + 1, step):
                w_end = w_start + window
                windows_list.append(eeg[w_start:w_end, :])
        return array(windows_list)

    epochs_1 = slice_epochs_sliding(eeg, idxs_1)

    return epochs_1

def get_full_epochs(data, intervals, trial_dur):
    """
    Slice multi-channel data into epochs based on start and end indices.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_channels)
        Input signal array.
    intervals : list of [start, end]
        List of intervals specifying the start (inclusive) and end (exclusive)
        indices for each epoch.

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_samples_in_epoch, n_channels)
        Array containing the extracted epochs.
    """


    epochs = []
    for start, end in intervals:
        epochs.append(data[end-trial_dur:end])
        
    return array(epochs)

def receive_epochs(events, event_code):
    return asarray(find_intervals(events, event_code))

def reveive_events_info(events, events_info=None):
    if events_info is None:
        assert "events_info is empty."

    for key in events_info:
        events_info[key]["num"] = count_any_transitions(events, events_info[key]["event_code"])
        events_info[key]["dur"] = get_duration(events_info[key]["trial_dur_ms"], events_info[key]["num"])

def get_duration(trial_dur, n_trial, degree=1):
    return float(round(trial_dur * n_trial / 1000 * degree, 1))

def count_any_transitions(arr, event_code=1):
    """
    Count transitions to bit in a discrete signal.

    Parameters
    ----------
    arr : array-like
        Input array of numbers.
    event_code: int
        Code of some event.

    Returns
    -------
    transitions : int
        Number of transitions.
    """
    arr = asarray(arr)
    # сдвинутый массив на 1 для сравнения текущего и предыдущего значения
    prev = arr[:-1]
    curr = arr[1:]
    return int(sum((prev != event_code) & (curr == event_code)))


def count_short_switches(arr, max_width, min_width=1, on_value=1):
    """
    Count short ``0 -> 1 -> 0`` pulses in a discrete signal.

    Parameters
    ----------
    arr : array-like
        Input binary-like signal.
    max_width : int
        Maximum pulse width in samples to count as a short switch.
    min_width : int, default=1
        Minimum pulse width in samples to include.
    on_value : int, default=1
        Active pulse value. Everything else is treated as the off state.

    Returns
    -------
    int
        Number of pulses whose width is in ``[min_width, max_width]``.
    """
    arr = asarray(arr)
    if arr.size == 0:
        return 0
    if min_width < 1:
        raise ValueError("min_width must be >= 1")
    if max_width < min_width:
        raise ValueError("max_width must be >= min_width")

    is_on = (arr == on_value).astype(int)
    starts = diff(is_on, prepend=0) == 1
    ends = diff(is_on, append=0) == -1

    start_idxs = starts.nonzero()[0]
    end_idxs = ends.nonzero()[0] + 1
    widths = end_idxs - start_idxs

    return int(sum((widths >= min_width) & (widths <= max_width)))
    

def find_intervals(arr, value):
    """
    Find intervals where a specific value occurs consecutively in an array.

    Parameters
    ----------
    arr : array-like
    value : int
        Value to search for.

    Returns
    -------
    intervals : list of [start, end]
        List of intervals where the value occurs consecutively.
        Each interval is [start_index, end_index] (inclusive start, exclusive end).
    """
    arr = asarray(arr)
    intervals = []
    in_interval = False
    start_idx = None

    for i, v in enumerate(arr):
        if v == value:
            if not in_interval:
                start_idx = i
                in_interval = True
        else:
            if in_interval:
                intervals.append([start_idx, i])
                in_interval = False

    # если массив заканчивается значением value, закрываем последний интервал
    if in_interval:
        intervals.append([start_idx, len(arr)])

    return intervals
