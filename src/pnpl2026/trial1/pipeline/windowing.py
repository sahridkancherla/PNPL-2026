



import numpy as np


#lower and upper window bounds
WINDOW_BOUNDS = (-0.32, 1.28)


def cut_windows(recording, onsets_s, sample_rate=250.0, bounds=WINDOW_BOUNDS):

    start_s, end_s = bounds

    #check recording shape
    if recording.ndim != 2:
        raise ValueError(f"Expected [n_channels, n_times], got shape {recording.shape}")

    #verify bound shape
    if end_s <= start_s:
        raise ValueError(f"bounds must have end > start, got {bounds}")

    #set up the data
    n_channels, n_times = recording.shape
    onsets_s = np.asarray(onsets_s, dtype=np.float64)

    #number of samples in every word record
    n_samples = np.rint((end_s - start_s) * sample_rate).astype(np.int64)

    #number of sample when the window of an individual word begins
    window_start = np.rint((onsets_s + start_s) * sample_rate).astype(np.int64)

    #some words are said at the very beginning and some at the very end
    #they do not fit the window constraints (e.g., word said as 0.05s can't have 0.05 - 0.28s)
    #thus they need filtering. v1 is just dropping them
    fits = (window_start >= 0) and (window_start + n_samples <= n_times)

    #update the list of windows starts we will work with
    kept_window_start_indices = np.flatnonzero(fits)
    kept_window_start = window_start[kept_window_start_indices]

    #prepare the shape of a windows array for a given recording
    windows = np.empty((len(kept_window_start), n_channels, n_samples), dtype=recording.dtype)

    #iterate through every window start that we keep
    for index, start in enumerate(kept_window_start):

        #mark the start and end of every window
        start = start
        end = start + n_samples

        #record the windows
        windows[index] = recording[:, start : start + end]

    return windows, kept_window_start
