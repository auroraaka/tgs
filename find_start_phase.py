import numpy as np
from scipy.signal import find_peaks

def find_start_phase(time: np.ndarray, amplitude: np.ndarray, grating: float, num_half_periods: int) -> float:

    if num_half_periods > 4:
        print('Cannot start this far out, defaulting to zero start')
        start_time = time[0]

    if grating > 6:
        idx = 20
        amplitude_segment = amplitude[idx:]
        time_segment = time[idx:]
        start_point = 0.5e-9
    else:
        idx = 1
        amplitude_segment = amplitude[idx:]
        time_segment = time[idx:]
        start_point = 0
    pos_peaks_indices, _ = find_peaks(amplitude_segment)
    neg_peaks_indices, _ = find_peaks(-amplitude_segment)

    num_required_peaks = 5
    if len(pos_peaks_indices) < num_required_peaks or len(neg_peaks_indices) < num_required_peaks:
        raise ValueError('Not enough peaks found in the data.')
    pos_locs = time_segment[pos_peaks_indices[:num_required_peaks]]
    neg_locs = time_segment[neg_peaks_indices[:num_required_peaks]]

    if neg_locs[0] < pos_locs[0]:
        check_length = pos_locs[0] - neg_locs[0]
        if neg_locs[0] - check_length / 2 < start_point:
            if num_half_periods == 1:
                start_time = neg_locs[0] + 0.5 * (pos_locs[0] - neg_locs[0])
            elif num_half_periods == 2:
                start_time = pos_locs[0] + 0.5 * (neg_locs[1] - pos_locs[0])
            elif num_half_periods == 3:
                start_time = neg_locs[1] + 0.5 * (pos_locs[1] - neg_locs[1])
            elif num_half_periods == 4:
                start_time = pos_locs[1] + 0.5 * (neg_locs[2] - pos_locs[1])
        else:
            if num_half_periods == 1:
                start_time = neg_locs[0] - 0.5 * (pos_locs[0] - neg_locs[0])
            elif num_half_periods == 2:
                start_time = neg_locs[0] + 0.5 * (pos_locs[0] - neg_locs[0])
            elif num_half_periods == 3:
                start_time = pos_locs[0] + 0.5 * (neg_locs[1] - pos_locs[0])
            elif num_half_periods == 4:
                start_time = neg_locs[1] + 0.5 * (pos_locs[1] - neg_locs[1])
    else:
        check_length = neg_locs[0] - pos_locs[0]
        if pos_locs[0] - check_length / 2 < start_point:
            if num_half_periods == 1:
                start_time = pos_locs[0] + 0.5 * (neg_locs[0] - pos_locs[0])
            elif num_half_periods == 2:
                start_time = neg_locs[0] + 0.5 * (pos_locs[1] - neg_locs[0])
            elif num_half_periods == 3:
                start_time = pos_locs[1] + 0.5 * (neg_locs[1] - pos_locs[1])
            elif num_half_periods == 4:
                start_time = neg_locs[1] + 0.5 * (pos_locs[2] - neg_locs[1])
        else:
            if num_half_periods == 1:
                start_time = pos_locs[0] - 0.5 * (neg_locs[0] - pos_locs[0])
            elif num_half_periods == 2:
                start_time = pos_locs[0] + 0.5 * (neg_locs[0] - pos_locs[0])
            elif num_half_periods == 3:
                start_time = neg_locs[0] + 0.5 * (pos_locs[1] - neg_locs[0])
            elif num_half_periods == 4:
                start_time = pos_locs[1] + 0.5 * (neg_locs[0] - pos_locs[1])

    return start_time