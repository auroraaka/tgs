import numpy as np
from scipy.signal import find_peaks

def find_start_phase(time: np.ndarray, amplitude: np.ndarray, grating: float, num_semiperiods: int) -> float:

    if num_semiperiods > 4:
        print('Cannot start this far out, defaulting to zero start')
        start_time = time[0]
    else:
        if grating > 6:
            amplitude_segment = amplitude[19:]
            time_segment = time[19:]
            start_point = 0.5e-9
        else:
            amplitude_segment = amplitude[1:]
            time_segment = time[1:]
            start_point = 0

        pos_peaks_indices, _ = find_peaks(amplitude_segment)
        neg_peaks_indices, _ = find_peaks(-amplitude_segment)

        num_required_peaks = 5
        if len(pos_peaks_indices) < num_required_peaks or len(neg_peaks_indices) < num_required_peaks:
            raise ValueError('Not enough peaks found in the data.')

        pos_peaks_indices = pos_peaks_indices[:5]
        neg_peaks_indices = neg_peaks_indices[:5]
        pos_locs = time_segment[pos_peaks_indices]
        neg_locs = time_segment[neg_peaks_indices]

        if neg_locs[0] < pos_locs[0]:
            check_length = pos_locs[0] - neg_locs[0]
            if neg_locs[0] - check_length / 2 < start_point:
                if num_semiperiods == 1:
                    start_time = neg_locs[0] + 0.5 * (pos_locs[0] - neg_locs[0])
                elif num_semiperiods == 2:
                    start_time = pos_locs[0] + 0.5 * (neg_locs[1] - pos_locs[0])
                elif num_semiperiods == 3:
                    start_time = neg_locs[1] + 0.5 * (pos_locs[1] - neg_locs[1])
                elif num_semiperiods == 4:
                    start_time = pos_locs[1] + 0.5 * (neg_locs[2] - pos_locs[1])
            else:
                if num_semiperiods == 1:
                    start_time = neg_locs[0] - 0.5 * (pos_locs[0] - neg_locs[0])
                elif num_semiperiods == 2:
                    start_time = neg_locs[0] + 0.5 * (pos_locs[0] - neg_locs[0])
                elif num_semiperiods == 3:
                    start_time = pos_locs[0] + 0.5 * (neg_locs[1] - pos_locs[0])
                elif num_semiperiods == 4:
                    start_time = neg_locs[1] + 0.5 * (pos_locs[1] - neg_locs[1])
        else:
            check_length = neg_locs[0] - pos_locs[0]
            if pos_locs[0] - check_length / 2 < start_point:
                if num_semiperiods == 1:
                    start_time = pos_locs[0] + 0.5 * (neg_locs[0] - pos_locs[0])
                elif num_semiperiods == 2:
                    start_time = neg_locs[0] + 0.5 * (pos_locs[1] - neg_locs[0])
                elif num_semiperiods == 3:
                    start_time = pos_locs[1] + 0.5 * (neg_locs[1] - pos_locs[1])
                elif num_semiperiods == 4:
                    start_time = neg_locs[1] + 0.5 * (pos_locs[2] - neg_locs[1])
            else:
                if num_semiperiods == 1:
                    start_time = pos_locs[0] - 0.5 * (neg_locs[0] - pos_locs[0])
                elif num_semiperiods == 2:
                    start_time = pos_locs[0] + 0.5 * (neg_locs[0] - pos_locs[0])
                elif num_semiperiods == 3:
                    start_time = neg_locs[0] + 0.5 * (pos_locs[1] - neg_locs[0])
                elif num_semiperiods == 4:
                    start_time = pos_locs[1] + 0.5 * (neg_locs[0] - pos_locs[1])
    return start_time