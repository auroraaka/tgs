from typing import Tuple

import numpy as np
from scipy.signal import find_peaks

from src.core.path import Paths
from src.core.utils import read_data
from src.core.plots import plot_signal_processed


PROMINENCE_FACTOR = 5
INITIAL_SAMPLES = 50

OFFSET_20NS = 19
END_TIME_20NS = 2e-7
OFFSET_50NS = 23
END_TIME_50NS = 5e-7

LARGE_GRATING_THRESHOLD = 6
LARGE_GRATING_INDEX = 20
LARGE_GRATING_START_POINT = 5e-10
SMALL_GRATING_INDEX = 1
SMALL_GRATING_START_POINT = 0

GRATING_SPACING_THRESHOLD = 8
TIME_OFFSET_INDEX = 20

def find_pump_time(pos_signal: np.ndarray, neg_signal: np.ndarray) -> Tuple[int, float]:
    """
    Approximate pump time index by analyzing the signal's second derivative.

    Locates the pump time index by finding the maximum of the second derivative
    of the differential signal (positive - negative). Includes adjustments for
    both 20 ns and 50 ns oscilloscope time divisions.

    Parameters:
        pos_signal (np.ndarray): positive signal array of shape (N, 2) where N is the number of samples
        neg_signal (np.ndarray): negative signal array of shape (N, 2) where N is the number of samples

    Returns:
        Tuple[int, float]: (time index, end time)
    """

    signal = np.column_stack((
        pos_signal[:, 0],
        pos_signal[:, 1] - neg_signal[:, 1]
    ))

    # Trace max method
    max_time = np.argmax(signal[:, 1])

    # Second derivative method
    first_derivative = np.gradient(signal[:, 1])
    second_derivative = np.gradient(first_derivative)

    max_second_derivative_idx = np.argmax(second_derivative[:max_time + 1])
    prominence = PROMINENCE_FACTOR * np.max(second_derivative[:INITIAL_SAMPLES])
    peak_idxs, _ = find_peaks(second_derivative[:max_time + 1], prominence=prominence)

    if len(peak_idxs) > 0:
        max_peak_idx = np.argmax(second_derivative[peak_idxs])
        if peak_idxs[max_peak_idx] < max_second_derivative_idx:
            max_second_derivative_idx = peak_idxs[max_peak_idx]

    # Adjust time index and determine end time
    time_len = len(signal[:, 0]) / 1e3
    if time_len < 5:
        # 20 ns oscilloscope
        end_time = END_TIME_20NS
        pump_time_idx = max_second_derivative_idx - OFFSET_20NS
    else:
        # 50 ns oscilloscope
        end_time = END_TIME_50NS
        pump_time_idx = max_second_derivative_idx - OFFSET_50NS

    return pump_time_idx, end_time

def find_start_time(signal: np.ndarray, grating_spacing: float, time_step: float, null_point: int = 2) -> Tuple[int, float]:
    """
    Determine the optimal start index and start time based on fixed null-point start.

    The TGS trace can exhibit four distinct morphologies depending on the acoustic 
    phase at time zero. This function:
    1. Analyzes the pattern of maxima and minima in the signal
    2. Identifies which of the four morphologies is present
    3. Calculates the appropriate start time based on the selected null-point

    Note:
        - Manual calculations are used instead of an automated search algorithm
        - Limited to the first four null-points (null_point ∈ {1, 2, 3, 4})
        - Recommended setting based on prior art: null_point = 2

    Parameters:
        signal (np.ndarray): signal array of shape (N, 2) where N is the number of samples
        grating_spacing (float): grating spacing of TGS probe [µm]
        time_step (float): time interval between consecutive signal samples [s]
        null_point (int, optional): null-point start (1-4)

    Returns:
        Tuple[int, float]: (start index, start time)
    """
    if null_point < 1 or null_point > 4:
        print('Null-point start must be between 1 and 4, defaulting to 0 start')
        start_time = signal[0, 0]

    if grating_spacing > LARGE_GRATING_THRESHOLD:
        idx = LARGE_GRATING_INDEX
        signal_segment = signal[idx:]
        start_point = LARGE_GRATING_START_POINT
    else:
        idx = SMALL_GRATING_INDEX
        signal_segment = signal[idx:]
        start_point = SMALL_GRATING_START_POINT
    
    pos_peak_idxs, _ = find_peaks(signal_segment[:, 1])
    neg_peak_idxs, _ = find_peaks(-signal_segment[:, 1])

    num_required_peaks = 5
    if len(pos_peak_idxs) < num_required_peaks or len(neg_peak_idxs) < num_required_peaks:
        raise ValueError('Not enough peaks found in the data for null-point start phase analysis.')
    
    pos_locs = signal_segment[pos_peak_idxs[:num_required_peaks], 0]
    neg_locs = signal_segment[neg_peak_idxs[:num_required_peaks], 0]

    # Find start time based on four morphologic cases
    if neg_locs[0] < pos_locs[0]:
        check_length = pos_locs[0] - neg_locs[0]
        if neg_locs[0] - check_length / 2 < start_point:
            if null_point == 1:
                start_time = neg_locs[0] + 0.5 * (pos_locs[0] - neg_locs[0])
            elif null_point == 2:
                start_time = pos_locs[0] + 0.5 * (neg_locs[1] - pos_locs[0])
            elif null_point == 3:
                start_time = neg_locs[1] + 0.5 * (pos_locs[1] - neg_locs[1])
            elif null_point == 4:
                start_time = pos_locs[1] + 0.5 * (neg_locs[2] - pos_locs[1])
        else:
            if null_point == 1:
                start_time = neg_locs[0] - 0.5 * (pos_locs[0] - neg_locs[0])
            elif null_point == 2:
                start_time = neg_locs[0] + 0.5 * (pos_locs[0] - neg_locs[0])
            elif null_point == 3:
                start_time = pos_locs[0] + 0.5 * (neg_locs[1] - pos_locs[0])
            elif null_point == 4:
                start_time = neg_locs[1] + 0.5 * (pos_locs[1] - neg_locs[1])
    else:
        check_length = neg_locs[0] - pos_locs[0]
        if pos_locs[0] - check_length / 2 < start_point:
            if null_point == 1:
                start_time = pos_locs[0] + 0.5 * (neg_locs[0] - pos_locs[0])
            elif null_point == 2:
                start_time = neg_locs[0] + 0.5 * (pos_locs[1] - neg_locs[0])
            elif null_point == 3:
                start_time = pos_locs[1] + 0.5 * (neg_locs[1] - pos_locs[1])
            elif null_point == 4:
                start_time = neg_locs[1] + 0.5 * (pos_locs[2] - neg_locs[1])
        else:
            if null_point == 1:
                start_time = pos_locs[0] - 0.5 * (neg_locs[0] - pos_locs[0])
            elif null_point == 2:
                start_time = pos_locs[0] + 0.5 * (neg_locs[0] - pos_locs[0])
            elif null_point == 3:
                start_time = neg_locs[0] + 0.5 * (pos_locs[1] - neg_locs[0])
            elif null_point == 4:
                start_time = pos_locs[1] + 0.5 * (neg_locs[0] - pos_locs[1])

    start_idx = int(start_time / time_step)
    return start_idx, start_time


def process_signal(paths: Paths, file_idx: int, pos_file: str, neg_file: str, grating_spacing: float, heterodyne: str = 'di-homodyne', null_point: int = 2, plot: bool = False) -> Tuple[np.ndarray, float, int, float]:
    """
    Process signals for analysis.

    This function reads the positive and negative TGS signal data, performs baseline corrections, determines the pump time,
    aligns the signals, and extracts the relevant portion of the signal for further analysis.It also calculates the start 
    index and start time for fitting procedures.

    Parameters:
        paths (Paths): paths to data, figures, and fit files
        file_idx (int): index of the positive signal file
        pos_file (str): positive signal file path
        neg_file (str): negative signal file path
        grating_spacing (float): grating spacing of TGS probe [µm]
        heterodyne (str, optional): detection scheme.
            - 'di-homodyne'
            - 'mono-homodyne'
        null_point (int, optional): null-point start (1-4)
        plot (bool, optional): whether to plot the processed signal

    Returns:
        Tuple[np.ndarray, float, int, float]:
            - signal (np.ndarray): processed signal of shape (N, 2) with time offsets and corrected amplitudes
            - max_time (float): time at which the maximum signal occurs after alignment
            - start_idx (int): index corresponding to the start time for fitting procedures
            - start_time (float): calculated start time for fitting procedures
    """
    pos = read_data(pos_file)
    if heterodyne == 'di-homodyne':
        neg = read_data(neg_file)   
        N = min(len(pos), len(neg))
    elif heterodyne == 'mono-homodyne':
        neg = np.zeros_like(pos)
        N = len(pos)
    else:
        raise ValueError('Invalid heterodyne setting, must be "di-homodyne" or "mono-homodyne"')
    pos, neg = pos[:N], neg[:N]

    # TODO: Add baseline bool functionality

    pos[:, 1] -= np.mean(pos[:INITIAL_SAMPLES, 1])
    neg[:, 1] -= np.mean(neg[:INITIAL_SAMPLES, 1])

    pump_time_idx, end_time = find_pump_time(pos, neg)
    time_step = pos[1, 0] - pos[0, 0]
    end_idx = int(end_time / time_step) - 36

    reference_time = neg[pump_time_idx, 0]
    if grating_spacing < GRATING_SPACING_THRESHOLD:
        offset_correction = np.mean(pos[:INITIAL_SAMPLES, 1] - neg[:INITIAL_SAMPLES, 1])
    else:
        offset_correction = 0

    signal = np.column_stack([
        pos[pump_time_idx:end_idx, 0] - reference_time,
        pos[pump_time_idx:end_idx, 1] - neg[pump_time_idx:end_idx, 1] - offset_correction
    ])

    # Determine start point for fitting
    max_idx = np.argmax(signal[TIME_OFFSET_INDEX:, 1]) + TIME_OFFSET_INDEX - 1
    max_time = signal[max_idx, 0]
    start_idx, start_time = find_start_time(signal[max_idx:], grating_spacing, time_step, null_point)

    if plot:
        plot_signal_processed(paths, file_idx, signal, max_time, start_time)

    return signal, max_time, start_time, start_idx