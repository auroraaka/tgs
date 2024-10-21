import numpy as np
from scipy.signal import find_peaks

def find_time_index(positive_signal: np.ndarray, negative_signal: np.ndarray) -> tuple[int, float]:
    """
    Approximate time index from signal.

    Parameters:
        positive_signal (np.ndarray): positive signal array
        negative_signal (np.ndarray): negative signal array

    Returns:
        tuple[int, float]: (time index, end time)
    """

    signal = np.column_stack((
        positive_signal[:, 0],
        positive_signal[:, 1] - negative_signal[:, 1]
    ))

    # Trace max method
    max_trace = np.max(signal[0:600, 1])
    max_time = np.argmax(signal[0:600, 1])

    # Second derivative method
    first_derivative = np.gradient(signal[:, 1])
    second_derivative = np.gradient(first_derivative)

    max_second_derivative_index = np.argmax(second_derivative[:max_time + 1])
    prominence = 5 * np.max(second_derivative[:50])
    peak_indices, _ = find_peaks(second_derivative[:max_time + 1], prominence=prominence)
    peak_values = second_derivative[peak_indices]

    if len(peak_indices) > 0:
        max_peak_index = np.argmax(peak_values)
        if peak_indices[max_peak_index] < max_second_derivative_index:
            max_second_derivative_index = peak_indices[max_peak_index]

    # Adjust time index and determine end time
    time_len = len(positive_signal[:, 0]) / 1000
    if time_len < 5:
        # 20 ns oscilloscope
        end_time = 2e-7
        offset = 23
        max_second_derivative_index -= offset
    else:
        # 50 ns oscilloscope
        end_time = 5e-7
        offset = 19
        max_second_derivative_index -= offset

    return max_second_derivative_index, end_time