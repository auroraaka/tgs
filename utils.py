import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.special import erfc
from scipy.signal import find_peaks

def tgs_function(start_time: float, grating: float,) -> Tuple[callable, callable]:
    """
    Build functional and thermal fit functions.

    Parameters:
        start_time (float): start time of TGS data [s] # TODO: check units
        grating (float): grating spacing of TGS probe [µm]

    Returns:
        Tuple[callable, callable]: (functional fit, thermal fit)
    """
    q = 2 * np.pi / (grating * 1e-6)

    def functional_fit(x, A, B, C, alpha, beta, theta, tau, freq, q=q):
        """
        Functional fit function.

        Equation:
            I(t) = A [erfc(q √(αt)) - (β/√t) e^(-q²αt)] + B sin(2πft + Θ) e^(-t/τ) + C

        Parameters:
            x (np.ndarray): time array [s] # TODO: check units
            A (float): constant [V] # TODO: check units
            B (float): constant [V] # TODO: check units
            C (float): constant [V] # TODO: check units
            alpha (α) (float): thermal diffusivity [m²/s] # TODO: check units
            beta (β) (float): thermal conductivity [W/(m·K)] # TODO: check units
            theta (Θ) (float): acoustic phase [rad] # TODO: check units
            tau (τ) (float): acoustic decay constant [s] # TODO: check units
            freq (f) (float): surface acoustic wave frequency [Hz] # TODO: check units
            q (float): excitation wave vector [rad/m] # TODO: check units

        Returns:
            np.ndarray: functional fit response [V] # TODO: check units
        """
        t = x + start_time
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        sinusoid = np.sin(2 * np.pi * freq * t + theta) * np.exp(-t / tau)
        return A * (displacement_field + thermal_field) + B * sinusoid + C

    def thermal_fit(x, A, B, C, alpha, beta, theta, tau, freq, q=q):
        """
        Thermal fit function.

        Equation:
            I(t) = A [erfc(q √(αt)) - (β/√t) e^(-q²αt)] + C

        Parameters:
            x (np.ndarray): time array [s] # TODO: check units
            A (float): constant [V] # TODO: check units
            C (float): constant [V] # TODO: check units
            alpha (α) (float): thermal diffusivity [m²/s] # TODO: check units
            beta (β) (float): thermal conductivity [W/(m·K)] # TODO: check units
            q (float): excitation wave vector [rad/m] # TODO: check units

        Returns:
            np.ndarray: thermal fit response [V] # TODO: check units
        """
        t = x + start_time
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        return A * (displacement_field + thermal_field) + C

    return functional_fit, thermal_fit

def find_time_index(positive_signal: np.ndarray, negative_signal: np.ndarray) -> Tuple[int, float]:
    """
    Approximate pump time index by analyzing the signal's second derivative.

    Locates the pump time index by finding the maximum of the second derivative
    of the differential signal (positive - negative). Includes adjustments for
    both 20 ns and 50 ns oscilloscope time divisions.

    Parameters:
        positive_signal (np.ndarray): positive signal array of shape (N, 2) where N is the number of samples
        negative_signal (np.ndarray): negative signal array of shape (N, 2) where N is the number of samples

    Returns:
        Tuple[int, float]: (time index, end time)
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

def find_start_phase(time: np.ndarray, amplitude: np.ndarray, grating: float, null_point: int) -> float:
    """
    Determine the optimal start time based on fixed null-point start.

    The TGS trace can exhibit four distinct morphologies depending on the acoustic 
    phase at time zero. This function:
    1. Analyzes the pattern of maxima and minima in the signal
    2. Identifies which of the four morphologies is present
    3. Calculates the appropriate start time based on the selected null-point

    Note:
        - Manual calculations are used instead of an automated search algorithm
        - Limited to the first four null-points (null_point ∈ {1, 2, 3, 4})
        - Recommended setting: null_point = 2

    Parameters:
        time (np.ndarray): time array [s]
        amplitude (np.ndarray): amplitude array [V]
        grating (float): grating spacing [µm]
        null_point (int): null-point start (1-4)

    Returns:
        float: start time [s]
    """

    if null_point < 1 or null_point > 4:
        print('Null-point start must be between 1 and 4, defaulting to 0 start')
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
        raise ValueError('Not enough peaks found in the data for null-point start phase analysis.')
    pos_locs = time_segment[pos_peaks_indices[:num_required_peaks]]
    neg_locs = time_segment[neg_peaks_indices[:num_required_peaks]]

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

    return start_time

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load a yaml configuration file and return a dictionary with the parsed data.

    Parameters:
        file_path (str): path to the yaml configuration file

    Returns:
        Dict[str, Any]: dictionary with the parsed data
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def save_csv(data: pd.DataFrame, save_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Parameters:
        data (pd.DataFrame): DataFrame to save
        save_path (str): path to save the CSV file
    """
    data.to_csv(save_path, index=False)

def save_json(data: List[List[float]], save_path: str) -> None:
    """
    Save a list of lists to a JSON file.

    Parameters:
        data (List[List[float]]): list of lists to save
        save_path (str): path to save the JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(data, f)

def read_fits(raw_folder_path: str, fit_path: str, interval: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Read the fit results and return a DataFrame with the parsed data.

    Parameters:
        raw_folder_path (str): path to the folder containing the raw data
        fit_path (str): path to the fit results CSV file
        interval (Optional[Tuple[float, float]]): time interval to filter the data

    Returns:
        pd.DataFrame: DataFrame with the parsed data
    """
    df = pd.read_csv(fit_path)
    dr = []
    for _, run in df.iterrows():
        path = os.path.join(raw_folder_path, run['run_name'])
        data = {}
        xs, ys = [], []
        with open(path, 'r') as file:
            for i, line in enumerate(file):
                if i < 14:
                    key, value = line.strip().split('\t', 1)
                    data[key] = value
                elif i == 15:
                    xn, yn = line.strip().split('\t')
                elif i > 15:
                    x, y = map(float, line.strip().split('\t'))
                    xs.append(x)
                    ys.append(y)
                
        data[xn], data[yn] = xs, ys
        dr.append(data)
    df = pd.concat([df, pd.DataFrame(dr)], axis=1)

    df['Timestamp'] = pd.to_datetime(df['time stamp (ms)'], format='%I:%M:%S %p')
    df['time[s]'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    df['time[min]'] = df['time[s]'] / 60

    if interval:
        start, end = interval
        df = df[(df['time[s]'] >= start) & (df['time[s]'] <= end)]
    return df

def load_data(postfit_path: str, prefit_path: str, x_raw_path: str, y_raw_path: str, x_fit_path: str, y_fit_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Load the raw and fit data then return a tuple of DataFrames and lists.

    Parameters:
        postfit_path (str): path to the postfit data CSV file
        prefit_path (str): path to the prefit data CSV file
        x_raw_path (str): path to the x raw data JSON file
        y_raw_path (str): path to the y raw data JSON file
        x_fit_path (str): path to the x fit data JSON file
        y_fit_path (str): path to the y fit data JSON file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
            (postfit_data, prefit_data, x_raw, y_raw, x_fit, y_fit)
    """
    postfit_data = pd.read_csv(postfit_path)
    prefit_data = pd.read_csv(prefit_path)
    with open(x_raw_path) as f:
        x_raw = json.load(f)
    with open(y_raw_path) as f:
        y_raw = json.load(f)
    with open(x_fit_path) as f:
        x_fit = json.load(f)
    with open(y_fit_path) as f:
        y_fit = json.load(f)
    return postfit_data, prefit_data, x_raw, y_raw, x_fit, y_fit