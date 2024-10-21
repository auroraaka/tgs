import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.special import erfc
from scipy.signal import find_peaks

def tgs_function(start_time: float, grating: float,) -> Tuple[callable, callable]:
    q = 2 * np.pi / (grating * 1e-6)

    def functional_fit(x, A, B, C, alpha, beta, theta, tau, freq):
        t = x + start_time
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        sinusoid = np.sin(2 * np.pi * freq * t + theta) * np.exp(-t / tau)
        return A * (displacement_field + thermal_field) + B * sinusoid + C

    def thermal_fit(x, A, B, C, alpha, beta, theta, tau, freq):
        t = x + start_time
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        return A * (displacement_field + thermal_field) + C

    return functional_fit, thermal_fit

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

def load_config(file_path: str) -> Dict[str, Any]:
    """Load a yaml configuration file and return a dictionary with the parsed data."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def save_csv(data: pd.DataFrame, save_path: str) -> None:
    """Save a pandas DataFrame to a CSV file."""
    data.to_csv(save_path, index=False)

def save_json(data: List[List[float]], save_path: str) -> None:
    """Save a list of lists to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(data, f)

def read_fits(raw_folder_path: str, fit_path: str, interval: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """Read the fit results and return a DataFrame with the parsed data."""
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
    """Load the raw and fit data then return a tuple of DataFrames and lists."""
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