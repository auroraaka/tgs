from typing import Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from src.core.process import process_signal
from src.analysis.functions import tgs_function
from src.analysis.lorentzian import lorentzian_fit
from src.analysis.fft import fft
from src.visualization.plots import plot_tgs


def tgs_fit(config: dict, pos_file: str, neg_file: str, grating_spacing: float, plot: bool = False) -> Tuple[Union[float, np.ndarray]]:
    """
    Fit transient grating spectroscopy (TGS) response equation to experimentally collected signal.

    This function processes the input TGS signal, performs thermal and acoustic fits, and returns 
    the fitted parameters along with their standard errors (1σ).

    Parameters:
        config (dict): configuration dictionary
        pos_file (str): positive signal file path
        neg_file (str): negative signal file path
        grating_spacing (float): grating spacing of TGS probe [µm]
        plot (bool, optional): whether to generate plots

    Returns:
        Tuple containing:
            start_time (float): start time of the fit [s]
            A (float): thermal signal amplitude [V]
            A_err (float): thermal signal amplitude error [V]
            B (float): acoustic signal amplitude [V]
            B_err (float): acoustic signal amplitude error [V]
            C (float): signal offset [V]
            C_err (float): signal offset error [V]
            alpha (float): thermal diffusivity [m²/s]
            alpha_err (float): thermal diffusivity error [m²/s]
            beta (float): displacement-reflectance ratio [dimensionless]
            beta_err (float): displacement-reflectance ratio error [dimensionless]
            theta (float): acoustic phase [rad]
            theta_err (float): acoustic phase error [rad]
            tau (float): acoustic decay time [s]
            tau_err (float): acoustic decay time error [s]
            f (float): surface acoustic wave frequency [Hz]
            f_err (float): surface acoustic wave frequency error [Hz]
            signal (np.ndarray): full processed signal [N, [time, amplitude]]
            fitted_signal (np.ndarray): truncated signal used for fitting [M, [time, amplitude]]

    Notes:
        The fitting process includes:
            1. Initial thermal fit
            2. FFT analysis and Lorentzian fit for acoustic parameters
            3. Iterative beta fitting
            4. Functional fit including thermal and acoustic components
    """
    # Process signal and build fit functions
    signal, max_time, start_time, start_idx, file_idx = process_signal(pos_file, neg_file, grating_spacing, **config['process'])
    functional_function, thermal_function = tgs_function(start_time, grating_spacing)

    # Thermal fit
    thermal_p0 = [0.05, 5e-6]
    thermal_bounds = ([0, 0], [1, 5e-4])
    popt, _ = curve_fit(lambda x, A, alpha: thermal_function(x, A, 0, 0, alpha, 0, 0, 0, 0), signal[:, 0], signal[:, 1], p0=thermal_p0, bounds=thermal_bounds)
    A, alpha = popt
    
    # Lorentzian fit on FFT of SAW signal
    saw_signal = np.column_stack([
        signal[:, 0], 
        signal[:, 1] - thermal_function(signal[:, 0], A, 0, 0, alpha, 0, 0, 0, 0)
    ])
    fft_signal = fft(saw_signal, **config['fft'])
    f, _, _, tau, _ = lorentzian_fit(fft_signal, file_idx, **config['lorentzian'])

    # Iteratively fit beta (displacement-reflectance ratio)
    q = 2 * np.pi / (grating_spacing * 1e-6)
    for _ in range(10):
        displacement = q * np.sqrt(alpha / np.pi)
        reflectance = (q ** 2 * alpha + 1 / (2 * max_time))
        beta = displacement / reflectance
        popt, _ = curve_fit(lambda x, A, alpha: thermal_function(x, A, 0, 0, alpha, beta, 0, 0, 0), signal[start_idx:, 0], signal[start_idx:, 1], p0=thermal_p0, bounds=thermal_bounds)
        A, alpha = popt

    # Functional fit
    functional_p0 = [0.05, 0.05, 0, alpha, beta, 0, tau, f]
    popt, pcov = curve_fit(functional_function, signal[start_idx:, 0], signal[start_idx:, 1], p0=functional_p0, maxfev=10000)
    A, B, C, alpha, beta, theta, tau, f = popt
    A_err, B_err, C_err, alpha_err, beta_err, theta_err, tau_err, f_err = np.sqrt(np.diag(pcov))

    if plot:
        plot_tgs(signal, start_idx, functional_function, thermal_function, popt, file_idx)

    return start_idx, start_time, grating_spacing, A, A_err, B, B_err, C, C_err, alpha, alpha_err, beta, beta_err, theta, theta_err, tau, tau_err, f, f_err, signal
