from typing import Tuple, Union

import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit

from process import process_signal
from lorentzian import lorentzian_fit
from fft import fft

def tgs_function(start_time: float, grating_spacing: float) -> Tuple[callable, callable]:
    """
    Build functional and thermal fit functions.

    Parameters:
        start_time (float): start time of TGS data [s] # TODO: check units
        grating_spacing (float): grating spacing of TGS probe [µm]

    Returns:
        Tuple[callable, callable]: (functional fit, thermal fit)
    """
    q = 2 * np.pi / (grating_spacing * 1e-6)

    def functional_fit(x, A, B, C, alpha, beta, theta, tau, f, q=q):
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
            beta (β) (float): displacement-reflectance ratio [dimensionless] # TODO: check units
            theta (Θ) (float): acoustic phase [rad] # TODO: check units
            tau (τ) (float): acoustic decay constant [s] # TODO: check units
            f (float): surface acoustic wave frequency [Hz] # TODO: check units
            q (float): excitation wave vector [rad/m] # TODO: check units

        Returns:
            np.ndarray: functional fit response [V] # TODO: check units
        """
        t = x + start_time
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        sinusoid = np.sin(2 * np.pi * f * t + theta) * np.exp(-t / tau)
        return A * (displacement_field + thermal_field) + B * sinusoid + C

    def thermal_fit(x, A, B, C, alpha, beta, theta, tau, f, q=q):
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
    signal, max_time, start_time, start_idx = process_signal(pos_file, neg_file, grating_spacing, **config['process'])
    functional_fit, thermal_fit = tgs_function(start_time, grating_spacing)

    # Thermal fit
    thermal_p0 = [0.05, 5e-6]
    thermal_bounds = ([0, 0], [1, 5e-4])
    popt, _ = curve_fit(lambda x, A, alpha: thermal_fit(x, A, 0, 0, alpha, 0, 0, 0, 0), signal[:, 0], signal[:, 1], p0=thermal_p0, bounds=thermal_bounds)
    A, alpha = popt
    
    # Lorentzian fit on FFT of SAW signal
    saw_signal = np.column_stack([
        signal[:, 0], 
        signal[:, 1] - thermal_fit(signal[:, 0], A, 0, 0, alpha, 0, 0, 0, 0)
    ])
    fft_signal = fft(saw_signal, **config['fft'])
    f, _, _, tau, _ = lorentzian_fit(fft_signal, **config['lorentzian'])

    # Iteratively fit beta (displacement-reflectance ratio)
    q = 2 * np.pi / (grating_spacing * 1e-6)
    for _ in range(10):
        displacement = q * np.sqrt(alpha / np.pi)
        reflectance = (q ** 2 * alpha + 1 / (2 * max_time))
        beta = displacement / reflectance
        popt, _ = curve_fit(lambda x, A, alpha: thermal_fit(x, A, 0, 0, alpha, beta, 0, 0, 0), signal[start_idx:, 0], signal[start_idx:, 1], p0=thermal_p0, bounds=thermal_bounds)
        A, alpha = popt

    # Functional fit
    functional_p0 = [0.05, 0.05, 0, alpha, beta, 0, tau, f]
    popt, pcov = curve_fit(functional_fit, signal[start_idx:, 0], signal[start_idx:, 1], p0=functional_p0, maxfev=10000)
    A, B, C, alpha, beta, theta, tau, f = popt
    A_err, B_err, C_err, alpha_err, beta_err, theta_err, tau_err, f_err = np.sqrt(np.diag(pcov))
    
    return start_time, grating_spacing, A, A_err, B, B_err, C, C_err, alpha, alpha_err, beta, beta_err, theta, theta_err, tau, tau_err, f, f_err, signal, signal[start_idx:]
