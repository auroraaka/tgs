from typing import Tuple

import numpy as np
from scipy.special import erfc

def tgs_function(start_time: float, grating_spacing: float) -> Tuple[callable, callable]:
    """
    Build functional and thermal fit functions.

    Notes:
        The thermal diffusivity (α) is set to a minimum value of 1e-10 to avoid numerical instability 
        in the square root term of the displacement field.

    Parameters:
        start_time (float): start time of TGS data [ns] # TODO: check units
        grating_spacing (float): grating spacing of TGS probe [µm]

    Returns:
        Tuple[callable, callable]: (functional fit, thermal fit)
    """
    q = 2 * np.pi / (grating_spacing * 1e-6)

    def functional_function(x, A, B, C, alpha, beta, theta, tau, f, q=q):
        """
        Functional fit function.

        Equation:
            I(t) = A [erfc(q √(αt)) - (β/√t) e^(-q²αt)] + B sin(2πft + Θ) e^(-t/τ) + C

        Parameters:
            x (np.ndarray): time array [ns]
            A (float): constant [V] # TODO: check units
            B (float): constant [V] # TODO: check units
            C (float): constant [V] # TODO: check units
            alpha (α) (float): thermal diffusivity [m²/s] # TODO: check units
            beta (β) (float): displacement-reflectance ratio [dimensionless] # TODO: check units
            theta (Θ) (float): acoustic phase [rad] # TODO: check units
            tau (τ) (float): acoustic decay constant [ns] # TODO: check units
            f (float): surface acoustic wave frequency [Hz] # TODO: check units
            q (float): excitation wave vector [rad/m] # TODO: check units

        Returns:
            np.ndarray: functional fit response [V] # TODO: check units
        """
        t = x + start_time
        alpha = max(alpha, 1e-10)
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        sinusoid = np.sin(2 * np.pi * f * t + theta) * np.exp(-t / tau)
        return A * (displacement_field + thermal_field) + B * sinusoid + C

    def thermal_function(x, A, B, C, alpha, beta, theta, tau, f, q=q):
        """
        Thermal fit function.

        Equation:
            I(t) = A [erfc(q √(αt)) - (β/√t) e^(-q²αt)] + C

        Parameters:
            x (np.ndarray): time array [ns] # TODO: check units
            A (float): constant [V] # TODO: check units
            C (float): constant [V] # TODO: check units
            alpha (α) (float): thermal diffusivity [m²/s] # TODO: check units
            beta (β) (float): thermal conductivity [W/(m·K)] # TODO: check units
            q (float): excitation wave vector [rad/m] # TODO: check units

        Returns:
            np.ndarray: thermal fit response [V] # TODO: check units
        """
        t = x + start_time
        alpha = max(alpha, 1e-10)
        displacement_field = erfc(q * np.sqrt(alpha * t))
        thermal_field = beta / np.sqrt(t) * np.exp(-q ** 2 * alpha * t)
        return A * (displacement_field + thermal_field) + C

    return functional_function, thermal_function

def lorentzian_function(x: np.ndarray, A: float, x0: float, W: float, C: float) -> np.ndarray:
    """
    Lorentzian function for peak fitting.

    Equation:
        L(x) = A / ((x - x0)² + W²) + C

    Parameters:
        x (np.ndarray): frequency array [GHz]
        A (float): amplitude of the peak
        x0 (float): center frequency [GHz]
        W (float): peak width parameter [GHz]
        C (float): vertical offset

    Returns:
        np.ndarray: Lorentzian function values
    """
    return A / ((x - x0) ** 2 + W ** 2) + C