import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lorentzian(x, A, x0, W, C):
    return (A / ((x - x0)**2 + W**2)) + C

def lorentzian_peak_fit(fft_data: np.ndarray, two_mode: bool = False, plotty: bool = False) -> tuple:
    freq_lowB = 0.1
    freq_hiB = 0.9
    percent_peak_fit = 1

    st_point = 1
    end_point = 12000

    fft_data = np.copy(fft_data)
    fft_data[:, 0] = fft_data[:, 0] / 1e9
    fft_data[:st_point, 1] = 0 
    max_val = np.max(fft_data[st_point - 1:, 1])
    peak_ind_rel = np.argmax(fft_data[st_point - 1:, 1])
    peak_ind = (st_point - 1) + peak_ind_rel
    peak_loc = fft_data[peak_ind, 0]

    if two_mode:
        st_two_mode = int(round(0.1 * peak_ind))
        end_two_mode = int(round(0.75 * peak_ind))

    fft_data[:, 1] = fft_data[:, 1] / max_val

    if percent_peak_fit != 1:
        pass
    else:
        neg_ind_final = st_point
        pos_ind_final = len(fft_data)

    neg_ind_final_python = neg_ind_final - 1
    pos_ind_final_python = pos_ind_final
        
    x_data = fft_data[neg_ind_final_python:pos_ind_final_python, 0]
    y_data = fft_data[neg_ind_final_python:pos_ind_final_python, 1]

    ST = [1e-4, 0.53, 0.01, 0]
    LB = [0, freq_lowB, 0.001, 0]
    UB = [1, freq_hiB, 0.05, 1]

    popt, pcov = curve_fit(lorentzian, x_data, y_data, p0=ST, bounds=(LB, UB))
    perr = np.sqrt(np.diag(pcov))

    peak = popt[1] * 1e9
    peak_err = perr[1] * 1e9
    fwhm = 2 * popt[2] * 1e9
    tau = 1 / (np.pi * fwhm)

    if two_mode:

        new_fft_amp = fft_data[:, 1] - lorentzian(fft_data[:, 0], *popt)
        peak_ind_2_rel = np.argmax(new_fft_amp[st_two_mode:end_two_mode])
        peak_ind_2 = st_two_mode + peak_ind_2_rel
        peak_loc_2 = fft_data[peak_ind_2, 0]

        ST1 = [1e-4, peak_loc_2, 0.01, 0]

        popt2, pcov2 = curve_fit(lorentzian, fft_data[:, 0], new_fft_amp, p0=ST1, bounds=(LB, UB))
        perr2 = np.sqrt(np.diag(pcov2))

        peak2 = popt2[1] * 1e9
        peak_err2 = perr2[1] * 1e9
        fwhm2 = 2 * popt2[2] * 1e9
        tau2 = 1 / (np.pi * fwhm2)

        peak = np.array([peak, peak2])
        peak_err = np.array([peak_err, peak_err2])
        fwhm = np.array([fwhm, fwhm2])
        tau = np.array([tau, tau2])

    fft_noise = fft_data[:, 1] - lorentzian(fft_data[:, 0], *popt)
    signal_power = np.mean(fft_data[:, 1] ** 2)
    noise_power = np.mean(fft_noise ** 2)
    SNR_fft = 10 * np.log10(signal_power / noise_power)


    return peak, peak_err, fwhm, tau, popt, SNR_fft