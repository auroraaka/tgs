import numpy as np
from scipy.signal import periodogram, savgol_filter, windows
import matplotlib.pyplot as plt

def TGS_phase_fft(SAW_only: np.ndarray, psd_out: int, truncate_fraction: float, plot_things: bool) -> np.ndarray:
    """
    Generates a filtered Fast Fourier Transform of SAW signal.
    
    Parameters:
    SAW_only : ndarray
        This should be the total_signal with an erfc fit subtracted from it so it's macroscopically flat.
    psd_out : int
        If psd_out=1, save out power spectrum; else save out fft magnitude.
    truncate_fraction : float
        What percentage of the signal should be analyzed (e.g., first 70%).
    plot_things : bool
        If True, the FFT will be plotted.
        
    Returns:
    fft_output : ndarray
        Array containing frequencies and the corresponding FFT amplitudes.
    """


    # Boolean options for plotting and processing
    derivative = 1
    copy = 0
    plotfft = plot_things
    saveout = 0
    cut_tails = 1000

    newlength = int(np.ceil(SAW_only.shape[0] * truncate_fraction))
    SAW_only_truncated = np.zeros((newlength, 2))
    SAW_only_truncated[:, 0] = SAW_only[:newlength, 0]
    SAW_only_truncated[:, 1] = SAW_only[:newlength, 1] / np.max(SAW_only[:newlength, 1])

    if copy:
        total_length = newlength * 2 - 1
        SAW_only_truncated_mirrored = np.zeros((total_length, 2))
        SAW_only_truncated_mirrored[:newlength, :] = SAW_only_truncated
        for i in range(newlength, total_length):
            idx = (2 * newlength - 2) - i
            SAW_only_truncated_mirrored[i, 0] = SAW_only_truncated_mirrored[i - 1, 0] + (SAW_only_truncated[1, 0] - SAW_only_truncated[0, 0])
            SAW_only_truncated_mirrored[i, 1] = SAW_only_truncated[idx, 1]
        SAW_only_truncated = SAW_only_truncated_mirrored

    tstep = SAW_only_truncated[-1, 0] - SAW_only_truncated[-2, 0]
    SAW_only_derivative = np.diff(SAW_only_truncated[:, 1]) / tstep
    SAW_only_derivative = SAW_only_derivative / np.max(SAW_only_derivative)
    if derivative:
        SAW_only_truncated = np.column_stack((SAW_only_truncated[:len(SAW_only_derivative), 0], SAW_only_derivative))

    num_points = SAW_only_truncated.shape[0]
    sampling_rate = num_points / (SAW_only_truncated[-1, 0] - SAW_only_truncated[0, 0])
    padding = 18
    padsize = 2 ** padding - num_points - 2
    pad_val = 0
    pad = np.full((padsize,), pad_val)
    padded_end_time = SAW_only_truncated[-1, 0] + tstep * np.arange(1, padsize + 1)
    SAW_only_padded = np.vstack((SAW_only_truncated, np.column_stack((padded_end_time, pad))))

    number_of_fft_points = SAW_only_padded.shape[0]
    window = windows.boxcar(number_of_fft_points)
    freq, power_spectral_density = periodogram(
        SAW_only_padded[:, 1],
        fs=sampling_rate,
        window=window,
        nfft=number_of_fft_points,
        scaling='density'
    )

    psd_length_adjust = int(np.ceil(len(power_spectral_density) * (1 / 5))) - 6 * cut_tails
    power_spectral_density[:cut_tails] = 0
    power_spectral_density[psd_length_adjust:] = 0
    power_spectral_density = power_spectral_density / np.max(power_spectral_density)

    if psd_out:
        output_amplitude = power_spectral_density
    else:
        output_amplitude = np.sqrt(power_spectral_density)

    window_length = 201
    polyorder = 5
    if window_length > len(output_amplitude):
        window_length = len(output_amplitude) // 2 * 2 + 1  # Ensure window_length is odd
    output_amplitude_smooth = savgol_filter(output_amplitude[:len(freq)], window_length, polyorder)

    fft_output = np.column_stack((freq[:len(output_amplitude_smooth)], output_amplitude_smooth))

    if saveout:
        np.savetxt('dat_spec.txt', fft_output)

    if plotfft:
        plt.figure()
        plt.plot(freq[:len(output_amplitude_smooth)], output_amplitude_smooth / np.max(output_amplitude), 'k', linewidth=2)
        plt.xlim(0, 1.0e9)
        plt.xlabel('Frequency [Hz]', fontsize=24, fontname='Times')
        plt.ylabel('Intensity [a.u.]', fontsize=24, fontname='Times')
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.spines['top'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.tick_params(width=3)
        plt.show()

    return fft_output