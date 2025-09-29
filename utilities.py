import numpy as np
import matplotlib.pyplot as plt
import math


def get_row_column_nums(number_bands_prime, minimum_row=4):
    def closestDivisors(n):
        a = round(math.sqrt(n))
        while n%a > 0: a -= 1
        row = min(a,n//a)
        column = max(a,n//a)
        return row, column

    def is_prime(n):
        for i in range(2,n):
            if (n%i) == 0:
                return False
        return True

    row_plot = 0
    while row_plot<minimum_row:
        number_bands_prime += 1
        if not is_prime(number_bands_prime):
            row_plot, column_plot = closestDivisors(number_bands_prime)
    return row_plot, column_plot


def plot_neurogram(neurogram, title='Neurogram'):
    fig = plt.figure()
    t = np.arange(neurogram.data.shape[1]) / (1/neurogram.dt)
    frequencies = neurogram.frequencies
    mesh = plt.pcolormesh(t, frequencies, neurogram.data, cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label('Spike rate [spikes/s]')
    plt.xlabel('Time [s]')
    plt.xlim(t[0], t[-1])
    plt.ylabel('Apical                     Fiber number                   Basal')
    plt.title(title)
    return fig


def plot_sum_neurogram(neurogram, title='Neurogram'):
    fig, axes = plt.subplots(2,1)
    plt.subplot(2,1,1)
    t = np.arange(neurogram.data.shape[1]) / (1/neurogram.dt)
    frequencies = neurogram.frequencies
    mesh = plt.pcolormesh(t, frequencies, neurogram.data, cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label('Spike rate [spikes/s]')
    plt.xlabel('Time [s]')
    plt.xlim(t[0], t[-1])
    plt.ylabel('Frequency')
    plt.title(title)

    plt.subplot(2,1,2)
    plt.plot(t, np.sum(neurogram.data,axis=0))
    plt.xlabel('Time [s]')
    plt.xlim(t[0], t[-1])

    return fig

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def compute_d_prime_2matrices_trials_freq(standard_matrix, inverted_matrix):
    """
    sources: 
    - https://eshedmargalit.com/dprime_calculator/
    - MacMillan 2005, "Detection Theory: A User's Guide" eq (12.2) page 303 Chapter components of Sensitivity
    - Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics. eq (3.10) page 69

    Compute d-prime across frequency channels.

    Parameters:
    - standard_matrix: np.ndarray of shape (n_trials, n_freqs)
    - neurograms_stim2: np.ndarray of shape (n_trials, n_freqs)
    
    Returns:
    - d_prime_vector: np.ndarray of shape (n_freqs,), one d′ per frequency channel
    """
    mu1 = np.mean(standard_matrix, axis=0)
    mu2 = np.mean(inverted_matrix, axis=0)
       
    var1 = np.var(standard_matrix, axis=0, ddof=1)
    var2 = np.var(inverted_matrix, axis=0, ddof=1)
    
    pooled_std = np.sqrt(0.5 * (var1 + var2))
    d_prime_vector = np.abs(mu1 - mu2) / (pooled_std + 1e-9)  # add small constant to avoid divide-by-zero
    
    return d_prime_vector