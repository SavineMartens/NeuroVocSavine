
import librosa
import numpy as np


def estimate_snr_simple(y, frame_length=2048, hop_length=512, noise_percentile=20):
    """
    Estimate SNR of a single-channel signal using frame energy statistics.
    Assumes lowest-energy frames are noise-only.
    """
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energies = np.mean(frames**2, axis=0)

    # Estimate noise power as the Nth percentile
    noise_power = np.percentile(energies, noise_percentile)
    signal_power = np.mean(energies)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def get_power(y, frame_length=2048, hop_length=512):
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energies = np.mean(frames**2, axis=0)
    return np.mean(energies)

def din_estimate_snr(y, sr):
    t = np.arange(len(y)) / sr
    mask = np.logical_and(t > .5, t < max(t) - 0.5)
    return 10 * np.log10(get_power(y[mask]) / get_power(y[~mask]))

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def rms_db(y):
    return 20 * np.log10(rms(y))

def scale_to_target_dbfs(y, target_dbfs):
    current_dbfs = rms_db(y)
    diff = target_dbfs - current_dbfs
    gain = 10 ** (diff / 20)
    return y * gain

