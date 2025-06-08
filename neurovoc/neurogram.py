import pickle
from dataclasses import dataclass, field
from fractions import Fraction

import librosa
import numpy as np
import scipy


@dataclass
class Neurogram:
    dt: float
    frequencies: np.ndarray = field(repr=None)
    data: np.ndarray = field(repr=None)
    source: str 
    shape: tuple = None

    def __post_init__(self):
        self.shape = self.data.shape

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def duration(self):
        return self.data.shape[1] * self.dt
    
    @property
    def sample_rate(self) -> int:
        return int(np.ceil(1 / self.dt))
    
    @property
    def min_freq(self) -> float:
        return np.min(self.frequencies)
    
    @property
    def max_freq(self) -> float:
        return np.max(self.frequencies)

    @staticmethod
    def load(path: str) -> "Neurogram":
        with open(path, "rb") as f:
            return pickle.load(f)


def mel_scale(n_mels: int, min_freq: int, max_freq: int):
    return librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)


def bin_over_y(
    data: np.ndarray, src_y: np.ndarray, tgt_y: np.ndarray, agg: callable = np.sum
):
    data_binned = np.zeros((len(tgt_y), data.shape[1]))
    bins = np.digitize(src_y, tgt_y)

    for i in range(len(tgt_y)):
        if not any(bins == i):
            continue
        data_binned[i] = agg(data[bins == i], axis=0)
    return data_binned


def smooth(
    data: np.ndarray,
    window_type: str = "hann",
    window_size: int = 2048,
    hop_length: int = None,
) -> np.ndarray:

    hop_length = hop_length or max(window_size // 4, 1)
    window = scipy.signal.get_window(window_type, window_size)
    wsum = window.sum()
    data = np.vstack(
        [
            (
                np.convolve(np.pad(data[i], (0, window_size)), window, mode="valid")[
                    : data[i].size
                ]
                / wsum
            )[::hop_length]
            for i in range(data.shape[0])
        ]
    )
    return data


def min_max_scale(
    data: np.ndarray,
    a: float = -80,
    b: float = 0,
    data_min: float = None,
    data_max: float = None,
):
    data_min = data_min or np.min(data)
    data_max = data_max or np.max(data)
    return a + (data - data_min) * (b - a) / (data_max - data_min)


def rebin_signal(signal, orig_sr, target_sr):
    signal = np.asarray(signal)
    frac = Fraction(target_sr, orig_sr).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    upsampled = np.zeros(len(signal) * up)
    upsampled[::up] = signal

    n_bins = len(upsampled) // down
    rebinned = upsampled[: n_bins * down].reshape(n_bins, down).sum(axis=1)

    return rebinned


def rebin_data(data: np.ndarray, dt_data: float, dt_tgt: float):
    src_sr = int(round(1 / dt_data))
    tgt_sr = int(round(1 / dt_tgt))
    return np.vstack([rebin_signal(x, src_sr, tgt_sr) for x in data])


def make_bins(n, data):
    if n == 1:
        return data
    return data[:, : len(data[0]) // n * n].reshape(data.shape[0], -1, n).sum(axis=2)
