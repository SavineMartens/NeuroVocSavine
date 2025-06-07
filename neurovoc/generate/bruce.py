import pathlib


import numpy as np
import brucezilany

from .neurogram import mel_scale, rebin_data, smooth, min_max_scale, Neurogram


def bruce(
    audio: str | pathlib.Path | np.ndarray,
    audio_fs: int = None,
    ref_db: int = 50,
    n_trials: int = 20,
    min_freq: int = 150,
    max_freq: int = 10_500,
    n_mels: int = 64,
    n_fibers_per_bin: int = 10,
    window_size: int = 1500,
    normalize: bool = True,
    seed: int = 42,
    n_threads: int = -1,
    binsize: float = 3.6e-05,
    n_rep: int = 1,
    remove_outliers: bool = True,
    **kwargs
):
    brucezilany.set_seed(seed)
    np.random.seed(seed)

    if isinstance(audio, (str, pathlib.Path)):
        stim = brucezilany.stimulus.from_file(audio, False, normalize=False)
    elif isinstance(audio, [np.ndarray]):
        duration = (1 / audio_fs) * len(audio)
        stim = brucezilany.stimulus.Stimulus(audio, audio_fs, duration)
    else:
        if audio_fs is None:
            raise TypeError(
                "Wrong audio signal type. If a numpy array is given, audio_fs must also be passed"
            )

    stim = brucezilany.stimulus.normalize_db(stim, ref_db)
    frequencies = mel_scale(n_mels, min_freq, max_freq)

    n_low = n_med = int(np.floor(n_fibers_per_bin / 5))
    n_high = n_fibers_per_bin - (2 * n_med)
    ng = brucezilany.Neurogram(
        frequencies,
        n_low=n_low,
        n_med=n_med,
        n_high=n_high,
        n_threads=n_threads,
    )
    ng.bin_width = stim.time_resolution
    ng.create(stim, n_rep=n_rep, n_trials=n_trials)
    neurogram_data = ng.get_output().sum(axis=1)
    neurogram_data = rebin_data(neurogram_data, stim.time_resolution, binsize)

    if window_size is not None:
        neurogram_data = smooth(neurogram_data, "hann", window_size, 1)

    if normalize:
        neurogram_data = min_max_scale(neurogram_data, 0, 1)

    if remove_outliers:
        neurogram_data.clip(0, np.quantile(neurogram_data.ravel(), 0.995))
        neurogram_data = min_max_scale(neurogram_data, 0, 1)

    return Neurogram(binsize, frequencies, neurogram_data, "brucezilany")
