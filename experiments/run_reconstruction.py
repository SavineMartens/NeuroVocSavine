import os
import argparse
import glob
import inspect
import random
from pathlib import Path

import librosa
import scipy
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from time import perf_counter

import phast

from notebooks import utils, plotting


N_FFT = 512
N_HOP = 32
REF_DB = 50.0
SR_PHAST = 50.0
WINDOW_SIZE = 1500
ONLY_LOW_SR = False
OUTPUT_FOLDER = "/scratch/jacob/output_folder"
TESTING = False

def echo(*args, **kwargs):
    if TESTING:
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        line_number = caller_frame.f_lineno
        name = caller_frame.f_code.co_qualname
        print(f"[{name},  line: {line_number}]:\t", *args, **kwargs)
        

def generate_ace():
    raise NotImplementedError()
    tp = phast.load_cochlear()
    (audio_signal, FS), pulse_train, neurogram = phast.ace_e2e(
        path,
        tp=tp,
        scaling_factor=scaling_factor,
        n_trials=n_trials,
    )


def pre_emphasis(signal: np.ndarray, factor: float = 0.97):
    return np.append(signal[0], signal[1:] - (factor * signal[:-1]))


def select_fibers(fiber_freq, mel_scale):
    grouped = np.digitize(fiber_freq, np.r_[mel_scale[1] - mel_scale[0], mel_scale], True)
    selected_fibers = []
    for fbin, nf in list(zip(*np.unique(grouped, return_counts=True)))[1:-1]:
        fibers = np.where(grouped == fbin)[0]
        if nf < 10:
            sf = np.r_[fibers, np.random.choice(fibers, 10 - nf, replace=False)]
        else:
            sf = np.random.choice(fibers, 10, replace=False)
        selected_fibers.extend(sorted(sf))
    return np.array(selected_fibers)


def get_fiber_freq_specres(tp, max_freq, power=25):
    channel_freq = phast.scs.ab.defaults.virtual_channel_frequencies(
        tp.i_det.shape[1], max_freq + 500
    )
    w = (-tp.i_det / tp.i_det.sum(axis=1).reshape(-1, 1)) + (2 / len(channel_freq))
    w = np.power(w, power) / np.power(w, power).sum(axis=1).reshape(-1, 1)
    fiber_freq = w @ channel_freq
    return fiber_freq


def get_fiber_freq_position(tp):
    from phast.scs.ab.defaults import ELECTRODE_FREQ_LOWER, ELECTRODE_FREQ_UPPER
    electrode_freq = np.r_[ELECTRODE_FREQ_LOWER, ELECTRODE_FREQ_UPPER[-1]]

    fiber_freq = np.interp(
        tp.position[::-1], 
        np.r_[tp.position[-1], tp.electrode.position[::-1], tp.position[0]], 
        np.r_[tp.greenwood_f.max(), electrode_freq[::-1], tp.greenwood_f.min()], 
    )[::-1]
    return fiber_freq
    



def generate_specres(path):
    n_trials = 20
    cs = True
    apply_premph = False
    min_freq = 150
    max_freq = 10_500
    n_mels = 64
    mel_scale = librosa.filters.mel_frequencies(n_mels, fmin=min_freq, fmax=max_freq)
    positional_freq = True
    tp = phast.load_df120()
    
    assert max_freq > phast.scs.ab.defaults.ELECTRODE_FREQ_UPPER[-1]
    
    if positional_freq:
        fiber_freq = get_fiber_freq_position(tp)
    else:
        fiber_freq = get_fiber_freq_specres(tp, max_freq)

    selected_fibers = select_fibers(fiber_freq, mel_scale)
    fiber_freq = fiber_freq[selected_fibers]
    
    audio_signal, audio_fs = phast.scs.ab.frontend.read_wav(path, stim_db=REF_DB)
    audio_signal += np.random.normal(0, 1e-20, size=len(audio_signal))
    if apply_premph:
        audio_signal = pre_emphasis(audio_signal[0]).reshape(1, -1)

    (audio_signal, FS), pulse_train, neurogram = phast.ab_e2e(
        audio_signal=audio_signal,
        audio_fs=audio_fs,
        tp=tp,
        current_steering=cs,
        scaling_factor=1.4,
        ramp_duration=(audio_signal.size / audio_fs) * 0.05,
        n_trials=n_trials,
        accommodation_amplitude=0.07,
        adaptation_amplitude=7.142,
        accommodation_rate=2,
        adaptation_rate=19.996,
        selected_fibers=selected_fibers,
        spont_activity=SR_PHAST,
        n_jobs=10 if not TESTING else -1,
        stim_db=REF_DB
    )
    audio_signal = audio_signal[0]
    neurogram_data = neurogram.data #/ n_trials
    binned_data = utils.bin_over_y(
        neurogram_data, fiber_freq, mel_scale, agg=np.sum
    )
    
    # fname = os.path.basename(path).split(".")[0]
    # np.save(f"{fname}_specres_spikes", binned_data)
    
    echo(f"ng bin width: {neurogram.binsize}")
    data = utils.smooth(binned_data, "hann", WINDOW_SIZE, 1)
    data = utils.min_max_scale(data, 0, 1)
    echo(data.min(), data.mean(), data.max())
    ret_data = data, neurogram.binsize, min_freq, max_freq, n_mels
    return ret_data


class BruceGenerator:
    def __init__(self, min_freq = 150, max_freq = 10_500):
        bruce.set_seed(22)

        self.n_trials = 20
        self.n_fibers = 10
        
        if ONLY_LOW_SR:
            self.n_low = 10
            self.n_med = 0
            self.n_high = 0
        else:
            self.n_low = self.n_med = int(np.floor(self.n_fibers / 5))
            self.n_high = self.n_fibers - (2 * self.n_med)
        
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_rep = 1
        self.n_mels = 64 
        
        self.mel_scale = librosa.filters.mel_frequencies(self.n_mels, fmin=self.min_freq, fmax=self.max_freq)

        self.ng = bruce.Neurogram(
            self.mel_scale,
            n_low=self.n_low,
            n_med=self.n_med,
            n_high=self.n_high,
            n_threads=self.n_fibers if not TESTING else -1,
        )
        self.dt_spikes = 1e-05 
        self.ng.bin_width = self.dt_spikes
        
        # min(1 / (max_freq * 2 * n_bins), 3.6e-05)
        # this needs to use the sampling frequency of the model
        # idk why but otherwise there is a shift
        # echo(f"ng bin width: {self.ng.bin_width}")
        
    @property
    def dt(self):
        return self.ng.bin_width
    
    def iter_fibers(self):
        for i in range(self.n_mels):
            for f in self.ng.get_fibers(i):
                yield f
    
    def get_sponts(self):
        return np.array([f.spont for f in self.iter_fibers()])
    
    def get_refr(self):
        return np.array([f.tabs + f.trel for f in self.iter_fibers()])
    
    
    def get_ng_data(self, stim):
        time = perf_counter()
        self.ng.bin_width = stim.time_resolution
        self.ng.create(stim, n_rep=self.n_rep, n_trials=self.n_trials)
        echo(perf_counter() - time)
        self.spikes = self.ng.get_output()
        return self.spikes
        
    def rebin_data(self):
        data_binned = np.zeros((self.data.shape[0], self.data.shape[1] * 10))
        data_binned[:, ::10] = self.data.copy()
        self.data = utils.make_bins(36, data_binned)
        self.ng.bin_width *= 3.6
        
        echo(self.ng.bin_width, int(1 / self.ng.bin_width))
        
        
    def average_response(self):
        "This somehow needs to divide by n_rep * n_trials * n_fibers"        
        self.data = utils.min_max_scale(self.data, 0, 1)
        echo(self.data.min(), self.data.max())  
        
    def smooth(self, window_size):
        echo(f"smoothing with hann window of size {window_size}")
        self.data = utils.smooth(self.data, "hann", window_size, 1)
        
    def generate_neurogram(self, path):
        
        
        self.stim = bruce.stimulus.from_file(path, False, normalize=False)
        echo(self.stim)
        echo(rms(self.stim.data))
        self.stim = bruce.stimulus.normalize_db(self.stim, REF_DB)
        echo(rms(self.stim.data))
        
        # self.stim = bruce.stimulus.normalize_db(self.stim, 65.)
        # echo(rms(self.stim.data))
        # breakpoint()
        # exit()
        
        self.spikes = self.get_ng_data(self.stim)
        self.data = self.spikes.sum(axis=1)
        # fname = os.path.basename(path).split(".")[0]
        # np.save(f"{fname}_burce_spikes", self.data)
                
        self.rebin_data()
        self.smooth(WINDOW_SIZE)
        self.average_response()
        self.data = self.data.clip(0, np.quantile(self.data.ravel(), 0.995))
        self.average_response()
        return self.data, self.ng.bin_width, self.min_freq, self.max_freq, self.n_mels
            
    def __call__(self, path):
        self.audio_signal, self.audio_fs = librosa.load(path, sr=None)
        return self.generate_neurogram(path)
        

def rms(x):
    return np.sqrt(np.mean(np.power(x, 2)))

def generate_bruce(path):
    bg = BruceGenerator()
    return bg(path)


def reconstruct_neurogram(
    M: np.ndarray, 
    sr: int, 
    min_freq: int, 
    max_freq: int, 
    n_fft: int, 
    n_hop: int
):
    """
    Parameters
    ----------
    M: np.ndarray
        A neurogram structure, scaled to a power spectrum, and downsampled by a factor
        of n_hop
    sr: int 
        The sampling rate of the original neurogram (before resampling with n_hop)
    min_freq: int 
        The lower bound of the filter bank
    max_freq: int
        The upper bound of the filter bank
    n_hop: int
        The number of hops that were applied to M
    """

    ### mel_to_stft ###
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft, 
        n_mels=M.shape[-2], 
        dtype=M.dtype, 
        fmin=min_freq,
        fmax=max_freq   
    )
    inverse = librosa.util.nnls(mel_basis, M)
    inverse = np.power(inverse, 1.0 / 2.0, out=inverse)  
    ### mel_to_stft ###

    ### mel_to_audio ###
    reconstructed = librosa.feature.inverse.griffinlim(
        inverse,
        n_iter=320,
        hop_length=n_hop,
        win_length=None,
        n_fft=n_fft,
        window="hann",
        center=True,
        dtype=np.float32,
        length=None,
        pad_mode="constant",
        momentum=0.99,
        init="random",
        random_state=None
    )
    ### mel_to_audio ###
    return reconstructed


def neurogram_to_wav(data, binsize, audio_size, min_freq, max_freq, resample="poly"):   
    # Resample
    
    from math import gcd
    
    n_s = int(np.ceil(data.shape[1] / N_HOP))
    echo(n_s, data.shape)
    
    if resample == 'fft':
        data = np.array([scipy.signal.resample(x, n_s) for x in data]).clip(0, 1)
    elif resample == 'poly':
        g = gcd(n_s, data.shape[1])
        data = np.array([
            scipy.signal.resample_poly(row, n_s // g, data.shape[1] // g)
            for row in data
        ]).clip(0, 1)
    else:
        raise TypeError
        
        
    echo(data.min(), data.max())
    echo(data.shape)
    
    # Scale to -80, 0
    data = utils.min_max_scale(data, data_min=0, data_max=1)
    echo(data.min(), data.max())
    
    # Apply smoothing filter
    # data = scipy.ndimage.median_filter(data, size=(1, 3))
    
    # Convert to power
    data = librosa.db_to_power(data, ref=REF_DB)
    echo(data.min(), data.max())
    
    reconstructed = reconstruct_neurogram(data, np.ceil(1 / binsize), min_freq, max_freq, N_FFT, N_HOP)

    echo(reconstructed.shape)
    reconstructed = scipy.signal.resample(reconstructed, audio_size)
    
    if reconstructed.max() > abs(reconstructed.min()):
        ub = 1
        lb = reconstructed.min() / reconstructed.max()
    else:
        lb = -1 
        ub = reconstructed.max() / abs(reconstructed.min())
    
    echo(reconstructed.min(), reconstructed.max())
    echo(lb, ub)
    
    reconstructed[reconstructed > 0] = utils.min_max_scale(reconstructed[reconstructed > 0] , 0, ub)
    reconstructed[reconstructed < 0] = utils.min_max_scale(reconstructed[reconstructed < 0] , lb, 0)
    return reconstructed


def plot(audio_signal, reconstructed_signal, audio_fs, n_mels, min_freq, max_freq, title, save=True):
    fig = plt.figure()
    fig.suptitle(title)
    gs = plt.GridSpec(2, 2, wspace=0.25, hspace=0.25) # 2x2 grid
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    
    din_healthy_word = []
    
    for i, (ax, sig) in enumerate(zip((ax0, ax1), (audio_signal, reconstructed_signal))):
        S = librosa.feature.melspectrogram(
            y=sig, 
            sr=audio_fs,
            n_mels=n_mels, 
            fmin=min_freq, 
            fmax=max_freq
        )
      
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=audio_fs,
                            fmin=min_freq, 
                            fmax=max_freq, ax=ax, vmax=0, vmin=-80)
        ax.set_xlabel(None)
        if i == 1:
            ax.set_title("reconstructed")
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
        else:
            ax.set_title("original")

    t = np.arange(audio_signal.size) / audio_fs
    ax2.plot(t, audio_signal, label='original', alpha=.6)
    ax2.plot(t, reconstructed_signal, label='reconstructed', alpha=.6)
    ax2.grid()
    ax2.legend()
    ax2.set_xlabel("time [s]")
    if save:
        plt.savefig("x.pdf")
        plt.savefig(f"{title}.pdf")
    
   
def rescale(stim, stimdb=50.0):
    return stim / rms(stim) * 20e-6 * 10**(stimdb / 20)
    
def reconstruct(path, neurogram_generator, output_path = None, scale_to_db:int = 70, overwrite: bool = False):
    path = os.path.realpath(path)
    assert os.path.isfile(path)
    name = os.path.basename(path).split(".")[0]

    neurogram_generator = NEUROGRAM_GENERATORS[neurogram_generator]
    method_name = neurogram_generator.__name__.replace("generate_", "")

    audio_signal, audio_fs = librosa.load(path, sr=None)
    
    neurogram_freq_bin, binsize, min_freq, max_freq, n_mels = neurogram_generator(path)
    reconstructed = neurogram_to_wav(
        neurogram_freq_bin, binsize, audio_signal.size, min_freq, max_freq
    )
    
    if scale_to_db:
        reconstructed = rescale(reconstructed, scale_to_db)
        audio_signal = rescale(audio_signal, scale_to_db)
        
    
    echo(rms(reconstructed))
    echo(rms(audio_signal))
    
    if TESTING:
        plot(audio_signal, reconstructed, audio_fs, n_mels, min_freq, max_freq, f"{method_name}_{name}")
        
            
    if output_path is not None:
        output_file = os.path.join(output_path, f"{name}.wav")
        if TESTING:
            output_file = os.path.join(output_path, f"{name}_{method_name}_{REF_DB}_{WINDOW_SIZE}.wav")
            
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if os.path.exists(output_file) and not overwrite:
            if not TESTING:
                return
            os.remove(output_file)
            
        sf.write(output_file, reconstructed, audio_fs, subtype="PCM_32")
        
    return audio_signal, reconstructed, audio_fs, n_mels, min_freq, max_freq, f"{method_name}_{name}"


NEUROGRAM_GENERATORS = [
    generate_specres,
    generate_bruce,
    generate_ace,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--neurogram_generator",
        default=0,
        choices=range(len(NEUROGRAM_GENERATORS)),
        type=int,
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--sr", type=float, default=SR_PHAST)
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--n_fft", type=int, default=N_FFT)
    parser.add_argument("--n_hop", type=int, default=N_HOP)
    parser.add_argument("--db", type=int, default=REF_DB)
    parser.add_argument("--testing", action='store_true')
    parser.add_argument("--only_low_sr", action='store_true')
    args = parser.parse_args()
    
    TESTING = args.testing
    SR_PHAST = args.sr
    WINDOW_SIZE = args.window_size
    REF_DB = args.db
    ONLY_LOW_SR = args.only_low_sr
    
    N_FFT = args.n_fft
    N_HOP = args.n_hop
    
    path = os.path.normpath(args.path)
    source_name = os.path.basename(path)
    
    NEUROGRAM_GENERATORS[args.neurogram_generator]
    gen_name = NEUROGRAM_GENERATORS[args.neurogram_generator].__name__.replace("generate_","")
    tgt_name = f"{source_name}_{gen_name}_{int(WINDOW_SIZE)}_{N_FFT}_{N_HOP}"
    if gen_name == 'specres':
        tgt_name = f"{tgt_name}_{int(SR_PHAST)}"
        
    elif args.only_low_sr:
        tgt_name = f"{tgt_name}_only_low_sr"
        
    if os.path.isdir(path):
        snrs = os.listdir(path)
        tgt_folder = os.path.join(os.path.dirname(path), tgt_name)

        for i, snr in enumerate(snrs, 1):
            src = os.path.join(path, snr)
            tgt = os.path.join(tgt_folder, snr)
            os.makedirs(tgt, exist_ok=True)
            files = os.listdir(src)
            random.shuffle(files)
            for j, f in enumerate(files, 1):
                perc = (i*j) / (len(snrs) * 120)
                src_file = os.path.join(src, f)
                tgt_file = os.path.join(tgt, f)
                if os.path.isfile(tgt_file):
                    continue
                Path(tgt_file).touch()
                reconstruct(src_file, args.neurogram_generator, tgt, overwrite=True)
                print(perc * 100, "%")
                
    elif os.path.isfile(path):
        reconstruct(
            args.path, args.neurogram_generator, args.output_path or OUTPUT_FOLDER
        )
