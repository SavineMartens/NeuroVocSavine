import matplotlib.pyplot as plt
import numpy as np

def PSTH(neurogram_MP, neurogram_M, MP_str='', M_str=''):
    fig = plt.figure()
    t = np.arange(neurogram_MP.shape[1]) / (1/neurogram_MP.dt)
    f_id = np.arange(int(np.argwhere(neurogram_MP.frequencies > 1500)[0][0]), int(np.argwhere(neurogram_MP.frequencies < 2500)[-1][0]))
    plt.plot(t, neurogram_MP.data[f_id].sum(axis=0), label='NH Neurogram - Masker + Probe')
    plt.plot(t, neurogram_M.data[f_id].sum(axis=0), label='NH Neurogram - Masker', linestyle='dotted')
    plt.plot(t, neurogram_MP.data[f_id].sum(axis=0) - neurogram_M.data[f_id].sum(axis=0), label='NH Neurogram - Masker + Probe - Masker', linestyle='dashed')
    plt.title('PSTH')
    plt.legend()
    plt.xlim(0, 0.25)
    plt.title(f'PSTH {MP_str} and {M_str}')


def PSTH_w_max(neurogram_MP, neurogram_M, neurogram_MP_max, fs, MP_str='', M_str='', ax=None):
    # fig = plt.figure()
    t = np.arange(neurogram_MP.shape[1]) / fs
    min_val = min(neurogram_MP_max.sum(axis=0) - neurogram_M.sum(axis=0))
    max_val = 2.2* np.median(neurogram_MP_max.sum(axis=0)) #max(neurogram_MP_max.sum(axis=0)[int(0.5*len(neurogram_MP_max.sum(axis=0))):])
    # f_id = np.arange(int(np.argwhere(neurogram_MP.frequencies > 1500)[0][0]), int(np.argwhere(neurogram_MP.frequencies < 2500)[-1][0]))
    if ax:
        ax.plot(t, neurogram_MP.sum(axis=0), label='MP')
        # ax.plot(t, neurogram_MP_max.data[f_id].sum(axis=0), label='MP$_{max}$', linestyle='dashdot')
        ax.plot(t, neurogram_M.sum(axis=0), label='M', linestyle='dotted')
        ax.plot(t, neurogram_MP.sum(axis=0) - neurogram_M.sum(axis=0), label='S (MP-M)', linestyle='dashed')
        ax.plot(t, neurogram_MP_max.sum(axis=0) - neurogram_M.sum(axis=0), label='S$_{max}$ (MP$_{max}$-M)', linestyle='solid' )
        ax.legend(ncol=2)
        ax.set_xlim(0.19, 0.22)
        ax.set_ylim((min_val, max_val))
        ax.set_title(f'PSTH {MP_str.replace(".wav" , "")}') 
        return ax
    else:
        fig = plt.figure()
        plt.plot(t, neurogram_MP.sum(axis=0), label='NH Neurogram - Masker + Probe')
        plt.plot(t, neurogram_MP_max.sum(axis=0), label='NH Neurogram - Masker + Probe max', linestyle='dashdot')
        plt.plot(t, neurogram_M.sum(axis=0), label='NH Neurogram - Masker', linestyle='dotted')
        plt.plot(t, neurogram_MP.sum(axis=0) - neurogram_M.sum(axis=0), label='S (MP-M)', linestyle='dashed')
        plt.plot(t, neurogram_MP_max.sum(axis=0) - neurogram_M.sum(axis=0), label='S$_{max}$ (MP$_{max}$-M)', linestyle='solid' )
        plt.title('PSTH')
        plt.legend()
        plt.xlim(0.19, 0.22)
        plt.ylim((min_val, max_val))
        plt.title(f'PSTH {MP_str} and {M_str}')
        return fig


def plot_sounds(original, reconstructed_NH, reconstructed_EH, fs):
    # plot sound files
    f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    t = np.arange(len(original)) / fs
    ax1.plot(t, original)
    ax1.set_title('Original sound')
    ax2.plot(t, reconstructed_NH)
    ax2.set_title('NH-vocoded sound')
    ax3.plot(t, reconstructed_EH)
    ax3.set_title('EH-vocoded sound')
    for a in ax1, ax2, ax3:
        a.grid()
    plt.tight_layout()


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