import numpy as np
import scipy
from neurovoc import bruce, reconstruct, specres
import os
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob
# from IPython.display import Audio
from utilities import *
from plotting import *

# To check
# [X] envelope of sound
# [X] moving correlation --> need larger temperature
# [X] reconstruct with dB in reconstruction
# [X] neurograms with None as stim dB
# [X] is the windowing happening as a noncausal filter??

# [ ] implement noise or use window size as noise?
# [ ] implement trials
# [ ] include more fibers
# [ ] random seed

# data_path = r"C:\python\MaskerProbePsychometricCurve\sounds\Experiment2_gap30ms"
# data_path = r"C:\python\MaskerProbePsychometricCurve\sounds\Experiment2"
data_path = r"C:\python\MaskerProbePsychometricCurve\sounds\Experiment2BruceReference91"

def neurovoc_sound(sound_name, 
                   hearing_type='NH', 
                   play_audio=False, 
                   plot_audio=False, 
                   use_envelope=False, 
                   show_neurogram=False,
                   window_size=150,
                   PSTH_as_RT=False):
    # load sound file
    original, fs = librosa.load(sound_name, sr=44100)

    # create neurograms
    if hearing_type == 'NH' and not PSTH_as_RT:
        neurogram, stim = bruce(sound_name, ref_db=masker_dB_Bruce, audio_fs=fs, window_size=window_size) #32 
    if hearing_type == 'NH' and PSTH_as_RT:
        if not random_seed:
            neurogram, stim = bruce(sound_name, 
                                    ref_db=None, 
                                    audio_fs=fs, 
                                    window_size=window_size, 
                                    normalize=False, 
                                    remove_outliers=False)
        else:
            neurogram, stim = bruce(sound_name, 
                                    ref_db=None, 
                                    audio_fs=fs, 
                                    window_size=window_size, 
                                    normalize=False, 
                                    remove_outliers=False, 
                                    seed=np.random.randint(1, 100 + 1), 
                                    n_fibers_per_bin=50)
            print('using random seed')
        # print('ref dB is None')
    elif hearing_type == 'EH':
        neurogram = specres(sound_name)
    if show_neurogram:
        plot_neurogram(neurogram, title=f'Neurogram of {os.path.basename(sound_name)}')

    # recreate sound
    reconstructed = reconstruct(neurogram, ref_db=masker_dB_Bruce)

    # upsample sound for comparison
    reconstructed = scipy.signal.resample(reconstructed, original.size)

    if plot_audio:
        plt.figure()
        t = np.arange(len(original)) / fs
        plt.plot(t, reconstructed, label='Reconstructed sound', linestyle='dotted')
        plt.plot(t, original, label='Original sound', linestyle='dotted')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlim(0, 0.21)
        # plt.ylim((-0.02, 0.02))
        plt.title(f'{hearing_type}-vocoded sound of {os.path.basename(sound_name)}')

    # only take envelope
    if use_envelope:
        reconstructed = scipy.signal.envelope(reconstructed, residual=None)

    return original, fs, reconstructed, neurogram


# parameters
plot_auditory_memory = False
hearing_type = 'NH'  # or 'EH'
norm_bool = True
PSTH_as_RT = True  # If True, use PSTH as RT representation, else use reconstructed sound
window_size = 15  # Window size for Bruce model
probe_period_only = False
metric = 'correlation'  # 'correlation' or 'rms'
random_seed = True

# plotting
plot_NIR = False
show_neurogram = False
plot_PSTH = True
plot_correlation_lags = False

masker_list = range(30, 95, 5)

if 'Reference' in data_path:
    remove_reference = True
    masker_dB_Bruce = 90
    print('Using Bruce reference at 91 dB SPL, removing reference part of sound for analysis')
else:
    remove_reference = False

if metric == 'rms':
    temperature_list = [0.02, 0.04, 0.06, 0.08]
    print('Psychometric curve based on RMS')
elif metric == 'correlation':
    temperature_list = np.arange(2, 12, 2)*1000 #[500, 1000, 1500, 2000, 2500, 3000] #np.array([0.0001, 0.001, 0.01, 0.1, 1])*2#np.array([4, 8, 16, 24, 32])
    print('Psychometric curve based on Correlation')

if len(masker_list)>8:
    minimum_row = 3
if len(masker_list)>4:
    minimum_row = 2
else:
    minimum_row = 1
rows, columns  = get_row_column_nums(len(masker_list), minimum_row=minimum_row)
fig_curve, axes_curve = plt.subplots(rows, columns, figsize=(15, 15))
axes_curve = axes_curve.flatten()
        
for m, masker_dB in enumerate(masker_list):
    masker_dB = str(masker_dB)
    print('masker:', masker_dB)

    if not remove_reference:
        masker_dB_Bruce = int(masker_dB)

    # get RT list
    RT_list = glob.glob(os.path.join(data_path, 'masker_' + masker_dB+ '*dB.wav'))
    sound_name_R = glob.glob(os.path.join(data_path, 'masker_' + masker_dB+ '.wav'))[0]
    sound_name_RT_max = RT_list[-1]
    RT_max_dB = str(sound_name_RT_max[-8:-6])

    # load RT_max and R
    original_RT_max, fs_sound, X_RT_max, neurogram_RT_max = neurovoc_sound(sound_name_RT_max, 
                                                                    hearing_type, 
                                                                    show_neurogram=show_neurogram,
                                                                    window_size=window_size, 
                                                                    PSTH_as_RT=PSTH_as_RT)
    _, _, X_R1, neurogram_R1 = neurovoc_sound(sound_name_R, 
                                            hearing_type, 
                                            show_neurogram=show_neurogram,
                                            window_size=window_size,
                                            PSTH_as_RT=PSTH_as_RT)

    if PSTH_as_RT:
        f_id = np.array([26, 27, 28, 29, 30, 31, 32, 33])
        X_RT_max = neurogram_RT_max.data[f_id].sum(axis=0)
        X_R1 = neurogram_R1.data[f_id].sum(axis=0)
        fs = 1/neurogram_RT_max.dt
        print('Using PSTH as RT representation')
    else:
        fs = fs_sound

    if remove_reference:
        original_length = 14051
        original_RT_max = original_RT_max[:len(original_RT_max)//2]
    else:
        original_length = len(X_R1)
    X_RT_max = X_RT_max[:original_length]
    X_R1 = X_R1[:original_length]


    t_sound_full = np.arange(len(original_RT_max)) / fs_sound
    t_sound_probe = t_sound_full[int(0.2*fs_sound): int(0.206*fs_sound)]

    # probe period = 0.2 to 0.206s
    probe_period = [0.203, 0.206]
    left, bottom, width, height = (probe_period[0], -1, probe_period[1] - probe_period[0], 200)
    rect = plt.Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.1)
    if probe_period_only:
        t_id = np.arange(int(probe_period[0]-0.003*fs), int(probe_period[0]+0.003*fs)) 
        Title_add = ' (probe period only)'
        print('Using only probe period for analysis')
    else:
        t_id = np.arange(original_length) 
        Title_add = ' (full sound)'
        print('Using full sound for analysis')


    # create time vector and select period if neccessary
    t_full = np.arange(len(X_RT_max)) / fs
    X_RT_max = X_RT_max[t_id]
    X_R1 = X_R1[t_id]
    t_probe = t_full[t_id]

    # S_max
    S_max = X_RT_max - X_R1

    # print('length RT max', len(X_RT_max), 'length R1', len(X_R1))

    # if remove_reference:
    #     S_max = S_max[:len(S_max)//2]
    #     X_R1 = X_R1[:len(X_R1)//2]
    #     t_full = t_full[:len(t_full)//2]

    if plot_PSTH:
        # fig = PSTH(neurogram_RT_max, neurogram_R1, MP_str=os.path.basename(sound_name_RT_max), M_str=os.path.basename(sound_name_R))
        fig = PSTH_w_max(neurogram_RT_max.data[f_id,:original_length], neurogram_R1.data[f_id,:original_length], neurogram_RT_max.data[f_id,:original_length], fs, MP_str=os.path.basename(sound_name_RT_max), M_str=os.path.basename(sound_name_R))
        fig2, axes = plt.subplots(4,4, figsize=(15, 15), sharex=True, sharey=True)
        axes = axes.flatten()

    probabilities = np.zeros((len(temperature_list), len(RT_list), 2))  # Store probabilities for R and RT
    correlations_matrix = np.zeros((len(RT_list), 2))  # Store correlations for R and RT
    dB_list = []
    rms_matrix = np.zeros((len(RT_list), 2))  # Store rms for R and RT

    for s, sound_name_RT in enumerate(RT_list):
        # maybe add trials per RT

        dB = sound_name_RT[-8:-6]
        dB_list.append(float(dB))
        print(dB)
        original_RT, _, X_RT, neurogram_RT = neurovoc_sound(sound_name_RT, 
                                                            hearing_type, 
                                                            show_neurogram=show_neurogram,
                                                            window_size=window_size, 
                                                            PSTH_as_RT=PSTH_as_RT)
        original_R, _, X_R, neurogram_R = neurovoc_sound(sound_name_R, 
                                                        hearing_type, 
                                                        window_size=window_size, 
                                                        PSTH_as_RT=PSTH_as_RT)   
        # use neural representation as RT
        if PSTH_as_RT:
            X_RT = neurogram_RT.data[f_id].sum(axis=0)
            X_R = neurogram_R.data[f_id].sum(axis=0)

        if original_length != len(X_R):
            X_R = X_R[:original_length]
        if original_length != len(X_RT):
            X_RT = X_RT[:original_length]

        if remove_reference:
            # X_RT = X_RT[:len(X_RT)//2]
            # X_R = X_R[:len(X_R)//2]
    
            original_RT = original_RT[:len(original_RT)//2]
            original_R = original_R[:len(original_R)//2]

        # probe period = 0.2 to 0.206s
        X_RT = X_RT[t_id]
        X_R = X_R[t_id]

        # print('length RT', len(X_RT), 'length R', len(X_R))


        if plot_PSTH:
            # fig = PSTH(neurogram_RT, neurogram_R, MP_str=os.path.basename(sound_name_RT), M_str=os.path.basename(sound_name_R))
            
            # fig = PSTH_w_max(neurogram_RT, neurogram_R, neurogram_RT_max, MP_str=os.path.basename(sound_name_RT), M_str=os.path.basename(sound_name_R))
            X = PSTH_w_max(neurogram_RT.data[f_id,:original_length], neurogram_R.data[f_id,:original_length], neurogram_RT_max.data[f_id,:original_length], fs, MP_str=os.path.basename(sound_name_RT), M_str=os.path.basename(sound_name_R), ax=axes[s])
            rect = plt.Rectangle((left, bottom), width, height,
                    facecolor="black", alpha=0.1)
            axes[s].add_patch(rect)    

        # plot R and RT
        if plot_NIR:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, figsize=(12, 8))
            # original sounds
            ax1.plot(t_sound_full, original_RT, label='RT', linestyle='dotted')
            ax1.plot(t_sound_full, original_R, label='R', linestyle='dotted')
            ax1.plot(t_sound_full, original_RT - original_R, label='RT - R', linestyle='dotted')
            ax1.plot(t_sound_full, original_RT_max, label='$RT_{max}$', linestyle='dotted')
            ax1.set_xlim((0.2, 0.2065)) 
            ax1.set_ylim((-1*max(original_RT_max), max(original_RT_max)))
            ax1.set_title(f'original sound of {os.path.basename(sound_name_RT).replace(".wav", '')}')

            # reconstructed sounds
            ax2.plot(t_probe, X_RT, label='RT', linestyle='dotted')
            ax2.plot(t_probe, X_RT_max, label = '$RT_{max}$' , linestyle='dotted')
            ax2.set_ylim(-2*max(X_RT_max[len(X_RT_max)//2:]), 2*max(X_RT_max[len(X_RT_max)//2:]))
            ax2.set_title('Reconstructed RT')

            # reconstructed sounds
            ax3.plot(t_probe, X_R, label='R', linestyle='dotted')
            ax3.plot(t_probe, X_R1, label='R1', linestyle='dotted')
            ax3.set_ylim(-2*max(X_RT_max[len(X_RT_max)//2:]), 2*max(X_RT_max[len(X_RT_max)//2:]))
            ax3.set_title('Reconstructed R')

            # analysis representation
            ax4.plot(t_probe, S_max, label='$S_{max}$', linestyle='dotted')
            ax4.plot(t_probe, X_RT - X_R, label='RT - R', linestyle='dotted')
            ax4.plot(t_probe, X_R1 - X_R, label='R1 - R', linestyle='dotted')
            ax4.set_title('Analysis Representation: R(T) - R')
            ax4.set_ylim((-2*max(S_max[len(S_max)//2:]), 2*max(S_max[len(S_max)//2:])))

            for a in ax1, ax2, ax3, ax4:
                a.grid()
                a.legend()
                a.set_xlim((0.19, 0.21)) 
                rect = plt.Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.1)
                a.add_patch(rect)
                
            plt.tight_layout()

        # cross-correlation
        corr_RT = np.max(scipy.signal.correlate(S_max, X_RT-X_R)) #np.corrcoef(S_max, X_RT-X_R)[1,0] # 
        corr_R = np.max(scipy.signal.correlate(S_max, X_R1-X_R)) #np.corrcoef(S_max, X_R1-X_R)[1,0]

        if plot_correlation_lags:
            # plotting correlation with lags:
            plt.subplots(2,2, sharex=False, sharey=False)
            lags = scipy.signal.correlation_lags(len(S_max), len(X_RT-X_R))
            corr = scipy.signal.correlate(S_max, X_RT-X_R)
            max_lag = lags[np.argmax(corr)]
            plt.subplot(2,2,1)
            plt.plot(t_probe, S_max, label='$S_{max}$')
            plt.plot(t_probe, X_RT - X_R, label='RT - R')
            # plt.plot(t, X_RT, label='RT')
            # plt.plot(t, X_R, label='R')
            # plt.plot(t, X_R1, label='R1')
            ax = plt.gca()
            ax.add_patch(Rectangle((2.03,-1), 0.06, 1,
                    edgecolor = 'grey',
                    facecolor = 'grey',
                    fill=True,
                    alpha=0.5) )
            plt.xlim((0, 0.02))
            # plt.ylim((-2*max(S_max), 2*max(S_max)))
            plt.legend()
            # plt.ylabel('S$_max$ and RT - R without lag correction')

            plt.subplot(2,2,2)
            # plt.plot(t, X_RT, label='RT')
            # plt.plot(t, X_R, label='R')
            # plt.plot(t, X_R1, label='R1')
            plt.legend()
            plt.plot(t_probe, S_max, label='$S_{max}$')
            plt.plot(t_probe, X_RT - X_R, label='RT - R')
            plt.xlim((0.19, 0.21))
            ax = plt.gca()
            ax.add_patch(Rectangle((2.03,-1), 0.06, 1,
                    edgecolor = 'blue',
                    facecolor = 'red',
                    fill=True,
                    alpha=0.5) )
            # plt.ylim((-2*max(S_max[len(S_max)//2:]), 2*max(S_max[len(S_max)//2:])))
            # plt.title('without lag correction, corr = ' + str(np.round(corr_RT, 3)))

            plt.subplot(2,2,3)
            plt.plot(t_probe, S_max, label='$S_{max}$')
            plt.plot(t_probe - max_lag/fs, X_RT - X_R, label='RT - R with lag correction')
            # plt.plot(t, X_RT_max, label='RT_max')
            # plt.plot(t, X_R1, label='R1')
            # plt.plot(t, X_R, label='R')
            plt.xlim((0, 0.02))
            plt.legend()
            # plt.ylim((-2*max(S_max), 2*max(S_max)))
            # plt.ylabel('S$_max$ and RT - R with lag correction')

            plt.subplot(2,2,4)
            # plt.plot(t, X_RT_max, label='RT_max')
            # plt.plot(t, X_R1, label='R1')
            # plt.plot(t, X_R, label='R')
            plt.legend()
            plt.plot(t_probe, S_max, label='$S_{max}$')
            plt.plot(t_probe - max_lag/fs, X_RT - X_R, label='RT - R with lag correction')
            plt.xlim((0.19, 0.21))
            # plt.ylim((-2*max(S_max[len(S_max)//2:]), 2*max(S_max[len(S_max)//2:])))
            # plt.suptitle(f'Correlation with lag correction of {os.path.basename(sound_name_RT).replace(".wav", "")}, max lag = {max_lag} samples')
            # plt.title('with lag correction, corr = ' + str(np.round(corr_RT, 3)))

        # rms
        if metric == 'rms':
            score_vector = np.array([rms(X_RT-X_R), rms(X_R1-X_R)])
            y_label_add = 'based on RMS'
        # correlations
        elif metric == 'correlation': 
            score_vector = np.array([corr_RT, corr_R])
            y_label_add = 'based on Correlation'
        rms_matrix[s,:] = np.array([rms(X_RT-X_R), rms(X_R1-X_R)])
        correlations_matrix[s,:] = score_vector

        # softmax 
        if norm_bool:
            # Subtract the maximum score for numerical stability. This prevents overflow in the exponentiation step
            max_score = np.max(score_vector) 
            score_vector -= max_score 

        for t, temperature in enumerate(temperature_list):#[8, 16, 24, 32]: #, 16, 24, 32
            print(f'Temperature: {temperature}')
            # Apply softmax formula
            expScores = np.exp(score_vector / temperature) # Using negative to invert the effect, lower MI -> higher score
            # if any value is inf, replace with really large number
            # if any value is inf, replace with 1 in probability matrix
            if (expScores == np.inf).any():
                index = np.where(expScores == np.inf)[0][0]
                inf_present = True
                print('Infinite present in exponential scores, overwriting to a probability of 1')
            else:
                inf_present = False
            probabilities[t, s,:] = expScores / np.sum(expScores)
            if inf_present:
                probabilities[t, s, index] = 1.0

    if plot_PSTH:        
        fig2.savefig(os.path.join(r'C:\python\MaskerProbePsychometricCurve\Figures\PSTH', f'PSTH_masker_{masker_dB}dB_window{window_size}_randomseed{random_seed}.png'))

    # plt.figure('correlations: ' + masker_dB + ' dB') 
    # plt.plot(dB_list, correlations_matrix[:, 0], label=f'RT Masker {masker_dB}dB and RT_max {RT_max_dB}dB')
    # plt.plot(dB_list, correlations_matrix[:, 1], label=f'R Masker {masker_dB}dB and RT_max {RT_max_dB}dB')
    # plt.legend()
    # plt.xlabel('dB')
    # plt.ylabel('Correlation RT with S_max ' + Title_add)

    # plt.figure('rms: ' + masker_dB + ' dB')
    # plt.plot(dB_list, rms_matrix[:, 0], label=f'RT with masker={masker_dB}dB and RT_max={RT_max_dB}dB')
    # plt.plot(dB_list, rms_matrix[:, 1], label=f'R with masker={masker_dB}dB and RT_max={RT_max_dB}dB')
    # plt.legend()
    # plt.ylabel('RMS ' + Title_add)
    # plt.xlabel('dB')

    ff = plt.figure('probabilities: ' + masker_dB + ' dB')
    for t, temperature in enumerate(temperature_list):
        plt.plot(dB_list, probabilities[t, :, 0], label=f'T={temperature}')
    plt.xlabel('dB')
    plt.ylabel('Probability ' + y_label_add)
    plt.title('Psychometric Curve ' + f' for masker {masker_dB}dB window: {window_size}'   + Title_add)
    if np.min(probabilities[:,:,0])>0.5:
        plt.ylim((0.49, 1.01))
    plt.legend()

    plt.figure(fig_curve)
    for t, temperature in enumerate(temperature_list):
        axes_curve[m].plot(dB_list, probabilities[t, :, 0], label=f'T={temperature}')
    axes_curve[m].set_xlabel('dB')
    axes_curve[m].set_ylabel('Probability ' + y_label_add)
    axes_curve[m].set_title(f' for masker {masker_dB}dB window: {window_size}'   + Title_add)
    axes_curve[m].legend()
    if np.min(probabilities[:,:,0])>0.5:
        axes_curve[m].set_ylim((0.49, 1.01))

    # plt.ylim((0.45, 1.01))

    ff.savefig(os.path.join(r'C:\python\MaskerProbePsychometricCurve\Figures', f'Psychometric_curve_masker_{masker_dB}dB_temp_{temperature_list[0]}_{temperature_list[-1]}_window{window_size}_probe_only{probe_period_only}_{metric}_randomseed{random_seed}.png'))


plt.figure(fig_curve)
plt.subplots_adjust(left=0.05, right=0.98, top=0.9)
plt.suptitle('Psychometric curves with window size ' + str(window_size) + Title_add)
plt.tight_layout()

plt.show()