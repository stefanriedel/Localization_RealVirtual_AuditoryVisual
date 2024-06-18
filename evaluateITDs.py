import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import os
import seaborn
import scipy.signal as signal

root_dir = dirname(__file__)
brir_dir = pjoin(root_dir, 'BRIR')
file_list = os.listdir(brir_dir)

if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')
if 'KU100_Native_ExpStudio_BRIR_fs_48000_numOutCH_25_numInCH_2_IRs' in file_list:
    file_list.remove('KU100_Native_ExpStudio_BRIR_fs_48000_numOutCH_25_numInCH_2_IRs')

num_participants = 16
num_directions = 25
orig_fs = 48000
start_idx = 1200

RESAMPLE = True

factor = 4
fs = orig_fs * factor


indiv_brirs = np.zeros((num_participants, int(fs/2), num_directions, 2))

# Store all individual BRIR for ITD evaluation
for subj in range(num_participants):
    brir_path = pjoin(brir_dir, file_list[subj])
    brir = np.load(brir_path)
    # normalize and remove time of flight
    brir /= np.max(np.abs(brir))
    brir = np.roll(brir, -start_idx, axis=0)

    if RESAMPLE:
        brir = signal.resample(brir, int(fs / orig_fs * brir.shape[0]), axis=0)

    if 0:
        plt.figure()
        plt.plot(brir[:, 0, 0], label='left ear brirs', color='red')
        plt.plot(brir[:, 0, 1], label='right ear brirs', color='blue')
        plt.show(block=True)

    indiv_brirs[subj, ...] = brir


def computeBroadBandITD(hrir):
    tau_limit = 0.001
    tau_range = np.arange(int(-fs * tau_limit), int(fs * tau_limit))
    Nfft = 4096 * factor
    hrtf_left = np.fft.rfft(hrir[:, 0], n=Nfft)
    hrtf_right = np.fft.rfft(hrir[:, 1], n=Nfft)

    cross_spectrum = np.conj(hrtf_left) * hrtf_right
    cross_correlation = np.real(np.fft.irfft(cross_spectrum))
    itd = (np.argmax(cross_correlation[tau_range]) / fs) - tau_limit

    return itd

# Convert to HRIRs by truncation of room reflections
hrir_len = 128 * factor
pre_delay = 32 * factor
indiv_hrirs = np.zeros((num_participants, int(hrir_len), num_directions, 2))
Nfadein = 16 * factor
Nfadeout = 32 * factor

indiv_itds =  np.zeros((num_participants, num_directions))
for subj in range(num_participants):
    for direction in range(num_directions):
        brir = indiv_brirs[subj, :, direction, :]
        peak_left = np.argmax(brir[:, 0], axis=0)
        peak_right = np.argmax(brir[:, 1], axis=0)

        if peak_left < peak_right:
            peak_idx = peak_left
        else:
            peak_idx = peak_right
        hrir = brir[(peak_idx-pre_delay):(peak_idx+hrir_len-pre_delay), :]
        sin_fadein = np.sin((np.arange(Nfadein)/Nfadein * np.pi/2))**2
        cos_fadeout = np.cos((np.arange(Nfadeout)/Nfadeout * np.pi/2))**2
        hrir[:Nfadein, :] *= sin_fadein[:, None]
        hrir[-Nfadeout:, :] *= cos_fadeout[:, None]

        if 0:
            plt.figure()
            plt.plot(hrir)
            plt.show(block=True)

        indiv_hrirs[subj, :, direction, :] = hrir
        itd = computeBroadBandITD(hrir)
        indiv_itds[subj, direction] = itd


# KU100
ku100_brir = np.load(pjoin(brir_dir, 'KU100_Native_ExpStudio_BRIR_fs_48000_numOutCH_25_numInCH_2_IRs.npy'))
# normalize and remove time of flight
ku100_brir /= np.max(np.abs(ku100_brir))
ku100_brir = np.roll(ku100_brir, -start_idx, axis=0)
if RESAMPLE:
        ku100_brir = signal.resample(ku100_brir, int(fs / orig_fs * ku100_brir.shape[0]), axis=0)
ku100_itds =  np.zeros(num_directions)
for direction in range(num_directions):
    brir = ku100_brir[:, direction, :]
    peak_left = np.argmax(brir[:, 0], axis=0)
    peak_right = np.argmax(brir[:, 1], axis=0)

    if peak_left < peak_right:
        peak_idx = peak_left
    else:
        peak_idx = peak_right
    hrir = brir[(peak_idx-pre_delay):(peak_idx+hrir_len-pre_delay), :]
    sin_fadein = np.sin((np.arange(Nfadein)/Nfadein * np.pi/2))**2
    cos_fadeout = np.cos((np.arange(Nfadeout)/Nfadeout * np.pi/2))**2
    hrir[:Nfadein, :] *= sin_fadein[:, None]
    hrir[-Nfadeout:, :] *= cos_fadeout[:, None]

    ku100_itds[direction] = computeBroadBandITD(hrir)
       
# Plot ITDs
channels = np.arange(0,20)


plt.figure()
plt.title('ITDs of individual subjects (violins) vs. KU100')
seaborn.violinplot(indiv_itds[:, channels] * 1e3, inner="stick", palette=['tab:orange'] * 8 + ['tab:red'] * 8 + ['tab:purple']  * 4)
plt.scatter(np.arange(0,channels.size), ku100_itds[channels] * 1e3, label='KU100 ITDs', marker='o', color='k')
plt.ylabel('ITD in ms')
plt.yticks(np.arange(-0.8,1.0, 0.2))
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)   
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
#plt.gca().yaxis.set_major_locator(AutoMinorLocator(0.2))   
plt.grid(True)
plt.xticks(ticks=np.arange(0,20), labels=np.arange(1,21))
plt.xlabel('Loudspeaker channel')
plt.legend()
plt.show(block=True)


percentages = np.sum(np.abs(indiv_itds[:, channels] - ku100_itds[channels]) > 0.000025, axis=0) / indiv_itds.shape[0] * 100.0
plt.figure()
plt.title('ITD error analysis of individual subjects vs. KU100')
plt.grid(True)
seaborn.barplot(percentages, palette=['tab:orange'] * 8 + ['tab:red'] * 8 + ['tab:purple']  * 4)
#plt.scatter(np.arange(0,channels.size), ku100_itds[channels] * 1e3, label='KU100 ITDs', marker='o', color='k')
plt.ylabel('Percentage of Delta-ITD > JND')
plt.xticks(ticks=np.arange(0,20), labels=np.arange(1,21))
plt.xlabel('Loudspeaker channel')
plt.legend()
plt.show(block=True)

print('done')

