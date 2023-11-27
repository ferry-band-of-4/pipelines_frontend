import os
import mne

import meegkit.asr as asr
import matplotlib.pyplot as plt
from ghostipy.spectral.wavelets import MorseWavelet

import ghostipy
import numpy as np
import subprocess
import gc
from memory_profiler import profile

if not os.path.exists('image_dir_no_axis'):
    os.makedirs('image_dir_no_axis')

if not os.path.exists('image_dir_no_axis/healthy'):
    os.makedirs('image_dir_no_axis/healthy')

if not os.path.exists('image_dir_no_axis/schizophrenic'):
    os.makedirs('image_dir_no_axis/schizophrenic')



@profile
def The_Creation_of_the_wavelet(dir_path):
    sfreq = 250
    asr_instance = asr.ASR(method="euclid")
    gmw = MorseWavelet(gamma=2, beta=1)
    raw_files_healthy = []
    raw_files_schizo = []
    file_list = os.listdir(dir_path)
    
    for files in file_list:
        if files.endswith('.edf') and files.startswith('h'):
            file_path = os.path.join(dir_path,files)
            raw = mne.io.read_raw_edf(file_path,preload=True)
            raw_files_healthy.append(raw)
            data_healthy = raw._data
            train_idx = np.arange(0, 30 * sfreq, dtype=int)
            asr_data_healthy, _ = asr_instance.fit(data_healthy[:, train_idx])

            channels = raw.ch_names
            dict_eeg = {channels[i] : asr_data_healthy[i] for i in range(19)}
            
            i = 0
            for channel_name, channel_data in dict_eeg.items():
                Wxh, *_  = ghostipy.spectral.cwt(channel_data, wavelet=gmw, voices_per_octave=10)
                # Plot the time frequency map
                plt.imshow(np.abs(Wxh), aspect='auto', cmap='turbo')
                file_name=files.replace('.edf',' ')
                
                image_filename = channel_name + '_' + file_name + str(i) + '.png'
                image_path = os.path.join(f"./image_dir_no_axis/healthy/", image_filename)
                plt.axis('off')
                plt.savefig(image_path, bbox_inches='tight', pad_inches=0, format='png')
                i = i + 1
                subprocess.run(['sync'])
                gc.collect()

        elif files.endswith('.edf') and files.startswith('s'):
            file_path = os.path.join(dir_path,files)
            raw = mne.io.read_raw_edf(file_path,preload=True)
            raw_files_schizo.append(raw)
            data_schizo = raw._data
            train_idx = np.arange(0, 30 * sfreq, dtype=int)
            asr_data_schizo, _ = asr_instance.fit(data_schizo[:, train_idx])
            channels = raw.ch_names
            dict_eeg = {channels[i] : asr_data_schizo[i] for i in range(19)}

            i = 0
            for channel_name, channel_data in dict_eeg.items():
                Wxs, *_  = ghostipy.spectral.cwt(channel_data, wavelet=gmw, voices_per_octave=10)
                # Plot the time frequency map
                plt.imshow(np.abs(Wxs), aspect='auto', cmap='turbo')
                file_name=files.replace('.edf',' ')
                plt.axis('off')
                image_filename = channel_name + '_' + file_name + str(i) + '.png'
                image_path = os.path.join(f"./image_dir_no_axis/schizophrenic/", image_filename)
                plt.savefig(image_path, bbox_inches='tight', pad_inches=0, format='png')
                i = i + 1
                subprocess.run(['sync'])
                gc.collect()
        gc.collect()
        plt.close('all')

        


if __name__ == "__main__":
    dir_path = "EEG_data/"
    The_Creation_of_the_wavelet(dir_path)
            