import torch
import random
import torchaudio
import numpy as np



class audio_augmentation():
    def __init__(self,volume_range = (0.7,1.3),noise_level = (0.001,0.005)):
        self.volume_range = volume_range
        self.noise_level = noise_level


    def apply_time_domain_augmentations(self, waveform, volume_range = None, noise_level = None):

        if volume_range is None:
            volume_range = self.volume_range
        if noise_level is None:
            noise_level = self.noise_level
        # randomly adjust the volume
        if random.random() < 0.5:
            volume = random.uniform(volume_range[0], volume_range[1])
            waveform = waveform * volume

        # add background noise
        if random.random() < 0.3:
            noise_level = random.uniform(noise_level[0], noise_level[1])
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        return waveform


    def apply_spectral_augmentations(self, mel_spectrogram):

        if random.random() < 0.8:
            time_mask_param = min(20, mel_spectrogram.shape[-1] // 8)
            if time_mask_param > 0:
                time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
                mel_spectrogram = time_mask(mel_spectrogram)


        if random.random() < 0.8:
            freq_mask_param = min(20, mel_spectrogram.shape[-2] // 4)
            if freq_mask_param > 0:
                freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
                mel_spectrogram = freq_mask(mel_spectrogram)


        if random.random() < 0.5:
            noise_std = random.uniform(0.001, 0.01)
            noise = torch.randn_like(mel_spectrogram) * noise_std
            mel_spectrogram = mel_spectrogram + noise


        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            mel_spectrogram = mel_spectrogram * gain


        if random.random() < 0.5:
            shift_amount = random.randint(-mel_spectrogram.shape[-1] // 4, mel_spectrogram.shape[-1] // 4)
            mel_spectrogram = torch.roll(mel_spectrogram, shifts=shift_amount, dims=-1)

        return mel_spectrogram


    def __call__(self, waveform, mel_spectrogram,volume_range = None, noise_level = None):

        waveform = self.apply_time_domain_augmentations(waveform, volume_range=volume_range,
                                                        noise_level=noise_level)

        mel_spectrogram = self.apply_spectral_augmentations(mel_spectrogram)

        waveform = torch.clamp(waveform, -1.0, +1.0)

        return waveform, mel_spectrogram