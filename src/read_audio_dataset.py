import random
import torch.utils.data as dataset
import os
import torchvision
import torchaudio
import torch
from tool.Audio_Augmentation import audio_augmentation
from tool.audio_process import AudioProcessor



class read_audio_dataset(dataset.Dataset):
    def __init__(self, root_path = 'data_path', label = 'train', train_mode= True,audio_duration = 5, sample_rate = 16000,n_mel = 64, transform = None):
        self.root_path = root_path
        self.label = label
        self.audio_duration = audio_duration
        self.set_sample_rate = sample_rate
        self.train_mode = train_mode
        self.n_mel = n_mel
        self.transform = transform
        self.audio_processor = AudioProcessor(sample_rate, n_mel)
        self.audio_aug = audio_augmentation()



        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.classes = sorted(os.listdir(os.path.join(self.root_path, self.label)))
        self.class_idx = lambda x: self.classes.index(x)
        self.dataset = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_path, self.label, class_name)
            for file_name in os.listdir(class_dir):
                self.dataset.append((os.path.join(class_dir, file_name), self.class_idx(class_name)))



    def load_and_preprocess_audio(self, audio_path):

        waveform, sample_rate = torchaudio.load(audio_path)


        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)


        if sample_rate != self.set_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.set_sample_rate)
            waveform = resampler(waveform)


        # Trim or pad to a fixed length
        target_length = int(self.audio_duration * self.set_sample_rate)

        if waveform.shape[1] > target_length:

            if self.train_mode:
                start_idx = random.randint(0, waveform.shape[1] - target_length)
                waveform = waveform[:, start_idx:start_idx + target_length]
            else:
                start_idx = (waveform.shape[1] - target_length) // 2
                waveform = waveform[:, start_idx:start_idx + target_length]
        elif waveform.shape[1] < target_length:
            # padding with 0
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, sample_rate



    # def generate_mel_spectrogram(self, waveform, sample_rate):
    #
    #     mel_transform = torchaudio.transforms.MelSpectrogram(
    #         sample_rate = sample_rate,
    #         n_mels = self.n_mel,
    #         power = 2.0
    #     )
    #
    #     mel_spectrogram = mel_transform(waveform)
    #
    #     # transfer to log_mel
    #     db_transform = torchaudio.transforms.AmplitudeToDB()
    #     mel_spectrogram = db_transform(mel_spectrogram)
    #
    #     return mel_spectrogram
    #
    #
    #
    # def normalize_spectrogram(self, mel_spectrogram):
    #
    #     mel_min = mel_spectrogram.min()
    #     mel_max = mel_spectrogram.max()
    #     if mel_max > mel_min:
    #         mel_spectrogram = (mel_spectrogram - mel_min) / (mel_max - mel_min)
    #
    #     mel_spectrogram = torch.clamp(mel_spectrogram, min=0, max=1)
    #
    #     return mel_spectrogram

    def __getitem__(self, idx):
        a_path, label = self.dataset[idx]

        try:
            waveform, sample_rate = self.load_and_preprocess_audio(a_path)

            # generate mel spectrogram
            mel_spectrogram = self.audio_processor.generate_mel_spectrogram(waveform, sample_rate)

            # apply  augmentations
            if self.train_mode:
                waveform, mel_spectrogram = self.audio_aug(waveform, mel_spectrogram,
                                                           volume_range = (0.7,1.3),
                                                           noise_level = (0.001,0.005))

            # normalise
            mel_spectrogram = self.audio_processor.generate_mel_spectrogram(mel_spectrogram)

            # need to transform to 3 channels
            mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)

            if self.transform:
                mel_spectrogram = self.transform(mel_spectrogram)

            return mel_spectrogram, label, a_path

        except Exception as e:

            print(f"error: can't load the file {a_path}: {str(e)}")
            dummy = torch.zeros(3, self.n_mel, int(self.audio_duration * self.set_sample_rate // 512) + 1)

            if self.transform:
                dummy = self.transform(dummy)

            return dummy, label, a_path

    def __len__(self):
        return len(self.dataset)