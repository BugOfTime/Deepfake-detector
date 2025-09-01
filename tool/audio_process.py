
import torch
import torchaudio
import torch.nn.functional as F

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mel=64, top_db=80.0):
        self.sample_rate   = sample_rate
        self.n_mel         = n_mel

        self.mel_spec      = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mel,
            power=2.0,
        )
        self.db_transform  = torchaudio.transforms.AmplitudeToDB(top_db=top_db)


        self.mean_db = -30.0
        self.std_db  = 20.0

    def waveform_to_mel(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        m = self.mel_spec(waveform)      # [C, n_mel, time_steps]
        m = self.db_transform(m)         # 同上
        return m

    def normalize(self, db_spec: torch.Tensor) -> torch.Tensor:

        return (db_spec - self.mean_db) / self.std_db

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        m = self.waveform_to_mel(waveform, sr)
        m = self.normalize(m)
        return m
