import os, cv2, torch, numpy as np, torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, Resample

class video_reader:
    def __init__(self,
                 device=torch.device('cpu'),
                 video_frames=16,
                 sample_rate=16000,
                 n_mels=64,
                 return_indices=True,
                 use_audio=True):
        self.device = device
        self.video_frames = int(video_frames)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.return_indices = return_indices
        self.use_audio = use_audio
        self.mel = MelSpectrogram(sample_rate=self.sample_rate, n_mels=self.n_mels)

    def linspace_idx(self, n, k):
        if n <= 0: return np.array([0]*k, dtype=np.int64)
        if n <= k: return np.arange(n, dtype=np.int64)
        xs = np.linspace(0, n - 1, num=k, endpoint=True, dtype=np.float64)
        return np.rint(xs).astype(np.int64)  # 用 round，跨平台稳定

    def load_media(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.mp4', '.avi', '.mov', '.mkv'):
            v, a, vid_idx, mel_idx = self.load_video_and_audio(path)
            if self.return_indices:
                return v, a, {'video_idx': vid_idx, 'mel_idx': mel_idx}
            return v, a
        elif ext in ('.wav', '.mp3', '.flac', '.m4a'):
            a, mel_idx = self.load_audio_only(path)
            if self.return_indices:
                return None, a, {'mel_idx': mel_idx}
            return None, a
        else:
            raise ValueError(f'Unsupported: {ext}')

    # —— 顺序解码视频，再从缓存等距抽样 ——
    def load_video_and_audio(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"open fail: {path}")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError("no frames")


        vid_idx = self.linspace_idx(len(frames), self.video_frames)
        sel = [frames[i] for i in vid_idx]
        arr = np.stack(sel, axis=0)                 # [T,H,W,3]
        video = torch.from_numpy(arr).permute(3,0,1,2).float().div(255.0)  # [3,T,H,W]
        video = video.to(self.device)


        if self.use_audio:
            try:
                waveform, sr = torchaudio.load(path)  # [C,L]
                if sr != self.sample_rate:
                    waveform = Resample(sr, self.sample_rate)(waveform)
                wav_mono = waveform.mean(dim=0, keepdim=True)  # [1,L]
                mel_full = self.mel(wav_mono)                  # [1,n_mels,Tmel]


                Tmel = mel_full.shape[-1]

                mel_pos = torch.linspace(0, Tmel - 1, steps=self.video_frames, dtype=torch.float32)
                mel_idx = torch.rint(mel_pos).long()
                mel = mel_full[:, :, mel_idx]
                # [1,n_mels,T] -> [3,T,n_mels]
                mel = mel.permute(2,0,1).repeat(1,3,1).permute(1,0,2).to(self.device)
            except Exception:
                mel = torch.zeros((3, self.video_frames, self.n_mels), device=self.device)
                mel_idx = torch.arange(self.video_frames)
        else:
            mel, mel_idx = None, None

        return video, mel, vid_idx.tolist(), None if mel is None else mel_idx.tolist()

    def load_audio_only(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = Resample(sr, self.sample_rate)(waveform)
        wav_mono = waveform.mean(dim=0, keepdim=True)
        mel_full = self.mel(wav_mono)                       # [1,n_mels,Tmel]
        Tmel = mel_full.shape[-1]
        mel_pos = torch.linspace(0, Tmel - 1, steps=self.video_frames, dtype=torch.float32)
        mel_idx = torch.rint(mel_pos).long()
        mel = mel_full[:, :, mel_idx]                       # [1,n_mels,T]
        mel = mel.permute(2,0,1).repeat(1,3,1).permute(1,0,2).to(self.device)
        return mel, mel_idx.tolist()
