import datasets
import torch.utils.data as dataset
import os
import torchvision
import torchaudio
from decord import VideoReader,cpu
import decord
import torch
import numpy as np
import random
from tool.Video_Augmentation import video_augmentation
from tool.audio_process import AudioProcessor
import torch.nn.functional as F



decord.bridge.set_bridge("torch")

'''
num_frames: how many frames per second. decided how many frames you what
v_sample_rate: the steps of sampling frames, only for video
num_threads: depended on your cpu, can set a number to speed up
a_sample_rate: the sample rate of audio,

'''


#pipline to read and process the data
class read_video_dataset(dataset.Dataset):
    def __init__(self, root_path = 'data', label = 'train',train_mode = True,
                 num_frames = 16, v_sample_rate = 2, num_threads = 1, a_sample_rate = 16000,audio_duration = 5,
                 n_mel = 64, sample_method = 'uniform', device = 'cpu'):
        self.root_path = root_path
        self.label = label
        self.train_mode = train_mode
        self.num_frames = num_frames
        self.v_sample_rate = v_sample_rate
        self.num_threads = num_threads
        self.a_sample_rate = a_sample_rate
        self.audio_duration = audio_duration
        self.audio_processor = AudioProcessor(a_sample_rate, n_mel)
        self.sample_method = sample_method
        self.device = device


        # if transform is None:
        #     self.transform = torchvision.transforms.Compose([
        #         torchvision.transforms.Resize((224, 224)),
        #         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        # else:
        #     self.transform = transform

        self.frame_resize = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),  # 得到 [0,1]，方便做增强
        ])
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        self.video_aug = video_augmentation() if self.train_mode else None


        self.classes = sorted(os.listdir(os.path.join(self.root_path, self.label)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.dataset = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_path, self.label, class_name)
            for file_name in os.listdir(class_dir):
                suffix = os.path.splitext(file_name)[1].lstrip('.').lower()
                path = os.path.join(class_dir, file_name)
                idx = self.class_to_idx[class_name]
                if suffix == "mp4":
                    self.dataset.append((path, idx,"video"))
                elif suffix == "wav":
                    self.dataset.append((path, idx,"audio"))



    def sound_process(self, audio):
        audio_data = audio[:]
        if isinstance(audio_data, list):
            if not audio_data:
                return None
            audio_data = np.concatenate(audio_data, axis=-1)

        # transfer to torch tensor
        if isinstance(audio_data, np.ndarray):
            waveform = torch.from_numpy(audio_data).float()
        elif isinstance(audio_data, torch.Tensor):
            waveform = audio_data.float()
        else:
            return None


        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        # if video have no audio or muted
        if waveform.numel() == 0 or waveform.abs().sum() < 1e-6:
            return None

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        sr = getattr(audio, 'sample_rate', self.a_sample_rate)
        if sr != self.a_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.a_sample_rate)(waveform)

        mel_full = self.audio_processor(waveform, self.a_sample_rate)
        _, n_mel, t_full = mel_full.shape

        if t_full >= self.num_frames:
            idx = torch.linspace(0, t_full - 1, steps=self.num_frames).long()
            mel = mel_full[:, :, idx]      # [1, n_mel, num_frames]
        else:
            pad = self.num_frames - t_full
            mel = F.pad(mel_full, (0, pad))  # [1, n_mel, num_frames]

        mel = mel.permute(2, 0, 1)
        mel3 = mel.repeat(1, 3, 1)

        return mel3.permute(1, 0, 2)




    def read_frames(self, path):

        if self.device == 'cpu':
            ctx = cpu(0)
        else:
            from decord import cuda
            ctx = cuda(int(self.device.replace('cuda:', '')))

        vr = VideoReader(path, ctx=ctx, num_threads=self.num_threads)
        tot_frame = len(vr)
        if tot_frame == 0:
            return None, "empty_video"

        # step = max(tot_frame // self.num_frames, self.v_sample_rate)
        # idx = list(range(0, tot_frame, step))[:self.num_frames]
        idx = np.linspace(0, tot_frame - 1, num=self.num_frames, dtype=np.int64)
        # frames = vr.get_batch(idx)
        try:
            frames = vr.get_batch(idx)
        except Exception as e:
            return None, f"get_batch_failed:{e}"

        if frames.shape[0] < self.num_frames:
            pad = self.num_frames - frames.shape[0]
            frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

        frames = frames.permute(0, 3, 1, 2)  # [T,3,H,W]

        return frames, None



    def read_audio(self, path):
        waveform, sr = torchaudio.load(path)  # [C, L]
        if sr != self.a_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.a_sample_rate)(waveform)
        return waveform

    def load_video(self, path):

        try:
            # frames = self.read_frames(path)
            # frames = frames.float() / 255.0
            frames, error = self.read_frames(path)
            if frames is None:
                raise RuntimeError(f"Video decode failed for {path}: {error}")
            frames = frames.float() / 255.0

        except Exception as e_video:
            print(f"video decode failed:({path}): {e_video}")
            frames = torch.zeros(self.num_frames, 3, 224, 224)


        if frames.is_cuda:
            frames = torch.nn.functional.interpolate(
                frames, size=(224, 224), mode='bilinear', align_corners=False
            )
        else:
            frames = torch.stack([self.frame_resize(f) for f in frames], dim=0)

        frames = frames.permute(1, 0, 2, 3)  # [C,T,H,W]

        if self.video_aug:
            frames = self.video_aug(frames)

        mean = self.norm_mean.to(frames.device, frames.dtype)
        std = self.norm_std.to(frames.device, frames.dtype)
        frames = (frames - mean) / std




        try:
            waveform = self.read_audio(path)
            mel = self.sound_process(waveform)
        except Exception as e_audio:
            # print(f"audio decode failed({path}): {e_audio}")
            mel = None

        if mel is None:
            mel = torch.zeros(3, self.num_frames, self.audio_processor.n_mel)


        return frames, mel


    def __getitem__(self, idx):
        path, label, mod = self.dataset[idx]
        if mod == 'video':
            video, mel = self.load_video(path)
        else:
            video = torch.zeros(3, self.num_frames, 224, 224)  # [0,1] 范围
            video = (video - self.norm_mean) / self.norm_std  # 归一化
            wav = self.read_audio(path)
            mel = self.sound_process(wav)

        if mel is None or video is None:
            return None

        return video, mel, label



    def __len__(self):
        return len(self.dataset)

    @property
    def labels(self):
        return [label for _, label, _ in self.dataset]


# v_without = "../Dataset/video_with"
# v_without_class = "train"
# dataset = read_video_dataset(v_without, v_without_class, True,16, 4, 4)
# video, mel, lable = dataset[0]
#
# print(type(video))
# print(len(dataset), dataset.classes)
# print(type(mel))
# print(mel.shape)
# print(video.shape)


