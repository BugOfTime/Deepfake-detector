
import torch
from torch.utils.data import Dataset

class unlabel(Dataset):
    def __init__(self, base_ds, dummy_label: int = -1):
        self.base = base_ds
        self.dummy_label = int(dummy_label)

        if hasattr(base_ds, "labels"):
            try:
                n = len(base_ds)
                base_ds.labels = [-1] * n
            except Exception:
                pass

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        if isinstance(item, dict):
            video = item.get("video", None)
            audio = item.get("audio", None)
            return video, audio, torch.tensor(self.dummy_label, dtype=torch.long)

        if isinstance(item, (tuple, list)):
            if len(item) >= 2:
                video, audio = item[0], item[1]
            else:
                video, audio = item[0], None
            return video, audio, torch.tensor(self.dummy_label, dtype=torch.long)

        video = item
        audio = None
        return video, audio, torch.tensor(self.dummy_label, dtype=torch.long)
