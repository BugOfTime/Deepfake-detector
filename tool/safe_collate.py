import torch
def safe_collate(batch):
    # 过滤掉任何 None 数据，或者 audio 为 None 的样本
    batch = [b for b in batch if b is not None and b[1] is not None]

    if len(batch) == 0:
        return None  # 所有样本都无效时

    videos, audios, labels = zip(*batch)

    videos = torch.stack([v.contiguous() for v in videos], dim=0)
    audios = torch.stack([a.contiguous() for a in audios], dim=0)
    labels = torch.as_tensor(labels, dtype=torch.long)

    return videos, audios, labels