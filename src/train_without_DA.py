import sys
import os

from model.CRNN_without import CRNN_without

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau  # << 学习率衰减

from read_video_dataset import read_video_dataset
from model.inital_model import inial_model
from tool.safe_collate import safe_collate
from tool.unlabel import unlabel
from tool.train_tool import train_tool
from tool.hot_config import hot_config
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
os.environ["FFMPEG_LOGLEVEL"] = "error"


# ---------- Config ----------
# source_root = 'D:/UCL/dissertation/PythonProject/Dataset/video_with'
# target_root = 'D:/UCL/dissertation/PythonProject/Dataset/new'
source_root   = "prepared_dataset/video_data"
target_root   = "prepared_dataset/Celeb-DF-v2"

epochs        = 25          # train times
batch_size   = 25           # batch
learning_rate = 1e-4
model_depth   = 50          # CRNN backbone it has "18", "34", "50", "10", "152"
num_classes   = 2           #
num_workers = 20  #train 28, train 0
prefetch_factor = 2   #train and train for 4
persistent_workers = True #only num worker >0 can be true
fusion_method = 'concat'    # fusion method：'concat', 'add' and 'attention'


# DA weight
lambda_da = 1.0  # DANN
lambda_mmd = 0.5  # MMD
grl_lambda = 1.0  # GRL 最
mmd_warmup_epochs = 1  # The first few epochs without MMD enabled

#  source domain with label
train_ds = read_video_dataset(root_path=source_root, label="train", train_mode=True)
val_ds = read_video_dataset(root_path=source_root, label="val", train_mode=False)

# target domain without label
tgt_ds_raw = read_video_dataset(root_path=target_root, label="test", train_mode=True)
tgt_ds = unlabel(tgt_ds_raw)


def build_loaders(bs: int):
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
        persistent_workers=persistent_workers,
    )

    target_loader = DataLoader(
        tgt_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, target_loader


class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad = 0

    @staticmethod
    def is_nan(x):
        return isinstance(x, float) and x != x

    def step(self, value):
        if value is None or self.is_nan(value):
            self.num_bad += 1
            return self.num_bad >= self.patience
        if self.best is None:
            self.best = value
            return False
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience


def make_loader(root, split, train_mode, batch_size, shuffle, unlabeled=False):
    ds_raw = read_video_dataset(root_path=root, label=split, train_mode=train_mode)
    ds = unlabel(ds_raw) if unlabeled else ds_raw
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        pin_memory=True,
                        drop_last=False,  # 避免数据量小被丢弃，导致 epoch 内 0 step
                        collate_fn=safe_collate,
                        persistent_workers=persistent_workers)
    return ds, loader


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, target_loader = build_loaders(batch_size)

    labels = np.array(train_ds.labels) if hasattr(train_ds, "labels") else np.array([])
    num_classes_local = int(labels.max()) + 1 if labels.size > 0 else num_classes
    cls_count = np.bincount(labels, minlength=num_classes_local) if labels.size > 0 else np.array([1, 1])
    safe_count = np.maximum(cls_count, 1)  # 防 0
    cls_weight = cls_count.sum() / (num_classes_local * safe_count)
    cls_weight = torch.tensor(cls_weight, dtype=torch.float32).to(device)

    print(f"[Debug] source train size: {len(train_ds)}, val size: {len(val_ds)}, target size: {len(tgt_ds)}")
    print(f"[Debug] initial batch_size: {batch_size}")

    # model
    model = CRNN_without(
        model_depth=model_depth,
        num_classes=num_classes_local,
        fusion_method=fusion_method
    ).to(device)

    trainer = inial_model(
        model=model,
        device=device,
        lr=learning_rate,
        batch_size=batch_size,
        class_weight=cls_weight,
        lambda_da=lambda_da,
        lambda_mmd=lambda_mmd,
        grl_lambda=grl_lambda,
    )

    # Hot parameters
    hot_defaults = {
        "max_epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "monitor": "auc",
        "patience": 5,
        "min_delta": 1e-4,
        "lr_factor": 0.5,
        "sched_patience": 1,
        "cooldown": 0,
        "min_lr": 1e-7,

        # DA/MMD
        "lambda_da": float(lambda_da),
        "lambda_mmd": float(lambda_mmd),
        "grl_lambda": float(grl_lambda),

        # Resume training: Fill in a .pth file (supports best_*.pth or last.pth)
        "resume_from": "",
        # External stop: set to true to exit after the current epoch ends
        "stop_now": False
    }
    hot = hot_config(path="without_hot_config.json", defaults=hot_defaults)

    tool = train_tool(
        trainer=trainer,
        epochs=epochs,  # Initial value (can be overridden by hot's max_epochs)
        model_type="without",
        log_dir="DA_log/without/continue",
        save_dir="DA_saved_model/without/continue",
        monitor="auc",
        patience=5,
        min_delta=1e-4,
        lr_factor=0.5,
        sched_patience=2,
        cooldown=0,
        min_lr=1e-7,

        hot=hot,
        hot_rebuild_loaders=build_loaders,
        save_last=True,
        # resume_from
        # resume_from="DA_saved_model/multi/last.pth",
    )

    tool.fit(train_loader, val_loader, target_loader=target_loader)


if __name__ == "__main__":
    main()


