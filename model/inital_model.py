import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,roc_auc_score
try:
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler as GradScalerCls

    def amp_autocast(enabled: bool):
        return _autocast_new('cuda', dtype=torch.float16, enabled=enabled)

except Exception:
    from torch.cuda.amp import autocast as _autocast_old
    from torch.cuda.amp import GradScaler as GradScalerCls

    def amp_autocast(enabled: bool):
        return _autocast_old(enabled=enabled)

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from tool.batch import parse_batch
import torch.nn.functional as F
from torch import nn
from itertools import cycle
import math


#  Gradient Reversal Layer
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GradReverse(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GRL.apply(x, self.lambd)

# Domain Adversarial Loss (DANN)
class DomainAdversarialLoss(nn.Module):

    def __init__(self, feature_dim, hidden_dim=1024, grl_lambda=1.0, p_drop=0.5):
        super().__init__()
        self.grl = GradReverse(lambd=grl_lambda)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, 1)  # logits
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, feat, domain_label):
        x = self.grl(feat)
        logits = self.domain_classifier(x).squeeze(1)  # [B]
        return self.criterion(logits, domain_label.float())

#  MMD (RBF kernel, multi-bandwidth)
def rbf_kernel(x, y, gammas):

    xx = (x**2).sum(dim=1, keepdim=True)       # [n,1]
    yy = (y**2).sum(dim=1, keepdim=True).T     # [1,m]
    dist = xx + yy - 2.0 * x @ y.T             # [n,m]
    k = 0
    for g in gammas:
        k = k + torch.exp(-g * dist)
    return k

def mmd_loss(source_features, target_features, gammas=(1, 2, 4, 8, 16)):

    # avoid empy batch error
    if source_features.numel() == 0 or target_features.numel() == 0:
        return torch.tensor(0.0, device=source_features.device, dtype=source_features.dtype)

    with torch.cuda.amp.autocast(enabled=False):
        s = source_features.float()
        t = target_features.float()

        # avoid inf and Nan
        s = torch.nan_to_num(s, nan=0.0, posinf=1e4, neginf=-1e4)
        t = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)


        s = torch.nn.functional.normalize(s, dim=1)
        t = torch.nn.functional.normalize(t, dim=1)

        # RBF
        g = torch.tensor(gammas, device=s.device, dtype=s.dtype)

        # pairwise
        xx = (s**2).sum(dim=1, keepdim=True)      # [n,1]
        yy = (t**2).sum(dim=1, keepdim=True).T    # [1,m]


        st = s @ t.T                               # [n,m]
        dist = (xx + yy - 2.0 * st).clamp_min(0.0)


        k = 0.0
        for gamma in g:
            k = k + torch.exp(-gamma * dist)

        ss = s @ s.T
        tt = t @ t.T
        dist_ss = ((s**2).sum(1, keepdim=True) + (s**2).sum(1, keepdim=True).T - 2.0 * ss).clamp_min(0.0)
        dist_tt = ((t**2).sum(1, keepdim=True) + (t**2).sum(1, keepdim=True).T - 2.0 * tt).clamp_min(0.0)

        k_ss = 0.0
        k_tt = 0.0
        for gamma in g:
            k_ss = k_ss + torch.exp(-gamma * dist_ss)
            k_tt = k_tt + torch.exp(-gamma * dist_tt)

        Kss = k_ss.mean()
        Ktt = k_tt.mean()
        Kst = k.mean()
        mmd = Kss + Ktt - 2.0 * Kst

        if not torch.isfinite(mmd):
            mmd = torch.tensor(0.0, device=source_features.device, dtype=source_features.dtype)

    return mmd.to(source_features.dtype)



class inial_model:
    def __init__(self, model, device="cuda", lr=1e-4, batch_size=16,
                 save_dir='../save_model', class_weight=None,
                 lambda_da=0.1, lambda_mmd=0.05, grl_lambda=1.0):

        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.lr = lr



        self.use_amp = (str(device).startswith("cuda") and torch.cuda.is_available())

        self.save_dir = save_dir if save_dir else "./saved_model"
        os.makedirs(self.save_dir, exist_ok=True)

        self.best_loss = float('inf')
        self.best_model_path = None

        # initialise GradScaler
        self.scaler = GradScalerCls(enabled=self.use_amp)

        # DANN head
        feat_dim = getattr(self.model, "fuse_dim", None)
        if feat_dim is None:
            raise AttributeError("model must expose `fuse_dim` as feature dimension for DA/MMD.")
        self.domain_head = DomainAdversarialLoss(feature_dim=feat_dim,
                                                 hidden_dim=1024,
                                                 grl_lambda=grl_lambda).to(device)

        # loss weights
        self.lambda_da_base = lambda_da
        self.lambda_mmd_base = lambda_mmd
        self.grl_lambda_base = grl_lambda  # glr maximize
        self.warmup_mmd_epochs = 1
        self.total_steps = None
        self.global_step = 0


        # optimizer must include domain_head params
        params = list(self.model.parameters()) + list(self.domain_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)

        self.criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)

    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']



    def train_epoch(self, train_loader, epoch, log_every=100, target_loader=None, max_batches=None,
                    hot_tool=None, hot_check_every=200):
        import math, numpy as np, torch
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
        from tqdm import tqdm

        self.model.train()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        actual_samples = 0

        # total_steps / global_step
        steps_per_epoch = max(1, len(train_loader))
        epochs_guess = max(1, getattr(self, "epochs", 1))
        if self.total_steps is None or not isinstance(self.total_steps, (int, float)) or self.total_steps <= 0:
            self.total_steps = steps_per_epoch * epochs_guess

        try:
            from itertools import cycle
            tgt_iter = cycle(target_loader) if (target_loader is not None and len(target_loader) > 0) else None
        except Exception:
            tgt_iter = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", total=len(train_loader), leave=True,
                    dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            if max_batches is not None and step >= max_batches:
                break

            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    video_s, audio_s, labels_s = batch[0], batch[1], batch[2]
                else:
                    video_s, audio_s, labels_s = parse_batch(batch)
            except Exception:
                continue

            video_s = video_s.to(self.device) if video_s is not None else None
            audio_s = audio_s.to(self.device) if audio_s is not None else None
            labels_s = labels_s.to(self.device).long().flatten()

            # target domain
            if tgt_iter is not None:
                try:
                    video_t, audio_t, _ = next(tgt_iter)
                    video_t = video_t.to(self.device) if video_t is not None else None
                    audio_t = audio_t.to(self.device) if audio_t is not None else None
                except Exception:
                    video_t = audio_t = None
            else:
                video_t = audio_t = None

            # GRL/MMD scheduling
            p = min(1.0, self.global_step / max(1, self.total_steps))
            # use hot parmater
            grl_k = getattr(self, "grl_k", 10.0)
            grl_now = (2.0 / (1.0 + math.exp(-grl_k * p)) - 1.0) * self.grl_lambda_base
            self.domain_head.grl.lambd = grl_now

            # use hot k_da
            lambda_da_eff = self.lambda_da_base * (grl_now / max(1e-8, self.grl_lambda_base))
            lambda_da_eff = lambda_da_eff * getattr(self, "k_da", 1.0)

            self.domain_head.grl.lambd = grl_now
            lambda_da_eff = self.lambda_da_base * (grl_now / max(1e-8, self.grl_lambda_base))
            if epoch <= self.warmup_mmd_epochs:
                lambda_mmd_eff = 0.0
            else:
                num = epoch - self.warmup_mmd_epochs
                den = max(1, getattr(self, "epochs", epoch) - self.warmup_mmd_epochs)
                lambda_mmd_eff = self.lambda_mmd_base * min(1.0, num / den)

            with amp_autocast(self.use_amp):
                logits_s, feat_s, *_ = self.model(video_s, audio_s)
                cls_loss = self.criterion(logits_s, labels_s)

                if (video_t is not None) or (audio_t is not None):
                    logits_t, feat_t, *_ = self.model(video_t, audio_t)
                    dom_s = torch.zeros(feat_s.size(0), device=self.device)
                    dom_t = torch.ones(feat_t.size(0), device=self.device)
                    da_loss = 0.5 * (self.domain_head(feat_s, dom_s) + self.domain_head(feat_t, dom_t))
                    mmd = mmd_loss(feat_s, feat_t)
                else:
                    da_loss = torch.tensor(0.0, device=self.device)
                    mmd = torch.tensor(0.0, device=self.device)
                    lambda_da_eff = 0.0
                    lambda_mmd_eff = 0.0

                if not torch.isfinite(da_loss):
                    print(f"[Warning] da_loss not finite at step {step}, skip DA this step.")
                    da_loss = torch.tensor(0.0, device=self.device)
                if not torch.isfinite(mmd):
                    print(f"[Warning] mmd not finite at step {step}, set mmd=0.")
                    mmd = torch.tensor(0.0, device=self.device)

                loss = cls_loss + lambda_da_eff * da_loss + lambda_mmd_eff * mmd

                if (step % 50 == 0) or (step == 0):
                    da_val = da_loss.detach().item() if torch.is_tensor(da_loss) else float(da_loss)
                    mmd_val = mmd.detach().item() if torch.is_tensor(mmd) else float(mmd)
                    print(f"[DA] step={step} grl_now={grl_now:.4f} λ_da={lambda_da_eff:.4f} "
                          f"λ_mmd={lambda_mmd_eff:.4f} da_loss={da_val:.4f} mmd={mmd_val:.4f}")

            # Backpropagation and optimization
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            bs = video_s.size(0) if video_s is not None else 0
            actual_samples += bs
            running_loss += float(loss.detach().item()) * bs


            preds = torch.argmax(logits_s, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_s.detach().cpu().numpy().astype(int).tolist())
            with torch.no_grad():
                if logits_s.shape[1] > 1:
                    probs_b = torch.softmax(logits_s, dim=1)[:, 1].detach().cpu().numpy()
                else:
                    probs_b = torch.sigmoid(logits_s.squeeze(1)).detach().cpu().numpy()
            all_probs.extend(probs_b.tolist())

            # clean cache
            del logits_s, feat_s, cls_loss, da_loss, mmd, loss
            if 'logits_t' in locals(): del logits_t
            if 'feat_t' in locals(): del feat_t
            if (step % 200 == 0) and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # load hot parameter
            if hot_tool and ((step + 1) % int(hot_check_every) == 0):
                hot_tool.apply_hot()
                # batch size will rebuild the dataloader
                if hot_tool.hot and hot_tool.hot.get("stop_now"):
                    print(f"[Hot] stop_now detected at step {step + 1}, exiting epoch early.")
                    break

        # overall metric
        total_samples = max(1, actual_samples)
        epoch_loss = running_loss / total_samples

        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0

        if len(set(all_labels)) > 1 and len(all_probs) == len(all_labels):
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = float('nan')

        return epoch_loss, acc, f1, rec, prec, auc

    def validate_epoch(self, val_loader, epoch, log_every=100, max_batches=None):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []  # ← 只存 float 概率
        actual_samples = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Validation]", total=len(val_loader), leave=True,
                    dynamic_ncols=True)
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                if max_batches is not None and step >= max_batches:
                    break
                try:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                        video, audio, labels = batch[0], batch[1], batch[2]
                    else:
                        video, audio, labels = parse_batch(batch)
                except Exception:
                    continue

                video = video.to(self.device) if video is not None else None
                audio = audio.to(self.device) if audio is not None else None
                labels = labels.to(self.device).long().flatten()

                logits, *_ = self.model(video, audio)
                loss = self.criterion(logits, labels)

                bs = video.size(0) if video is not None else 0
                actual_samples += bs
                running_loss += float(loss.item()) * bs

                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

                if logits.shape[1] > 1:
                    probs_b = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                else:
                    probs_b = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
                all_probs.extend(probs_b.tolist())


                del logits, loss
                if (step % 200 == 0) and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_samples = max(1, actual_samples)
        epoch_loss = running_loss / total_samples

        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0

        # AUC
        if len(set(all_labels)) > 1 and len(all_probs) == len(all_labels):
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = float('nan')

        return epoch_loss, acc, f1, rec, prec, auc


    def save_model(self, epoch, val_loss, val_acc, model_type="default",
                   val_auc=None, save_dir=None, extra: dict = None):

        os.makedirs(self.save_dir, exist_ok=True)
        metric_part = f"_auc{val_auc:.4f}" if (val_auc is not None) else ""
        fname = f"best_{model_type}_epoch{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}{metric_part}.pth"

        if save_dir is None:
            save_path = os.path.join(self.save_dir, fname)
        else:

            is_dir_like = (os.path.isdir(save_dir)
                           or save_dir.endswith(os.sep)
                           or os.path.splitext(save_dir)[1] == "")
            if is_dir_like:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, fname)
            else:

                os.makedirs(os.path.dirname(save_dir) or ".", exist_ok=True)
                save_path = save_dir


        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_auc": (None if val_auc is None else float(val_auc)),

            "global_step": getattr(self, "global_step", 0),
            "total_steps": getattr(self, "total_steps", None),

            "domain_head_state_dict": getattr(self.domain_head, "state_dict", lambda: {})(),
            "lambda_da": getattr(self, "lambda_da_base", None),
            "lambda_mmd": getattr(self, "lambda_mmd_base", None),
            "grl_lambda": getattr(self, "grl_lambda_base", None),

            "scaler_state_dict": getattr(self, "scaler", None).state_dict() if getattr(self, "scaler", None) else None,
        }

        if extra:
            ckpt.update(extra)


        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(ckpt, save_path)
        print(f"[Model Saved] {save_path}")
        self.best_model_path = save_path
        return save_path

    def load_checkpoint(self, path, monitor: str = "auc"):

        import torch, os
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt.get("model_state_dict", {}), strict=True)
        if hasattr(self, "domain_head") and ckpt.get("domain_head_state_dict"):
            try:
                self.domain_head.load_state_dict(ckpt["domain_head_state_dict"], strict=True)
            except Exception as e:
                print(f"[Resume] domain_head load failed: {e}")

        if ckpt.get("optimizer_state_dict"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Resume] optimizer load failed: {e}")
        if ckpt.get("scaler_state_dict") and getattr(self, "scaler", None):
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                print(f"[Resume] scaler load failed: {e}")


        self.global_step = ckpt.get("global_step", 0)
        self.total_steps = ckpt.get("total_steps", None)

        for name in ["lambda_da", "lambda_mmd", "grl_lambda"]:
            if ckpt.get(name) is not None:
                setattr(self, f"{name}_base", ckpt[name])

        epoch = int(ckpt.get("epoch", 0))
        best_metric = None
        if monitor == "auc" and (ckpt.get("val_auc") is not None):
            best_metric = float(ckpt["val_auc"])
        elif monitor == "acc" and (ckpt.get("val_acc") is not None):
            best_metric = float(ckpt["val_acc"])
        return epoch, best_metric



