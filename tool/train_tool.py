import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import json
from .hot_config import hot_config

class train_tool:
    def __init__(self,
                 trainer,
                 epochs=10,
                 model_type="with_attention",
                 log_dir=None,
                 save_dir="../saved_model",
                 # Early stopping
                 monitor="auc",  # "auc" 或 "acc"
                 patience=5,
                 min_delta=1e-4,
                 # LR scheduler (ReduceLROnPlateau)
                 lr_factor=0.5,
                 sched_patience=2,
                 cooldown=0,
                 min_lr=1e-7,
                 hot: hot_config | None = None,
                 resume_from: str | None = None,
                 save_last: bool = True,
                 hot_rebuild_loaders=None
                 ):

        self.trainer = trainer
        self.epochs = int(epochs)
        self.save_dir = save_dir
        self.model_type = model_type
        self.hot = hot
        self.resume_from = resume_from
        self.save_last_enabled = save_last
        self.hot_rebuild_loaders = hot_rebuild_loaders



        # monitor metric
        self.monitor = monitor.lower()
        assert self.monitor in ("auc", "acc")
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        #
        self.scheduler = ReduceLROnPlateau(
            self.trainer.optimizer, mode="max" if self.monitor in ("auc", "acc") else "min",
            factor=float(lr_factor), patience=int(sched_patience),
            threshold=self.min_delta, cooldown=int(cooldown),
            min_lr=float(min_lr)
        )

        # TB（可选）
        self.writer = None
        if log_dir:
            if isinstance(log_dir, bytes):
                log_dir = log_dir.decode("utf-8", errors="ignore")

            log_dir = os.path.abspath(os.path.normpath(log_dir))
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            setattr(self.trainer, "writer", self.writer)

        self.best_metric = float("-inf")
        self.bad_epochs = 0

    def apply_hot(self):
        if not self.hot:
            return {}
        changed = self.hot.reload()
        cfg = self.hot.to_dict()
        if not cfg:
            return {}

        # learnning rate update
        lr = cfg.get("learning_rate")
        if isinstance(lr, (int, float)) and lr > 0:
            for g in self.trainer.optimizer.param_groups:
                g["lr"] = float(lr)

        for name in ["lambda_da", "lambda_mmd", "grl_lambda"]:
            v = cfg.get(name)
            if isinstance(v, (int, float)):
                setattr(self.trainer, f"{name}_base", float(v))

        v = cfg.get("grl_k")
        if isinstance(v, (int, float)):
            setattr(self.trainer, "grl_k", float(v))

        v = cfg.get("k_da")
        if isinstance(v, (int, float)):
            setattr(self.trainer, "k_da", float(v))

        mon = cfg.get("monitor")
        if mon in ("auc", "acc"):
            self.monitor = mon

        pat = cfg.get("patience")
        if isinstance(pat, int) and pat > 0:
            self.patience = pat

        md = cfg.get("min_delta")
        if isinstance(md, (int, float)) and md >= 0:
            self.min_delta = float(md)


        max_epochs = cfg.get("max_epochs")
        if isinstance(max_epochs, int) and max_epochs > 0:
            self.epochs = max_epochs

        bs = cfg.get("batch_size")
        rebuilt = None
        if self.hot_rebuild_loaders and isinstance(bs, int) and bs > 0:
            try:
                rebuilt = self.hot_rebuild_loaders(bs)
            except Exception as e:
                print(f"[Hot] failed to rebuild loaders with bs={bs}: {e}")
                rebuilt = None

        sf = cfg.get("lr_factor")
        if isinstance(sf, (int, float)) and sf > 0:
            self.lr_factor = float(sf)
            if hasattr(self, "scheduler"):
                try:
                    self.scheduler.factor = float(sf)
                except Exception as e:
                    print(f"[Hot] update scheduler.factor failed: {e}")

        sp = cfg.get("sched_patience")
        if isinstance(sp, int) and sp > 0:
            self.sched_patience = sp
            if hasattr(self, "scheduler"):
                try:
                    self.scheduler.patience = int(sp)
                except Exception as e:
                    print(f"[Hot] update scheduler.patience failed: {e}")

        cd = cfg.get("cooldown")
        if isinstance(cd, int) and cd >= 0:
            self.cooldown = cd
            if hasattr(self, "scheduler"):
                try:
                    self.scheduler.cooldown = int(cd)
                except Exception as e:
                    print(f"[Hot] update scheduler.cooldown failed: {e}")

        mlr = cfg.get("min_lr")
        if isinstance(mlr, (int, float)) and mlr >= 0:
            self.min_lr = float(mlr)
            if hasattr(self, "scheduler") and hasattr(self.scheduler, "optimizer"):
                try:
                    groups = self.scheduler.optimizer.param_groups
                    # PyTorch 内部通常是 scheduler.min_lrs
                    if hasattr(self.scheduler, "min_lrs"):
                        self.scheduler.min_lrs = [float(mlr)] * len(groups)
                except Exception as e:
                    print(f"[Hot] update scheduler.min_lrs failed: {e}")

        return {"rebuild": rebuilt, "cfg": cfg}

    def resume_if_needed(self):
        path = None

        if isinstance(self.resume_from, str) and self.resume_from:
            path = self.resume_from
        elif self.hot:
            p = self.hot.get("resume_from")
            path = p if isinstance(p, str) and p else None

        if not path:
            return 0, None

        if not os.path.isfile(path):
            print(f"[Resume] file not found: {path}")
            return 0, None

        if hasattr(self.trainer, "load_checkpoint"):
            ep, best_metric = self.trainer.load_checkpoint(path, monitor=self.monitor)
            print(f"[Resume] loaded '{path}' (epoch={ep}, best={best_metric})")
            return ep, best_metric
        else:
            print("[Resume] trainer has no load_checkpoint(); skip")
            return 0, None

    def save_last(self, epoch, val_loss=None, val_acc=None, val_auc=None):
        if not self.save_last:
            return
        try:
            save_dir = self.save_dir or "./saved_model"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "last.pth")
            extra = {"tag": "last"}
            if hasattr(self.trainer, "save_model"):
                self.trainer.save_model(epoch=epoch, val_loss=(val_loss or 0.0), val_acc=(val_acc or 0.0),
                                        val_auc=val_auc, model_type=self.model_type, save_dir=path, extra=extra)
        except Exception as e:
            print(f"[Warn] save last failed: {e}")

    def get_lr(self):
        return self.trainer.optimizer.param_groups[0]["lr"]

    def pick_metric(self, val_auc, val_acc):
        return val_auc if self.monitor == "auc" else val_acc

    def is_nan(self, x):
        return isinstance(x, float) and x != x

    def fit(self, train_loader, val_loader, target_loader=None):
        setattr(self.trainer, "epochs", self.epochs)

        steps_per_epoch = max(1, len(train_loader))
        self.trainer.total_steps = steps_per_epoch * self.epochs
        if not hasattr(self.trainer, "global_step") or self.trainer.global_step is None:
            self.trainer.global_step = 0

        start_epoch, best_from_ckpt = self.resume_if_needed()
        if best_from_ckpt is not None:
            self.best_metric = float(best_from_ckpt)
        start_epoch = int(start_epoch) + 1 if isinstance(start_epoch, int) else 1

        for epoch in range(start_epoch, self.epochs + 1):
            hot_result = self.apply_hot()
            if hot_result and hot_result.get('rebuild'):
                train_loader, val_loader, target_loader = hot_result['rebuild']
            tr_loss, tr_acc, tr_f1, tr_rec, tr_prec, tr_auc = self.trainer.train_epoch(
                train_loader,
                epoch=epoch,
                target_loader=target_loader,
                hot_tool=self,
                hot_check_every=200,  # ← 每 200 步检查一次 JSON
            )

            val_loss, val_acc, val_f1, val_rec, val_prec, val_auc = self.trainer.validate_epoch(
                val_loader, epoch=epoch
            )


            print(f"[Train] loss:{tr_loss:.4f} acc:{tr_acc:.4f} f1:{tr_f1:.4f} rec:{tr_rec:.4f} prec:{tr_prec:.4f} auc:{tr_auc}")
            print(f"[Valid] loss:{val_loss:.4f} acc:{val_acc:.4f} f1:{val_f1:.4f} rec:{val_rec:.4f} prec:{val_prec:.4f} auc:{val_auc}")


            if self.writer:
                self.writer.add_scalar("Train/loss", tr_loss, epoch)
                self.writer.add_scalar("Train/acc",  tr_acc,  epoch)
                self.writer.add_scalar("Train/auc", tr_auc if not self.is_nan(tr_auc) else 0.0, epoch)

                self.writer.add_scalar("Valid/loss", val_loss, epoch)
                self.writer.add_scalar("Valid/acc",  val_acc,  epoch)
                self.writer.add_scalar("Valid/auc", val_auc if not self.is_nan(val_auc) else 0.0, epoch)

                self.writer.add_scalar("LR/base", self.get_lr(), epoch)

            #checkout point
            self.save_last(epoch, val_loss=val_loss, val_acc=val_acc, val_auc=val_auc)


            metric = self.pick_metric(val_auc, val_acc)
            metric_for_sched = metric if (metric is not None and not self.is_nan(metric)) else self.best_metric


            self.scheduler.step(metric_for_sched)

            # save the best model
            if metric is not None and not self.is_nan(metric) and metric > self.best_metric + self.min_delta:
                self.best_metric = metric
                self.bad_epochs = 0
                self.trainer.save_model(epoch=epoch, val_loss=val_loss, val_acc=val_acc,val_auc=val_auc, model_type=self.model_type,save_dir=self.save_dir)
                print(f"[Save] best {self.monitor} improved to {self.best_metric:.6f} | lr={self.get_lr():.2e}")
            else:
                self.bad_epochs += 1
                print(f"[Info] no improvement on {self.monitor}. bad_epochs={self.bad_epochs}/{self.patience} | lr={self.get_lr():.2e}")

            if self.hot and self.hot.get('stop_now'):
                print('[Hot] stop_now detected; exiting training loop.')
                break

            # 早停
            if self.bad_epochs >= self.patience:
                print(f"[EarlyStop] {self.monitor} no improvement for {self.patience} epochs. Stop at epoch {epoch}.")
                break

        if self.writer:
            self.writer.flush()
            self.writer.close()

