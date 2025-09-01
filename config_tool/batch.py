import torch
import gc
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.atten_CRNN_Single import atten_CRNN_Single
from model.atten_CRNN_multi import atten_CRNN_multi
from model.CRNN_without import CRNN_without

try:
    # if Pytorch >2.0
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler as _GradScalerNew
    def amp_autocast(use_amp: bool, is_cuda: bool):
        return _autocast_new('cuda', dtype=torch.float16, enabled=(use_amp and is_cuda))
    def GradScalerCls(use_amp: bool, is_cuda: bool):
        return _GradScalerNew('cuda', enabled=(use_amp and is_cuda))
except Exception:
    from torch.cuda.amp import autocast as _autocast_old
    from torch.cuda.amp import GradScaler as _GradScalerOld
    def amp_autocast(use_amp: bool, is_cuda: bool):
        return _autocast_old(enabled=(use_amp and is_cuda))
    def GradScalerCls(use_amp: bool, is_cuda: bool):
        return _GradScalerOld(enabled=(use_amp and is_cuda))

# the parameter you can change
V_T, V_H, V_W   = 16, 112, 112      # video -> [B, 3, T, H, W]
A_T, A_H, A_W   = 16, 112, 112      # audio -> [B, 1, T, H, W] 先给 5D，更通用
NUM_CLASSES     = 2
MODEL_DEPTH     = 50
FUSION          = "concat"
USE_AMP         = True
MAX_TRY_CAP     = 4096

def make_fake_batch(bs, device="cuda"):
    video = torch.randn(bs, 3, V_T, V_H, V_W, device=device)   # [B,3,T,H,W]
    audio = torch.randn(bs, 3, A_T, A_H, A_W, device=device)   # ⭐ [B,3,T,H,W] —— 关键：3通道，保持5D
    label = torch.randint(0, NUM_CLASSES, (bs,), device=device)
    return video, audio, label


def transfer_to_4d(audio_5d):
    return audio_5d.mean(dim=2) if audio_5d.dim() == 5 else audio_5d

def transfer_to_3ch(audio_4d):
    return audio_4d.repeat(1, 3, 1, 1) if (audio_4d.dim() == 4 and audio_4d.size(1) == 1) else audio_4d

#check if the model need 3 channel
def check_3ch(model):
    try:
        conv1 = getattr(getattr(model, "audio_resnet", None), "conv1", None)
        if conv1 is not None and getattr(conv1, "in_channels", None) == 3:
            return True
    except Exception:
        pass
    return False

def try_one_step(model, bs, device="cuda"):
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    scaler = GradScalerCls(USE_AMP, is_cuda)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    video, audio, label = make_fake_batch(bs, device=device)
    try:
        optim.zero_grad(set_to_none=True)
        with amp_autocast(USE_AMP, is_cuda):
            try:
                logits, feat, *_ = model(video, audio)
            except RuntimeError as e:
                msg = str(e).lower()
                need_3ch = ("to have 3 channels" in msg) or ("in_channels=3" in msg) or ("64, 3, 7, 7" in msg)
                if need_3ch and audio.size(1) == 1:
                    audio_fixed = audio.repeat(1, 3, 1, 1, 1)
                    logits, feat, *_ = model(video, audio_fixed)
                else:
                    raise

            loss = torch.nn.CrossEntropyLoss()(logits, label)

        if USE_AMP and is_cuda:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        # clean the cache
        del video, audio, label, logits, feat, loss
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            del video, audio, label
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()
            return False
        else:
            raise

def find_upper_bound(model, start=8, device="cuda"):
    bs, ok = max(1, start), 0
    while bs <= MAX_TRY_CAP:
        if try_one_step(model, bs, device=device):
            ok = bs
            bs *= 2
        else:
            return ok, bs
    return ok, min(bs, MAX_TRY_CAP)

def find_max_batch_for_model(model, device="cuda", warmup_bs=8):
    model.train()
    model.to(device)
    # avodi false OOM
    try_one_step(model, warmup_bs, device=device)

    lo, hi = find_upper_bound(model, start=warmup_bs, device=device)
    if hi == 0:
        return 0
    best, left, right = lo, max(1, lo+1), max(lo+1, hi-1)
    while left <= right:
        mid = (left + right) // 2
        if try_one_step(model, mid, device=device):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best

def build_model(kind: str, device="cuda"):
    kind = kind.lower()
    if kind == "without":
        return CRNN_without(model_depth=MODEL_DEPTH, num_classes=NUM_CLASSES, fusion_method=FUSION).to(device)
    elif kind == "atten_single":
        return atten_CRNN_Single(model_depth=MODEL_DEPTH, num_classes=NUM_CLASSES, fusion_method=FUSION).to(device)
    elif kind == "atten_multi":
        return atten_CRNN_multi(model_depth=MODEL_DEPTH, num_classes=NUM_CLASSES, fusion_method=FUSION).to(device)
    else:
        raise ValueError(f"unknown model: {kind}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model_types = ["without", "atten_single", "atten_multi"]
    results = {}

    for m in model_types:
        print(f"\n=== test: {m} | import: video=[B,3,{V_T},{V_H},{V_W}], audio=[B,3,{A_T},{A_H},{A_W}] | AMP={USE_AMP} ===")
        model = build_model(m, device=device)
        try:
            max_bs = find_max_batch_for_model(model, device=device, warmup_bs=8)
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        results[m] = max_bs
        print(f" {m} maximise batch_size = {max_bs}")

    print("\n==== summary ====")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
