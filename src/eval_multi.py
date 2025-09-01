import os, sys, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tool.safe_collate import safe_collate
from model.CRNN_without import CRNN_without
from model.atten_CRNN_multi import atten_CRNN_multi
from model.atten_CRNN_Single import atten_CRNN_Single
from read_video_dataset import read_video_dataset
from contextlib import nullcontext
amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")




source_root = 'prepared_dataset/video_data'            # 源域
target_root = 'prepared_dataset/Celeb-DF-v2'
# source_root = 'D:/UCL/dissertation/PythonProject/Dataset/video_with'
# target_root = 'D:/UCL/dissertation/PythonProject/Dataset/new'
base_pic_dir = 'DA_multi_cross'             # 每个模型会建子文件夹
os.makedirs(base_pic_dir, exist_ok=True)

batch_size      = 20  #train 36 or 16(only attention block)
num_classes     = 2
num_workers     = 10    #train 28, train 0
prefetch_factor = 2  #train and train for 4
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cost curve
COST_FN = 30
COST_FP = 1
thr_grid = np.linspace(0, 1, 101)

# robustness tset
noise_levels = [0, 0.01, 0.02, 0.05, 0.1]

# DI parameter
di_steps       = 20
di_subset_size = 2000
di_use_amp     = True

thr_grid = np.linspace(0.01, 0.99, 99)


# three model configs
model_configs = [
    {
        "tag": "atten_multi",
        "arch": "atten_multi",
        "ckpt": "DA_saved_model/multi/continue/best_multi_epoch3_loss0.4086_acc0.8858_auc0.9675.pth",   # ← 修改成你的路径
        "model_depth": 50,
        "fusion_method": "concat",
        "pooling_method": "avg",
        "has_attention": True
    },

    {
        "tag": "atten_single",
        "arch": "atten_single",  # atten_single | atten_multi | without
        "ckpt": "best_atten_single_model_epoch7_loss0.1887_acc0.9616.pth",
        "model_depth": 50,
        "fusion_method": "concat",
        "pooling_method": "avg",  # 仅无注意力模型用
        "has_attention": True
    },

    {
        "tag": "without",
        "arch": "crnn_without",
        "ckpt": "best_without_attention_model_epoch6_loss0.1765_acc0.9739.pth",
        "model_depth": 50,
        "fusion_method": "concat",
        "pooling_method": "avg",
        "has_attention": False
    },



]

# tool function
def load_model_from_config(cfg):
    arch = cfg["arch"]
    if arch == "atten_single":
        model = atten_CRNN_Single(
            model_depth=cfg["model_depth"],
            num_classes=num_classes,
            fusion_method=cfg["fusion_method"]
        )
    elif arch == "atten_multi":
        model = atten_CRNN_multi(
            model_depth=cfg["model_depth"],
            num_classes=num_classes,
            fusion_method=cfg["fusion_method"]
        )
    elif arch == "crnn_without":
        model = CRNN_without(
            model_depth=cfg["model_depth"],
            num_classes=num_classes,
            fusion_method=cfg["fusion_method"],
            pooling_method=cfg.get("pooling_method", "avg")
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # load checkpoint
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")

    state = ckpt.get("model_state_dict", ckpt)

    #
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[load_state_dict] Missing keys (showing up to 20): {missing[:20]}"
              + (" ..." if len(missing) > 20 else ""))
    if unexpected:
        print(f"[load_state_dict] Unexpected keys (showing up to 20): {unexpected[:20]}"
              + (" ..." if len(unexpected) > 20 else ""))

    model.to(DEVICE)
    model.eval()
    return model

def pick_pos_idx_by_auc(y_true: np.ndarray, y_prob_2col: np.ndarray) -> int:

    assert y_prob_2col.shape[1] == 2, "Only supports binary classification (2 columns)."
    auc0 = roc_auc_score(y_true, y_prob_2col[:, 0])
    auc1 = roc_auc_score(y_true, y_prob_2col[:, 1])
    return int(auc1 >= auc0)

def best_cost_threshold(y_true: np.ndarray, pos_probs: np.ndarray,
                        cost_fn: float, cost_fp: float,
                        grid: np.ndarray = None):

    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for thr in grid:
        pred = (pos_probs >= thr).astype(int)
        fn = np.sum((pred == 0) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        costs.append(cost_fn * fn + cost_fp * fp)
    costs = np.array(costs)
    best_i = int(np.argmin(costs))
    return grid[best_i], costs

def best_f1_threshold(y_true: np.ndarray, pos_probs: np.ndarray, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)  # 避免0/1
    best_t, best = 0.5, -1
    for t in grid:
        pred = (pos_probs >= t).astype(int)
        if pred.min() == pred.max():  # 避免退化
            continue
        f1 = f1_score(y_true, pred, average="binary", zero_division=0, pos_label=1)
        if f1 > best:
            best, best_t = f1, t
    return best_t

def probs_1_from_logits_no_nan(logits: torch.Tensor) -> np.ndarray:
    p = torch.softmax(logits, dim=1)[:, 1]
    # clean inf and NaN
    p = p.detach().float().cpu()
    p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)  # 兜底策略可按需调整
    return p.numpy()

def safe_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    if (y_true.min() == y_true.max()):
        return 0.5
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5

def fit_temperature_on_logits(logits: np.ndarray, labels: np.ndarray, max_iter=500):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.tensor(logits, dtype=torch.float32, device=dev)
    y = torch.tensor(labels, dtype=torch.long, device=dev)

    T = torch.nn.Parameter(torch.ones(1, device=dev))
    opt = torch.optim.LBFGS([T], lr=0.5, max_iter=50, line_search_fn='strong_wolfe')

    def nll_loss():
        zT = z / T.clamp_min(1e-3)
        logp = F.log_softmax(zT, dim=1)
        return F.nll_loss(logp, y)

    def closure():
        opt.zero_grad()
        loss = nll_loss()
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)
    return float(T.detach().cpu().item())

def logits_from_probs_binary(y_prob_2col: np.ndarray) -> np.ndarray:

    p = np.clip(y_prob_2col, 1e-6, 1-1e-6)
    a = np.log(p[:, 1]) - np.log(p[:, 0])   # log-odds
    z0 = np.zeros_like(a)
    z1 = a
    return np.stack([z0, z1], axis=1)

def apply_temperature_to_logits(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / max(T, 1e-3)

def make_loader(root_path, split='test'):
    ds = read_video_dataset(root_path=root_path, label=split, train_mode=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        pin_memory=True, collate_fn=safe_collate
    )
    return ds, loader



@torch.no_grad()
def eval_single_domain(model, dataset, loader, out_prefix, pic_dir,model_tag,T: float | None = None):
    all_labels, all_preds, all_probs,all_logits = [], [], [],[]
    for batch in tqdm(loader, desc=f"Evaluating [{out_prefix}]"):
        video, audio, labels = batch
        video  = video.to(DEVICE) if video is not None else None
        audio  = audio.to(DEVICE) if audio is not None else None
        labels = labels.to(DEVICE)

        outputs = model(video, audio)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        all_logits.append(logits.detach().cpu().numpy())
        # ：
        logits_for_prob = logits / max(1e-3, float(T)) if T is not None else logits
        probs = torch.softmax(logits_for_prob, dim=1)
        preds  = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)  # shape [N, 2]


    pos_idx = pick_pos_idx_by_auc(y_true, y_prob)
    pos_probs = y_prob[:, pos_idx]

    q = np.quantile(pos_probs, [0, 0.01, 0.5, 0.99, 1.0])
    pos_rate_05 = float((pos_probs >= 0.5).mean())
    print(f"[{out_prefix}] pos_idx={pos_idx}, prob quantiles={q}, pos_rate@0.5={pos_rate_05:.2f}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    auc = roc_auc_score(y_true, pos_probs)
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, pos_probs)
    auprc = average_precision_score(y_true, pos_probs)


    # CM
    cm = confusion_matrix(y_true, y_pred)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({model_tag}) ({out_prefix})')
    plt.savefig(os.path.join(pic_dir, f'cm_{out_prefix}.png'), dpi=300);
    plt.close()

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', alpha=0.5)
    plt.xlabel('FPR');
    plt.ylabel('TPR');
    plt.title(f'ROC ({model_tag}) ({out_prefix})');
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, f'roc_{out_prefix}.png'), dpi=300);
    plt.close()

    # PR
    plt.figure()
    plt.plot(rec_vals, prec_vals, label=f'AUPRC={auprc:.4f}')
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.title(f'PR ({model_tag}) ({out_prefix})');
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(pic_dir, f'pr_{out_prefix}.png'), dpi=300);
    plt.close()

    # cost curve
    best_thr, costs = best_cost_threshold(y_true, pos_probs, COST_FN, COST_FP, thr_grid)

    pi = float((y_true == 1).mean())
    thr_bayes = (COST_FP * (1 - pi)) / (COST_FN * pi + COST_FP * (1 - pi) + 1e-12)
    print(f"[{out_prefix}] pi={pi:.3f}, bayes_thr≈{thr_bayes:.3f}")


    plt.figure()
    plt.plot(thr_grid, costs, label='Total Cost')
    plt.axvline(best_thr, linestyle='--', label=f'Recommended thr={best_thr:.2f}')
    plt.axvline(thr_bayes, linestyle=':', label=f'Bayes thr≈{thr_bayes:.2f}')
    plt.xlabel('Threshold');
    plt.ylabel('Total Cost');
    plt.title(f'Cost Curve ({model_tag}) ({out_prefix})');
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(pic_dir, f'cost_{out_prefix}.png'), dpi=300);
    plt.close()

    # robustness
    aucs_gauss, aucs_uniform = [], []
    for nl in noise_levels:
        noisy_gauss = np.clip(pos_probs + np.random.normal(0, nl, size=len(pos_probs)), 0, 1)
        noisy_uniform = np.clip(pos_probs + np.random.uniform(-nl, nl, size=len(pos_probs)), 0, 1)
        aucs_gauss.append(roc_auc_score(y_true, noisy_gauss))
        aucs_uniform.append(roc_auc_score(y_true, noisy_uniform))

    plt.figure()
    plt.plot(noise_levels, aucs_gauss, marker='o', label='Gaussian Noise')
    plt.plot(noise_levels, aucs_uniform, marker='s', label='Uniform Noise')
    plt.xlabel('Noise Level');
    plt.ylabel('AUC');
    plt.title(f'Robustness ({model_tag}) ({out_prefix})');
    plt.legend()
    plt.grid(True, alpha=0.3);
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, f'robustness_{out_prefix}.png'), dpi=300);
    plt.close()

    best_thr_f1 = best_f1_threshold(y_true, pos_probs, thr_grid)
    logits_np = np.concatenate(all_logits, axis=0)

    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'auc': auc, 'auprc': auprc,
        'best_thr': best_thr,
        'costs': costs, 'thr_grid': thr_grid,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,  # 原始两列保留
        'pos_idx': pos_idx, 'y_prob_pos': pos_probs,  # ★ 新增：正类列与其概率
        'best_thr_f1': best_thr_f1, 'temperature': T,
        'logits': logits_np
    }


# attention and DI
@torch.no_grad()
def collect_temporal_attention(model, loader):
    ys, ps, atts, pads = [], [], [], []

    for video, audio, label in loader:
        v = video.to(DEVICE) if video is not None else None
        a = audio.to(DEVICE) if audio is not None else None

        out = model(v, a)
        if not isinstance(out, (list, tuple)) or len(out) < 2:
            raise RuntimeError("Model output is not a tuple with enough elements to collect attention.")

        logits = out[0]

        # Take temporal attention weight
        t_attn = None
        if len(out) >= 3 and isinstance(out[2], torch.Tensor):
            t_attn = out[2]
        elif isinstance(out[1], torch.Tensor) and out[1].dim() >= 2:
            t_attn = out[1]
        else:
            raise RuntimeError("Temporal attention tensor not found in model output (expected at out[2]).")

        # key_padding_mask
        pad = None
        if len(out) >= 5 and isinstance(out[4], torch.Tensor):
            pad = out[4]                    # atten_multi: key_padding_mask
        elif len(out) >= 4 and isinstance(out[3], torch.Tensor):
            pad = out[3]


        prob1 = torch.softmax(logits, dim=1)[:, 1]


        ys.append(np.asarray(label, dtype=int))
        ps.append(prob1.detach().cpu().numpy())


        if t_attn.dim() == 3 and t_attn.size(1) == 1:
            t_attn = t_attn.squeeze(1)
        atts.append(t_attn.detach().cpu())

        if pad is not None:
            if pad.dim() == 3 and pad.size(1) == 1:
                pad = pad.squeeze(1)
            pads.append(pad.detach().cpu())

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    attn_T = torch.cat(atts, dim=0).numpy()                 # [N, L]
    pad_mask = torch.cat(pads, dim=0).numpy() if pads else None  # [N, L] or None
    return y_true, y_prob, attn_T, pad_mask


def build_top_idx_list(attn_T, pad_mask=None):
    N, L = attn_T.shape
    top_idx_list = []
    for i in range(N):
        att = attn_T[i].copy()
        if pad_mask is not None: att[pad_mask[i].astype(bool)] = -np.inf
        order = np.argsort(-att)
        if np.isneginf(att[order]).all():
            order = np.array([], dtype=int)
        else:
            valid = ~np.isneginf(att[order]); order = order[valid]
        top_idx_list.append(order)
    return top_idx_list

@torch.no_grad()
def di_curves_batched(model, dataset, loader, attn_T, out_prefix, pic_dir,model_tag,
                      pad_mask=None, steps=20, subset_size=2000, batch_size=16, num_workers=0, use_amp=True):
    N, L = attn_T.shape
    idxs = list(range(N))
    if subset_size is not None and subset_size < N:
        random.seed(42); idxs = random.sample(idxs, subset_size)

    sub_attn = attn_T[idxs]
    sub_pad  = pad_mask[idxs] if pad_mask is not None else None
    top_idx_list = build_top_idx_list(sub_attn, sub_pad)

    sub_ds = Subset(dataset, idxs)
    sub_loader = DataLoader(
        sub_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(DEVICE.type=='cuda'), collate_fn=safe_collate
    )

    cached, ptr = [], 0
    for video, audio, label in tqdm(sub_loader, desc=f"Caching DI subset [{out_prefix}]"):
        B = label.shape[0] if isinstance(label, torch.Tensor) else len(label)
        cur_top = top_idx_list[ptr: ptr+B]; ptr += B
        cached.append((video, audio, label, cur_top))

    fractions = np.linspace(0, 1, steps)
    del_scores, ins_scores = [], []

    from contextlib import nullcontext
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if (use_amp and DEVICE.type=='cuda') else nullcontext()

    def apply_mask(x, keep, mode):
        if x is None: return None
        if x.dim() == 5:   # [B,C,T,H,W]
            return x * (1.0 - keep) if mode == 'deletion' else x * keep
        elif x.dim() == 4: # [B,C,T,F]
            m = keep.squeeze(-1)
            return x * (1.0 - m) if mode == 'deletion' else x * m
        else:
            raise RuntimeError(f"Unexpected tensor shape: {x.shape}")

    for frac in tqdm(fractions, desc=f"DI (batched) [{out_prefix}]"):
        k = int(round(frac * L))
        y_del_all, p_del_all, y_ins_all, p_ins_all = [], [], [], []
        for video, audio, label, cur_top in cached:
            v = video.to(DEVICE) if video is not None else None
            a = audio.to(DEVICE) if audio is not None else None
            y_np = label.cpu().numpy().astype(int) if isinstance(label, torch.Tensor) else np.array(label, dtype=int)
            B = y_np.shape[0]

            if k > 0:
                keep_mask = torch.zeros((B,1,L,1,1), device=DEVICE, dtype=torch.float32)
                for b in range(B):
                    top_idx = cur_top[b][:k] if len(cur_top[b]) >= k else cur_top[b]
                    if len(top_idx) > 0:
                        keep_mask[b,0, torch.as_tensor(top_idx, device=DEVICE),0,0] = 1.0
            else:
                keep_mask = torch.zeros((B,1,L,1,1), device=DEVICE, dtype=torch.float32)

            with amp_ctx, torch.inference_mode():
                v_del = apply_mask(v, keep_mask, 'deletion') if v is not None else None
                a_del = apply_mask(a, keep_mask, 'deletion') if a is not None else None
                out = model(v_del, a_del)
                logits = out[0] if isinstance(out, tuple) else out
                p1 = probs_1_from_logits_no_nan(logits)
                y_del_all.append(y_np)
                p_del_all.append(p1)

                v_ins = apply_mask(v, keep_mask, 'insertion') if v is not None else None
                a_ins = apply_mask(a, keep_mask, 'insertion') if a is not None else None
                out = model(v_ins, a_ins)
                logits = out[0] if isinstance(out, tuple) else out
                p1 = probs_1_from_logits_no_nan(logits)
                y_ins_all.append(y_np)
                p_ins_all.append(p1)

        y_del = np.concatenate(y_del_all)
        p_del = np.concatenate(p_del_all)
        y_ins = np.concatenate(y_ins_all)
        p_ins = np.concatenate(p_ins_all)

        del_scores.append(safe_auc_binary(y_del, p_del))
        ins_scores.append(safe_auc_binary(y_ins, p_ins))

    del_scores = np.array(del_scores); ins_scores = np.array(ins_scores)

    plt.figure(figsize=(8,6))
    plt.plot(fractions, del_scores, marker='o', label='Deletion')
    plt.plot(fractions, ins_scores, marker='s', label='Insertion')
    plt.axhline(0.5, linestyle='--', alpha=0.5)
    plt.xlabel('Fraction of most important frames'); plt.ylabel('AUC')
    plt.title(f'Temporal Deletion/Insertion ({model_tag}) ({out_prefix})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, f'di_{out_prefix}.png'), dpi=300); plt.close()

    return fractions, del_scores, ins_scores

# cross domain report
def summarize_cross_domain(src_metrics, tgt_metrics, tag='default', out_dir='.'):
    def retention(x_src, x_tgt): return 100.0 * (x_tgt / max(1e-12, x_src))
    auc_drop   = src_metrics['auc']  - tgt_metrics['auc']
    f1_drop    = src_metrics['f1']   - tgt_metrics['f1']
    acc_drop   = src_metrics['acc']  - tgt_metrics['acc']
    auprc_drop = src_metrics['auprc']- tgt_metrics['auprc']
    auc_ret   = retention(src_metrics['auc'],   tgt_metrics['auc'])
    f1_ret    = retention(src_metrics['f1'],    tgt_metrics['f1'])
    acc_ret   = retention(src_metrics['acc'],   tgt_metrics['acc'])
    auprc_ret = retention(src_metrics['auprc'], tgt_metrics['auprc'])

    # The cost of the optimal threshold in the source domain on the target domain
    src_best_thr = src_metrics['best_thr']
    y_true_tgt, y_prob_tgt = tgt_metrics['y_true'], tgt_metrics['y_prob']
    preds_tgt_srcThr = (y_prob_tgt[:,1] >= src_best_thr).astype(int)
    fn = np.sum((preds_tgt_srcThr==0) & (y_true_tgt==1))
    fp = np.sum((preds_tgt_srcThr==1) & (y_true_tgt==0))
    tgt_cost_with_srcThr = COST_FN * fn + COST_FP * fp
    tgt_best_cost = np.min(tgt_metrics['costs'])
    tgt_best_thr  = tgt_metrics['thr_grid'][np.argmin(tgt_metrics['costs'])]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
==== Cross-Domain Summary ({tag}) ====
Timestamp: {ts}
Source:
  ACC={src_metrics['acc']:.4f}  F1={src_metrics['f1']:.4f}
  AUC={src_metrics['auc']:.4f}  AUPRC={src_metrics['auprc']:.4f}
  BestThr={src_metrics['best_thr']:.2f}
  BestThr(cost)={src_metrics['best_thr']:.2f}  BestThr(F1)={src_metrics['best_thr_f1']:.2f}

Target:
  ACC={tgt_metrics['acc']:.4f}  F1={tgt_metrics['f1']:.4f}
  AUC={tgt_metrics['auc']:.4f}  AUPRC={tgt_metrics['auprc']:.4f}
  BestThr={tgt_best_thr:.2f}
  BestThr(cost)={tgt_best_thr:.2f}  BestThr(F1)={tgt_metrics['best_thr_f1']:.2f}

Diff (Source - Target):
  ΔACC={acc_drop:.4f}  ΔF1={f1_drop:.4f}  ΔAUC={auc_drop:.4f}  ΔAUPRC={auprc_drop:.4f}

Retention (Target / Source):
  ACC={acc_ret:.2f}%  F1={f1_ret:.2f}%  AUC={auc_ret:.2f}%  AUPRC={auprc_ret:.2f}%

Threshold Transfer (Source Thr → Target):
  Target cost @SourceThr = {tgt_cost_with_srcThr:.0f}
  Target best cost       = {tgt_best_cost:.0f}
  Cost gap (↑ means miscalibration risk) = {tgt_cost_with_srcThr - tgt_best_cost:.0f}
========================================
"""
    # print
    print(report)

    # wirte in the folder
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{tag}_cross_domain_summary.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(report + "\n")


    # csv_path = os.path.join(out_dir, f"{tag}_cross_domain_summary.csv")
    # header = ("timestamp,src_acc,src_f1,src_auc,src_auprc,src_best_thr,"
    #           "tgt_acc,tgt_f1,tgt_auc,tgt_auprc,tgt_best_thr,"
    #           "d_acc,d_f1,d_auc,d_auprc,"
    #           "r_acc,r_f1,r_auc,r_auprc,"
    #           "tgt_cost_srcThr,tgt_best_cost,cost_gap")
    # line = (f"{ts},{src_metrics['acc']:.6f},{src_metrics['f1']:.6f},{src_metrics['auc']:.6f},{src_metrics['auprc']:.6f},{src_metrics['best_thr']:.4f},"
    #         f"{tgt_metrics['acc']:.6f},{tgt_metrics['f1']:.6f},{tgt_metrics['auc']:.6f},{tgt_metrics['auprc']:.6f},{tgt_best_thr:.4f},"
    #         f"{acc_drop:.6f},{f1_drop:.6f},{auc_drop:.6f},{auprc_drop:.6f},"
    #         f"{acc_ret:.2f},{f1_ret:.2f},{auc_ret:.2f},{auprc_ret:.2f},"
    #         f"{tgt_cost_with_srcThr:.0f},{tgt_best_cost:.0f},{(tgt_cost_with_srcThr - tgt_best_cost):.0f}")
    # if not os.path.exists(csv_path):
    #     with open(csv_path, "w", encoding="utf-8") as f:
    #         f.write(header + "\n")
    #         f.write(line + "\n")
    # else:
    #     with open(csv_path, "a", encoding="utf-8") as f:
    #         f.write(line + "\n")


def main():
    # source domain
    src_val_ds,  src_val_loader  = make_loader(source_root, split='val')
    src_test_ds, src_test_loader = make_loader(source_root, split='test')

    # target domain
    tgt_test_ds, tgt_test_loader = make_loader(target_root,  split='test')

    for cfg in model_configs:
        tag = cfg["tag"]
        pic_dir = os.path.join(base_pic_dir, tag)
        os.makedirs(pic_dir, exist_ok=True)
        print(f"\n===== Running model: {tag} =====")

        model = load_model_from_config(cfg)

        src_uncal_val = eval_single_domain(model, src_val_ds, src_val_loader,
                                           out_prefix='SOURCE_VAL', pic_dir=pic_dir, model_tag=cfg['tag'], T=None)

        logits_for_fit = src_uncal_val['logits'] if 'logits' in src_uncal_val and src_uncal_val['logits'] is not None \
                         else logits_from_probs_binary(src_uncal_val['y_prob'])
        T_src = fit_temperature_on_logits(logits_for_fit, src_uncal_val['y_true'])
        print(f"[Calib] Fitted temperature on SOURCE-VAL: T={T_src:.3f}")

        src_cal = eval_single_domain(model, src_test_ds, src_test_loader,
                                     out_prefix='SOURCE_TEST_CAL', pic_dir=pic_dir, model_tag=cfg['tag'], T=T_src)
        tgt_cal = eval_single_domain(model, tgt_test_ds, tgt_test_loader,
                                     out_prefix='TARGET_TEST_CAL', pic_dir=pic_dir, model_tag=cfg['tag'], T=T_src)

        if cfg.get("has_attention", False):
            try:
                for which, ds, loader in [('SOURCE_TEST', src_test_ds, src_test_loader),
                                          ('TARGET_TEST', tgt_test_ds, tgt_test_loader)]:
                    y_true, y_prob, attn_T, pad_mask = collect_temporal_attention(model, loader)
                    di_curves_batched(
                        model, ds, loader, attn_T, out_prefix=which, pic_dir=pic_dir, model_tag=cfg['tag'],
                        pad_mask=pad_mask, steps=di_steps, subset_size=di_subset_size,
                        batch_size=batch_size, num_workers=num_workers, use_amp=di_use_amp
                    )
            except RuntimeError as e:
                print(f"[DI skipped for {tag}] {e}")

        summarize_cross_domain(src_cal, tgt_cal, tag=tag + "_CAL", out_dir=pic_dir)
        print("========================================\n")
        print("eval complete!")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
