import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from model.atten_CRNN_multi import atten_CRNN_multi
from video_reader import video_reader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def to_uint8_img(frame_chw: torch.Tensor):
    x = frame_chw.detach().cpu().float().clone()
    if x.max() <= 1.0 + 1e-6:
        x = x * 255.0
    x = x.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return x

def even_indices(n_frames, k=8):
    if n_frames <= k:
        return list(range(n_frames))
    return [int(round(i*(n_frames-1)/(k-1))) for i in range(k)]

def norm01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

def ensure_spatial_maps(s_attn, T_expected=None):
    if s_attn is None:
        return None
    if isinstance(s_attn, (tuple, list)):
        s_attn = s_attn[0] if s_attn[0] is not None else s_attn[1]
        if s_attn is None: return None
    if not isinstance(s_attn, torch.Tensor):
        return None

    a = s_attn.detach().float().cpu()
    if a.ndim == 4 and a.shape[1] == 1:  # [T,1,H,W] or [B*T,1,H,W]
        if T_expected and a.shape[0] > T_expected:
            a = a[:T_expected]
        return a
    if a.ndim == 4 and a.shape[0] == 1:  # [1,L,H,W]
        return a.squeeze(0).unsqueeze(1)
    if a.ndim == 5:  # [1,L,1,H,W]
        return a.squeeze(0)
    return None

def temporal_to_color_overlays(t_attn, T, size_hw):
    w = t_attn.detach().float().cpu().view(-1).numpy()
    w = w / (w.sum() + 1e-6)
    xs = np.linspace(0, len(w)-1, num=len(w))
    xt = np.linspace(0, len(w)-1, num=T)
    wt = np.interp(xt, xs, w)
    wt = norm01(wt)

    H, W = size_hw
    overlays = []
    for i in range(T):
        mask = np.ones((H, W), dtype=np.float32) * wt[i]
        rgb = (cm.jet(mask)[:, :, :3] * 255).astype(np.uint8)
        overlays.append(Image.fromarray(rgb))
    return overlays

def blend(orig_img_pil: Image.Image, overlay_pil: Image.Image, alpha: float):
    return Image.blend(orig_img_pil.convert('RGBA'),
                       overlay_pil.convert('RGBA'), alpha).convert('RGB')


@torch.inference_mode()
def export_heatmaps(model, media_path, out_dir="heatmap_out",
                    num_frames=8, display_size=(400, 400),
                    alpha=0.45, device=None):

    os.makedirs(out_dir, exist_ok=True)
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device).eval()

    # load video frame
    reader = video_reader(device=device, video_frames=16)  # 改成你训练时的帧数
    res = reader.load_media(media_path)

    if isinstance(res, (tuple, list)):
        vid = res[0]  # video
        aud = res[1] if len(res) > 1 else None
        info = res[2] if len(res) > 2 else None
    elif isinstance(res, dict):
        vid = res.get("video", None)
        aud = res.get("audio", None)
        info = res.get("info", None)
    else:
        vid, aud, info = res, None, None

    if vid is None:
        raise RuntimeError("can't load file")

    frames = [to_uint8_img(vid[:, t]) for t in range(vid.shape[1])]  # [C,T,H,W]
    T = len(frames)

    out = model(vid.unsqueeze(0), None)
    if isinstance(out, (tuple, list)):
        if len(out) >= 4 and isinstance(out[2], torch.Tensor):  # 新版
            logits, context, t_attn, s_attn = out[0], out[1], out[2], out[3]
        elif len(out) >= 3:  # 旧版
            logits, t_attn, s_attn = out[0], out[1], out[2]
        else:
            logits, t_attn, s_attn = out[0], None, None
    else:
        logits, t_attn, s_attn = out, None, None


    s_maps = ensure_spatial_maps(s_attn, T_expected=T)

    idxs = even_indices(T, k=num_frames)

    images_to_save = []
    for i, t in enumerate(idxs):
        base = Image.fromarray(frames[t]).resize(display_size, Image.LANCZOS)

        if s_maps is not None:  # 空间注意力
            att = s_maps[t].squeeze(0).numpy()
            att = norm01(att)
            att_img = Image.fromarray((cm.jet(att)[:, :, :3] * 255).astype(np.uint8)).resize(display_size, Image.LANCZOS)
        elif isinstance(t_attn, torch.Tensor):  # 时序注意力
            H, W = display_size[1], display_size[0]
            att_img = temporal_to_color_overlays(t_attn, T, (H, W))[t]
        else:  # 没有注意力
            att_img = Image.fromarray(np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8))

        blended = blend(base, att_img, alpha=alpha)
        images_to_save.append(blended)
        save_name = f"{os.path.splitext(os.path.basename(media_path))[0]}_attn_{i:02d}.png"
        blended.save(os.path.join(out_dir, save_name))

    print(f"saved {len(images_to_save)} heatmaps to : {os.path.abspath(out_dir)}")
    return images_to_save


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = atten_CRNN_multi(
    model_depth=50,
    num_classes=2,
    fusion_method='concat'
).to(device)



import torch
from collections import OrderedDict

def load_checkpoint_flex(model, ckpt_path, device='cpu', strict=False, verbose=True):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt

    elif isinstance(ckpt, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:

            try:
                model = ckpt
                model.to(device).eval()
                if verbose:
                    print(f"[load_checkpoint_flex] Loaded whole model object from {ckpt_path}")
                return model
            except Exception as e:
                raise RuntimeError(
                    f"Unrecognized checkpoint structure. Top-level keys: {list(ckpt.keys())}"
                ) from e
    else:

        try:
            model = ckpt
            model.to(device).eval()
            if verbose:
                print(f"[load_checkpoint_flex] Loaded whole model object from {ckpt_path}")
            return model
        except Exception as e:
            raise RuntimeError("Unsupported checkpoint format") from e


    fixed_sd = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith('module.') else k
        fixed_sd[new_k] = v

    missing, unexpected = model.load_state_dict(fixed_sd, strict=strict)
    model.to(device).eval()

    if verbose:
        print(f"[load_checkpoint_flex] Loaded weights from: {ckpt_path}")
        if missing:
            print("[load_checkpoint_flex] MISSING keys:", missing)
        if unexpected:
            print("[load_checkpoint_flex] UNEXPECTED keys:", unexpected)

    return model

model = load_checkpoint_flex(
    model,
    "../version2_data/new_saved_model/best_atten_multi_model_epoch8_loss0.1939_acc0.9654.pth",
    device=device,
    strict=False,
    verbose=True
)


export_heatmaps(
    model,
    "../Dataset/predict/FakeVideo-FakeAudio_Caucasian (European)_men_id01157_00048_id00554_crjq1dFN1Ko_faceswap_id01058_wavtolip.mp4",
    out_dir="No_heatmaps_0.55",
    num_frames=16,
    display_size=(400,400),
    alpha=0.55
)