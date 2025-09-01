import threading
from doctest import master
from tkinter import ttk, StringVar, DoubleVar, Menu
import os
import io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from tkinter import (
    ttk,
    Tk,
    Button,
    Label,
    Checkbutton,
    BooleanVar,
    filedialog,
    messagebox,
    Frame
)
from PIL import Image, ImageTk

from model.atten_CRNN_Single import atten_CRNN_Single
from model.atten_CRNN_multi import atten_CRNN_multi
from model.CRNN_without import CRNN_without
from video_reader import video_reader
from scipy.ndimage import gaussian_filter




class AttentionVisualizer:
    def tensor_to_images(attn: torch.Tensor):
        a = attn.detach().cpu()
        print(f"[Debug] attention tensor ndim={a.ndim}, shape={tuple(a.shape)}")
        images = []

        if a.ndim == 4 and a.shape[1] == 1 and a.shape[0] > 1:
            seq = a.squeeze(1)
            for t in range(seq.shape[0]):
                mat = seq[t].numpy()
                mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)

                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                # ax.imshow(mat, cmap='jet')
                ax.imshow(mat, cmap='jet', interpolation='bilinear')
                ax.axis('off')
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8) \
                    .reshape(h, w, 4)
                buf = buf_argb[:, :, 1:4]
                plt.close(fig)
                images.append(Image.fromarray(buf))
            return images

        if a.ndim == 3:
            a = a.unsqueeze(0)  # now [1, L, H, W]

        if a.ndim == 2:
            vec = a[0].numpy()
            vec = vec / (vec.sum() + 1e-6)

            fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
            ax.bar(np.arange(len(vec)), vec)
            ax.set_title("Temporal Attention")
            ax.set_xlabel("Frame Index")
            fig.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            buf = buf_argb[:, :, 1:4]
            plt.close(fig)
            images.append(Image.fromarray(buf))
            return images

        # Spatial attention to  sequence of heatmaps
        if a.ndim in (4, 5):
            spa = a.squeeze(2) if a.ndim == 5 else a  # ensure [1, L, H, W]
            L = spa.shape[1]
            for t in range(L):
                mat = spa[0, t].numpy()
                mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)

                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                ax.imshow(mat, cmap='jet')
                ax.axis('off')
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
                buf = buf_argb[:, :, 1:4]
                plt.close(fig)
                images.append(Image.fromarray(buf))
            return images

        raise RuntimeError(f"Unsupported attention tensor shape: ndim={a.ndim}, shape={tuple(a.shape)}")

    def audio_attention_image(mel: torch.Tensor, t_attn: torch.Tensor):

        a = t_attn.squeeze().cpu().numpy()
        print(f"[Debug] attention tensor ndim={a.ndim}, shape={tuple(a.shape)}")
        if np.ndim(a) == 0:
            a = np.array([float(a)])
        a = a / (a.sum() + 1e-6)

        time_steps = np.arange(len(a))
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(time_steps, a, linewidth=2)
        ax.fill_between(time_steps, 0, a, alpha=0.3)
        ax.set_title("Temporal Attention")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attention Weight")
        ax.set_ylim(0, a.max() * 1.1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)


    def mel_spectrogram_image(mel: torch.Tensor):

        m = mel.squeeze().cpu().numpy()
        if m.ndim == 2:

            m = m.T
        else:
            m = m[0].T if m.ndim == 3 else m.T


        try:

            S_db = librosa.power_to_db(m, ref=np.max)
        except ImportError:
            S_db = 10 * np.log10(np.maximum(m, 1e-10))

        S_db = gaussian_filter(S_db, sigma=1.5)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

        ax.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')

        ax.set_title("Log-Mel Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Bins")


        fig.tight_layout()

        # PIL
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)




# build inital model
def build_models(device: torch.device):
    m1 = atten_CRNN_Single(model_depth=50, num_classes=2, fusion_method='concat').to(device)
    m2 = CRNN_without(model_depth=50, num_classes=2, fusion_method='concat', pooling_method='avg').to(device)
    m3 = atten_CRNN_multi(model_depth=50, num_classes=2, fusion_method='concat').to(device)
    return {'single_atten': m1, 'CRNN_without_atten': m2, 'multi_atten': m3}


def load_data(path: str, device: torch.device):
    reader = video_reader(device=device, video_frames=16, sample_rate=16000, n_mels=64,
                          return_indices=True, use_audio=True)

    out = reader.load_media(path)
    # video, mel, meta
    if len(out) == 3:
        video, mel, meta = out
        print("video_idx:", meta.get('video_idx'))
        print("mel_idx:", meta.get('mel_idx'))
    else:
        video, mel = out

    if mel is not None:
        mel = mel.unsqueeze(0)
    if video is not None:
        video = video.unsqueeze(0)
    return video, mel




def predict(model, media_path: str, device: torch.device):

    video, mel = load_data(media_path, device)
    model.eval()

    with torch.no_grad():
        if mel is None and video is not None:
            outputs = model.forward_video(video)
            return_mel = None

        elif video is None and mel is not None:
            outputs = model.forward_audio(mel)
            return_mel = mel

        else:
            outputs = model(video, mel)
            return_mel = mel


        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]


            if len(outputs) >= 4 and outputs[2] is not None:
                context = outputs[1]
                t_attn = outputs[2]  # [B, L]
                s_attn = outputs[3]  # v_spa 或 (v_spa, a_spa)

            elif len(outputs) >= 3:
                t_attn = outputs[1]
                s_attn = outputs[2]
            else:
                t_attn = s_attn = None
        else:
            logits, t_attn, s_attn = outputs, None, None

        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred = int(probs.argmax())


    if return_mel is not None:
        return pred, probs, t_attn, s_attn, return_mel
    else:
        return pred, probs, t_attn, s_attn


class PredictApp:
    DISPLAY_SIZE = (400, 400)

    def __init__(self, master):
        self.master = master
        master.title("Media Classification")
        master.geometry("950x950")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = build_models(self.device)
        self.selected_key = None
        self.selected_model = None
        self.ckpt_path = ''
        self.media_path = ''
        self.orig_images = []
        self.attn_images = []
        self.current_idx = 0


        self.status_var = StringVar(value="Ready")
        try:
            self.alpha_var = DoubleVar(value=0.5)
            self.frame_index_var = DoubleVar(value=0)
        except Exception:
            from tkinter import DoubleVar as _DV
            self.alpha_var = _DV(value=0.5)
            self.frame_index_var = _DV(value=0)

        # Progress bar (indeterminate) and status
        try:
            self.progress = ttk.Progressbar(master, mode="indeterminate")
            self.progress.pack(fill='x', padx=20, pady=6)
        except Exception:
            pass

        # Bottom controls: Prev/Next + Frame slider + Alpha slider
        bottom = Frame(master)
        bottom.pack(fill='x', padx=10, pady=6)

        Button(bottom, text="⟨ Prev", command=self.show_prev).pack(side='left', padx=4)
        Button(bottom, text="Next ⟩", command=self.show_next).pack(side='left', padx=4)

        try:
            Label(bottom, text="Frame").pack(side='left', padx=(10,4))
            self.frame_scale = ttk.Scale(bottom, from_=0, to=0, variable=self.frame_index_var, command=lambda _: self.on_seek_frame())
            self.frame_scale.pack(side='left', fill='x', expand=True, padx=6)

            Label(bottom, text="Alpha").pack(side='left', padx=(10,4))
            self.alpha_scale = ttk.Scale(bottom, from_=0.0, to=1.0, variable=self.alpha_var, command=lambda _: self.on_alpha_change())
            self.alpha_scale.pack(side='left', fill='x', expand=False, padx=6, ipadx=40)
        except Exception:
            pass

        # Menu (File/Open/Save/Export)
        try:
            menubar = Menu(master)
            filem = Menu(menubar, tearoff=0)
            filem.add_command(label="Open Media", command=self.load_media, accelerator="Ctrl+O")
            filem.add_command(label="Load Weights", command=self.load_ckpt, accelerator="Ctrl+L")
            filem.add_separator()
            filem.add_command(label="Save Current", command=self.save_current, accelerator="Ctrl+S")
            filem.add_command(label="Export All Frames...", command=self.export_all_frames)
            filem.add_separator()
            filem.add_command(label="Exit", command=master.quit)
            menubar.add_cascade(label="File", menu=filem)
            master.config(menu=menubar)

            master.bind("<Control-o>", lambda e: self.load_media())
            master.bind("<Control-l>", lambda e: self.load_ckpt())
            master.bind("<Control-s>", lambda e: self.save_current())
            master.bind("<Left>", lambda e: self.show_prev())
            master.bind("<Right>", lambda e: self.show_next())
            master.bind("<space>", lambda e: self.toggle_attn())
        except Exception:
            pass


        # Model and checkpoint controls
        ctrl_frame = Frame(master)
        ctrl_frame.pack(pady=10)

        Button(ctrl_frame, text="Use CRNN Model", command=lambda: self.select_model('CRNN_without_atten')).pack(side='left', padx=5)
        Button(ctrl_frame, text="Use Atten Model", command=lambda: self.select_model('single_atten')).pack(side='left', padx=5)
        Button(ctrl_frame, text="Use Multi atten Model", command=lambda: self.select_model('multi_atten')).pack(side='left', padx=5)
        Button(ctrl_frame, text="Load trained model", command=self.load_ckpt).pack(side='left', padx=20)
        self.model_label = Label(master, text="No model selected", fg='gray')
        self.model_label.pack()
        self.ckpt_label = Label(master, text="No checkpoint loaded", fg='gray')
        self.ckpt_label.pack()

        # Media loading
        Button(master, text="Load Media", command=self.load_media).pack(fill='x', padx=20, pady=5)
        self.media_label = Label(master, text="No media selected", fg='gray')
        self.media_label.pack(fill='x', padx=20)

        # Options and predict
        self.show_attn = BooleanVar(master, value=False)
        Checkbutton(master, text="Show Attention", variable=self.show_attn).pack(pady=5)
        Button(master, text="Predict", command=self.start_predict_thread).pack(fill='x', padx=20, pady=10)
        self.result_label = Label(master, text='', justify='left', font=(None,12))
        self.result_label.pack(fill='x', padx=20)

        # Image display
        img_frame = Frame(master)
        img_frame.pack(fill='both', expand=True)
        orig_frame = Frame(img_frame)
        orig_frame.pack(side='left', padx=10, pady=10)
        Label(orig_frame, text="Original").pack()
        self.orig_label = Label(orig_frame)
        self.orig_label.pack()
        attn_frame = Frame(img_frame)
        attn_frame.pack(side='right', padx=10, pady=10)
        Label(attn_frame, text="Attention").pack()
        self.attn_label = Label(attn_frame)
        self.attn_label.pack()

        nav = Frame(master)
        nav.pack(pady=5)
        Button(nav, text="< Prev", command=self.show_prev).pack(side='left', padx=20)
        Button(nav, text="Next >", command=self.show_next).pack(side='right', padx=20)

    def select_model(self, key):
        self.selected_key = key
        self.selected_model = self.models[key]
        self.model_label.config(text=f"Selected Model: {key}", fg='black')
        # Reset checkpoint label
        self.ckpt_label.config(text="No checkpoint loaded for this model", fg='gray')
        self.ckpt_path = ''

    def load_ckpt(self):
        if not self.selected_model:
            messagebox.showwarning('No Model', 'Please select a model first.')
            return
        path = filedialog.askopenfilename(
            title="Select checkpoint (.pth)",
            filetypes=[('Checkpoint','*.pth *.pt'),('All files','*.*')]
        )
        if path:
            self.ckpt_path = path
            self.ckpt_label.config(text=os.path.basename(path), fg='black')
            # Load state dict into selected model
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            self.selected_model.load_state_dict(state, strict=False)
            self.selected_model.eval()

    def load_media(self):
        path = filedialog.askopenfilename(
            title="Select media file",
            filetypes=[('Media','*.mp4 *.avi *.wav *.mp3'),('All files','*.*')]
        )
        if path:
            self.media_path = path
            self.media_label.config(text=os.path.basename(path), fg='black')

    def run_predict(self):

        if not self.selected_model or not self.ckpt_path or not self.media_path:
            messagebox.showwarning('Missing Input', 'please select model, checkpoint and media files first.。')
            return

        ext = os.path.splitext(self.media_path)[1].lower()
        is_audio = ext in ('.wav', '.mp3', '.flac', '.m4a')

        self.orig_images = []
        self.attn_images = []

        try:
            predict_result = predict(self.selected_model, self.media_path, self.device)


            if len(predict_result) == 5:
                pred, probs, t_attn, s_attn, mel = predict_result

            else:
                pred, probs, t_attn, s_attn = predict_result
                mel = None


            label_map = {0: 'fake', 1: 'true'}
            self.result_label.config(
                text=f"Predicted: {label_map.get(pred, 'unknown')} ({pred})\n"
                     f"Probs: {', '.join(f'{p:.3f}' for p in probs)}"
            )

            if is_audio:
                self.audio_visualization(mel, t_attn, s_attn)
            else:
                self.video_visualization(t_attn, s_attn)

        except Exception as e:
            messagebox.showerror('Prediction Error', f'error happen：{str(e)}')
            print(f"error details: {e}")
            return

        self.current_idx = 0

        # --- Added: UI vars and widgets ---
        self.status_var = StringVar(value="Ready")
        try:
            self.alpha_var = DoubleVar(value=0.5)
            self.frame_index_var = DoubleVar(value=0)
        except Exception:
            from tkinter import DoubleVar as _DV
            self.alpha_var = _DV(value=0.5)
            self.frame_index_var = _DV(value=0)

        # Progress bar (indeterminate) and status
        try:
            self.progress = ttk.Progressbar(master, mode="indeterminate")
            self.progress.pack(fill='x', padx=20, pady=6)
        except Exception:
            pass

        # Bottom controls: Prev/Next + Frame slider + Alpha slider
        bottom = Frame(master)
        bottom.pack(fill='x', padx=10, pady=6)

        Button(bottom, text="⟨ Prev", command=self.show_prev).pack(side='left', padx=4)
        Button(bottom, text="Next ⟩", command=self.show_next).pack(side='left', padx=4)

        try:
            Label(bottom, text="Frame").pack(side='left', padx=(10,4))
            self.frame_scale = ttk.Scale(bottom, from_=0, to=0, variable=self.frame_index_var, command=lambda _: self.on_seek_frame())
            self.frame_scale.pack(side='left', fill='x', expand=True, padx=6)

            Label(bottom, text="Alpha").pack(side='left', padx=(10,4))
            self.alpha_scale = ttk.Scale(bottom, from_=0.0, to=1.0, variable=self.alpha_var, command=lambda _: self.on_alpha_change())
            self.alpha_scale.pack(side='left', fill='x', expand=False, padx=6, ipadx=40)
        except Exception:
            pass

        # Menu (File/Open/Save/Export)
        try:
            menubar = Menu(master)
            filem = Menu(menubar, tearoff=0)
            filem.add_command(label="Open Media", command=self.load_media, accelerator="Ctrl+O")
            filem.add_command(label="Load Weights", command=self.load_ckpt, accelerator="Ctrl+L")
            filem.add_separator()
            filem.add_command(label="Save Current", command=self.save_current, accelerator="Ctrl+S")
            filem.add_command(label="Export All Frames...", command=self.export_all_frames)
            filem.add_separator()
            filem.add_command(label="Exit", command=master.quit)
            menubar.add_cascade(label="File", menu=filem)
            master.config(menu=menubar)

            master.bind("<Control-o>", lambda e: self.load_media())
            master.bind("<Control-l>", lambda e: self.load_ckpt())
            master.bind("<Control-s>", lambda e: self.save_current())
            master.bind("<Left>", lambda e: self.show_prev())
            master.bind("<Right>", lambda e: self.show_next())
            master.bind("<space>", lambda e: self.toggle_attn())
        except Exception:
            pass

        self.display_images()

    def audio_visualization(self, mel, t_attn, s_attn):
        try:
            if mel is None:
                raise ValueError("can't get the mel Spectrogram")

            orig_img = AttentionVisualizer.mel_spectrogram_image(mel)

            if self.show_attn.get() and t_attn is not None:
                attn_img = AttentionVisualizer.audio_attention_image(mel, t_attn)
            else:
                attn_img = Image.new('RGB', orig_img.size, color=(128, 128, 128))


            orig_img = orig_img.resize(self.DISPLAY_SIZE, Image.LANCZOS)
            attn_img = attn_img.resize(self.DISPLAY_SIZE, Image.LANCZOS)

            self.orig_images = [orig_img]
            self.attn_images = [attn_img]

        except Exception as e:
            print(f"visualization failed: {e}")
            placeholder = Image.new('RGB', self.DISPLAY_SIZE, color=(128, 128, 128))
            self.orig_images = [placeholder]
            self.attn_images = [placeholder]

    def video_visualization(self, t_attn, s_attn):
        try:
            reader = video_reader(device=self.device, video_frames=16, sample_rate=16000, n_mels=64,
                                  return_indices=True, use_audio=True)
            out = reader.load_media(self.media_path)

            if isinstance(out, (list, tuple)) and len(out) == 3:
                orig_vid, _mel, _meta = out
            else:
                orig_vid, _mel = out

            if isinstance(orig_vid, torch.Tensor) and orig_vid.ndim == 5:  # [B,3,T,H,W]
                orig_vid = orig_vid[0]

            if orig_vid is not None:
                arr = orig_vid.permute(1, 0, 2, 3).cpu().numpy()  # [T, C, H, W]
                self.orig_images = [
                    Image.fromarray((f * 255).astype(np.uint8).transpose(1, 2, 0)).resize(self.DISPLAY_SIZE,
                                                                                          Image.LANCZOS)
                    for f in arr
                ]
            else:
                raise ValueError("can't load")

            if self.show_attn.get():
                self.generate_attention_heatmaps(t_attn, s_attn)
            else:
                self.attn_images = self.orig_images.copy()

        except Exception as e:
            print(f"video process error: {e}")
            placeholder = Image.new('RGB', self.DISPLAY_SIZE, color=(128, 128, 128))
            self.orig_images = [placeholder]
            self.attn_images = [placeholder]

    def normalize_spatial_map(self, s_attn):

        attn = s_attn[0] if isinstance(s_attn, (tuple, list)) else s_attn
        if not isinstance(attn, torch.Tensor):
            return None

        a = attn.detach().cpu()
        if a.ndim == 4 and a.shape[1] == 1:  # [N,1,h,w], N 可能是 T 或 B*T
            T = len(self.orig_images)
            if a.shape[0] >= T:
                return a[:T]
            last = a[-1:].repeat(T - a.shape[0], 1, 1, 1)
            return torch.cat([a, last], dim=0)

        if a.ndim == 4 and a.shape[0] == 1:  # [1,L,H,W] -> [L,1,H,W]
            a = a.squeeze(0).unsqueeze(1)
            return a

        if a.ndim == 5 and a.shape[0] == 1:  # [1,L,1,H,W] -> [L,1,H,W]
            return a.squeeze(0)

        return None

    def generate_attention_heatmaps(self, t_attn, s_attn):
        if s_attn is not None:
            s_attn = self.normalize_spatial_map(s_attn)

        self.last_t_attn = t_attn
        self.last_s_attn = s_attn
        attn = None

        if s_attn is not None:
            if isinstance(s_attn, tuple):
                v_map, a_map = s_attn
                attn = v_map if v_map is not None else a_map
            else:
                attn = s_attn
        elif t_attn is not None:
            attn = t_attn

        if attn is not None:
            try:
                heatmaps = AttentionVisualizer.tensor_to_images(attn)
                blended = []

                for idx, hm in enumerate(heatmaps):
                    hm_rgba = hm.resize(self.DISPLAY_SIZE, Image.LANCZOS).convert('RGBA')
                    orig_rgba = self.orig_images[idx].convert('RGBA')

                    merged = Image.blend(orig_rgba, hm_rgba, alpha=float(self.alpha_var.get()))
                    blended.append(merged.convert('RGB'))

                self.attn_images = blended


                self.sync_attention_images()

            except Exception as e:
                print(f"attention heatmap error: {e}")
                self.attn_images = self.orig_images.copy()
        else:
            self.attn_images = self.orig_images.copy()

    def sync_attention_images(self):
        if not self.orig_images or not self.attn_images:
            return

        orig_len = len(self.orig_images)
        attn_len = len(self.attn_images)

        if orig_len == attn_len:
            return
        elif attn_len == 1:
            self.attn_images = [self.attn_images[0]] * orig_len
        elif attn_len > orig_len:
            self.attn_images = self.attn_images[:orig_len]
        else:
            last_img = self.attn_images[-1]
            while len(self.attn_images) < orig_len:
                self.attn_images.append(last_img)

    def display_images(self):
        if self.orig_images:
            o = ImageTk.PhotoImage(self.orig_images[self.current_idx])
            self.orig_label.config(image=o)
            self.orig_label.image = o
        if self.attn_images:
            a = ImageTk.PhotoImage(self.attn_images[self.current_idx])
            self.attn_label.config(image=a)
            self.attn_label.image = a
        else:
            self.attn_label.config(image='')


    def start_predict_thread(self):
        if not self.selected_model or not self.ckpt_path or not self.media_path:
            messagebox.showwarning('Missing Input', 'please select model, checkpoint and media files first。')
            return
        self.set_busy(True, "Running inference...")
        t = threading.Thread(target=self.predict_worker, daemon=True)
        t.start()

    def predict_worker(self):
        try:
            # Heavy part: run model.predict only; rest will be done on main thread
            result = predict(self.selected_model, self.media_path, self.device)
            self._predict_result = result
            self.master.after(0, self.after_predict_success)
        except Exception as e:
            self._predict_error = e
            self.master.after(0, self.after_predict_fail)

    def after_predict_success(self):
        try:
            ext = os.path.splitext(self.media_path)[1].lower()
            is_audio = ext in ('.wav', '.mp3', '.flac', '.m4a')

            pr = self._predict_result
            if pr is None:
                raise RuntimeError("Empty predict result")

            # Unpack with optional mel
            if len(pr) == 5:
                pred, probs, t_attn, s_attn, mel = pr
            else:
                pred, probs, t_attn, s_attn = pr
                mel = None

            label_map = {0: 'fake', 1: 'true'}
            try:
                probs_list = [float(x) for x in (probs.tolist() if hasattr(probs, 'tolist') else probs)]
            except Exception:
                probs_list = list(probs)
            self.result_label.config(
                text=f"Predicted: {label_map.get(int(pred), 'unknown')} ({int(pred)})\n"
                     f"Probs: {', '.join(f'{p:.3f}' for p in probs_list)}"
            )

            # Use original visualization functions to preserve appearance
            if is_audio:
                self.audio_visualization(mel, t_attn, s_attn)
            else:
                self.video_visualization(t_attn, s_attn)

            # Update frame slider
            try:
                n = max(0, len(self.orig_images) - 1)
                if hasattr(self, 'frame_scale'):
                    self.frame_scale.configure(to=n)
                self.frame_index_var.set(0)
            except Exception:
                pass

            self.current_idx = 0
            self.display_images()
        finally:
            self.set_busy(False, "Done.")

    def after_predict_fail(self):
        messagebox.showerror('Prediction Error', f'error happen：{str(self._predict_error)}')
        self.set_busy(False, "Failed.")

    def set_busy(self, busy: bool, msg: str):
        try:
            self.status_var.set(msg)
        except Exception:
            pass
        try:
            # disable predict button if we can find it (best-effort)
            # (If you give the button a name, you can reference it directly.)
            pass
        except Exception:
            pass
        if hasattr(self, 'progress'):
            if busy:
                try:
                    self.progress.start(50)
                except Exception:
                    pass
            else:
                try:
                    self.progress.stop()
                except Exception:
                    pass

    # --- Added: Frame seek & alpha change handlers ---
    def on_seek_frame(self):
        if not self.orig_images:
            return
        try:
            idx = int(self.frame_index_var.get())
        except Exception:
            idx = self.current_idx
        self.current_idx = max(0, min(idx, len(self.orig_images) - 1))
        self.display_images()

    def on_alpha_change(self):
        # Re-generate attention overlays with new alpha
        try:
            # Try to reuse last attention tensors if available
            # If not, fall back to current attn_images
            if hasattr(self, '_last_t_attn') and hasattr(self, 'last_s_attn') and self.show_attn.get():
                self.generate_attention_heatmaps(getattr(self, 'last_t_attn', None),
                                                 getattr(self, 'last_s_attn', None))
                self.display_images()
        except Exception:
            pass

    def toggle_attn(self):
        self.show_attn.set(not self.show_attn.get())
        self.on_toggle_attn()

    def on_toggle_attn(self):
        # If attention is already generated, just refresh
        if self.attn_images:
            self.display_images()

    # --- Added: Save / Export helpers ---
    def save_current(self):
        if not self.orig_images:
            messagebox.showinfo('Save', 'Nothing to save yet.')
            return
        idx = self.current_idx
        base, _ = os.path.splitext(os.path.basename(self.media_path or 'frame'))
        default = f"{base}_frame{idx:03d}_{'attn' if (self.show_attn.get() and self.attn_images) else 'orig'}.png"
        fp = filedialog.asksaveasfilename(
            title='Save current frame',
            defaultextension='.png',
            initialfile=default,
            filetypes=[('PNG Image', '*.png'), ('JPEG Image', '*.jpg;*.jpeg'), ('All files', '*.*')]
        )
        if not fp:
            return
        img = (self.attn_images[idx] if (self.show_attn.get() and self.attn_images) else self.orig_images[idx])
        try:
            img.save(fp)
        except Exception as e:
            messagebox.showerror('Save Error', str(e))

    def export_all_frames(self):
        if not self.orig_images:
            messagebox.showinfo('Export', 'Nothing to export yet.')
            return
        outdir = filedialog.askdirectory(title='Select export folder')
        if not outdir:
            return
        use_attn = self.show_attn.get() and self.attn_images
        base = os.path.splitext(os.path.basename(self.media_path or 'frames'))[0]
        try:
            for i in range(len(self.orig_images)):
                img = self.attn_images[i] if use_attn else self.orig_images[i]
                name = f"{base}_{'attn' if use_attn else 'orig'}_{i:03d}.png"
                img.save(os.path.join(outdir, name))
        except Exception as e:
            messagebox.showerror('Export Error', str(e))
    def show_prev(self):
        if not self.orig_images:
            return
        self.current_idx = (self.current_idx - 1) % len(self.orig_images)
        self.display_images()

    def show_next(self):
        if not self.orig_images:
            return
        self.current_idx = (self.current_idx + 1) % len(self.orig_images)
        self.display_images()


def main():
    root = Tk()
    PredictApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
