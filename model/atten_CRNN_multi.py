from torchvision.models import (resnet50, resnet18, resnet34, resnet101, resnet152,
                                ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights)
import torch
from pytorch_tcn import TCN
from model.attention_block import (
    masked_MHAFFN, additive_temporal_attention, spatial_attention2D, lengths_to_mask
)
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['atten_CRNN_multi']


class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 1x1 conv to generate spatial attention map
        self.conv1 = torch.nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn_map = self.conv1(x)             # [B, 1, H, W]
        B, _, H, W = attn_map.shape
        attn_flat = attn_map.view(B, -1)     # [B, H*W]
        attn_norm = torch.nn.functional.softmax(attn_flat, dim=1)
        attn_map = attn_norm.view(B, 1, H, W)
        # weighted pooling
        out = (attn_map * x).sum(dim=[2, 3]) # [B, C]
        return out, attn_map


class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = torch.nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, encoder_outputs):
        B, T, D = encoder_outputs.size()
        attn_energies = torch.bmm(
            encoder_outputs,                  # [B, T, D]
            self.attn_weights.unsqueeze(0).expand(B, -1, -1)  # [B, D, 1]
        ).squeeze(2)                          # [B, T]
        attn_weights = torch.nn.functional.softmax(attn_energies, dim=1)
        context = torch.bmm(
            attn_weights.unsqueeze(1),        # [B,1,T]
            encoder_outputs                   # [B,T,D]
        ).squeeze(1)                         # [B, D]
        return context, attn_weights


class model_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3, bidirectional=True):
        super(model_encoder, self).__init__()
        self.bi_GRU = torch.nn.GRU(input_dim,
                                   hidden_dim,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   batch_first=True,
                                   dropout=dropout)
        self.tcn = TCN(num_inputs=hidden_dim*2,
                       num_channels=[hidden_dim*2, hidden_dim*2],
                       kernel_size=3,
                       dropout=dropout)

    def forward(self, x):
        # x: [B, L, input_dim]
        x, _ = self.bi_GRU(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return x


class atten_CRNN_multi(torch.nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 model_depth: int = 101,  # choose the model depth
                 fusion_method: str = "concat",
                 nhead: int = 4):
        super().__init__()
        self.fusion = fusion_method
        self.num_classes = num_classes

        resnet_dict = {
            18: (resnet18, ResNet18_Weights.DEFAULT, 512),
            34: (resnet34, ResNet34_Weights.DEFAULT, 512),
            50: (resnet50, ResNet50_Weights.DEFAULT, 2048),
            101: (resnet101, ResNet101_Weights.DEFAULT, 2048),
            152: (resnet152, ResNet152_Weights.DEFAULT, 2048)
        }
        assert model_depth in resnet_dict, f"Unsupported depth: {model_depth}"
        resnet_fn, weights, resnet_dim = resnet_dict[model_depth]

        base = resnet_fn(weights=weights)

        # self.backbone = nn.Sequential(
        #     base.conv1, base.bn1, base.relu, base.maxpool,
        #     base.layer1, base.layer2, base.layer3, base.layer4
        # )
        base_v = resnet_fn(weights=weights)
        base_a = resnet_fn(weights=weights)

        self.v_backbone = nn.Sequential(
            base_v.conv1, base_v.bn1, base_v.relu, base_v.maxpool,
            base_v.layer1, base_v.layer2, base_v.layer3, base_v.layer4
        )
        self.a_backbone = nn.Sequential(
            base_a.conv1, base_a.bn1, base_a.relu, base_a.maxpool,
            base_a.layer1, base_a.layer2, base_a.layer3, base_a.layer4
        )

        # do attention pooing on the convolutional feature
        self.video_spatial_attn = spatial_attention2D(in_channels=resnet_dim)
        self.audio_spatial_attn = spatial_attention2D(in_channels=resnet_dim)

        # time encoder to the same size
        self.video_encoder = model_encoder(input_dim=resnet_dim)
        self.audio_encoder = model_encoder(input_dim=resnet_dim)

        # Optional: Attention during cross-modal fusion (only used when fusion_method=“attn”)
        self.cross_modal_mha = nn.MultiheadAttention(
            embed_dim=512, num_heads=nhead, batch_first=True
        )

        # the dimention after fusion
        self.fuse_dim = 512 * 2 if self.fusion in ("concat", "cat") else 512

        # Residual + Normalized MHA + FFN Stabilization Block (Time Dimension)
        self.mha_block = masked_MHAFFN(d_model=self.fuse_dim, nhead=nhead, dropout=0.2)

        # Temporal Attention Pooling
        self.temporal_attn = additive_temporal_attention(d_model=self.fuse_dim, hidden=128, dropout=0.2)


        # Two-layer MLP classification head
        hidden = max(self.fuse_dim // 2, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fuse_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, video: torch.Tensor = None, audio: torch.Tensor = None, lengths: torch.Tensor = None):

        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided.")

        # get the batch size and device
        if video is not None:
            B = video.size(0)
            device = video.device
        else:
            B = audio.size(0)
            device = audio.device

        # initialize device
        v_seq = None
        a_seq = None
        v_attn_map = None
        a_attn_map = None

        # video part process
        if video is not None:
            _, _, T, H, W = video.shape
            # reshape as [B*T, 3, H, W]
            v_in = video.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
            # use the share backbone
            # fmap_v = self.backbone(v_in)  # [B*T, resnet_dim, h', w']
            fmap_v = self.v_backbone(v_in)

            # spatial Attention Pooling
            v_vec, v_attn_map = self.video_spatial_attn(fmap_v)  # [B*T, resnet_dim]

            # reshape back to sequence form and pass through the temporal encoder
            v_seq = v_vec.view(B, T, -1)  # [B, T, resnet_dim]
            v_seq = self.video_encoder(v_seq)  # [B, T, 512]


        #audio part process
        if audio is not None:
            if audio.dim() == 4:
                B_a, C_a, T_a, F_a = audio.shape
                # [B,3,T,n_mel]  to [B,3,n_mel,T] to  [B,3,224,224]
                a_img = audio.permute(0, 1, 3, 2)  # [B, 3, n_mel, T]
                a_img = F.interpolate(a_img, size=(224, 224), mode='bilinear', align_corners=False)
                fmap_a = self.a_backbone(a_img)
                a_vec, a_attn_map = self.audio_spatial_attn(fmap_a)
                T = v_seq.size(1) if (video is not None) else T_a
                a_seq = a_vec.unsqueeze(1).expand(-1, T, -1)
                a_seq = self.audio_encoder(a_seq)
            elif audio.dim() == 5:
                # [B,3,T,H,W]
                B_a, C_a, T_a, H_a, W_a = audio.shape
                a_in = audio.permute(0, 2, 1, 3, 4).reshape(B_a * T_a, C_a, H_a, W_a)
                fmap_a = self.a_backbone(a_in)
                a_vec, a_attn_map = self.audio_spatial_attn(fmap_a)
                a_seq = a_vec.view(B_a, T_a, -1)
                a_seq = self.audio_encoder(a_seq)
            else:
                raise ValueError(f"Unsupported audio shape: {audio.shape}")


        # feature fusion
        H_seq = self.fusion_features(v_seq, a_seq)  # [B, L, fuse_dim]

        # padding mask
        if lengths is None:
            key_padding_mask = torch.zeros(B, H_seq.size(1), dtype=torch.bool, device=device)
        else:
            key_padding_mask = lengths_to_mask(lengths.to(device), max_len=H_seq.size(1))

        # Multi-head attention residual block
        H_seq = self.mha_block(H_seq, key_padding_mask=key_padding_mask)  # [B, L, fuse_dim]

        # Temporal Attention Pooling
        context, attn = self.temporal_attn(H_seq, key_padding_mask=key_padding_mask)  # [B, fuse_dim]

        # classifier
        logits = self.classifier(context)  # [B, num_classes]

        # return the context as a domain alignment feature
        return logits, context, attn, (v_attn_map, a_attn_map), key_padding_mask

    def fusion_features(self, v_seq, a_seq) -> torch.Tensor:
        # print(f"[DEBUG] v_seq shape: {None if v_seq is None else v_seq.shape}")
        # print(f"[DEBUG] a_seq shape: {None if a_seq is None else a_seq.shape}")
        if v_seq is None and a_seq is None:
            raise ValueError("Both streams are None.")

        if v_seq is None:
            v_seq = torch.zeros_like(a_seq)
        if a_seq is None:
            a_seq = torch.zeros_like(v_seq)

        # 对齐长度
        Lv, La = v_seq.size(1), a_seq.size(1)
        if Lv != La:
            if Lv == 1:
                v_seq = v_seq.expand(-1, La, -1)
            elif La == 1:
                a_seq = a_seq.expand(-1, Lv, -1)
            else:
                raise ValueError(f"Length mismatch: {Lv} vs {La}")

        if self.fusion in ("concat", "cat"):
            return torch.cat([v_seq, a_seq], dim=-1)  # [B, L, 1024]
        elif self.fusion == "add":
            return v_seq + a_seq  # [B, L, 512]
        else:
            # Cross-modal attention fusion (query=v, key/value=a), then averaged with the reverse (q=a, k=v)
            # 跨模态注意力融合（query=v，key/value=a），再与反向（q=a,k=v）平均
            v2, _ = self.cross_modal_mha(v_seq, a_seq, a_seq)  # [B, L, 512]
            a2, _ = self.cross_modal_mha(a_seq, v_seq, v_seq)  # [B, L, 512]
            return (v2 + a2) / 2.0


    def extract_video_seq(self, video: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T,3,H,W]
        fmap = self.backbone(x)  # [B*T,C',h',w']

        v_vec, _ = self.video_spatial_attn(fmap)  # [B*T,C']
        v_seq = v_vec.view(B, T, -1)  # [B,T,resnet_dim]
        return v_seq

    def extract_audio_seq(self, audio: torch.Tensor) -> torch.Tensor:

        if audio.dim() == 4:
            B, C, T, Fm = audio.shape
            x = audio.permute(0, 2, 1, 3).unsqueeze(-1)  # [B,T,3,n_mel,1]
            x = x.reshape(B * T, C, Fm, 1)  # [B*T,3,n_mel,1]
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        elif audio.dim() == 5:
            B, C, T, H, W = audio.shape
            x = audio.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        fmap = self.backbone(x)  # [B*T,C',h',w']
        a_vec, _ = self.audio_spatial_attn(fmap)  # [B*T,C']
        a_seq = a_vec.view(B, T, -1)  # [B,T,resnet_dim]
        return a_seq




    def forward_video(self, video):
        return self.forward(video=video, audio=None)

    def forward_audio(self, audio):
        return self.forward(video=None, audio=audio)

    def forward_video_audio(self, video, audio):
        return self.forward(video=video, audio=audio)



# if __name__ == "__main__":
#     model = MultiModal_CRNN_with_TCN(model_depth=50, num_classes=2, fusion_method='concat')
#
#     batch_size = 4
#     num_frame = 16
#
#     video = torch.randn(batch_size, 3, num_frame, 224, 224)
#     audio = torch.randn(batch_size, 3, num_frame, 224, 224)
#
#     print("v+a")
#     output1 = model(video=video, audio=audio)
#     print(f"output shape: {output1.shape}")
#     print("video")
#     output2 = model(video=video, audio=None)
#     print(f"output shape: {output2.shape}")
#
#     print("audio")
#     output3 = model(video=None, audio=audio)
#     print(f"output shape: {output3.shape}")
#
#     print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
#     print(f"\ntrain parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


