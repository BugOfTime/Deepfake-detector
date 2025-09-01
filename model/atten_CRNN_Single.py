from torchvision.models import (resnet50, resnet18, resnet34, resnet101, resnet152,
                                ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights)
import torch
import torch.nn.functional as F
from pytorch_tcn import TCN

__all__ = ['atten_CRNN_Single']


class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 1x1 conv to generate spatial attention map
        self.conv1 = torch.nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
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
        # print("input shape to model_encoder:", x.shape)
        # x: [B, L, input_dim]
        x, _ = self.bi_GRU(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return x


class atten_CRNN_Single(torch.nn.Module):
    def __init__(self, model_depth, num_classes, fusion_method='concat'):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # Setup ResNet backbone up to layer4
        resnet_dict = {
            18:  (resnet18,  ResNet18_Weights.DEFAULT,  512),
            34:  (resnet34,  ResNet34_Weights.DEFAULT,  512),
            50:  (resnet50,  ResNet50_Weights.DEFAULT, 2048),
            101: (resnet101, ResNet101_Weights.DEFAULT, 2048),
            152: (resnet152, ResNet152_Weights.DEFAULT, 2048),
        }
        assert model_depth in resnet_dict, f"Unsupported ResNet depth: {model_depth}"
        fn, weights, dim = resnet_dict[model_depth]
        base_v = fn(weights=weights)
        base_a = fn(weights=weights)
        self.v_backbone = torch.nn.Sequential(base_v.conv1, base_v.bn1, base_v.relu, base_v.maxpool,
                                              base_v.layer1, base_v.layer2, base_v.layer3, base_v.layer4)
        self.a_backbone = torch.nn.Sequential(base_a.conv1, base_a.bn1, base_a.relu, base_a.maxpool,
                                              base_a.layer1, base_a.layer2, base_a.layer3, base_a.layer4)

        self.v_global_pool = base_v.avgpool
        self.a_global_pool = base_a.avgpool
        self.dropout = torch.nn.Dropout(0.3)

        # spatial attention on conv features
        self.v_spatial_attn = SpatialAttention(dim)  # dim 与 backbone 输出通道一致
        self.a_spatial_attn = SpatialAttention(dim)

        # sequence encoders
        self.video_encoder = model_encoder(input_dim=dim)
        self.audio_encoder = model_encoder(input_dim=dim)

        # fusion dimension and optional attention
        if fusion_method == 'concat':
            self.fuse_dim = 512 * 2
        elif fusion_method == 'add':
            self.fuse_dim = 512
        elif fusion_method == 'attention':
            self.fuse_dim = 512
            self.fusion_attention = torch.nn.MultiheadAttention(
                embed_dim=512, num_heads=4, batch_first=True
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # temporal attention and classifier
        self.temporal_attn = Attention(self.fuse_dim)
        self.classifier_fc = torch.nn.Linear(self.fuse_dim, self.num_classes)

    def extract_video_features(self, video):
        # video: [B, C, L, H, W]
        B, C, L, H, W = video.shape
        x = video.transpose(1, 2).reshape(B*L, C, H, W)
        feat = self.v_backbone(x)                   # [B*L, D, H', W']
        seq, spa = self.v_spatial_attn(feat)
        seq = self.dropout(seq)
        seq = seq.view(B, L, -1)                  # [B, L, D]
        # spa = spa.view(B, L, 1, spa.size(2), spa.size(3))
        return seq, spa

    # def extract_audio_features(self, audio):
    #     print("[DEBUG] audio input:", audio.shape)
    #     if audio.dim() == 4:
    #         B, C, T_a, Fm = audio.shape
    #         a_img = audio.permute(0, 1, 3, 2)  # [B,3,n_mel,T]
    #         print("[DEBUG] a_img after permute:", a_img.shape)
    #         a_img = F.interpolate(a_img, size=(224, 224), mode='bilinear', align_corners=False)
    #         print("[DEBUG] a_img after interpolate:", a_img.shape)
    #         feat = self.a_backbone(a_img)  # [B, D, h', w']
    #         print("[DEBUG] feat:", feat.shape)
    #         a_vec, _ = self.a_spatial_attn(feat)  # [B, D]
    #         print("[DEBUG] a_vec after spatial_attn:", a_vec.shape)
    #         a_vec = self.dropout(a_vec)
    #         a_vec = a_vec.view(B, T_a, -1)  # <--- 必须用B、T_a和最后D！
    #         print("[DEBUG] a_vec after view:", a_vec.shape)
    #         return a_vec, None


    # def extract_audio_features(self, audio):
    #     if audio.dim() == 4:
    #     # [B, C, H, W] -> treat W as time L, H as mel bins
    #         B, C, H, W = audio.shape
    #     # permute so time is third dim, then add a dummy width=1
    #         audio = audio.permute(0, 1, 3, 2).reshape(B, C, W, H, 1)
    #     # now audio.shape == [B, C, L=W, H, W=1]
    #     B, C, L, H, W = audio.shape
    #     x = audio.transpose(1, 2).reshape(B * L, C, H, W)
    #     feat = self.a_backbone(x)  # [B*L, D, H', W']
    #     pooled = self.a_global_pool(feat).view(B*L, -1)
    #     seq = pooled.view(B, L, -1)               # [B, L, D]
    #     return seq
    def extract_audio_features(self, audio):

        if audio is None:
            return None, None

        if audio.dim() == 4:
            B, C, T, n_mel = audio.shape

            # [B, C, T, n_mel] -> [B, C, n_mel, T]
            a_img = audio.permute(0, 1, 3, 2)

            # get the same size
            a_img = F.interpolate(a_img, size=(224, 224), mode='bilinear', align_corners=False)

            feat = self.a_backbone(a_img)  # [B, D, h', w']

            a_vec, a_spa = self.a_spatial_attn(feat)  # [B, D], [B, 1, h', w']
            a_vec = self.dropout(a_vec)  # [B, D]

            a_seq = a_vec.unsqueeze(1).expand(-1, T, -1)

            return a_seq, a_spa

        raise ValueError(f"Unexpected audio dim: {audio.dim()}, expected 4D [B, C, T, n_mel].")


    def fusion_features(self, v_seq, a_seq):
        # handle missing streams
        if v_seq is None and a_seq is None:
            raise ValueError("Both streams missing")
        if v_seq is None:
            v_seq = torch.zeros_like(a_seq)
        if a_seq is None:
            a_seq = torch.zeros_like(v_seq)
        # align lengths
        Lv, La = v_seq.size(1), a_seq.size(1)
        if Lv != La:
            if Lv == 1:
                v_seq = v_seq.expand(-1, La, -1)
            elif La == 1:
                a_seq = a_seq.expand(-1, Lv, -1)
            else:
                # raise ValueError(f"Length mismatch: {Lv} vs {La}")
                a_seq = F.interpolate(
                    a_seq.permute(0, 2, 1),
                    size=Lv,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
                La = Lv
        # fuse
        if self.fusion_method == 'concat':
            return torch.cat([v_seq, a_seq], dim=2)
        if self.fusion_method == 'add':
            return v_seq + a_seq
        # attention fusion
        v2, _ = self.fusion_attention(v_seq, a_seq, a_seq)
        a2, _ = self.fusion_attention(a_seq, v_seq, v_seq)
        return (v2 + a2) / 2

    def forward(self, video=None, audio=None):
        # v_seq, v_spa = (None, None) if video is None else self.extract_video_features(video)
        # # print('[Debug] v_seq:', None if v_seq is None else v_seq.shape)
        # a_seq = None if audio is None else self.extract_audio_features(audio)
        # # print('[Debug] a_seq:', None if a_seq is None else a_seq.shape)
        # v_enc = None if v_seq is None else self.video_encoder(v_seq)
        # # print('[Debug] v_enc:', None if v_enc is None else v_enc.shape)
        # a_enc = None if a_seq is None else self.audio_encoder(a_seq)
        # # print('[Debug] a_enc:', None if a_enc is None else a_enc.shape)
        # assert video.dim() == 5, f"video shape wrong: {video.shape}"

        v_seq, v_spa = (None, None) if video is None else self.extract_video_features(video)
        a_seq, a_spa = (None, None) if audio is None else self.extract_audio_features(audio)
        # print('before model_encoder, v_seq shape:', None if v_seq is None else v_seq.shape)
        # print('before model_encoder, a_seq shape:', None if a_seq is None else a_seq.shape)
        v_enc = None if v_seq is None else self.video_encoder(v_seq)
        a_enc = None if a_seq is None else self.audio_encoder(a_seq)


        fused = self.fusion_features(v_enc, a_enc)
        context, t_weights = self.temporal_attn(fused)
        context = self.dropout(context)
        logits = self.classifier_fc(context)  # [B, num_classes]

        return logits, context, t_weights, v_spa

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


