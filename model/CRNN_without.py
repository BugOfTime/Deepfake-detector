import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights


__all__ = ['CRNN_without']

class model_encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3, bidirectional=True):
        super(model_encoder, self).__init__()
        self.bi_GRU = torch.nn.GRU(input_dim,
                                   hidden_dim,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   batch_first=True,
                                   dropout=dropout
                                   )


    def forward(self, x):
        x, _ = self.bi_GRU(x)
        return x


class CRNN_without(torch.nn.Module):
    def __init__(self, model_depth, num_classes, fusion_method='concat', pooling_method='avg'):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.pooling_method = pooling_method

        resnet_dict = {
            18: (resnet18, ResNet18_Weights.DEFAULT, 512),
            34: (resnet34, ResNet34_Weights.DEFAULT, 512),
            50: (resnet50, ResNet50_Weights.DEFAULT, 2048),
            101: (resnet101, ResNet101_Weights.DEFAULT, 2048),
            152: (resnet152, ResNet152_Weights.DEFAULT, 2048)
        }
        assert model_depth in resnet_dict, f"can't find ResNet depth weight: {model_depth}"
        resnet_fn, weights, resnet_dim = resnet_dict[model_depth]


        self.video_resnet = resnet_fn(weights=weights)
        self.video_resnet.fc = torch.nn.Identity()

        self.audio_resnet = resnet_fn(weights=weights)
        self.audio_resnet.fc = torch.nn.Identity()

        self.video_encoder = model_encoder(input_dim=resnet_dim)
        self.audio_encoder = model_encoder(input_dim=resnet_dim)


        if fusion_method == 'concat':
            fusion_dim = 512 * 2  # Bidirectional GRU output dimension * 2
        elif fusion_method == 'add':
            fusion_dim = 512
        elif fusion_method == 'attention':
            fusion_dim = 512
            self.fusion_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fuse_dim = fusion_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(fusion_dim, fusion_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(fusion_dim // 2, self.num_classes)
        )

        self.register_buffer("video_available", torch.tensor(True))
        self.register_buffer("audio_available", torch.tensor(True))

    def extract_video_features(self, video):
        if video is None:
            return None

        device = next(self.video_resnet.parameters()).device
        video = video.to(device)

        B, C, L, H, W = video.shape  # L是帧数
        video = video.transpose(1, 2).contiguous()
        video = video.view(B * L, C, H, W)
        video_features = self.video_resnet(video)
        video_features = video_features.view(B, L, -1)
        return video_features

    def extract_audio_features(self, audio):
        if audio is None:
            return None

        device = next(self.audio_resnet.parameters()).device
        audio = audio.to(device)

        if audio.dim() == 4:
            audio = audio.unsqueeze(-1)

        B, C, L, H, W = audio.shape
        audio = audio.transpose(1, 2).contiguous()
        audio = audio.view(B * L, C, H, W)

        audio_features = self.audio_resnet(audio)
        audio_features = audio_features.view(B, L, -1)

        return audio_features

    def fusion_features(self, video_features, audio_features):
        if video_features is None and audio_features is None:
            raise ValueError("Both video and audio features are None")

        if video_features is None:
            video_features = torch.zeros_like(audio_features).to(audio_features.device)
        if audio_features is None:
            audio_features = torch.zeros_like(video_features).to(video_features.device)

        if self.fusion_method == 'concat':
            return torch.cat([video_features, audio_features], dim=2)
        elif self.fusion_method == 'add':
            return video_features + audio_features
        elif self.fusion_method == 'attention':
            video_features, _ = self.fusion_attention(video_features, audio_features, audio_features)
            audio_features, _ = self.fusion_attention(audio_features, video_features, video_features)
            fusion_features = (video_features + audio_features) / 2
            return fusion_features

    def temporal_pooling(self, features):
        # alternative attention mechanism
        if self.pooling_method == 'avg':
            # global pooling
            return torch.mean(features, dim=1)

        elif self.pooling_method == 'max':
            # maximize pooling
            return torch.max(features, dim=1)[0]

        # elif self.pooling_method == 'last':
        #     # 取最后一个时间步
        #     return features[:, -1, :]
        #
        # elif self.pooling_method == 'first':
        #     # 取第一个时间步
        #     return features[:, 0, :]

        #Combination of average pooling and max pooling
        elif self.pooling_method == 'avg_max':
            avg_pool = torch.mean(features, dim=1)
            max_pool = torch.max(features, dim=1)[0]
            return (avg_pool + max_pool) / 2
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def forward(self, video=None, audio=None):
        # extract feature
        video_features = self.extract_video_features(video)
        audio_features = self.extract_audio_features(audio)

        video_encoder = None
        audio_encoder = None

        if video_features is not None:
            video_encoder = self.video_encoder(video_features)
        if audio_features is not None:
            audio_encoder = self.audio_encoder(audio_features)

        fusion_features = self.fusion_features(video_encoder, audio_encoder)

        pooled_features = self.temporal_pooling(fusion_features)

        # classifier
        output = self.classifier(pooled_features)



        return output,pooled_features

    def forward_video(self, video):
        # video only
        return self.forward(video=video, audio=None)

    def forward_audio(self, audio):
        # audio only
        return self.forward(audio=audio, video=None)

    def forward_video_audio(self, video, audio):
        # video and audio
        return self.forward(video=video, audio=audio)



# if __name__ == "__main__":
#
#     model = Multi_CRNN_with_TCN(
#         model_depth=50,
#         num_classes=10,
#         fusion_method='concat',
#         pooling_method='avg'
#     )
#
#     # Simulate input data
#     batch_size = 2
#     video_input = torch.randn(batch_size, 3, 16, 224, 224)  # [B, C, T, H, W]
#     audio_input = torch.randn(batch_size, 3, 16, 224, 224)  # [B, C, T, H, W]
#
#
#     output = model(video_input, audio_input)
#     print(f"Output shape: {output.shape}")  # [B, num_classes]
#
#
#     video_output = model.forward_video(video_input)
#     print(f"Video-only output shape: {video_output.shape}")
#
#     # 只使用音频
#     audio_output = model.forward_audio(audio_input)
#     print(f"Audio-only output shape: {audio_output.shape}")