import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.mixins import BinaryEvaluationMixin
from forgery_detection.models.utils import SequenceClassificationModel


class SqueezeModule(nn.Module):
    def __init__(self, times, squeeze=True, dim=1):
        super().__init__()
        self.times = times
        self.squeeze = squeeze
        self.dim = dim

        if squeeze:
            self.func = lambda x: x.squeeze(dim)
        else:
            self.func = lambda x: x.unsqueeze(dim)

    def forward(self, x):
        return self.func(x)


class SmallVideoNetworkPooledEmbedding(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.layer2 = nn.Identity()
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()  # output is 64

        self.video_pooling = nn.Sequential(
            SqueezeModule(1, squeeze=False, dim=1),
            nn.MaxPool1d(8, 8),
            SqueezeModule(1, squeeze=True, dim=1),
            nn.Dropout(0.3),
        )

    def forward(self, video):
        video = video.permute(0, 2, 1, 3, 4)
        video = self.r2plus1(video)  # bs x 64
        video = self.video_pooling(video)  # bs x 8
        return video


class SmallEmbeddingSpace(BinaryEvaluationMixin, SequenceClassificationModel):
    def __init__(self, num_classes, sequence_length=8, pretrained=True):
        super().__init__(
            num_classes=2, sequence_length=sequence_length, contains_dropout=False
        )

        self.r2plus1 = SmallVideoNetworkPooledEmbedding(pretrained=pretrained)

        self.sync_net = PretrainedSyncNet()
        self._set_requires_grad_for_module(self.sync_net, requires_grad=False)
        # use standard audio_extractor with 1024 -> then max pooling? or average?
        # to lets say 8 values
        self.audio_pooling = nn.Sequential(
            SqueezeModule(1, squeeze=False, dim=1),
            nn.MaxPool1d(128, 128),
            SqueezeModule(1, squeeze=True, dim=1),
        )

        self.out = nn.Sequential(
            nn.Linear(16, 50), nn.BatchNorm1d(50), nn.LeakyReLU(0.2), nn.Linear(50, 2)
        )

    def forward(self, x):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 4 x 13
        # syncnet only uses 5 frames
        audio = audio[:, 2:-1]
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)

        audio = self.sync_net.audio_extractor(audio)  # bs x 1024
        audio = self.audio_pooling(audio)  # bs x 8

        video = self.r2plus1(video)  # bs x 8
        embedding = torch.cat((video, audio), dim=1)
        return self.out(embedding)

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        return super().training_step((x, target[0] // 4), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for output in outputs:
            output["target"] = output["target"][0] // 4
        return super().aggregate_outputs(outputs, system)


class SmallVideoNetwork(BinaryEvaluationMixin, SequenceClassificationModel):
    def __init__(self, num_classes, sequence_length=8, pretrained=True):
        super().__init__(
            num_classes=2, sequence_length=sequence_length, contains_dropout=False
        )

        self.r2plus1 = SmallVideoNetworkPooledEmbedding(pretrained=pretrained)

        self.out = nn.Sequential(
            nn.Linear(8, 50), nn.BatchNorm1d(50), nn.LeakyReLU(0.2), nn.Linear(50, 2)
        )

    def forward(self, x):

        video = self.r2plus1(x)  # bs x 8
        return self.out(video)

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        return super().training_step((x, target // 4), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for output in outputs:
            output["target"] = output["target"] // 4
        return super().aggregate_outputs(outputs, system)
