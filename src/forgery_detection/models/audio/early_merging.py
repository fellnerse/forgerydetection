import torch
from torch import nn
from torchvision.models.video import r2plus1d_18
from torchvision.models.video.resnet import Conv2Plus1D

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.utils import SequenceClassificationModel


class EarlyMergeNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = r2plus1d_18(pretrained=True)

        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.sync_net = PretrainedSyncNet()
        self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()

        self.padding = nn.ReflectionPad2d((0, 1, 0, 0))
        self.upsample = nn.Upsample(size=(8, 56, 56))

        self.merge_conv: nn.Module = nn.Sequential(
            Conv2Plus1D(128, 64, 144, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Linear(128, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )

    def audio_forward(self, audio):
        features = self.sync_net.netcnnaud[:15](audio)  # bs x 256 x 11 x 15
        features = self.padding(features)  # bs x 256 x 11 x 16
        features = features.reshape((-1, 256, 11, 8, 2))  # bs x 256 x 11 x 8 x 2
        features = features.permute((0, 1, 3, 2, 4))  # bs x 256 x 8 x 11 x 2
        features = features.reshape((-1, 64, 8, 11, 8))  # bs x 64 x 8 x 11 x 8
        features = self.upsample(features)  # bs x 64 x 8 x 56 x 56
        return features

    def video_forward(self, video):
        x = self.r2plus1.stem(video)

        x = self.r2plus1.layer1(x)  # bs x 64 x 8 x 56 x 56
        return x

    def merge_modalities(self, video, audio):
        merged_features = torch.cat((video, audio), dim=1)  # bs x 128 x 8 x 56 x 56
        merged_features = self.merge_conv(merged_features)  # # bs x 64 x 8 x 56 x 56
        return merged_features

    def final_forward(self, x):
        x = self.r2plus1.layer2(x)
        x = self.r2plus1.layer3(x)
        x = self.r2plus1.layer4(x)
        x = self.r2plus1.avgpool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29
        # video = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29

        video = video.transpose(1, 2)
        video = self.video_forward(video)

        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)
        audio = self.audio_forward(audio)

        merged_features = self.merge_modalities(video, audio)
        merged_features = self.final_forward(merged_features)

        out = self.out(merged_features)
        return out

    def training_step(self, batch, batch_nb, system):
        x, (target, _) = batch
        return super().training_step((x, target), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for x in outputs:
            x["target"] = x["target"][0]
        return super().aggregate_outputs(outputs, system)


class EarlyMergeNetBinary(EarlyMergeNet):
    def __init__(self, num_classes=2):
        super().__init__(num_classes=2)

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        return super().training_step((x, target // 4), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for output in outputs:
            output["target"] = output["target"] // 4
        return super().aggregate_outputs(outputs, system)
