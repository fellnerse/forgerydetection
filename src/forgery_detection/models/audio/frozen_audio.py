import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.mixins import BinaryEvaluationMixin
from forgery_detection.models.utils import SequenceClassificationModel


class FrozenR2plus1(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(num_classes=2, sequence_length=8, contains_dropout=False)
        self.r2plus1 = r2plus1d_18(pretrained=True)
        self.r2plus1.fc = nn.Identity()
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)

        # self.sync_net = PretrainedSyncNet()
        # self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Linear(512, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )
        self._init = False

    def forward(self, x):
        video = x.transpose(1, 2)
        video = self.r2plus1(video)

        # # syncnet only uses 5 frames
        # audio = audio[:, 2:-1]
        # audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)
        # audio = self.sync_net.audio_extractor(audio)
        # flat = torch.cat((video, audio), dim=1)

        out = self.out(self.relu(video))  # todo dont use relu
        return out

    def training_step(self, batch, batch_nb, system):
        x, target = batch
        return super().training_step((x, target // 4), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        if not self._init:
            self._init = True
            system.file_list.class_to_idx = {"fake": 0, "youtube": 1}
            system.file_list.classes = ["fake", "youtube"]
        for x in outputs:
            x["target"] = x["target"] // 4
        return super().aggregate_outputs(outputs, system)


class FrozenR2plus1Audio(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(num_classes=2, sequence_length=8, contains_dropout=False)
        self.r2plus1 = r2plus1d_18(pretrained=True)
        self.r2plus1.fc = nn.Identity()
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)

        self.sync_net = PretrainedSyncNet()
        # self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Linear(512 + 1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )
        self._init = False

    def forward(self, x):
        video, audio = x
        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        # # syncnet only uses 5 frames
        audio = audio[:, 2:-1]
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)
        audio = self.sync_net.audio_extractor(audio)
        flat = torch.cat((video, audio), dim=1)

        out = self.out(self.relu(flat))  # todo dont use relu
        return out

    def training_step(self, batch, batch_nb, system):
        x, (target, aud_noisy) = batch
        return super().training_step((x, target // 4), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        if not self._init:
            self._init = True
            system.file_list.class_to_idx = {"fake": 0, "youtube": 1}
            system.file_list.classes = ["fake", "youtube"]
        for x in outputs:
            x["target"] = x["target"][0] // 4
        return super().aggregate_outputs(outputs, system)


class FrozenR2plus1AudioResnet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(num_classes=2, sequence_length=8, contains_dropout=False)
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.fc = nn.Identity()
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)

        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)
        self.resnet.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )
        self._init = False

    def forward(self, x):
        video, audio = x
        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        audio = self.resnet(
            audio.reshape((audio.shape[0], -1, 13))
            .unsqueeze(1)
            .expand(-1, 3, -1, -1)
            .repeat((1, 1, 1, 4))
        )

        flat = torch.cat((video, audio), dim=1)

        out = self.out(self.relu(flat))  # todo dont use relu
        return out

    def training_step(self, batch, batch_nb, system):
        x, (target, aud_noisy) = batch
        return super().training_step((x, target // 4), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        if not self._init:
            self._init = True
            system.file_list.class_to_idx = {"fake": 0, "youtube": 1}
            system.file_list.classes = ["fake", "youtube"]
        for x in outputs:
            x["target"] = x["target"][0] // 4
        return super().aggregate_outputs(outputs, system)


class FrozenR2Plus1BNLeakyRelu(BinaryEvaluationMixin, FrozenR2plus1):
    def __init__(self, num_classes=2):
        super().__init__(num_classes=2)

        self.relu = nn.Identity()

        self.out = nn.Sequential(
            nn.Linear(512, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, self.num_classes),
        )


class R2plus1UnfrozenBaseline(FrozenR2Plus1BNLeakyRelu):
    def __init__(self, num_classes=2):
        super().__init__(num_classes=2)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=True)
