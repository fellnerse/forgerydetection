import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.mixins import BinaryEvaluationMixin
from forgery_detection.models.utils import SequenceClassificationModel


class NoisySyncAudioNet(BinaryEvaluationMixin, SequenceClassificationModel):
    def __init__(self, num_classes, pretrained=True):
        super().__init__(num_classes=2, sequence_length=8, contains_dropout=False)

        self.r2plus1 = self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.layer2 = nn.Identity()
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.sync_net = PretrainedSyncNet()
        # self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(64 + 1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )
        self._init = False

    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29
        # video = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29

        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        # syncnet only uses 5 frames
        audio = audio[:, 2:-1]
        audio = (audio.reshape((audio.shape[0], -1, 13)).unsqueeze(1)).transpose(-2, -1)
        audio = self.sync_net.audio_extractor(audio)

        flat = torch.cat((video, audio), dim=1)
        out = self.out(self.relu(flat))
        return out

    def training_step(self, batch, batch_nb, system):
        x, (target, aud_noisy) = batch
        return super().training_step((x, aud_noisy), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        if not self._init:
            self._init = True
            system.file_list.class_to_idx = {"fake": 0, "youtube": 1}
            system.file_list.classes = ["fake", "youtube"]
        for x in outputs:
            x["target"] = x["target"][1]
        return super().aggregate_outputs(outputs, system)


class FrozenNoisySyncAudioNet(NoisySyncAudioNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)
        self.r2plus1 = r2plus1d_18(pretrained=True)
        self.r2plus1.fc = nn.Identity()
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)

        self.out = nn.Sequential(
            nn.Linear(512 + 1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )


class BigNoisySyncAudioNet(NoisySyncAudioNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)
        self.r2plus1 = r2plus1d_18(pretrained=True)
        self.r2plus1.fc = nn.Identity()
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)

        self.sync_net = resnet18(pretrained=True, num_classes=1000)
        self.sync_net.fc = nn.Identity()
        self.relu = nn.Identity()

        self.out = nn.Sequential(
            nn.Linear(512 + 512, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, self.num_classes),
        )

    def forward(self, x):
        video, audio = x
        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        audio = self.sync_net(
            audio.reshape((audio.shape[0], -1, 13))
            .unsqueeze(1)
            .expand(-1, 3, -1, -1)
            .repeat((1, 1, 1, 4))
        )

        flat = torch.cat((video, audio), dim=1)

        out = self.out(self.relu(flat))  # todo dont use relu
        return out


class FilterNoisySyncAudioNet(BigNoisySyncAudioNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)

        self.filter = nn.Sequential(  # b x 512 x 9
            nn.Conv1d(
                512, 128, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 16 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                128, 32, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 8 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                32, 8, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 4 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                8, 2, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 2 x seq_len
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                2, 1, kernel_size=3, stride=1, padding=1, bias=True
            ),  # b x 1 x seq_len
            nn.LeakyReLU(0.02, True),
        )

        self.attention = self.attentionNet = nn.Sequential(
            nn.Linear(9, 9, bias=True), nn.Softmax(dim=1)
        )

    def filter_audio(self, audio: torch.Tensor):
        # audio shape: b x 16 x 4 x 13
        bs = audio.shape[0]

        audio = audio.reshape((-1, 8, 4, 13))

        audio = (
            audio.reshape((audio.shape[0], -1, 13))
            .unsqueeze(1)
            .expand(-1, 3, -1, -1)
            .repeat((1, 1, 1, 4))
        )  # (bs*9) x 3 x 32 x 52

        audio: torch.Tensor = self.sync_net(audio)  # (bs*9) x 512
        audio = audio.reshape(bs, 9, 512).transpose(2, 1)  # bs x 512 x 9
        # todo maybe need pooling here to reduce number of params
        weights = self.filter(audio)  # bs x 1 x 8
        attention = self.attention(weights.squeeze()).unsqueeze(-1)  # bs x 9 x 1

        filtered_audio = torch.bmm(audio, attention).squeeze()  # bs x 512

        return filtered_audio

    def forward(self, x):
        video, audio = x
        # def forward(self, video, audio):
        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        audio = self.filter_audio(audio)

        flat = torch.cat((video, audio), dim=1)

        out = self.out(self.relu(flat))  # todo dont use relu
        return out


class FilterNoisySyncAudioNetUnfrozen(FilterNoisySyncAudioNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=True)
        # # self.r2plus1.layer3 = nn.Identity()
        # self.r2plus1.layer4 = nn.Identity()
        #
        # self.out = nn.Sequential(
        #     nn.Linear(256 + 512, 50),
        #     nn.BatchNorm1d(50),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(50, self.num_classes),
        # )

    def training_step(self, batch, batch_nb, system):
        x, (target, aud_noisy) = batch
        return super(NoisySyncAudioNet, self).training_step(
            (x, target // 4), batch_nb, system
        )

    def aggregate_outputs(self, outputs, system):
        if not self._init:
            self._init = True
            system.file_list.class_to_idx = {"fake": 0, "youtube": 1}
            system.file_list.classes = ["fake", "youtube"]
        for x in outputs:
            x["target"] = x["target"][0] // 4
        return super(NoisySyncAudioNet, self).aggregate_outputs(outputs, system)


class FilterNoisySyncAudioNetUnfrozen2VideoLayer(NoisySyncAudioNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=2)
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()

        self.out = nn.Sequential(
            nn.Linear(128 + 512, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, self.num_classes),
        )

    def training_step(self, batch, batch_nb, system):
        x, (target, aud_noisy) = batch
        return super(NoisySyncAudioNet, self).training_step(
            (x, target // 4), batch_nb, system
        )

    def aggregate_outputs(self, outputs, system):
        if not self._init:
            self._init = True
            system.file_list.class_to_idx = {"fake": 0, "youtube": 1}
            system.file_list.classes = ["fake", "youtube"]
        for x in outputs:
            x["target"] = x["target"][0] // 4
        return super(NoisySyncAudioNet, self).aggregate_outputs(outputs, system)
