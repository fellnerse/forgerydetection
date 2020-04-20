import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.mixins import PretrainedNet
from forgery_detection.models.utils import SequenceClassificationModel


class AudionetUtils(SequenceClassificationModel):
    def forward(self, x):
        # def forward(self, video, audio):
        video, audio = x  # bs x 8 x 3 x 112 x 112 , bs x 8 x 29

        video = video.transpose(1, 2)
        video = self.r2plus1(video)

        audio = self.resnet(audio.unsqueeze(1).expand(-1, 3, -1, -1))

        flat = torch.cat((video, audio), dim=1)
        out = self.out(self.relu(flat))
        return out

    def training_step(self, batch, batch_nb, system):
        x, (target, _) = batch
        return super().training_step((x, target), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for x in outputs:
            x["target"] = x["target"][0]
        return super().aggregate_outputs(outputs, system)


class AudioNet(AudionetUtils):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = r2plus1d_18(pretrained=pretrained)
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(256, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )


class AudioNetFrozen(AudioNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet, requires_grad=False)


class PretrainedAudioNet(
    PretrainedNet("/data/hdd/model_checkpoints/audionet/13_epochs/model.ckpt"), AudioNet
):
    pass


class AudioNetLayer2Unfrozen(PretrainedAudioNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.r2plus1.layer2, requires_grad=True)
        self._set_requires_grad_for_module(self.resnet, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet.layer2, requires_grad=True)


class SyncAudioNet(SequenceClassificationModel):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=True,
        )
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.sync_net = PretrainedSyncNet()
        self._set_requires_grad_for_module(self.sync_net, requires_grad=False)

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(128 + 1024, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )

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
        x, (target, _) = batch
        return super().training_step((x, target), batch_nb, system)

    def aggregate_outputs(self, outputs, system):
        for x in outputs:
            x["target"] = x["target"][0]
        return super().aggregate_outputs(outputs, system)


class PretrainSyncAudioNet(
    PretrainedNet(
        "/mnt/raid/sebastian/log/runs/TRAIN/sync_audio_net/version_1/checkpoints/_ckpt_epoch_23.ckpt"
    ),
    SyncAudioNet,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)


class AudioNet34(AudionetUtils):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(
            num_classes=num_classes, sequence_length=8, contains_dropout=False
        )
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=True,
        )
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Identity()

        self.resnet = resnet18(pretrained=pretrained, num_classes=1000)
        self.resnet.layer3 = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(256, 50), nn.ReLU(), nn.Linear(50, self.num_classes)
        )


class PretrainingAudioNet34(AudioNet34):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__(num_classes=num_classes)

        # load weight of MLP from audionet trained with smaller r2plus1 net
        loaded_state = torch.load(
            "/data/hdd/model_checkpoints/audionet/13_epochs/model.ckpt",
            map_location=lambda storage, loc: storage,
        )["state_dict"]

        self_state = self.state_dict()
        for name, param in loaded_state.items():
            if "out" in name:
                self_state[name.replace("model.", "")].copy_(param)

        self._set_requires_grad_for_module(self.r2plus1, requires_grad=False)
        self._set_requires_grad_for_module(self.resnet, requires_grad=False)

    def forward(self, x):
        video, audio = x
        return super().forward((video, audio.reshape((audio.shape[0], -1, 13))))
