import torch
from torch import nn
from torchvision.models.video import r2plus1d_18

from forgery_detection.models.audio.similarity_stuff import PretrainedSyncNet
from forgery_detection.models.utils import SequenceClassificationModel


class NoisySyncAudioNet(SequenceClassificationModel):
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
