import torch
import torch.multiprocessing as mp
from torch import nn

from forgery_detection.models.image.multi_class_classification import Resnet182D
from forgery_detection.models.utils import SequenceClassificationModel


class AudioNet(SequenceClassificationModel):
    def __init__(self, pretrained=True):
        super().__init__(num_classes=5, sequence_length=8, contains_dropout=False)
        self.r2plus1 = torch.hub.load(
            "moabitcoin/ig65m-pytorch",
            "r2plus1d_34_8_kinetics",
            num_classes=400,
            pretrained=pretrained,
        )
        self.r2plus1.layer3 = nn.Identity()
        self.r2plus1.layer4 = nn.Identity()
        self.r2plus1.fc = nn.Linear(128, 256)

        self.resnet = Resnet182D(
            num_classes=256, sequence_length=1, pretrained=pretrained
        )
        self.resnet.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.resnet.layer2 = nn.Identity()
        self.resnet.resnet.layer3 = nn.Identity()
        self.resnet.resnet.fc = nn.Linear(64, 256)

        self.out = nn.Linear(512, self.num_classes)

    def forward(self, x):
        video, audio = x
        video = video.transpose(1, 2)

        processes = []
        p = mp.Process(target=self.r2plus1, args=(video,))
        p.start()
        processes.append(p)
        video = self.r2plus1(video)
        audio = self.resnet(audio.unsqueeze(1))
        flat = torch.cat((video, audio), dim=1)
        out = self.out(flat)
        return out
