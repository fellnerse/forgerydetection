import torch
from facenet_pytorch import InceptionResnetV1
from torch import nn
from torchvision import models


class SlicedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

    def forward(self, X, slices=2):
        h = self.slice1(X)
        h = self.slice2(h)
        if slices > 2:
            h = self.slice3(h)
        if slices > 3:
            h = self.slice4(h)
        return h


class Vgg16(SlicedNet):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


class FaceNet(SlicedNet):
    def __init__(self, requires_grad=False):
        super(FaceNet, self).__init__()
        inception_resnet_features = InceptionResnetV1(pretrained="vggface2")
        self.slice1 = torch.nn.Sequential(
            inception_resnet_features.conv2d_1a,
            inception_resnet_features.conv2d_2a,
            inception_resnet_features.conv2d_2b,
        )
        self.slice2 = torch.nn.Sequential(
            inception_resnet_features.maxpool_3a,
            inception_resnet_features.conv2d_3b,
            inception_resnet_features.conv2d_4a,
            inception_resnet_features.conv2d_4b,
            inception_resnet_features.repeat_1,
        )
        self.slice3 = torch.nn.Sequential(
            inception_resnet_features.mixed_6a, inception_resnet_features.repeat_2
        )
        self.slice4 = torch.nn.Sequential(
            inception_resnet_features.mixed_7a, inception_resnet_features.repeat_3
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
