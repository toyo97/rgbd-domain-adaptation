from torchvision import models
import torch
import torch.nn as nn
from torch.autograd import Function
import copy


class LambdaRev(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.mul(ctx.lamda), None


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.rgb_feature_extractor = nn.Sequential(
            *(list(copy.deepcopy(self.resnet18).children())[:-2]))  # Eliminate last fc and avg pool
        self.depth_feature_extractor = nn.Sequential(
            *(list(copy.deepcopy(self.resnet18).children())[:-2]))  # Eliminate last fc and avg pool

        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output has shape (batch_size, num_featues, 1, 1)
            nn.Flatten(),
            nn.Linear(1024, 1000),  # wants first dimension = batch_size
            # In resnet18 it is 512 * block.expansion -> ???
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes),
            nn.Softmax(dim=1)
        )

        # Xavier initialization
        self.main_head.apply(init_weights)

        self.pretext_head = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=100, kernel_size=(1, 1), bias=True, stride=1, padding=0),
            # output img = 100 channels of 7x7
            nn.BatchNorm2d(num_features=100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(3, 3), bias=True, stride=2, padding=0),
            # output img = 100 channels of 98x98
            nn.BatchNorm2d(num_features=100),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(900, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Linear(100, 4),
            nn.Softmax(dim=1)
        )

        self.pretext_head.apply(init_weights)

    def forward(self, x, y, lamda=None):  # x is the rgb batch, y the depth batch

        rgb_features = self.rgb_feature_extractor(x)  # list of rgb filters of the batch (list_size = batch_size)
        depth_features = self.depth_feature_extractor(y)

        tot_features = torch.cat((depth_features, rgb_features), 1)  # To check: concatenate along right direction?
        # size_allFeatures = (batch_size, number_filters, height, width)
        # number_filters = 512
        # height = width = 7 if input of network is 224x224

        if lamda is None:

            class_scores = self.main_head(tot_features)  # class scores of the batch
            return class_scores

        else:
            out2 = self.pretext_head(tot_features)
            out2 = LambdaRev.apply(out2, lamda)  # lambda mul in backward pass
            return out2
