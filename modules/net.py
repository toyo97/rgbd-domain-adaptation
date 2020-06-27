import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class ModifiedResNet(nn.Module):
    """
    Pytorch implementation of Resnet without the last two layers (fully connected and pooling)
    """

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ModifiedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # removed last two layers
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# V2
class Net(nn.Module):
    def __init__(self, num_classes, single_mod=None):
        """
        RGBD Domain Adaptation network based on Resnet18

        Args:
            num_classes: number of output classes
            single_mod: specify `RGB` or `depth` if only one modality is being passed through the network
                        otherwise leave it to None
        """
        super(Net, self).__init__()

        if single_mod not in ['RGB', 'depth', None]:
            raise ValueError('single_mod parameter not valid. Please choose between `RGB` or `depth`, otherwise leave '
                             'it to None')

        self.single_mod = single_mod
        num_maps = 1024
        if self.single_mod is not None:
            num_maps = 512

        state_dict = load_state_dict_from_url(model_urls['resnet18'])

        self.rgb_feature_extractor = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
        self.rgb_feature_extractor.load_state_dict(state_dict, strict=False)
        self.depth_feature_extractor = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
        self.depth_feature_extractor.load_state_dict(state_dict, strict=False)

        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output has shape (batch_size, num_featues, 1, 1)
            nn.Flatten(),
            nn.Linear(num_maps, 1000),  # wants first dimension = batch_size
            # In resnet18 it is 512 * block.expansion -> ???
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes)
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
            nn.Dropout(p=0.5),
            nn.Linear(100, 4)
        )

        self.pretext_head.apply(init_weights)

    def forward(self, x=None, y=None, pretext=None):  # x is the rgb batch, y the depth batch
        if self.single_mod == 'RGB':
            tot_features = self.rgb_feature_extractor(x)

        if self.single_mod == 'depth':
            tot_features = self.depth_feature_extractor(y)

        if self.single_mod is None:
            rgb_features = self.rgb_feature_extractor(x)  # list of rgb filters of the batch (list_size = batch_size)
            depth_features = self.depth_feature_extractor(y)
            tot_features = torch.cat((depth_features, rgb_features), 1)

        if pretext is None:
            class_scores = self.main_head(tot_features)
            return class_scores

        else:
            out2 = self.pretext_head(tot_features)
            return out2


class AFNNet(nn.Module):
    def __init__(self, num_classes, single_mod=None, dropout_p=0.5, rescale_dropout = True):
        """
        RGBD Domain Adaptation network based on Resnet18
        @param num_classes: number of output classes
        @param single_mod: specify `RGB` or `depth` if only one modality is being passed through the network
                        otherwise leave it to None
        """
        super(AFNNet, self).__init__()

        if single_mod not in ['RGB', 'depth', None]:
            raise ValueError('single_mod parameter not valid. Please choose between `RGB` or `depth`, otherwise leave '
                             'it to None')

        self.rescale_dropout = rescale_dropout
        self.single_mod = single_mod
        self.dropout_p = dropout_p
        num_maps = 1024
        if self.single_mod is not None:
            num_maps = 512

        state_dict = load_state_dict_from_url(model_urls['resnet18'])

        self.rgb_feature_extractor = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
        self.rgb_feature_extractor.load_state_dict(state_dict, strict=False)
        self.depth_feature_extractor = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
        self.depth_feature_extractor.load_state_dict(state_dict, strict=False)

        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output has shape (batch_size, num_featues, 1, 1)
            nn.Flatten(),
            nn.Linear(num_maps, 1000),  # wants first dimension = batch_size
            # In resnet18 it is 512 * block.expansion -> ???
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.fc2 = nn.Linear(1000, num_classes)

        # Xavier initialization
        self.main_head.apply(init_weights)
        self.fc2.apply(init_weights)

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
            nn.Dropout(p=self.dropout_p),
            nn.Linear(100, 4)
        )

        self.pretext_head.apply(init_weights)

    def forward(self, x=None, y=None, pretext=None):  # x is the rgb batch, y the depth batch
        if self.single_mod == 'RGB':
            tot_features = self.rgb_feature_extractor(x)

        if self.single_mod == 'depth':
            tot_features = self.depth_feature_extractor(y)

        if self.single_mod is None:
            rgb_features = self.rgb_feature_extractor(x)  # list of rgb filters of the batch (list_size = batch_size)
            depth_features = self.depth_feature_extractor(y)
            tot_features = torch.cat((depth_features, rgb_features), 1)

        if pretext is None:
            out = self.main_head(tot_features)

            if self.rescale_dropout:
                if self.training:
                    out.mul_(math.sqrt(1 - self.dropout_p))

            class_scores = self.fc2(out)
            return out, class_scores

        else:
            out2 = self.pretext_head(tot_features)
            return out2


class ablationAFNNet(nn.Module):
    def __init__(self, num_classes, single_mod=None, dropout_p=0.5):
        """
        RGBD Domain Adaptation network based on Resnet18
        @param num_classes: number of output classes
        @param single_mod: specify `RGB` or `depth` if only one modality is being passed through the network
                        otherwise leave it to None
        """
        super(ablationAFNNet, self).__init__()

        if single_mod not in ['RGB', 'depth', None]:
            raise ValueError('single_mod parameter not valid. Please choose between `RGB` or `depth`, otherwise leave '
                             'it to None')

        self.single_mod = single_mod
        self.dropout_p = dropout_p
        num_maps = 1024
        if self.single_mod is not None:
            num_maps = 512

        state_dict = load_state_dict_from_url(model_urls['resnet18'])

        self.rgb_feature_extractor = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
        self.rgb_feature_extractor.load_state_dict(state_dict, strict=False)
        self.depth_feature_extractor = ModifiedResNet(BasicBlock, [2, 2, 2, 2])
        self.depth_feature_extractor.load_state_dict(state_dict, strict=False)

        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output has shape (batch_size, num_featues, 1, 1)
            nn.Flatten(),
            nn.Linear(num_maps, 1000),  # wants first dimension = batch_size
            # In resnet18 it is 512 * block.expansion -> ???
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.fc2 = nn.Linear(1000, num_classes)

        # Xavier initialization
        self.main_head.apply(init_weights)
        self.fc2.apply(init_weights)

        self.pretext_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output has shape (batch_size, num_featues, 1, 1)
            nn.Flatten(),
            nn.Linear(num_maps, 1000),  # wants first dimension = batch_size
            # In resnet18 it is 512 * block.expansion -> ???
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(1000, 4)
        )

        self.pretext_head.apply(init_weights)

    def forward(self, x=None, y=None, pretext=None):  # x is the rgb batch, y the depth batch
        if self.single_mod == 'RGB':
            tot_features = self.rgb_feature_extractor(x)

        if self.single_mod == 'depth':
            tot_features = self.depth_feature_extractor(y)

        if self.single_mod is None:
            rgb_features = self.rgb_feature_extractor(x)  # list of rgb filters of the batch (list_size = batch_size)
            depth_features = self.depth_feature_extractor(y)
            tot_features = torch.cat((depth_features, rgb_features), 1)

        if pretext is None:
            out = self.main_head(tot_features)

            if self.training:
                    out.mul_(math.sqrt(1 - self.dropout_p))

            class_scores = self.fc2(out)
            return out, class_scores

        else:
            out2 = self.pretext_head(tot_features)
            return out2
