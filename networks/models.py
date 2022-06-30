# +

import math
import networks.utils as utils
import networks.models_utils as models_utils
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models

# -
__all__ = ['ResNet', 'resnet50']


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = FCL(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class DAN(nn.Module):
    def __init__(self, num_head, num_class=8, pretrained=True):
        super(DAN, self).__init__()
        
        resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict = True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)


    def forward(self, x):
        #x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        
        return x, heads


class VGGFACE2_DAN(nn.Module):
    def __init__(self,  num_head, pretrained, num_class):
        super(VGGFACE2_DAN, self).__init__()
        

        self.num_class = num_class
        self.num_head = num_head
        self.pretrained = pretrained
        
        resnet = resnet50(pretrained_checkpoint_path="./models/resnet50_ft_weight.pkl", num_classes=8631, include_top=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(512, self.num_class)
        self.bn = nn.BatchNorm1d(self.num_class)
        
        self.model = DAN(num_class = 8, num_head = self.num_head , pretrained = self.pretrained)
        
        if pretrained :
            print("Load pre-trained weights ...")

            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict = True)

            print("Done !!")

        #self.dan = nn.Sequential(*list(self.model.children()))
        

    def forward(self, x):

        x=self.features(x)
        
        x=self.conv1x1_1(x) 
        x=self.conv1x1_2(x)
        
        x, heads = self.model(x)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
        

        return out, x, heads


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        
        return out 

class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = models_utils.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = models_utils.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        print("block.expansion : ", block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def load_from_pretrain(self, model, pretrained_checkpoint_path):
        model = utils.load_state_dict(model, pretrained_checkpoint_path)

        return model


class FCL(nn.Module) :
    
    def __init__(self, block, layers, pretrained_checkpoint_path, num_classes, include_top=True, freeze_base=True) :
        
        super(FCL, self).__init__()
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        
        self.base = ResNet(block, layers, num_classes, include_top)
        
        self.base.load_from_pretrain(self.base, self.pretrained_checkpoint_path)
        if freeze_base:
            for param in self.base.parameters():
                #print
                param.requires_grad = True
        self.base = nn.Sequential(*(list(self.base.children())))
        self.base = self.base[:-2]

        self.fc = nn.Linear(512 * block.expansion, 2048, bias=True)

        
        self.init_weights()

    def init_weights(self):
        models_utils.init_layer(self.fc)
        
    def load_from_pretrain(self, model, pretrained_checkpoint_path):
        model = utils.load_state_dict(model, pretrained_checkpoint_path)

        return model


    def forward(self, x) :

        x = self.base(x)

        return x

class DINO(nn.Module):
    def __init__(self, pretrained_weights, checkpoint_key, arch, patch_size, num_class) :
        
        super(DINO, self).__init__()
        #self.num_class = num_class
        self.pretrained_checkpoint_path = pretrained_weights
        self.checkpoint_key = checkpoint_key
        self.arch = arch
        self.patch_size = patch_size
        self.num_class = num_class
        
        self.model = models.resnet50(num_classes=self.num_class)
        self.fc = nn.Linear(2048, self.num_class, bias=True)
        #print(self.model)
        utils.dino_load_pretrained_weights(self.model, self.pretrained_checkpoint_path, self.checkpoint_key, self.arch, self.patch_size)
        



    def forward(self, x) :

        x = self.model(x)
        #x = self.fc(x)
        return x


class DINO_DAN(nn.Module):
    def __init__(self, pretrained_weights, checkpoint_key, arch, patch_size, num_class, pretrained, num_head) :
        
        super(DINO_DAN, self).__init__()
    
        self.pretrained_checkpoint_path = pretrained_weights
        self.checkpoint_key = checkpoint_key
        self.arch = arch
        self.patch_size = patch_size
        self.num_class = num_class
        self.pretrained = pretrained
        self.num_head = num_head
        
        self.features = models.resnet50(num_classes=self.num_class)
        self.features = nn.Sequential(*(list(self.features.children())))
        
        
        #print(self.features)
        utils.dino_load_pretrained_weights(self.features, self.pretrained_checkpoint_path, self.checkpoint_key, self.arch, self.patch_size)
        self.features = self.features[:-1]

        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(512, self.num_class)
        self.bn = nn.BatchNorm1d(self.num_class)
        
        self.model = DAN(num_class = 8, num_head = self.num_head , pretrained = self.pretrained)
        
        if pretrained :
            print("Load pre-trained weights ...")

            checkpoint = torch.load('./models/affecnet8_epoch5_acc0.6209.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict = True)

            print("Done !!")
        



    def forward(self, x) :

        x = self.features(x)
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)

        x, heads = self.model(x)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
        
        return out, x, heads