# -*- coding: utf-8 -*-
# +

from ast import Pass
import math
from re import X
import networks.utils as utils
import networks.models_utils as models_utils
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models

import networks.vision_transformer as vits
from mobilevitv2.cvnets import get_model
__all__ = ['ResNet', 'resnet50']


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = FCL(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def mtl_resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = mtl_FCL(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
class DAN(nn.Module):
    def __init__(self, num_head, num_class=8, pretrained=True):
        super(DAN, self).__init__()
        
        resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('../models/resnet18_msceleb.pth')
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
        #print(np.shape(x), np.shape(heads))
        return x, heads


class VGGFACE2_DAN(nn.Module):
    def __init__(self, pretrained_checkpoint_path, num_head, pretrained, num_class):
        super(VGGFACE2_DAN, self).__init__()
        

        self.num_class = num_class
        self.num_head = num_head
        self.pretrained = pretrained
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        
        resnet = resnet50(pretrained_checkpoint_path = self.pretrained_checkpoint_path, num_classes = 8631, include_top = True)

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
        
        
        
        self.model = DAN(num_class = 8, num_head = self.num_head , pretrained = self.pretrained)
        self.fc = nn.Linear(512, self.num_class)
        self.bn = nn.BatchNorm1d(self.num_class)
        if pretrained :
            print("Load pre-trained weights ...")

            checkpoint = torch.load('../models/affecnet8_epoch5_acc0.6209.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict = True)

            print("Done !!")

        #self.dan = nn.Sequential(*list(self.model.children()))
        

    def forward(self, x):
        #print("잘들어옴1", np.shape(x))
        x=self.features(x)
        #print("잘들어옴2", np.shape(x))
        x=self.conv1x1_1(x) 
        x=self.conv1x1_2(x)
        
        x, heads = self.model(x)
        #print(np.shape(x))
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

    def resnet50(**kwargs):
        """Constructs a ResNet-50 model.
        """
        model = FCL(Bottleneck, [3, 4, 6, 3], **kwargs)
        return model


class mtl_FCL(nn.Module) :
    
    def __init__(self, block, layers, pretrained_checkpoint_path, num_classes, include_top=True, freeze_base=True) :
        
        super(mtl_FCL, self).__init__()
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        
        self.base = ResNet(block, layers, num_classes, include_top)
        
        self.base.load_from_pretrain(self.base, self.pretrained_checkpoint_path)
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = True
        self.base = nn.Sequential(*(list(self.base.children())))
        self.base = self.base[:-1]
        #self.fc = nn.Linear(512 * block.expansion, 2048, bias=True)
        #print(self.base)
        
        #self.init_weights()

    #def init_weights(self):
        #models_utils.init_layer(self.fc)
        
    def load_from_pretrain(self, model, pretrained_checkpoint_path):
        model = utils.load_state_dict(model, pretrained_checkpoint_path)

        return model


    def forward(self, x) :
        x = self.base(x)
        #x = self.fc(x)
        return x

class FCL(nn.Module) :
    
    def __init__(self, block, layers, pretrained_checkpoint_path, num_classes, include_top=True, freeze_base=True) :
        
        super(FCL, self).__init__()
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        
        self.base = ResNet(block, layers, num_classes, include_top)
        
        self.base.load_from_pretrain(self.base, self.pretrained_checkpoint_path)
        if freeze_base:
            for param in self.base.parameters():
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
        
        self.pretrained_checkpoint_path = pretrained_weights
        self.checkpoint_key = checkpoint_key
        self.arch = arch
        self.patch_size = patch_size
        self.num_class = num_class
        
        if self.arch == "vit_base":
            self.model = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
            self.fc = nn.Linear(768, self.num_class, bias=True)
            if self.pretrained_checkpoint_path is not None:
                utils.dino_load_pretrained_weights(self.model, self.pretrained_checkpoint_path, self.checkpoint_key, self.arch, self.patch_size)
        else:
            self.model = models.resnet50(num_classes=self.num_class)
            if self.pretrained_checkpoint_path is not None:
                utils.dino_load_pretrained_weights(self.model, self.pretrained_checkpoint_path, self.checkpoint_key, self.arch, self.patch_size)
        
        

    def forward(self, x) :

        x = self.model(x)
        if self.arch == "vit_base":
            x = self.fc(x)
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
        if self.pretrained_checkpoint_path is not None :
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

            checkpoint = torch.load('../models/affecnet8_epoch5_acc0.6209.pth')
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


class Finetuning_models(nn.Module) :
    def __init__(self, model_name=None, pretrained_weights=None, checkpoint_key=None, arch=None, patch_size=None, 
                            num_class=None, pretrained=None, num_head=None) :
        super(Finetuning_models, self).__init__() 
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights
        self.pretrained_model_num_class = num_class
        self.pretrained = pretrained
        self.num_class = num_class
        self.model = None
        self.fc = nn.Linear(768, num_class, bias=True)
        if self.model_name == "DINO" :
            print(self.pretrained_weights)
            self.checkpoint_key = checkpoint_key
            self.arch = arch
            self.patch_size = patch_size
            print("Loading pretrain model of DINO for finetuning ...")
    
            self.model = vits.__dict__["vit_base"](patch_size=self.patch_size, num_classes=self.num_class)
            
            utils.dino_load_pretrained_weights(self.model, self.pretrained_weights, self.checkpoint_key, self.arch, self.patch_size)
            #print(self.model)
            print("Done !")
            #print(self.model)
            
        
        elif self.model_name == "DINO_VIT":
#             print(self.pretrained_weights)
            self.checkpoint_key = checkpoint_key
            self.arch = arch
            self.patch_size = patch_size
            print("Loading pretrain model of DINO for finetuning ...")
    
            self.model = vits.__dict__["vit_base"](patch_size=self.patch_size, num_classes=0)
            self.fc = nn.Linear(768, self.num_class, bias=True)
            
            utils.dino_load_pretrained_weights(self.model, self.pretrained_weights, self.checkpoint_key, self.arch, self.patch_size)
            print("Done !")

        
        elif self.model_name == "DINO_RESNET":
            self.checkpoint_key = checkpoint_key
            self.arch = arch
            self.patch_size = patch_size
            print("Loading pretrain model of DINO for finetuning ...")
    
            self.model = models.resnet50(num_classes=6)
            
            utils.dino_load_pretrained_weights(self.model, self.pretrained_weights, self.checkpoint_key, self.arch, self.patch_size)
            print("Done !")
            
        
        
        else :
            from collections import OrderedDict
            print(self.pretrained_weights)
            self.num_head = num_head 
            self.model = VGGFACE2_DAN(self.num_head, False, self.pretrained_model_num_class)
            checkpoint = torch.load(self.pretrained_weights)
            
            state_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict() 
            for k, v in state_dict.items(): 
                name = k[7:] 
                new_state_dict[name] = v 
            self.model.load_state_dict(new_state_dict, strict = True)
            self.fc =  nn.Linear(512, 6) 
            self.bn = nn.BatchNorm1d(6)   
            
            
            
    
    def forward(self, x) :        
        if self.model_name == "DINO" or self.model_name == "DINO_RESNET" :
            x = self.model(x)  
            return x
    
        elif self.model_name == "DINO_VIT" :
            x = self.model(x)  
            x = self.fc(x)
            #print(np.shape(x))
            return x
    
        else:
            out, feat, heads = self.model(x)
            if self.pretrained_model_num_class == 8 :
                out = self.fc(heads.sum(dim=1))
                out = self.bn(out)
            
                return out, feat, heads
            else :
                return out, feat, heads



class mobile_vit2(nn.Module) :
    def __init__(self, task) :
        super(mobile_vit2, self).__init__() 
        print("Loading MobileViT v2 !!")
        self.task = task
        opts = './mobilevitv2/config/classification/finetune_in21k_to_1k/mobilevit_v2.yaml'
        #our_pretrined_model_path = '/abaw_4th/DAN/jy_scripts/mobilevitv2/mobilevitv2_results_in21k_ft_256/width_2_0_0/run_1/checkpoint_ema_best.pt'
        imagenet_pretrained_model_path = '/abaw_4th/DAN/scripts/mobilevitv2/checkpoints/mobilevitv2-2.0.pt'
        dev_id = getattr(opts, "dev.device_id", None)
        #device = getattr(opts, "dev.device", torch.device("cpu"))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(dev_id, device)
        if dev_id is None:
            model_state = torch.load(imagenet_pretrained_model_path, map_location=device)
            #model_state = torch.load(our_pretrined_model_path, map_location=device)
        #else:
            #model_state = torch.load(imagenet_pretrained_model_path, map_location="cuda:{}".format(dev_id))
            #model_state = torch.load(our_pretrined_model_path, map_location="cuda:{}".format(dev_id))
        self.model = get_model(opts)
        #self.model = self.model.to(device=device)
        #print(self.model)
        #model_state = torch.load(our_pretrined_model_path)
        model_state = torch.load(imagenet_pretrained_model_path)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state, strict = False)
        else:
            self.model.load_state_dict(model_state, strict = False)

        self.model = nn.Sequential(*(list(self.model.children())))
        #print(self.model)
        self.model = self.model[:-1]
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(1024, 6) 
        #print(self.model)
        print("Done !")
        
        #print(self.model)

    def forward(self, x) :
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        x = self.model(x)
        if self.task == 0 :
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

class MTL_classfier(nn.Module):
    def __init__(self, num_class=7):
        super(MTL_classfier, self).__init__()
        
        self.sharedfc1 = nn.Linear(512,256)
        self.sharedbn1 = nn.BatchNorm1d(256)
        self.sharedfc2 = nn.Linear(256,128)
        self.sharedbn2 = nn.BatchNorm1d(128)
        self.encode1 = nn.Linear(128,68)
        self.encode2 = nn.Linear(128,68)
        
        self.fc = nn.Linear(128, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x1, x2):
        
        out1 = self.sharedfc1(x1)
        out1 = self.sharedbn1(out1)
        out1 = self.sharedfc2(out1)
        out1 = self.sharedbn2(out1)
        out1 = self.fc(out1)
        out1 = self.bn(out1)

        out2 = self.sharedfc1(x2)
        out2 = self.sharedbn1(out2)
        out2 = self.sharedfc2(out2)
        out2 = self.sharedbn2(out2)   
        ox = self.encode1(out2)    
        oy = self.encode2(out2)
        ox=ox.unsqueeze(2)
        oy=oy.unsqueeze(2)

        out2 = torch.cat([ox,oy], dim=2)  
            
   
        return out1, out2
include_top = True 
N_IDENTITY = 8631 

class MTL_ResNet18(nn.Module):
    def __init__(self, num_head, num_class=8, pretrained=True):
        super(MTL_ResNet18, self).__init__()
        
        resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('../models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict = True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)


    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        
        return x, heads

class landmark_ResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super(landmark_ResNet18, self).__init__()
        
        self.resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('../models/resnet18_msceleb.pth')
            self.resnet.load_state_dict(checkpoint['state_dict'], strict = True)

        self.fc =  nn.Linear(1000,512)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
        
class MTL_ResNet50(nn.Module):
    def __init__(self,num_head=4,pretrained_path=""):
        super(MTL_ResNet50, self).__init__()
    
        
        self.resnet = ResNet.resnet50(pretrained_checkpoint_path=pretrained_path, num_classes=N_IDENTITY, include_top=include_top)
        self.features = nn.Sequential(*list(self.resnet.children())[0][:-1])

        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
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
    def forward(self, x):
        x = self.features(x)
        x=self.conv1x1_1(x) 
        x=self.conv1x1_2(x)

        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
        
        return x , heads

class Landmark_ResNet50(nn.Module) :
    def __init__(self,pretrained_path=""):
        super(Landmark_ResNet50, self).__init__()
        self.resnet = ResNet.resnet50(pretrained_checkpoint_path=pretrained_path, num_classes=N_IDENTITY, include_top=include_top)

        self.lmfc = nn.Linear(2048,512)
        self.lmbn = nn.BatchNorm1d(512)

    def forward(self, x):
        x2 = self.resnet(x)
        x2 = x2.squeeze()
        
        x2 = self.lmfc(x2)
        x2 = self.lmbn(x2)
        return x2

class MTL_finetuning(nn.Module) :
    def __init__(self, em_model_name = None, lm_model_name = None, em_pretrained_weights = None, lm_pretrained_weights = None,
                             checkpoint_key = None, em_arch = None, lm_arch = None, patch_size = None, num_class = None, pretrained = None, num_head = None) :
        super(MTL_finetuning, self).__init__() 
        self.em_model_name = em_model_name
        self.lm_model_name = lm_model_name
        self.em_pretrained_weights = em_pretrained_weights
        self.lm_pretrained_weights = lm_pretrained_weights
        self.num_class = num_class
        self.pretrained = pretrained

        
        if self.em_model_name == "RESNET50" :
            self.em_feature = VGGFACE2_DAN(pretrained_checkpoint_path=em_pretrained_weights, num_head = 4, pretrained = self.pretrained, num_class = self.num_class)

        elif self.em_model_name == "DINO_RESNET" :



            self.checkpoint_key = checkpoint_key
            self.em_arch = lm_arch
            self.patch_size = patch_size

            print("Loading pretrain model of DINO for finetuning ...")
            
            self.em_feature = models.resnet50(num_classes=512)
            utils.dino_load_pretrained_weights(self.em_feature, self.em_pretrained_weights, self.checkpoint_key, self.em_arch, self.patch_size)
            #print(self.em_feature)
            self.em_feature = nn.Sequential(*list(self.em_feature.children())[:-2])
            self.conv1x1_em = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.avgpool_ = nn.AvgPool2d(7, stride=1)
            
            print("Done !")

        elif self.em_model_name == "DINO_VIT" :
            self.checkpoint_key = checkpoint_key
            self.em_arch = em_arch
            self.patch_size = patch_size

            print("Loading pretrain model of DINO for finetuning ...")
            
            self.em_feature = vits.__dict__["vit_base"](patch_size = self.patch_size, num_classes = 0)
            self.em_fc = nn.Linear(768, 512)
            utils.dino_load_pretrained_weights(self.em_feature, self.em_pretrained_weights, self.checkpoint_key, self.em_arch, self.patch_size)
            print("Done !")
            
        elif self.em_model_name == "MobileVITv2" :
            self.em_feature = mobile_vit2(1)
            self.em_feature = nn.Sequential(*list(self.em_feature.children())[:-1])
            #print(self.em_feature)
            #print(self.em_feature)
            self.conv1x1_em = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            #self.avgpool = nn.AvgPool2d(8, stride=1)
            self.em_fc = nn.Linear(1024, 512)
        else :
            print("No models")
            pass
            


        if self.lm_model_name == "RESNET50" :
            self.lm_feature = mtl_resnet50(pretrained_checkpoint_path = self.lm_pretrained_weights, num_classes = 8631, include_top = True)
            self.conv1x1_lm = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )

        elif self.lm_model_name == "DINO_RESNET" :
            
            self.checkpoint_key = checkpoint_key
            self.lm_arch = em_arch
            self.patch_size = patch_size

            print("Loading pretrain model of DINO for finetuning ...")
            
            self.lm_feature = models.resnet50(num_classes = 512)
            utils.dino_load_pretrained_weights(self.lm_feature, self.lm_pretrained_weights, self.checkpoint_key, self.lm_arch, self.patch_size)

    
            print("Done !")

        elif self.lm_model_name == "DINO_VIT" :
            self.checkpoint_key = checkpoint_key
            self.lm_arch = lm_arch
            self.patch_size = patch_size

            print("Loading pretrain model of DINO for finetuning ...")
            
            self.lm_feature = vits.__dict__["vit_base"](patch_size = self.patch_size, num_classes = 0)
            
            self.lm_fc = nn.Linear(768, 512)
            utils.dino_load_pretrained_weights(self.lm_feature, self.lm_pretrained_weights, self.checkpoint_key, self.lm_arch, self.patch_size)
           
            print("Done !")

        elif self.lm_model_name == "MobileVITv2" :
            self.lm_feature = mobile_vit2(1)
            self.lm_feature = nn.Sequential(*list(self.lm_feature.children())[:-1])
            self.conv1x1_lm = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            #self.avgpool = nn.AvgPool2d(8, stride=1)
            self.lm_fc = nn.Linear(1024, 512)
            #print(self.lm_feature)
        else :
            print("No models")
            pass
        
        print(self.em_model_name)
        print(self.lm_model_name)
        
        self.classfier = MTL_classfier(num_class=num_class)
    
        
    def forward(self, x1, x2): 

        em_img = x1
        lm_img = x2
        em_x = None
        lm_x = None
        if (self.em_model_name == "RESNET50" and self.lm_model_name is not "RESNET50") or self.em_model_name == "RESNET50" and self.lm_model_name == "RESNET50" : 
            _, feat, heads = self.em_feature(em_img)
            
        elif (self.em_model_name == "DINO_RESNET" and self.lm_model_name is not "DINO_RESNET") or self.em_model_name == "DINO_RESNET" and self.lm_model_name == "DINO_RESNET" :
            #print("왜 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            feat = self.em_feature(em_img)
            feat = self.conv1x1_em(feat)
            #print("왜 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            feat_ = self.avgpool_(feat)
            
            em_x = feat_.view(feat_.size(0), -1)
            

        elif (self.em_model_name == "DINO_VIT" and self.lm_model_name is not "DINO_VIT") or self.em_model_name == "DINO_VIT" and self.lm_model_name == "DINO_VIT" :
            feat = self.em_feature(em_img)
            em_x = self.em_fc(feat)
            
        elif (self.em_model_name == "MobileVITv2" and self.lm_model_name is not "MobileVITv2") or self.em_model_name == "MobileVITv2" and self.lm_model_name == "MobileVITv2" :
            feat = self.em_feature(em_img)
            #print(feat.size())
            feat = self.conv1x1_em(feat)
            #feat_ = self.avgpool(feat)
            em_x = feat.view(feat.size(0), -1)


        if (self.lm_model_name == "RESNET50" and self.em_model_name is not "RESNET50") or self.lm_model_name == "RESNET50" and self.em_model_name == "RESNET50" :
            lm_x = self.lm_feature(lm_img)
            lm_x = self.conv1x1_lm(lm_x)
            lm_x = lm_x.view(lm_x.size(0), -1)

        elif (self.lm_model_name == "DINO_RESNET" and self.em_model_name is not "DINO_RESNET") or self.lm_model_name == "DINO_RESNET" and self.em_model_name == "DINO_RESNET" :
            lm_x = self.lm_feature(lm_img)

        elif (self.lm_model_name == "DINO_VIT" and self.em_model_name is not "DINO_VIT") or self.lm_model_name == "DINO_VIT" and self.em_model_name == "DINO_VIT" :
            lm_x = self.lm_feature(lm_img)
            lm_x = self.lm_fc(lm_x)

        elif (self.lm_model_name == "MobileVITv2" and self.em_model_name is not "MobileVITv2") or self.em_model_name == "MobileVITv2" and self.lm_model_name == "MobileVITv2" :
            lm_x = self.lm_feature(lm_img)
            lm_x = self.conv1x1_lm(lm_x)
            #lm_x = self.avgpool(lm_x)
            lm_x = lm_x.view(lm_x.size(0), -1)
                
        #print(lm_x)

        if self.em_model_name == "RESNET50" :    
            out1, out2 = self.classfier(heads.sum(dim = 1), lm_x)
        
            return out1, out2, feat, heads

        elif self.em_model_name == "DINO_VIT" :        
            out1, out2 = self.classfier(em_x, lm_x)

            return out1, out2
        
        else :
            feat = feat
            #print(feat.size(), em_x.size(), lm_x.size())
            
            out1, out2 = self.classfier(em_x, lm_x)
        
            return out1, out2, feat
            
