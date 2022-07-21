from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from networks.prev_code.resnet import ResNet
import numpy as np
N_IDENTITY = 8631 
class DAN(nn.Module):
    def __init__(self, num_class=7,num_head=4, pretrained=True):
        super(DAN, self).__init__()
        
        # self.resnet = models.resnet18(pretrained)
        
        # if pretrained:
        #     checkpoint = torch.load('./models/resnet18_msceleb.pth')    
        #     self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        include_top = True 
        self.resnet = ResNet.resnet50(pretrained_checkpoint_path="./models/resnet50_ft_weight.pkl", num_classes=N_IDENTITY, include_top=include_top)
        print(self.resnet)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        #print(self.features)
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
        
        self.lmfc = nn.Linear(1000,512)
        self.lmbn = nn.BatchNorm1d(512)
        
        self.sharedfc1 = nn.Linear(512,256)
        self.sharedbn1 = nn.BatchNorm1d(256)
        self.sharedfc2 = nn.Linear(256,128)
        self.sharedbn2 = nn.BatchNorm1d(128)
        self.encode1 = nn.Linear(128,68)
        self.encode2 = nn.Linear(128,68)
        
        self.fc = nn.Linear(128, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        

    def forward(self, x):
        x1 = self.features(x)
        print(np.shape(x1))
        x2 = self.resnet(x)
        print(np.shape(x2))
        x1=self.conv1x1_1(x1) 
        x1=self.conv1x1_2(x1)

        
        x2 = self.lmfc(x2)
        x2 = self.lmbn(x2)
        
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x1))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
            
        out1 = self.sharedfc1(heads.sum(dim=1))
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
        #print(np.shape(ox),np.shape(oy))
        out2 = torch.cat([ox,oy], dim=2)  
            
        #print(np.shape(out1),np.shape(out2))
        return out1, x1, heads, out2 

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
        #print(np.shape(sa))
        y = self.attention(sa)
        out = sa * y
        
        return out

