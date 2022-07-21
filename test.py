# -*- coding: utf-8 -*-
import os
import glob
from tqdm import tqdm
import argparse
import itertools
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
#from natsorted import natsorted
from networks.models_tmp import MTL_finetuning
from networks.loss import AffinityLoss, PartitionLoss
from natsort import natsorted
import sys

from torch.autograd import Variable
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning) 

test_csv_path = "../src/test1.csv"
img_path = "../cropped_aligned/"



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testconfig', type=str, default='testconfig.txt', help='testsetting')
    parser.add_argument('--test_path', type=str, default='path/to/test/', help='test dataset path.')
    parser.add_argument('--model', type=str, default='path/to/model/', help='test model path.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    return parser.parse_args()


class Testloader(data.Dataset):
    def __init__(self, path, transform = None):

        self.transform = transform
        self.file_paths1= natsorted(glob.glob(path))
        
        

    def __len__(self):
        return len(self.file_paths1)

    def __getitem__(self, idx):
        path = self.file_paths1[idx]
        name = path.replace("../../test_data/","")
        image1 = Image.open(path).convert('RGB')

        if self.transform is not None:
            image1 = self.transform(image1)
        
        return image1, name


class Model() :
    def __init__(self, device) :
        self.device = device
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])   
        self.all_models = []
        
    def load_all_models(self, weight_list):
        
        print("Load all models ... ")

        em_model = "RESNET50"
        
        
        for model_name in weight_list:
            em_arch = None
            lm_arch = None
            em_pretrained_weights = "../models/resnet50_ft_weight.pkl"
            lm_pretrained_weights=None
            pretrained_path = args.model 

            # load model from file
            if "DINO_RESNET" in model_name :
                print("DINO_RESNET************")
                lm_pretrained_weights = "../models/checkpoint0060.pth"
                lm_arch = 'resnet50'
                model = MTL_finetuning(em_model_name = em_model, lm_model_name = "DINO_RESNET",
                        em_pretrained_weights = em_pretrained_weights, lm_pretrained_weights = lm_pretrained_weights,
                        checkpoint_key = "student", em_arch = em_arch, lm_arch = lm_arch, patch_size = 16, num_class = 6, pretrained = True, num_head = 4)
                checkpoint = torch.load(pretrained_path + model_name)
                
            elif "MobileVITv2" in model_name:
                print("MobileVit************")
                lm_pretrained_weights = None
                #args.batch_size = 128
                model = MTL_finetuning(em_model_name = em_model, lm_model_name = "MobileVITv2",
                        em_pretrained_weights = em_pretrained_weights, lm_pretrained_weights = lm_pretrained_weights,
                        checkpoint_key = "student", em_arch = em_arch, lm_arch = lm_arch, patch_size = 16, num_class = 6, pretrained = True, num_head = 4)
                checkpoint = torch.load(pretrained_path + model_name)
                
            else:
                print("RESNET************")
                lm_pretrained_weights = "../models/resnet50_ft_weight.pkl"
                model = MTL_finetuning(em_model_name = em_model, lm_model_name = "RESNET50",
                        em_pretrained_weights = em_pretrained_weights, lm_pretrained_weights = lm_pretrained_weights,
                        checkpoint_key = "student", em_arch = em_arch, lm_arch = lm_arch, patch_size = 16, num_class = 6, pretrained = True, num_head = 4)
                checkpoint = torch.load(pretrained_path + model_name)
            

            if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
                print('Multi GPU activate')
                model = nn.DataParallel(model)
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            
            model.to(self.device)
            model.eval()    
            model_ = model
            
            self.all_models.append(model_)
            print('>loaded %s' % model_name)

        return self.all_models

    def fit(self, img):
        
        with torch.set_grad_enabled(False):
            img = img.to(self.device)

            DANloss = 0
            mse = 0
            nmse = 0
            outs = None

            for i in self.all_models:
                out, out2, feat, heads = i(img, img)
                if(outs == None):
                    outs=out
                else:
                    outs+=out
                
            _, pred = torch.max(outs,1)
            index = pred

            return index, outs.size(0), DANloss, mse, nmse

if __name__ == "__main__":
    args = parse_args()
    weight_list = os.listdir(args.pretrained)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Model(device)
    model.load_all_models(weight_list)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) 


    print("data Loader working !")

    val_dataset = Testloader(args.test_path+"*.jpg",transform=data_transforms_val)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True,
                                            )
    test = []
    with open(args.testconfig, 'w') as file:
        file.writelines("image,expression\n")
        with torch.no_grad():
            for (imgs,names) in tqdm(val_loader):
                
                imgs = imgs.float()

                
                predicts, _, _, _, _ = model.fit(imgs)
                for name ,pred in zip(names,predicts.cpu().detach().numpy()):
                    file.writelines(name+","+str(pred)+"\n")
                break
