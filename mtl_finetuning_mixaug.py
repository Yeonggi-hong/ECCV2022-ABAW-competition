# -*- coding: utf-8 -*-
import os
import sys
import glob
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.autograd import Variable

from networks.models_tmp import MTL_finetuning
from networks.utils import mixaug_data, mixaug_criterion
from networks.loss import AffinityLoss, PartitionLoss
eps = sys.float_info.epsilon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path1', type=str, default='/path/to/dataset/', help='Dataset path.')
    parser.add_argument('--aff_path2', type=str, default='/path/to/landmark/', help='Landmark path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=12, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=6, help='Number of class.')
    parser.add_argument('--em_model', type=str, default="MobileVITv2" ,help = 'Pretrained model')
    parser.add_argument('--lm_model', type=str, default="MobileVITv2" ,help = 'Pretrained model')
    parser.add_argument('--aug', type=str, default="True" ,help = 'Augmentation?')
    return parser.parse_args()


class MTL_loader(data.Dataset):
    def __init__(self, path1,path2,num_class, transform = None):
        filepath="/file/path"
        self.transform = transform
        self.file_paths1=[]
        self.label1=[]
        self.label2=[]
        for i in range(0,num_class):
            if(path1 == filepath):
                imagepath1 = path1 +"/"+str(i+1)+"/"
                imagepath2 = path2 +"/"+str(i+1)+"/"
            else:
                imagepath1 = path1 +"/"+str(i)+"/"
                imagepath2 = path2 +"/"+str(i)+"/"
           
            filst1 = sorted(glob.glob(imagepath1+"*.jpg"))
            filst2 = sorted(glob.glob(imagepath2+"*.txt"))
            self.file_paths1+=filst1
            
            for j in range(len(filst2)):    
                f = pd.read_csv(filst2[j], sep=" ", engine='python', header=None)
                lines = f.values
                self.label1.append(i)
                self.label2.append(lines)


    def __len__(self):
        return len(self.label2)

    def __getitem__(self, idx):
        path = self.file_paths1[idx]
        image1 = Image.open(path).convert('RGB')
        label1 = self.label1[idx]
        label2 = self.label2[idx]
        label2 = np.array(label2)
        if self.transform is not None:
            image1 = self.transform(image1)
        
        return image1, label1, label2


def run_training():
    args = parse_args()
    torch.manual_seed(17)   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    print("finetuning START ! !")
    print(args.em_model, args.lm_model)
    
    weight_name =  str(args.data) + "_Mixaug_Emotion_model_" + str(args.em_model) + "_Landmark_model_" + str(args.lm_model) +"_AUG_" + str(args.aug)
    
    print(weight_name)
    
    em_arch = None
    lm_arch = None
    em_pretrained_weights=None
    lm_pretrained_weights=None

    if args.em_model == "RESNET50" :
        em_pretrained_weights = "../models/resnet50_ft_weight.pkl"

    elif args.em_model == "DINO_RESNET":
        em_pretrained_weights = "../models/checkpoint0060.pth"
        em_arch = 'resnet50'

    elif args.em_model == "MobileVITv2":
        em_pretrained_weights = None
        args.batch_size = 128
    
    if args.lm_model == "RESNET50" :
        lm_pretrained_weights = "../models/resnet50_ft_weight.pkl"  
        
    elif args.lm_model == "DINO_RESNET":      
        lm_pretrained_weights = "../models/checkpoint0060.pth"
        lm_arch = 'resnet50'
    
    elif args.lm_model == "MobileVITv2":
        lm_pretrained_weights = None
        args.batch_size = 128


    model = MTL_finetuning(em_model_name = args.em_model, lm_model_name = args.lm_model,
                        em_pretrained_weights = em_pretrained_weights, lm_pretrained_weights = lm_pretrained_weights,
                        checkpoint_key = "student", em_arch = em_arch, lm_arch = lm_arch, patch_size = 16, num_class = 6, pretrained = True, num_head = 4)

    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        model = nn.DataParallel(model)
        model = model.cuda()
        
    model.to(device)

    em_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
                ], p=0.7),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing()
            ])

    lm_transforms = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            ])

    if args.aug == "False" :
        data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
    else :
        print("Data augmentation")
        data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor()
            
            ])

    data_transforms_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ]) 

    print("Raw dataset")
    train_dataset = MTL_loader(args.aff_path1 + 'train', args.aff_path2 + 'train', transform = data_transforms)   
    print("DONE !")
    
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True, 
                                               pin_memory = True,)

    val_dataset1 = MTL_loader(args.aff_path1 + 'val', args.aff_path2 + 'val', transform = data_transforms_val)  # loading dynamically

    print('Validation set size:', val_dataset1.__len__())

    val_loader1 = torch.utils.data.DataLoader(val_dataset1,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True,)
    
    nSamples = [18286, 15150, 10923, 73285, 144631, 14976] 
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(normedWeights).to(device) 
    
    criterion_af = AffinityLoss(device, num_class = args.num_class, feat_dim=512)
    criterion_pt = PartitionLoss()
    criterion_mse = torch.nn.MSELoss().to(device)
    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.Adam(params, args.lr, weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)

    
    best_acc = 0
    best_f1 = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        running_nmse = 0.0
        
        for (imgs1, targets1, targets2) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            em_img = em_transforms(imgs1) 
            em_img = em_img.to(device)
            lm_img = lm_transforms(imgs1)
            lm_img =lm_img.to(device)

            if args.em_model == "RESNET50" :   
                emimgs_mixed,imgs_a,imgs_b, targets_a, targets_b, lam = mixaug_data(em_img, targets1, alpha = 0.1, use_cuda = True)
                emimgs_mixed,imgs_a,imgs_b, targets_a, targets_b = map(Variable, (emimgs_mixed,imgs_a,imgs_b, targets_a, targets_b))
                out, out2, feat, heads = model(emimgs_mixed,lm_img)
                out_a, _ , feat_a, heads_a = model(imgs_a,lm_img)
                out_b, _ ,feat_a, heads_b = model(imgs_b,lm_img)
                mixaug_loss = mixaug_criterion(criterion_cls, out,out_a,out_b, targets_a, targets_b, lam)
                DANloss = criterion_af(feat, targets_b) + criterion_pt(heads) +criterion_af(feat_a, targets_a) + criterion_pt(heads_a)+ criterion_af(feat, targets_b) + criterion_pt(heads_b) +mixaug_loss 
                

            elif args.em_model == "DINO_VIT":
                imgs_mixed,imgs_a,imgs_b, targets_a, targets_b, lam = mixaug_data(em_img, targets1, alpha = 0.1, use_cuda = True)
                imgs_mixed,imgs_a,imgs_b, targets_a, targets_b = map(Variable, (imgs_mixed,imgs_a,imgs_b, targets_a, targets_b))
                out,out2 = model(imgs_mixed,lm_img)
                out_a,_ = model(imgs_a,lm_img)
                out_b,_ = model(imgs_b,lm_img)
                DANloss = mixaug_criterion(criterion_cls, out,out_a,out_b, targets_a, targets_b, lam)

            elif args.em_model == "MobileVITv2" or args.em_model == "MobileVITv2_ImageNet"  :  
                imgs_, targets_a, targets_b, lam = mixaug_data(em_img, targets1, alpha = 0.1, use_cuda = True)
                imgs_, targets_a, targets_b = map(Variable, (imgs_, targets_a, targets_b))
                out, out2, feat = model(imgs_, lm_img)

                mixup_loss = mixaug_criterion(criterion_cls, out, targets_a, targets_b, lam) 
                
                DANloss =  criterion_af(feat, targets_b) + mixup_loss

            else :
                
                imgs_mixed,imgs_a,imgs_b, targets_a, targets_b, lam = mixaug_data(em_img, targets1, alpha = 0.1, use_cuda = True)
                imgs_mixed,imgs_a,imgs_b, targets_a, targets_b = map(Variable, (imgs_mixed,imgs_a,imgs_b, targets_a, targets_b))
                out,out2 = model(imgs_mixed,lm_img)
                out_a,_ = model(imgs_a,lm_img)
                out_b,_ = model(imgs_b,lm_img)
                mixaug_loss = mixaug_criterion(criterion_cls, out,out_a,out_b, targets_a, targets_b, lam)
                
                DANloss =  criterion_af(feat, targets_b) + mixaug_loss
                
                
            mse = criterion_mse(out2.float(),targets2.float())
            nmse = mse / criterion_mse(out2, torch.zeros(out2.size(0), out2.size(1), out2.size(2)).to(device))
            loss = DANloss + mse 

            loss.backward()
            optimizer.step()
            
            running_loss += DANloss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets1).sum()
            correct_sum += correct_num
            running_nmse += nmse
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        running_nmse = running_nmse/iter_cnt
          
        tqdm.write('[Epoch %d] Training accuracy: %.4f. DANLoss: %.3f. LR %.6f NMSE %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr'], running_nmse))
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            running_nmse = 0.0
            running_emotion_loss = 0.0
            running_landmark_loss = 0.0
            model.eval()
            temp_exp_pred = []
            temp_exp_target = []
            p_ = []
            t_ = []
            for (imgs1, targets1, targets2) in tqdm(val_loader1):
        
                imgs1 = imgs1.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device) 
                em_img = imgs1

                if args.em_model == "RESNET50" :   
                    out, out2, feat, heads = model(em_img, imgs1)

                    DANloss = criterion_cls(out,targets1) + criterion_af(feat,targets1) + criterion_pt(heads)

                elif args.em_model == "DINO_VIT":
                    out, out2 = model( em_img, imgs1)

                    DANloss =  criterion_cls(out,targets1)
                    
                else :
                    out, out2, feat = model( em_img, imgs1)

                    DANloss =  criterion_cls(out, targets1) + criterion_af(feat, targets1)

                mse = criterion_mse(out2.float(),targets2.float())
                nmse = mse /criterion_mse(out2, torch.zeros(out2.size(0), out2.size(1), out2.size(2)).to(device))

                running_emotion_loss += DANloss
                running_landmark_loss += mse
                loss = DANloss + mse 

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets1)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                running_nmse += nmse
                for p, t in zip(predicts, targets1) :
                    p_.append(p.cpu())
                    t_.append(t.cpu())
            
            running_landmark_loss = running_landmark_loss/iter_cnt
            running_emotion_loss = running_emotion_loss/iter_cnt
            running_loss = running_loss/iter_cnt 
            running_nmse = running_nmse/iter_cnt  
            scheduler.step()
            
            f1=[]             
            temp_exp_pred = np.array(p_)
            temp_exp_target = np.array(t_)
            temp_exp_pred = torch.eye(6)[temp_exp_pred]
            temp_exp_target = torch.eye(6)[temp_exp_target]
            
            for i in range(0, 6):
                exp_pred = temp_exp_pred[:, i]
                exp_target = temp_exp_target[:, i]
                f1.append(f1_score(exp_pred,exp_target))

                
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            running_f1 = np.mean(f1)

            tqdm.write("F1 score by classes: %.4f %.4f %.4f %.4f %.4f %.4f" %(f1[0], f1[1], f1[2], f1[3], f1[4], f1[5]))
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f F1 score: %.4f" % (epoch, acc, running_loss, running_f1))
            tqdm.write("best_acc:" + str(best_acc))
            tqdm.write("best_f1: " + str(max(best_f1, running_f1)))

            if running_f1>best_f1 :
                best_f1 = running_f1
                best_epoch = str(epoch)               
                best_model_info = {'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
                torch.save(best_model_info,
                            "../checkpoint/" + best_epoch + "_epoch_" + weight_name + ".pth")
                tqdm.write('Model saved.')
                

if __name__ == "__main__":                    
    run_training()
