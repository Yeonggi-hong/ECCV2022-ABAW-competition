
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn import functional as F

from networks.utils import ImbalancedDatasetSampler, save_plt, plot_confusion_matrix
from networks.loss import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../3th_dataset/datasets', help='AfectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=1500, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=6, help='Number of class.')
    parser.add_argument('--pretrained', type=str, default=True ,help = 'pretrained model')
    parser.add_argument('--model', type=str, default="VGGFACE_DAN" ,help = 'pretrained model')

    return parser.parse_args()


def run_training():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    if args.pretrained == "True" :
        pretrain_flag = True
    else :
        pretrain_flag = False
    weight_name = str(args.num_class) + "_class_pretrain+"+str(args.model)+"_model_" + "num_head_" + str(args.num_head) + "_weightinit_" + str(args.pretrained)
    plt_name = str(args.num_class) + "_class_pretrain+"+str(args.model)+"_model_" + "num_head_" + str(args.num_head) + "_weightinit_" + str(args.pretrained)
    print(weight_name)
    print(plt_name)
    if args.model == "DINO" :
        
        from networks.models import DINO
        if pretrain_flag:
            model = DINO("./models/dino_resnet50_40epoch.pth", "student", "resnet50", 8, args.num_class)
        else:
            model = DINO("./models/dino_resnet50_40epoch.pth", "taecher", "resnet50", 8, args.num_class)
        params = list(model.parameters())

    elif args.model == "DINO_DAN" :
        from networks.loss import AffinityLoss, PartitionLoss
        from networks.models import DINO_DAN
        model = DINO_DAN("./models/dino_resnet50_40epoch.pth", "student", "resnet50", 8, pretrained = pretrain_flag, num_head = args.num_head, num_class = args.num_class)
        
        criterion_af = AffinityLoss(device, num_class=args.num_class)
        criterion_pt = PartitionLoss()
        params = list(model.parameters()) + list(criterion_af.parameters())

    else :
        from networks.loss import AffinityLoss, PartitionLoss
        from networks.models import VGGFACE2_DAN
        model = VGGFACE2_DAN(pretrained = pretrain_flag, num_head = args.num_head, num_class = args.num_class)
        criterion_af = AffinityLoss(device, num_class=args.num_class)
        criterion_pt = PartitionLoss()
        params = list(model.parameters()) + list(criterion_af.parameters())
    
    criterion_cls = FocalLoss()
    model.to(device)
        
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    print(args.data_path)
    train_dataset = datasets.ImageFolder(f'{args.data_path}/train/', transform = data_transforms)   # loading statically

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False, 
                                               pin_memory = True,
                                               drop_last=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])      

    val_dataset = datasets.ImageFolder(f'{args.data_path}/val/', transform = data_transforms_val)    # loading statically

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True,
                                               drop_last=True)


    # Set Optimizer
    
    optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    
    
    #train
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        model = nn.DataParallel(model)
        model = model.cuda()
    
    history = {"epoch" : [], "train_acc" : [], "train_loss" : [], "lr" : [], "val_acc" : [], "val_loss" : [], "train_f1" : [], "val_f1" : []}
    best_f1 = 0
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        p_ = []
        t_ = []

        for (imgs, targets) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.float().to(device)
            targets = targets.to(device)
            if args.model == "DINO" :
                out = model(imgs)
                loss = criterion_cls(out, targets)
            else :
                out,feat,heads = model(imgs)
                loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            for p, t in zip(predicts, targets) :
                p_.append(p.cpu())
                t_.append(t.cpu())

        f1=[]
        temp_exp_pred = np.array(p_)
        temp_exp_target = np.array(t_)
        temp_exp_pred = torch.eye(args.num_class)[temp_exp_pred]
        temp_exp_target = torch.eye(args.num_class)[temp_exp_target]

        for i in range(0, args.num_class):
            exp_pred = temp_exp_pred[:, i]
            exp_target = temp_exp_target[:, i]
            
            f1.append(f1_score(exp_pred,exp_target))
        print(f1)
        running_f1 = np.mean(f1)
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.4f. F1 score: %.4f. LR %.6f' % (epoch, acc, running_loss, running_f1, optimizer.param_groups[0]['lr']))

        history["epoch"].append(epoch)        
        history["train_acc"].append(acc.item()*100.0)
        history['train_loss'].append(running_loss.item())
        history["lr"].append(optimizer.param_groups[0]['lr'])
        history["train_f1"].append(running_f1*100.0)

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            temp_exp_pred = []
            temp_exp_target = []
            p_ = []
            t_ = []
            pred = []
            corr = []
            for imgs, targets in tqdm(val_loader):
        
                imgs = imgs.float().to(device)
                targets = targets.to(device)
                if args.model == "DINO" :
                    out = model(imgs)
                    loss = criterion_cls(out, targets)
                else :
                    out,feat,heads = model(imgs)
                    loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
            
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                for p, t in zip(predicts, targets) :
                    p_.append(p.cpu())
                    t_.append(t.cpu())
                
            running_loss = running_loss/iter_cnt   
            scheduler.step()
            
            

            f1=[]
            temp_exp_pred = np.array(p_)
            temp_exp_target = np.array(t_)
            temp_exp_pred = torch.eye(args.num_class)[temp_exp_pred]
            temp_exp_target = torch.eye(args.num_class)[temp_exp_target]

            for i in range(0, args.num_class):
                exp_pred = temp_exp_pred[:,i]
                exp_target = temp_exp_target[:,i]
                
                f1.append(f1_score(exp_pred,exp_target))
            print(f1)

            
            for p, t in zip(predicts, targets) :
                pred.append(p.cpu())
                corr.append(t.cpu())
            #print(np.shape(p_))
            #print(np.shape(t_))

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)
            cm = confusion_matrix(t_, p_) #confusion matrix
            running_f1 = np.mean(f1)
            history["val_acc"].append(acc*100.0)
            history["val_loss"].append(running_loss.item())
            history["val_f1"].append(running_f1*100.0)

            if args.num_class == 6:
                tqdm.write("F1 score by classes: %.4f %.4f %.4f %.4f %.4f %.4f" %(f1[0], f1[1], f1[2], f1[3], f1[4], f1[5]))
            else :
                tqdm.write("F1 score by classes: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" %(f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7]))

            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f F1 score: %.4f" % (epoch, acc, running_loss, running_f1))
            tqdm.write("best_acc:" + str(best_acc))
            tqdm.write("best_f1: " + str(max(best_f1, running_f1)))

            #print("\n", temp_exp_pred, temp_exp_pred,  temp_exp_target, temp_exp_target)


            #model saving
            if running_f1>best_f1 :
                best_f1 = running_f1
                best_epoch = str(epoch)
                best_cm = cm
                best_history = pd.DataFrame({"model" : plt_name, "val_acc" : history["val_acc"][-1], "val_loss" : history["val_loss"][-1], "val_f1" : history["val_f1"][-1]}, index = [0])
                best_model_info = {'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
    print(history)
    torch.save(best_model_info,
                "../pretrain_checkpoints/"+best_epoch+"epoch_"+weight_name+".pth")
    tqdm.write('Model saved.')

    save_plt(plt_name, history)
    #save_pickle(plt_name+'.pickle', history)

    if args.num_class == 6 :
        plot_confusion_matrix(cm, plt_name, 
        target_names=['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise'], cmap=None, normalize=True, labels=True, title=None)
    else :
        plot_confusion_matrix(cm, plt_name, 
        target_names=['Neutral', 'Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Other'], cmap=None, normalize=True, labels=True, title=None)

    #os.path.exists('../result/csv/')
    if not os.path.exists('../result/csv/pretrain.csv'):
        best_history.to_csv('../result/csv/pretrain.csv', mode='w', header=True, index=False)
    else:
        best_history.to_csv('../result/csv/pretrain.csv', mode='a', header=False, index=False)
    
    
if __name__ == "__main__":                    
    run_training()
