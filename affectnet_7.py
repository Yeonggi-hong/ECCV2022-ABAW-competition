import os
import sys
import glob
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

# +
import torch
import torch.nn as nn 
import torch.utils.data as data
from torchvision import transforms, datasets
from sklearn.metrics import f1_score

from networks.dan_p import DAN_ab
from torch.nn import functional as F
from torch.autograd import Variable
eps = sys.float_info.epsilon
# -

import result_visualize as visualize
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../3th_dataset/train', help='train dataset path.')
    parser.add_argument('--val_path', type=str, default='../3th_dataset/val', help='val dataset path.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=15, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of class.')
    parser.add_argument('--pretrained', type=str2bool, default=True ,help = 'pretrained model?')

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
class AffectNet(data.Dataset):
    def __init__(self, aff_path, phase, use_cache = True, transform = None):
        self.phase = phase
        self.transform = transform
        self.aff_path = aff_path
        
        if use_cache:
            cache_path = os.path.join(aff_path,'affectnet.csv')
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
            else:
                df = self.get_df()
                df.to_csv(cache_path)
        else:
            df = self.get_df()

        self.data = df[df['phase'] == phase]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

    def get_df(self):
        train_path = os.path.join(self.aff_path,'train_set/')
        val_path = os.path.join(self.aff_path,'val_set/')
        data = []
        
        for anno in glob.glob(train_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(train_path,f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['train',img_path,label])
        
        for anno in glob.glob(val_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(val_path,f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['val',img_path,label])
        
        return pd.DataFrame(data = data,columns = ['phase','img_path','label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            ## add eps to avoid empty var case
            loss = torch.log(1+num_head/(var+eps))
        else:
            loss = 0
            
        return loss


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def run_training():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN_ab(num_class=7, num_head=4,pretrained=args.pretrained)
    
    model.to(device)
    if ((device.type == 'cuda') and (torch.cuda.device_count()>1)):
        print('Multi GPU activate')
        model = nn.DataParallel(model)
        model = model.cuda()    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    train_dataset = datasets.ImageFolder(f'{args.train_path}', transform = data_transforms)   # loading statically
    # if args.num_class == 7:   # ignore the 8-th class
    #     idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] != 7]
    #     train_dataset = data.Subset(train_dataset, idx)

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False, 
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])      

    val_dataset = datasets.ImageFolder(f'{args.val_path}', transform = data_transforms_val)    # loading statically

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)


    criterion_cls = FocalLoss()
    criterion_af = AffinityLoss(device, num_class=args.num_class)
    criterion_pt = PartitionLoss()

    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    
    model_info={"epoch":[], "train_acc":[], "train_loss":[], "lr":[], "val_acc":[], "val_loss":[], "f1":[]}
    
    best_acc = 0
    best_f1 = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)

            # targets = targets[:]-1
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)

            loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        model_info["epoch"].append(epoch)        
        model_info["train_acc"].append(acc.item())
        model_info['train_loss'].append(running_loss.item())
        model_info["lr"].append(optimizer.param_groups[0]['lr'])
        
        
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
            for imgs, targets in tqdm(val_loader):
        
                imgs = imgs.to(device)
                imgs = imgs.float()
                targets_ =targets
                targets_=targets_.to(device)
                
        
                out,feat,heads = model(imgs)

                loss = criterion_cls(out,targets_) + criterion_af(feat,targets_) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
            
                correct_num  = torch.eq(predicts,targets_)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                for p, t in zip(predicts, targets_) :
                    p_.append(p.cpu())
                    t_.append(t.cpu())
               
                
            running_loss = running_loss/iter_cnt   
            scheduler.step()
            f1=[]
       
            temp_exp_pred = np.array(p_)
            temp_exp_target = np.array(t_)
            temp_exp_pred = torch.eye(7)[temp_exp_pred]
            temp_exp_target = torch.eye(7)[temp_exp_target]
            
            for i in range(0,7):
                
                exp_pred = temp_exp_pred[:,i]
                exp_target = temp_exp_target[:,i]
                
                f1.append(f1_score(exp_pred,exp_target))  
                
                
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)
            running_f1 = np.mean(f1)
            
            
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f. f1: %.4f" % (epoch, acc, running_loss,running_f1))
            tqdm.write("best_acc:" + str(best_acc))
            tqdm.write("best_f1:" + str(max(running_f1, best_f1)))

            model_info["val_acc"].append(acc)
            model_info["val_loss"].append(running_loss.item())
            model_info["f1"].append(running_f1)

            print(model_info)


            if running_f1>best_f1 :
                model_name="model_"+str(epoch)+"th_model"+"_batch_"+str(args.batch_size)+"_pretrained_"+str(args.pretrained)+"_f1_"+str(running_f1)
                
                
                best_f1 = running_f1
                
                y_hat = temp_exp_pred
                y_test = temp_exp_target
                cm = confusion_matrix(y_test.argmax(axis=1), y_hat.argmax(axis=1))
      
                
                visualize.plot_confusion_matrix(cm, model_name, 
                                                target_names=['Neutral','Anger', 'Disgust', 'Fear', 'Happiness','Sadness','Surprise'], 
                                                title="DAN")
                
                if args.pretrained:
                    torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                           os.path.join('checkpoints/pretrain/finetuning', model_name+".pth"))
                else:
                    torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                           os.path.join('checkpoints/pretrain/scratch', model_name+".pth"))
                tqdm.write('Model saved.')
                
    visualize.save_plt(model_name, model_info)

if __name__ == "__main__":                    
    run_training()
