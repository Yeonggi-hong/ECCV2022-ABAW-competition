import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch.utils.data as data
from torchvision import datasets
import matplotlib.pyplot as plt
import itertools
def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe
    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict


def dino_load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weight")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

 
def val_accuracy(output, target, p_, t_, f1, running_accuracy, total):

    _, pred = torch.max(output, 1)
    pred = torch.nn.functional.one_hot(pred, num_classes=8).float()
    tmp = f1_score(target.cpu(), pred.cpu(), average="macro")

    f1 = np.append(f1, 100*tmp)

    
    for p, t in zip(pred, target) :
        p_.append(p.argmax().cpu())
        t_.append(t.argmax().cpu())
       
    correct = accuracy_score(target.cpu(), pred.cpu())
    #print(f1)
    running_accuracy += correct
    total += target.size(0)
    res = [p_, t_]

    return running_accuracy, total, res, f1

def accuracy(output, target, f1, running_accuracy, total):

    _, pred = torch.max(output, 1)
    
    
    pred = torch.nn.functional.one_hot(pred, num_classes=8).float()
    tmp = f1_score(target.cpu(), pred.cpu(), average="macro")
    f1 = np.append(f1, 100*tmp)
    #print(f1_)
    #print(f1)
    #print(f1_)
    correct = accuracy_score(target.cpu(), pred.cpu())
    #print(f1)
    running_accuracy += correct
    total += target.size(0)
    
    return running_accuracy, total, f1   



def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

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



def isexist(path):
    return os.path.exists(path)

def get_eq(pred, out) :
    tmp = torch.eq(pred, out).sum()
    return tmp

def save_plt(model_name, model_info) :
    print("Save plt ...")
    path = "../result/plt/"
    isexist_ = isexist(path)
    print(isexist_)
    if isexist_ is False:
        os.mkdir(path)

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.set_ylim(0.0,5.00)
    acc_ax.set_ylim(0.0,100.0)

    loss_ax.plot(model_info['train_loss'], 'y', label='train loss')
    loss_ax.plot(model_info['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(model_info['train_f1'], 'b', label='train f1')
    acc_ax.plot(model_info['val_f1'], 'g', label='val f1')
    acc_ax.set_ylabel('F1 score')
    acc_ax.legend(loc='lower left')

    plt.savefig(path+model_name+".png")

    print("Done !")

def plot_confusion_matrix(cm, model_name, target_names=None, cmap=None, normalize=True, labels=True, title=None):
    print("Save CM ...")
    path = "../result/cm/"
    isexist_ = isexist(path)
    print(isexist_)
    if isexist_ is False:
        os.mkdir(path)

    #fig1 = plt.gcf()
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(model_name + " confusion matrix")
    #plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    plt.savefig(path+model_name+".png")

    print("Done !")
    plt.close()

def save_pickle(pickle_name, history):
    print("Saving pickle...")
    path = '../result/pickle/'
    print(path+pickle_name)
    with open(path+pickle_name, 'wb') as f:
        pickle.dump(history, f)
    print("Done !")