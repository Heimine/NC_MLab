import os
import torch 
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import trange
import torch.nn.functional as F
from models import *
from define_data import MNIST_Combine, get_svhn
from utility import *
import pickle

def parse_eval_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', type=str, default='/scratch/qingqu_root/qingqu1/xlxiao/DL/data/mnist_combine.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='saved/param/')
    parser.add_argument('--arch', type=str, default='resnet18', help='Model Architecture')
    parser.add_argument('--num_classes', type=int, default=10, help='# of classes in this dataset')
    parser.add_argument("--etf", dest='etf', action="store_true")
    parser.add_argument("--fix_dim", dest='fix_dim', action="store_true")
    parser.add_argument("--imbalance", dest='imbalance', action="store_true")
    parser.add_argument('--p_name', type=str, default="info.pkl")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_eval_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Dataset
    if "c10" in args.dataset_root:
        print("Using Multi-label Cifar10 dataset! \n")
        args.in_ch = 3
        args.dataset = "cifar10"
    elif "mnist" in args.dataset_root:
        print("Using Multi-label MNIST dataset! \n")
        args.dataset = "mnist"
        args.in_ch = 1
    elif "svhn" in args.dataset_root:
        print("Using SVHN dataset! \n")
        args.dataset = "svhn"
        args.in_ch = 3
        args.num_classes = 9
    
    if args.dataset == "svhn":
        trainset, testset, trainloader, testloader = get_svhn(128)
    else:
        trainset = MNIST_Combine(args.dataset_root, train=True)
        trainloader = DataLoader(trainset, batch_size=200, shuffle=False)
    
    # Model
    if args.arch == "vgg16":
        model = VGG('VGG16',
                    in_ch=args.in_ch,
                    num_classes=args.num_classes, 
                    ETF_fc=args.etf,
                    fixdim=args.fix_dim).to(device)
    elif args.arch == "vgg19":
        model = VGG('VGG19',
                    in_ch=args.in_ch,
                    num_classes=args.num_classes, 
                    ETF_fc=args.etf,
                    fixdim=args.fix_dim).to(device)
    elif args.arch == "resnet18":
        model = resnet18(in_ch=args.in_ch,
                         num_classes=args.num_classes, 
                         ETF_fc=args.etf,
                         SOTA=True,
                         fixdim=args.fix_dim).to(device)
    elif args.arch == "resnet50":
        model = resnet50(in_ch=args.in_ch,
                         num_classes=args.num_classes, 
                         ETF_fc=args.etf,
                         SOTA=True,
                         fixdim=args.fix_dim).to(device)

    # Prepare dicts to save info
    info_dict = {'nc1': [], "nc2_w": [], "nc2_h": [], "nc3": [], "angle_metric": [], 'imb_als': []}
    
    # Register hooks
    for i in range(0,210,20): 
        print(f"Now processing epoch {i}")
        if i != 200: #200
            model.load_state_dict(torch.load(args.checkpoint_path + 'model_epoch_' + str(i) + '.pth')["state_dict"])
        else:
            model.load_state_dict(torch.load(args.checkpoint_path + 'model_last' + '.pth')["state_dict"])
        model.eval()
        
        if args.imbalance:
            nc1, nc2_w, nc2_h, nc3, angle_diff, angle_m, als, feature_dict = calculate_nc_stats(model, trainloader, device, imbalance=True)
        else:
            nc1, nc2_w, nc2_h, nc3, angle_diff, angle_m, feature_dict = calculate_nc_stats(model, trainloader, device)
            als = 0
        info_dict['nc1'].append(nc1)
        info_dict['nc2_w'].append(nc2_w)
        info_dict['nc2_h'].append(nc2_h)
        info_dict['nc3'].append(nc3)
        info_dict['angle_metric'].append(angle_m)
        info_dict['imb_als'].append(als)
        
    
    print(info_dict)

    args.p_name = f"info_{args.dataset}.pkl"
    with open(args.checkpoint_path + args.p_name, 'wb') as f: 
        pickle.dump(info_dict, f)

    save_feature = {"features": feature_dict}
    with open(args.checkpoint_path + "feature_dict", 'wb') as f: 
        pickle.dump(save_feature, f)

if __name__ == "__main__":
    main()