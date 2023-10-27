"""
For the parameter efficient experiments
"""
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
        
def parse_eval_args():
    parser = argparse.ArgumentParser()

    # Directory Setting
    # Note dataset mean which dataset to transfer learning on
    parser.add_argument('--dataset_root', type=str, default='/scratch/qingqu_root/qingqu1/xlxiao/DL/data/mnist_combine.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='saved/param/')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=200, help='Max Epochs')
    parser.add_argument('--arch', type=str, default='resnet18', help='Model Architecture')
    parser.add_argument('--num_classes', type=int, default=10, help='# of classes in this dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lamb_w', type=float, default=5e-4, help='weight decay weight')
    parser.add_argument('--lamb_h', type=float, default=1e-3, help='feature decay weight') #resnet50 for 1e-4
    parser.add_argument("--etf", dest='etf', action="store_true")
    parser.add_argument("--fix_dim", dest='fix_dim', action="store_true")

    args = parser.parse_args()

    return args

def training(args, model, trainset, trainloader, criterion, optimizer, epoch):
    model.train()
    num_samples = len(trainset)

    iou = 0
    loss_cur_epoch = 0
    for i, (data, label) in enumerate(trainloader):

        data, label = data.to(args.device).float(), label.to(args.device).float()

        pred, feature = model(data)

        loss = criterion(pred, label) 
        loss_total = loss + args.lamb_h * torch.linalg.norm(feature)**2

        loss_cur_epoch += loss.item()

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        iou_cur = iou_measure(pred, label)

        iou += torch.sum(iou_cur).item()
    
    print(f"Training Epoch: {epoch}, Current loss is {loss_cur_epoch / num_samples}, Current iou is {iou / num_samples}")
    
    return loss_cur_epoch / num_samples, iou / num_samples

def testing(args, model, testset, testloader, epoch):
    model.eval()
    num_samples = len(testset)

    iou = 0
    loss_cur_epoch = 0
    for i, (data, label) in enumerate(testloader):
    
        data, label = data.to(args.device).float(), label.to(args.device).float()
        
        with torch.no_grad():
            pred, feature = model(data)

        iou_cur = iou_measure(pred, label)

        iou += torch.sum(iou_cur).item()
    
    print(f"Testing Epoch: {epoch}, Current test iou is {iou / num_samples}")
    
    return iou / num_samples

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
    else:
        raise ValueError("Please give a valid dataset path")
    
    if "imbalance" in args.dataset_root:
        print("Imbalance Data Training! \n")
        args.imbalance = True
    else:
        args.imbalance = False
    
    if args.dataset == "svhn":
        trainset, testset, trainloader, testloader = get_svhn(args.batch_size)
    else:
        trainset = MNIST_Combine(args.dataset_root, train=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        testset = MNIST_Combine(args.dataset_root, train=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    print(args)
    
    # Model, optimizer, criterion
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
    else:
        raise ValueError("Model type not supported!")
        
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    print(len(list(filter(lambda x: x.requires_grad, model.parameters()))))
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=args.lamb_w)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, eta_min=args.lr/1000)
    
    # Save path 
    if args.imbalance:
        checkpoint_dir = args.checkpoint_path + "/" + f"imbalance_m_{args.dataset}_{args.arch}_etf_{args.etf}_fixdim_{args.fix_dim}/"
    else:
        checkpoint_dir = args.checkpoint_path + "/" + f"{args.dataset}_{args.arch}_etf_{args.etf}_fixdim_{args.fix_dim}_lambh_with_mul3/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    # Training
    train_iou_all = []
    test_iou_all = []
    best_iou = 0
    epochs = args.epochs # How many epochs to run
    for epoch in range(args.epochs):
        is_best = False
        train_loss, train_iou = training(args, model, trainset, trainloader, criterion, optimizer, epoch)
        test_iou = testing(args, model, testset, testloader, epoch)
        scheduler.step()
        train_iou_all.append(train_iou)
        test_iou_all.append(test_iou)
        
        if test_iou > best_iou:
            best_iou = test_iou
            is_best = True
            
        if is_best:
            print("Save current model (best)")
            state = {
                    'args': args,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'test_acc': test_iou
                }
            path = checkpoint_dir + 'model_best.pth'
            torch.save(state, path)
        if epoch % 20 == 0:
            nc1, nc2_w, nc2_h, nc3, angle_diff, angle_m, feature_dict = calculate_nc_stats(model, trainloader, device)
            print("Current angle metric", angle_m)
            print(f"Save current model (epoch:{epoch})")
            state = {
                    'args': args,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'test_acc': test_iou
                }
            path = checkpoint_dir + f'model_epoch_{epoch}.pth'
            torch.save(state, path)
        elif epoch + 1 == args.epochs:
            print("Save current model (last)")
            state = {
                    'args': args,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'test_acc': test_iou
                }
            path = checkpoint_dir + 'model_last.pth'
            torch.save(state, path)
    
    print(train_iou_all)
    print(test_iou_all)
    print(f"\n Training done, best test iou: {best_iou}")
        
        
if __name__ == "__main__":
    main()