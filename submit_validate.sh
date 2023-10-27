#!/bin/bash

module purge
source /scratch/qingqu_root/qingqu1/xlxiao/DL/dl_env/bin/activate

# SVHN
python validate_nc.py --checkpoint_path saved/param/svhn_resnet18_etf_False_fixdim_False_lambh_with_mul3/ --arch resnet18 --dataset_root svhn
python validate_nc.py --checkpoint_path saved/param/svhn_resnet50_etf_False_fixdim_False_lambh_with_mul3/ --arch resnet50 --dataset_root svhn

# mnist
python validate_nc.py --checkpoint_path saved/param/mnist_resnet18_etf_False_fixdim_False_lambh/ --arch resnet18 --dataset_root /scratch/qingqu_root/qingqu1/xlxiao/DL/data/mnist_combine.pkl
python validate_nc.py --checkpoint_path saved/param/mnist_resnet50_etf_False_fixdim_False_lambh/ --arch resnet50 --dataset_root /scratch/qingqu_root/qingqu1/xlxiao/DL/data/mnist_combine.pkl
python validate_nc.py --checkpoint_path saved/param/mnist_vgg16_etf_False_fixdim_False_lambh/ --arch vgg16 --dataset_root /scratch/qingqu_root/qingqu1/xlxiao/DL/data/mnist_combine.pkl
python validate_nc.py --checkpoint_path saved/param/mnist_vgg19_etf_False_fixdim_False_lambh/ --arch vgg19 --dataset_root /scratch/qingqu_root/qingqu1/xlxiao/DL/data/mnist_combine.pkl