#!/bin/bash

module purge
source /scratch/qingqu_root/qingqu1/xlxiao/DL/dl_env/bin/activate

arch=$1
data_path=$2

python train_param_eff.py --arch $arch --etf --fix_dim --dataset_root $data_path --epochs 200 --lr 0.1