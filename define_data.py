import torch
import torchvision
import numpy as np
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import pandas as pd
import ast

class SVHN(torch.utils.data.Dataset):
    """
    SVHN dataset, remove all 0s
    """
    def __init__(
        self,
        index_name,
        train=True,
        transform=None):
        super().__init__()
        
        root_dir = "/scratch/qingqu_root/qingqu1/xlxiao/DL/data/SVHNformat1/"
        self.transform = transform

        if train:
            label_data = pd.read_csv(root_dir + "labels_train.csv")
            with open(root_dir + index_name, 'rb') as f: 
                self.ids = pickle.load(f)["indices_train"]
            self.root_dir = root_dir + "train/"
        else:
            label_data = pd.read_csv(root_dir + "labels_test.csv")
            with open(root_dir + index_name, 'rb') as f: 
                self.ids = pickle.load(f)["indices_test"]
            self.root_dir = root_dir + "test/"

        self.image_files = label_data.name.values
        self.targets = label_data.unique_label.values

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        path = self.root_dir + self.image_files[id]
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target_list = ast.literal_eval(self.targets[id])
        target_list = np.array(target_list) - 1

        one_hot_target = np.zeros(9)
        one_hot_target[target_list] = 1

        if self.transform is not None:
            image = self.transform(image)

        return image, one_hot_target
    
class MNIST_Combine(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        super().__init__()
        
        self.root=root
        with open(root, 'rb') as f:
            contents = pickle.load(f)
        if train:
            self.data = contents['train_data']
            self.targets = contents['train_label']
        else:
            self.data = contents['test_data']
            self.targets = contents['test_label']
        print("data size", self.data.shape)
    
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        if "mnist" in self.root:
            return self.data[idx].unsqueeze(0), self.targets[idx]
        else:
            return self.data[idx], self.targets[idx]


def get_svhn(batch_size):

    transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    svhn_train = SVHN("no_zero_indices_mul1_balance.pkl", train=True, transform=transform_train)
    svhn_test = SVHN("no_zero_indices_mul1_balance.pkl", train=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
                    svhn_train, batch_size=batch_size, shuffle=True, num_workers=16)
    valloader = torch.utils.data.DataLoader(
                    svhn_test, batch_size=batch_size, shuffle=False, num_workers=16)
    
    return svhn_train, svhn_test, trainloader, valloader