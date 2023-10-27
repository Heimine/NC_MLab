import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Model
class MLP(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_layers, num_classes=6, 
                 non_linear=True, bn=True, etf=False, fix_dim=False):
        super(MLP, self).__init__()
        
        print(f"Creating model of width {hidden_dim}, depth {num_layers+1}, use non_linear? {non_linear}, bn? {bn}, etf? {etf}, fix dim? {fix_dim} \n")
        
        self.num_classes = num_classes
        # First layer, from data dim to hidden dim
        layers = [] 
        for i in range(1, num_layers):
            if non_linear:
                # Add relu in between
                if bn:
                    if i == 1:
                        layers.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))
                    elif i == num_layers-1:
                        if fix_dim:
                            layers.append(nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.ReLU(), nn.BatchNorm1d(num_classes)))
                        else:
                            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))
                    else:
                        layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)))
                else:
                    if i == 1:
                        layers.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU()))
                    elif i == num_layers-1:
                        if fix_dim:
                            layers.append(nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.ReLU()))
                        else:
                            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
                    else:
                        layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
            else: 
                # Totoally linear model
                if i == 1:
                    layers.append(nn.Sequential(nn.Linear(data_dim, hidden_dim)))
                elif i == num_layers-1:
                    if fix_dim:
                        layers.append(nn.Sequential(nn.Linear(hidden_dim, num_classes)))
                    else:
                        layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim)))
                else:
                    layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim)))
            
        self.layers = nn.ModuleList(layers)
        self.non_linear = non_linear
        # Final classifier
        
        if etf:
            print("ETF Classifier!")
            # self.fc = nn.Linear(hidden_dim, num_classes, bias=False)
            self.fc = nn.Linear(hidden_dim, num_classes)
            weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
            
            if not fix_dim:
                weight = weight @ torch.eye(num_classes, hidden_dim)
            
            self.fc.weight.data = weight
            self.fc.weight.requires_grad_(False)
        else:
            self.fc = nn.Linear(hidden_dim, num_classes)
            
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        out = self.fc(x)
        return out, x
    
# Dataset 
class MultiLabelDataset(Dataset):

    def __init__(self, data, target):
        super(MultiLabelDataset, self).__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]