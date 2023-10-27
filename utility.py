import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io.arff as sioarff
import scipy.linalg as scilin

def f1_score(tp,fp,fn):
    """
    p is mul-1, n is mul-n
    tp is predicted mul-1, gt mul-1
    tn is predicted mul-n, gt mul-n
    fp is predicted mul-1, gt mul-n
    fn is predicted mul-n, gt mul-1
    """
    return (2 * tp) / (2*tp+fp+fn)

def iou_measure(pred, target):
    num_class_sample = torch.sum(target != 0, dim=1) # Number of classes each sample belong to
    pred_prob = F.softmax(pred, dim=1)
    if torch.sum(torch.sum(target, dim=1) == 0) != 0:
        print("warning: all entries of target is 0...")
    target_prob = target / torch.sum(target, dim=1)[:,None]
    target_prob = torch.nan_to_num(target_prob)
    iou = torch.sum(pred_prob * target_prob, dim=1) / 2 
    iou = iou * num_class_sample * 2
    # print(iou)
    
    return iou

def preprocess(file_name, num_classes, train=True): 
    
    # Get data
    if train:
        file_folder = f'/scratch/qingqu_root/qingqu1/shared_data/{file_name}/{file_name}-train.arff'
    else:
        file_folder = f'/scratch/qingqu_root/qingqu1/shared_data/{file_name}/{file_name}-test.arff'
    total_data = sioarff.loadarff(file_folder)[0]
    print(total_data.shape)
    num_data = total_data.shape[0]
    total_length = len(total_data[0])
    im_len = total_length-num_classes
    
    # Actual pre-process part
    # Now we preprocess the data to ensure balanceness for multiplicity 1
    image_data_all, label_data_all = [], []
    label_sum = np.zeros(num_classes+1)  # Changed
    count = 0
    for i in range(num_data):
        data = total_data[i]
        image_data = np.array([data[i] for i in range(im_len)]).astype(float)
        label_data = np.array([int(str(data[i])[-2]) for i in range(im_len,total_length)])
        label_data = np.array([0, *label_data]) # Added (no label)
        
        if np.sum(label_data) == 0:
            count += 1
            label_data[0] = 1 # Added (no label)
        else:
            if np.sum(label_data) == 1:
                label_sum += label_data
            # image_data_all.append(image_data)
            # label_data_all.append(label_data)
        image_data_all.append(image_data)
        label_data_all.append(label_data)
    print("no label count", count)
    
    image_data_all, label_data_all = np.stack(image_data_all, axis=0), np.stack(label_data_all, axis=0)
    
    # Check balanceness
    try:
        assert np.std(label_sum) == 0
        print(f"All class have {label_sum[0]} samples of multiplicity 1!")
    except:
        print(f"Here's the label counts (m1): {label_sum}")
        print(f"Here's the label counts (m all): {np.sum(label_data_all, axis=0)}")
    
    return image_data_all, label_data_all, im_len

# To make sure all class occurs the same time **in the multiplicity 1 case**
def preprocess_v2(file_name, num_classes): 
    
    def find_min_samples(label_data_total, num_classes):
        # First find the argmin and where's the argmin
        label_sum = np.zeros(num_classes) # Help us to keep track of how many samples each label have
        for i in range(len(label_data_total)):
            label_data = label_data_total[i]
            if np.sum(label_data) == 1:
                label_sum += label_data

        min_samples = np.min(label_sum)
        print()
        print(f"The distribution of classes for multiplicity 1 is {label_sum}, with the minimum {min_samples}")
        if min_samples < 2:
            print("Change min samples to 2")
            min_samples = 2
        return min_samples
    
    # Get data
    file_folder = f'/scratch/qingqu_root/qingqu1/shared_data/{file_name}/{file_name}-train.arff'
    total_data = sioarff.loadarff(file_folder)[0]
    # file_folder = f'/scratch/qingqu_root/qingqu1/shared_data/{file_name}/{file_name}-test.arff'
    # total_data_2 = sioarff.loadarff(file_folder)[0]
    # total_data = np.hstack([total_data, total_data_2])
    print(total_data.shape)
    num_data = total_data.shape[0]
    total_length = len(total_data[0])
    im_len = total_length-num_classes
    
    # Prepare part
    label_data_total = [] # For finding # of labels in multiplicity 1 case
    for i in range(num_data):
        data = total_data[i]
        label_data = np.array([int(str(data[i])[-2]) for i in range(im_len,total_length)])
        label_data_total.append(label_data)
    
    min_samples = find_min_samples(label_data_total, num_classes)
    
    # Actual pre-process part
    # Now we preprocess the data to ensure balanceness for multiplicity 1
    image_data_all, label_data_all = [], []
    label_sum = np.zeros(num_classes) 
    for i in range(num_data):
        data = total_data[i]
        image_data = np.array([data[i] for i in range(im_len)]).astype(float)
        label_data = np.array([int(str(data[i])[-2]) for i in range(im_len,total_length)])
        
        if np.sum(label_data) == 1 and label_sum[label_data==1] >= min_samples:
            pass # In this case, we don't accept this sample to ensure balance
        else:
            image_data_all.append(image_data)
            label_data_all.append(label_data)
            if np.sum(label_data) == 1:
                label_sum += label_data
        
    min_samples = find_min_samples(label_data_all, num_classes)
    
    image_data_all, label_data_all = np.stack(image_data_all, axis=0), np.stack(label_data_all, axis=0)
    
    # Check balanceness
    try:
        assert np.std(label_sum) == 0
        print(f"All class have {label_sum[0]} samples of multiplicity 1!")
    except:
        print(f"Here's the label counts: {label_sum}")
        print("Samples of multiplicity 1 are roughly equal")
    
    return image_data_all, label_data_all, im_len

def compute_info(model, dataloader, device):
    """
    Aim: Store a dictionary of two layers contain all features
         The first layer is about multiplicity
             (i.e., Dict[m] will give a new dict containing all features for multiplicity m)
         The second layer is about class
             (i.e., Dict[m][k] will be an array containing all features with multiplicity m and k 
                    represents the label set, e.g., k could be '12' or '24', etc)
    """
    model.eval()
    num_data = 0
    mu_G = 0 # Only care about multiplicity 1
    mu_c_dict = dict() # Only care about multiplicity 1
    num_class_dict = dict() # Only care about multiplicity 1
    before_class_dict = dict() # Store features (care about all multiplicity)
    
    all_features = []
    all_labels = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        with torch.no_grad():
            output, features = model(inputs)
            
        all_features.extend(features.cpu())
        all_labels.extend(targets.cpu())
        num_data += targets.shape[0]
    
    # Process these features according to the label
    for i in range(len(all_labels)):
        cur_label = all_labels[i]
        cur_feature = all_features[i]
        
        multip = torch.sum(cur_label != 0).item() # multiplicity
        if multip not in before_class_dict:
            before_class_dict[multip] = {}
        if multip == 1: # multiplicity 1
            mu_G += cur_feature.numpy()
            y = torch.argmax(cur_label).item()
            if y not in mu_c_dict:
                mu_c_dict[y] = cur_feature.numpy()
                before_class_dict[multip][y] = [cur_feature.numpy()]
                num_class_dict[y] = 1
            else:
                mu_c_dict[y] += cur_feature.numpy()
                before_class_dict[multip][y].append(cur_feature.numpy())
                num_class_dict[y] = num_class_dict[y] + 1
        
        else: # Other multiplicity
            y = str(torch.argwhere(cur_label).squeeze().detach().cpu().numpy())
            if y not in before_class_dict[multip]:
                before_class_dict[multip][y] = [cur_feature.numpy()]
            else:
                before_class_dict[multip][y].append(cur_feature.numpy())
        
    mu_G = mu_G / num_data
    for cla in mu_c_dict:
        mu_c_dict[cla] /= num_class_dict[cla]
    print(num_data, num_class_dict)
    return mu_G, mu_c_dict, before_class_dict

# Within-class covariance matrix
def compute_Sigma_W(before_class_dict, mu_c_dict, device):
    num_data = 0
    Sigma_W = 0
    
    for target in before_class_dict.keys():
        class_feature_list = torch.from_numpy(np.array(before_class_dict[target])).float().to(device)
        class_mean = torch.from_numpy(mu_c_dict[target]).float().to(device)
        for feature in class_feature_list:
            diff = feature - class_mean
            Sigma_W += torch.outer(diff,diff)
            num_data += 1
    Sigma_W /= num_data
    
    return Sigma_W.cpu().numpy()

# Between-class covariance matrix
def compute_Sigma_B(mu_c_dict, mu_G, device):
    mu_G = torch.from_numpy(mu_G).float().to(device)
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        class_mean = torch.from_numpy(mu_c_dict[i]).float().to(device)
        diff = class_mean - mu_G
        Sigma_B += torch.outer(diff,diff)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = torch.from_numpy(mu_c_dict[i] - mu_G).float()

    WH = torch.mm(W, H.cuda())
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda()

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H

def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub.to(W.device), p='fro')
    return ETF_metric.detach().cpu().numpy().item()

def angle_metric(m1_feature_dict, m2_feature_dict):
    numerator = []
    denominator = []
    
    m1_feature_means = []
    m2_feature_means = []
    for m1_key in m1_feature_dict:
        feature_cla = np.stack(m1_feature_dict[m1_key], axis=0)
        feature_cla_mean = np.mean(feature_cla, axis=0)
        m1_feature_means.append(feature_cla_mean)
        
    for m_key in m2_feature_dict:
        # m1
        key1, key2 = m_key[1:-1].split(' ')[-2:]
        key1, key2 = int(key1), int(key2)
        feature_key1 = np.stack(m1_feature_dict[key1], axis=0)
        feature_key2 = np.stack(m1_feature_dict[key2], axis=0)
        feature_key1_mean = np.mean(feature_key1, axis=0)
        feature_key2_mean = np.mean(feature_key2, axis=0)
        sum_feature_key12 = feature_key1_mean + feature_key2_mean
        sum_feature_key12 /= np.linalg.norm(sum_feature_key12)
        
        # m2
        m_feature = np.stack(m2_feature_dict[m_key], axis=0)
        m_feature_mean = np.mean(m_feature, axis=0)
        m_feature_mean /= np.linalg.norm(m_feature_mean)
        m2_feature_means.append(m_feature_mean)

        inner = np.sum(sum_feature_key12 * m_feature_mean)
        angle = np.degrees(np.arccos(inner))
        numerator.append(angle)
        
    # Get denominator
    for i in range(len(m1_feature_means)):
        for j in range(i+1, len(m1_feature_means)):
            for k in range(len(m2_feature_means)):
                fi_mean = m1_feature_means[i]
                fj_mean = m1_feature_means[j]
                fm2_mean = m2_feature_means[k] # m2
                fij_combine = fi_mean + fj_mean
                fij_combine /= np.linalg.norm(fij_combine) # m1
                inner = np.sum(fij_combine * fm2_mean)
                angle = np.degrees(np.arccos(inner))
                denominator.append(angle)
    print(len(numerator), len(denominator))
    return np.mean(numerator) / np.mean(denominator)

def imbalance_angle_stat(m1_feature_dict, m2_feature_dict, lengths = [500,50,5]):
    # For multiplicity 1 balance, multiplicity 2 not balance case
    cur_angle_list_1 = []
    cur_angle_list_2 = []
    cur_angle_list_3 = []
    denominator = []

    m1_feature_means = []
    m2_feature_means = []
    for m1_key in m1_feature_dict:
        feature_cla = np.stack(m1_feature_dict[m1_key], axis=0)
        feature_cla_mean = np.mean(feature_cla, axis=0)
        m1_feature_means.append(feature_cla_mean)

    for m_key in m2_feature_dict:
        key1, key2 = m_key[1:-1].split(' ')[-2:]
        key1, key2 = int(key1), int(key2)
        m_feature = np.stack(m2_feature_dict[m_key], axis=0)
        feature_key1 = np.stack(m1_feature_dict[key1], axis=0)
        feature_key2 = np.stack(m1_feature_dict[key2], axis=0)
        feature_key1_mean = np.mean(feature_key1, axis=0)
        feature_key2_mean = np.mean(feature_key2, axis=0)
        m_feature_mean = np.mean(m_feature, axis=0)
        sum_feature_key12 = feature_key1_mean + feature_key2_mean
        sum_feature_key12 /= np.linalg.norm(sum_feature_key12)
        m_feature_mean /= np.linalg.norm(m_feature_mean)
        m2_feature_means.append(m_feature_mean)

        inner = np.sum(sum_feature_key12 * m_feature_mean)
        angle = np.degrees(np.arccos(inner))
        
        if len(m2_feature_dict[m_key]) == lengths[0]:
            cur_angle_list_1.append(angle)
        elif len(m2_feature_dict[m_key]) == lengths[1]:
            cur_angle_list_2.append(angle)
        elif len(m2_feature_dict[m_key]) == lengths[2]:
            cur_angle_list_3.append(angle)
        else:
            raise ValueError("Check Length!")

    # Get denominator
    for i in range(len(m1_feature_means)):
        for j in range(i+1, len(m1_feature_means)):
            for k in range(len(m2_feature_means)):
                fi_mean = m1_feature_means[i]
                fj_mean = m1_feature_means[j]
                fm2_mean = m2_feature_means[k] # m2
                fij_combine = fi_mean + fj_mean
                fij_combine /= np.linalg.norm(fij_combine) # m1
                inner = np.sum(fij_combine * fm2_mean)
                angle = np.degrees(np.arccos(inner))
                denominator.append(angle)
    
    return np.mean(cur_angle_list_1)/np.mean(denominator),np.mean(cur_angle_list_2)/np.mean(denominator),np.mean(cur_angle_list_3)/np.mean(denominator)

def calculate_nc_stats(model, trainloader, device, imbalance=False):
    mu_G, mu_c_dict, feature_dict = compute_info(model, trainloader, device)
    Sigma_W = compute_Sigma_W(feature_dict[1], mu_c_dict, device)
    Sigma_B = compute_Sigma_B(mu_c_dict, mu_G, device)
    collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict)
    
    # NC2
    try:
        nc2_w = compute_ETF(model.fc.weight.data)
    except:
        nc2_w = compute_ETF(model.classifier.weight.data)
    
    feature_means = [mu_c_dict[i] for i in mu_c_dict.keys()]
    feature_means = torch.from_numpy(np.array(feature_means))
    nc2_h = compute_ETF(feature_means)
    
    # NC3 
    try:
        nc3,_ = compute_W_H_relation(model.fc.weight.data, mu_c_dict, mu_G)
    except:
        nc3,_ = compute_W_H_relation(model.classifier.weight.data, mu_c_dict, mu_G)
    
    # Angle_diff_list
    m1_feature_dict = feature_dict[1]
    m2_feature_dict = feature_dict[2]
    
    if imbalance:
        al1,al2,al3 = imbalance_angle_stat(m1_feature_dict, m2_feature_dict, lengths = [500,50,5])
    
    cur_angle_list = []
    for m_key in m2_feature_dict:
        key1, key2 = m_key[1:-1].split(' ')[-2:]
        key1, key2 = int(key1), int(key2)
        m_feature = np.stack(m2_feature_dict[m_key], axis=0)
        feature_key1 = np.stack(m1_feature_dict[key1], axis=0)
        feature_key2 = np.stack(m1_feature_dict[key2], axis=0)
        feature_key1_mean = np.mean(feature_key1, axis=0)
        feature_key2_mean = np.mean(feature_key2, axis=0)
        m_feature_mean = np.mean(m_feature, axis=0)
        sum_feature_key12 = feature_key1_mean + feature_key2_mean
        sum_feature_key12 /= np.linalg.norm(sum_feature_key12)

        inner = np.sum(sum_feature_key12 * (m_feature_mean / np.linalg.norm(m_feature_mean)))
        angle = np.degrees(np.arccos(inner))
        cur_angle_list.append(angle)
    
    # Angle metric
    angle_m = angle_metric(m1_feature_dict, m2_feature_dict)
    if imbalance:
        return collapse_metric, nc2_w, nc2_h, nc3, np.mean(cur_angle_list), angle_m, [al1,al2,al3], feature_dict
    else:
        return collapse_metric, nc2_w, nc2_h, nc3, np.mean(cur_angle_list), angle_m, feature_dict