import csv

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.fusion_model import FModel
from utils import *
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler



# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.y.view(-1, 1).float().to(device))
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis','kiba'][int(sys.argv[1])]]


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


# 定义一个用于拆分数据集的函数
def split_dataset(dataset, num_folds):
    folds = []
    fold_size = len(dataset) // num_folds

    # 随机打乱数据集中的样本
    indices = torch.randperm(len(dataset))

    # 将数据集拆分成 num_folds 折
    for i in range(num_folds):
        # 选择当前折的起始和终止索引
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(dataset)

        # 将当前折的数据和标签放入一个元组中
        fold_data = dataset[indices[start_idx:end_idx]]
        folds.append(fold_data)

    return folds


# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ',  dataset)
    processed_data_file = '/root/autodl-tmp/processed/' + dataset + '.pt'
    if (not os.path.isfile(processed_data_file)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        data = TestbedDataset(root='/root/autodl-tmp/', dataset=dataset)
        # kfold = KFold(n_splits=5, shuffle=True)
        folds = split_dataset(data, num_folds=5)
        for fold_idx, test_fold in enumerate(folds):
            print(f"Fold {fold_idx + 1}")
            # 创建训练集和验证集
            train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
            train_data = torch.utils.data.ConcatDataset(train_folds)
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

            test_data = test_fold
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)

            # training the model
            model = FModel().to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            best_mse = 1000
            best_ci = 0
            best_rm2 = 0
            best_epoch = -1
            model_file_name = 'model_' +  dataset +  str(fold_idx+1) + '.pth'
            result_file_name = 'result_' + dataset + '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch + 1)
                G, P = predicting(model, device, test_loader)
                # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
                ret = [rmse(G, P), mse(G, P), ci(G, P), rm2(G, P)]

                with open(result_file_name, 'a') as f:
                    f.write(str(fold_idx + 1) + ',' + str(epoch) + ',' + ','.join(map(str, ret)) + '\n')  # 把train_loss写在文件最后一列
                if ret[1] < best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    best_epoch = epoch + 1
                    best_mse = ret[1]
                    best_ci = ret[-2]
                    best_rm2 = ret[-1]
                    print('mse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,dataset)
                else:
                    print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,dataset)







