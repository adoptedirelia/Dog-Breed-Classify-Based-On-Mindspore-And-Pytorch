import os
import sys
import re
import math
import time
import random
import requests
import hashlib
import collections
import zipfile
import tarfile
import shutil
import mindspore
import mindcv
import numpy as np
import pandas as pd
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore import Tensor, Parameter
import mindspore.common.initializer as initializer

from matplotlib import pyplot as plt

def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename.

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """Download and extract a zip/tar file.

    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def reorg_dog_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

def copyfile(filename, target_dir):
    """Copy a file into a target directory.

    Defined in :numref:`sec_kaggle_cifar10`"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary.

    Defined in :numref:`sec_kaggle_cifar10`"""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set.

    Defined in :numref:`sec_kaggle_cifar10`"""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction.

    Defined in :numref:`sec_kaggle_cifar10`"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
        

def get_net(devices):
    finetune_net = nn.SequentialCell()
    finetune_net.feature = mindcv.create_model('convnext_tiny',pretrained=True)
    #finetune_net.append(feature)
    # 定义一个新的输出网络，共有120个输出类别
    output_new = nn.SequentialCell([nn.Dense(1000, 512),
                  nn.ReLU(),
                  nn.Dropout(p=0.8),                  
                  nn.Dense(512,256),
                  nn.ReLU(),
                  nn.Dropout(p=0.8),
                  nn.Dense(256, 120)])
    for name, cell in output_new.cells_and_names():
        if isinstance(cell, nn.Dense):
            k = 1 / cell.in_channels
            k = k ** 0.5

            cell.weight.set_data(
                initializer.initializer(initializer.Uniform(k), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(
                    initializer.initializer(initializer.Uniform(k), cell.bias.shape, cell.bias.dtype))

    finetune_net.append(output_new)
    #finetune_net.append(output_new)
    # 冻结参数
    for param in finetune_net.feature.get_parameters():
        param.requires_grad = False
    return finetune_net


def get_resnet(devices):
    finetune_net = nn.SequentialCell()
    finetune_net.feature = mindcv.create_model('resne34',pretrained=True)
    #finetune_net.append(feature)
    # 定义一个新的输出网络，共有120个输出类别
    output_new = nn.SequentialCell([nn.Dense(1000, 512),
                  nn.ReLU(),
                  nn.Dropout(p=0.8),                  
                  nn.Dense(512,256),
                  nn.ReLU(),
                  nn.Dropout(p=0.8),
                  nn.Dense(256, 120)])
    for name, cell in output_new.cells_and_names():
        if isinstance(cell, nn.Dense):
            k = 1 / cell.in_channels
            k = k ** 0.5

            cell.weight.set_data(
                initializer.initializer(initializer.Uniform(k), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(
                    initializer.initializer(initializer.Uniform(k), cell.bias.shape, cell.bias.dtype))

    finetune_net.append(output_new)
    #finetune_net.append(output_new)
    # 冻结参数
    for param in finetune_net.feature.get_parameters():
        param.requires_grad = False
    return finetune_net