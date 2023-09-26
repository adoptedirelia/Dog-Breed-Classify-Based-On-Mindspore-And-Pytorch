import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'


import torch
import torchvision
from torch import nn
import utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import config
import argparse
import time 
from time import sleep
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch version')
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def prepare_data(args):
    data_dir = '../data/'
    utils.reorg_dog_data(data_dir,args.ratio)

    transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 添加随机噪声
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    

    transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, args.batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

    valid_iter = torch.utils.data.DataLoader(valid_ds, args.batch_size, shuffle=False,
                                            drop_last=True)

    test_iter = torch.utils.data.DataLoader(test_ds, args.batch_size, shuffle=False,
                                            drop_last=False)
    return train_iter,train_valid_iter,valid_iter,test_iter

def evaluate_loss(data_iter, net, devices,loss):
    train_l_sum,train_acc_sum,n,c = 0.0,0.0,0,0

    for features, labels in data_iter:
        net.eval()
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        train_l_sum +=l.sum().item()
        train_acc_sum += (outputs.argmax(dim=1)==labels).sum().item()
        n +=labels.shape[0]
        c +=1
    return (train_l_sum / c),(train_acc_sum / n)

def train(args,train_iter,valid_iter,net):
    
    
    loss = nn.CrossEntropyLoss(reduction='none')

    net = nn.DataParallel(net, device_ids=args.devices).to(args.devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=args.lr,
                              momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, args.lr_period, args.lr_decay)
    num_batches= len(train_iter)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(args.epoch):
        train_l_sum,train_acc_sum,n,c = 0.0,0.0,0,0
        begin = time.time()
        for features, labels in tqdm(train_iter):
            net.train()
            features, labels = features.to(args.devices[0]), labels.to(args.devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            train_l_sum +=l.item()
            train_acc_sum += (output.argmax(dim=1)==labels).sum().item()
            n +=labels.shape[0]
            c +=1
        
        
        if valid_iter is not None:
            valid_loss,valid_acc = evaluate_loss(valid_iter, net, args.devices,loss)

        end = time.time()
        print(f'epoch {epoch+1}, train loss {train_l_sum/c:.4f}, valid_loss {valid_loss:.4f}, train_acc {train_acc_sum/n:.3f}, valid_acc {valid_acc:.3f} time cost {(end-begin):.4f}s')
        train_loss.append(train_l_sum/c)
        train_acc.append(train_acc_sum/n)
        val_loss.append(valid_loss)
        val_acc.append(valid_acc)

        scheduler.step()
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(net.state_dict(), f"{args.model_save_path}/model_epoch_{epoch + 1}.pth")
            print(f"model saved in {args.model_save_path}/model_epoch_{epoch + 1}.pth")
            
    np.save('./train_l.npy',np.array(train_loss))
    np.save('./train_a.npy',np.array(train_acc))
    np.save('./val_l.npy',np.array(val_loss))
    np.save('./val_a.npy',np.array(val_acc))

    return net


def test(net,data_iter,devices):
    train_l_sum,train_acc_sum,n,c = 0.0,0.0,0,0
    loss = nn.CrossEntropyLoss(reduction='none')

    for features, labels in data_iter:
        net.eval()
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        train_l_sum +=l.sum().item()
        train_acc_sum += (outputs.argmax(dim=1)==labels).sum().item()
        n +=labels.shape[0]
        c +=1
    return (train_l_sum / c),(train_acc_sum / n)

def main(args):
    train_iter,train_valid_iter,valid_iter,test_iter = prepare_data(args)
    net = utils.get_net(args.devices)
    if args.ckpt != None:
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.ckpt).items()})
    if args.train:
        net = train(args,train_iter,valid_iter,net)
    
    l,acc = test(net,valid_iter,args.devices)
    print(f'test loss: {l:.4f}, test acc: {acc:.4f}')

if __name__ == '__main__':
    args = get_parser()
    print(args)
    print(args.ckpt,args.train)
    main(args)