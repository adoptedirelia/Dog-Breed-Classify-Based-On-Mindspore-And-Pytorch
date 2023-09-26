import os
import pandas as pd
import mindspore
import mindcv
import mindspore.nn as nn
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from tqdm import tqdm
import config
import argparse
import utils
import numpy as np
import time 
from time import sleep


def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch version')
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def prepare_data(args):
    data_dir = '../data/'
    
    #data_dir = '../data/'
    utils.reorg_dog_data(data_dir,args.ratio)

    transform_train = transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    vision.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    vision.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    vision.RandomColorAdjust(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
     # 标准化图像的每个通道
    vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], 
                     std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
    vision.HWC2CHW()])

    transform_test = transforms.Compose([
    vision.Resize(256),
    # 从图像中心裁切224x224大小的图片
    vision.CenterCrop(224),
    vision.Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                     [0.229 * 255, 0.224 * 255, 0.225 * 255]),
    vision.HWC2CHW()])

    train_ds, train_valid_ds = [mindspore.dataset.ImageFolderDataset(
    os.path.join(data_dir, 'train_valid_test', folder), shuffle=True, decode=True) 
                            for folder in ['train', 'train_valid']]
    train_ds = train_ds.map(transform_train, 'image')
    train_valid_ds = train_valid_ds.map(transform_train, 'image')


    valid_ds, test_ds = [mindspore.dataset.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder), shuffle=False, decode=True) 
                        for folder in ['valid', 'test']]
    valid_ds = valid_ds.map(transform_test, 'image')
    test_ds = test_ds.map(transform_test, 'image')


    train_iter, train_valid_iter = [dataset.batch(batch_size=args.batch_size, drop_remainder=True)
                                for dataset in (train_ds, train_valid_ds)]

    valid_iter = valid_ds.batch(batch_size=args.batch_size, drop_remainder=True)

    test_iter = test_ds.batch(batch_size=args.batch_size, drop_remainder=False)
    
    return train_iter,train_valid_iter,valid_iter,test_iter

def evaluate_loss(data_iter, net, loss):

    train_l_sum,train_acc_sum,n,c = 0.0,0.0,0,0

    net.set_train(False)
    
    for features, labels in data_iter:
        logits = net(features)
        l = loss(logits, labels).sum()
        train_l_sum += l.asnumpy()
        predicted_labels = np.argmax(logits.asnumpy(), axis=1)
        train_acc_sum += np.sum(predicted_labels == labels.asnumpy())
        n +=labels.shape[0]
        c +=1

    net.set_train(True)
    # print(train_l_sum,train_acc_sum)
    return (train_l_sum / c), (train_acc_sum/n)

def train(args,train_iter,valid_iter,net):
    loss = nn.CrossEntropyLoss(reduction='none')

    devices = None
    lr_list = mindspore.Tensor([args.lr*(args.lr_decay**(i//args.lr_period)) 
                          for i in range(args.epoch) 
                          for j in range(train_iter.get_dataset_size())])
    trainer = nn.SGD((param for param in net.get_parameters() if param.requires_grad), 
                     learning_rate=lr_list, momentum=0.9, weight_decay=args.wd)

    def forward_fn(inputs, targets):
        logits = net(inputs)
        # print(logits.shape, targets.shape)
        l = loss(logits, targets)
        return l, logits
    
    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, trainer.parameters, has_aux=True)
    
    def train_step(inputs, targets):
        (l, logits), grads = grad_fn(inputs, targets)
        trainer(grads)
        return l.sum(), logits
    
    num_batches= train_iter.get_dataset_size()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(args.epoch):
        net.set_train()
        train_l_sum,train_acc_sum,n,c = 0.0,0.0,0,0
        begin = time.time()
        for (features, labels) in tqdm(train_iter):
            
            l, logits = train_step(features, labels)
            train_l_sum += l.asnumpy()
            predicted_labels = np.argmax(logits.asnumpy(), axis=1)
            train_acc_sum += np.sum(predicted_labels == labels.asnumpy())
            n +=labels.shape[0]
            c +=1
            
        
        if valid_iter is not None:
            valid_loss,valid_acc = evaluate_loss(valid_iter, net, loss)
        
        end = time.time()


        print(f'epoch {epoch+1}, train loss {train_l_sum/c:.4f}, valid_loss {valid_loss:.4f}, train_acc {train_acc_sum/n:.3f}, valid_acc {valid_acc:.3f}, time cost {(end-begin):.4f}s')
        train_loss.append(train_l_sum/c)
        train_acc.append(train_acc_sum/n)
        val_loss.append(valid_loss)
        val_acc.append(valid_acc)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            checkpoint_file = f"./{args.model_save_path}/model_epoch_{epoch + 1}.ckpt"
            mindspore.save_checkpoint(net,checkpoint_file)
            
    np.save('./train_l.npy',np.array(train_loss))
    np.save('./train_a.npy',np.array(train_acc))
    np.save('./val_l.npy',np.array(val_loss))
    np.save('./val_a.npy',np.array(val_acc))
    return net

def test(net,data_iter):
    train_l_sum,train_acc_sum,n,c = 0.0,0.0,0,0

    loss = nn.CrossEntropyLoss(reduction='none')
    net.set_train(False)
    
    for features, labels in data_iter:
        logits = net(features)
        l = loss(logits, labels).sum()
        train_l_sum += l.asnumpy()
        predicted_labels = np.argmax(logits.asnumpy(), axis=1)
        train_acc_sum += np.sum(predicted_labels == labels.asnumpy())
        n +=labels.shape[0]
        c +=1

    net.set_train(True)
    # print(train_l_sum,train_acc_sum)
    return (train_l_sum / c), (train_acc_sum/n)

def main(args):
    train_iter,train_valid_iter,valid_iter,test_iter = prepare_data(args)

    net = utils.get_net(None)

    if args.ckpt != None:
        param_dict = mindspore.load_checkpoint(f"{args.ckpt}")
        mindspore.load_param_into_net(net, param_dict)
    if args.train:
        net = train(args,train_iter,valid_iter,net)

    l,acc = test(net,valid_iter)
    print(f'test loss: {l:.4f}, test acc: {acc:.4f}')



if __name__ == '__main__':
    args = get_parser()
    print(args)
    print(args.ckpt,args.train)
    main(args)