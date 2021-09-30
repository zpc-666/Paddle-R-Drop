# coding=utf-8
from __future__ import absolute_import, division, print_function

from paddle.vision import transforms, datasets
from paddle.io import DataLoader


def get_loader(args):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "imagenet":
        trainset =datasets.ImageFolder(root='../ILSVRC2012/train',transform=transform_train)
        
        testset = datasets.ImageFolder(root='../ILSVRC2012/val',transform=transform_test)
        
    elif args.dataset == 'cifar100':
        trainset = datasets.Cifar100(data_file="./data/cifar-100-python.tar.gz",
                                     mode='train',
                                     download=True,
                                     transform=transform_train)
        testset = datasets.Cifar100(data_file="./data/cifar-100-python.tar.gz",
                                    mode='test',
                                    download=True,
                                    transform=transform_test)

    train_loader = DataLoader(trainset,
                              shuffle=True,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              use_buffer_reader=True)
    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             use_buffer_reader=True)

    print('Number of train data:', len(trainset))
    print('Number of test data:', len(testset))
    return train_loader, test_loader
