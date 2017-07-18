from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import time
import shutil
import os
import numpy as np
from dataset.dataset_imagenet import ImageNet
from torchvision import utils

# Training settings
parser = argparse.ArgumentParser(description='ECE281 CNN Image classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='Number of workers if CUDA is used (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                        help='path to latest checkpoint (default: none)')


def main():
	args = parser.parse_args()
	bestPrecision = 0

    # Use CUDA if available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('Cuda used: ', args.cuda)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
   
    # Set up the data loaders
    kwargs = {'num_workers': args.workers,
              'pin_memory': True} if args.cuda else {}
    trainLoader = torch.utils.data.DataLoader(
        ImageNet('data', mode='train', transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.478571, 0.44496, 0.392131], [
                                 0.26412, 0.255156, 0.269064])
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    valLoader = torch.utils.data.DataLoader(
        ImageNet('data', mode='validate', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.478571, 0.44496, 0.392131], [
                                 0.26412, 0.255156, 0.269064])
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # Set up the model, optimizer and loss function
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(num_ftrs, 100)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum)
    optimizer = optim.SGD(model.fc.parameters(),
                          lr=args.lr, momentum=args.momentum)

    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()
    startEpoch = 1

    testAcc = []
    testLoss = []
    valAcc = []
    valLoss = []
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            startEpoch = checkpoint['epoch']
            bestPrecision = float(checkpoint['best_precision'])
            print('Best prediction: ', bestPrecision)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            testAcc = checkpoint['testAcc']
            testLoss = checkpoint['testLoss']
            valAcc = checkpoint['valAcc']
            valLoss = checkpoint['valLoss']

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    # Train
    for epoch in range(startEpoch, startEpoch + args.epochs + 1):
        exp_lr_scheduler(optimizer, epoch, args.lr)
        startTime = time.clock()
        tAcc, tLoss = train(trainLoader, model, criterion, optimizer, epoch)
        endTime = time.clock()
        print('Time used training for epoch: ', (endTime - startTime))
        vAcc, vLoss = validate(valLoader, model, criterion)
        isBest = False
        if vAcc > bestPrecision:
            bestPrecision = vAcc
            isBest = True
        print('Precision:', vAcc)
        print('Best precision:', bestPrecision)
        testAcc.append(tAcc)
        testLoss.append(tLoss)
        valAcc.append(vAcc)
        valLoss.append(vLoss)
        saveCheckpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_precision': bestPrecision,
            'optimizer': optimizer.state_dict(),
            'testAcc': testAcc,
            'testLoss': testLoss,
            'valAcc': valAcc,
            'valLoss': valLoss,
        }, isBest)
        print()


def train(trainLoader, model, criterion, optimizer, epoch):
    model.train()
    trainLoss = 0
    correct = 0
    for batchIdx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        trainLoss += loss.data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        loss.backward()
        optimizer.step()
        if batchIdx % args.log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader), loss.data[0]))
    # loss function already averages over batch size
    trainLoss /= len(trainLoader)
    return correct / len(trainLoader.dataset), trainLoss


def validate(valLoader, model, criterion):
    model.eval()
    valLoss = 0
    correct = 0
    for idx, (data, target) in enumerate(valLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valLoss += criterion(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    valLoss /= len(valLoader)  # loss function already averages over batch size
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        valLoss, correct, len(valLoader.dataset),
        100. * correct / len(valLoader.dataset)))
    return correct / len(valLoader.dataset), valLoss


def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch=20):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def saveCheckpoint(state, isBest, filename='saved_models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if isBest:
        shutil.copy(filename, 'saved_models/model_best.pth.tar')


if __name__ == '__main__':
    main()
