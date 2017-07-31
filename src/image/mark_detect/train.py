import os
import time
import argparse
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
from torch.autograd import Variable

import models.custom_model as custom
from dataset import tttoe_data
# Training settings
parser = argparse.ArgumentParser(description='ECE281 CNN Image classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--decay-epochs', type=int, default=10, metavar='N',
                    help='divide lr by 10 every decay-epochs (default: 10)')
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
    best_precision = 0
    dataset_size = 4096
    image_size = 64
    text_size = 50
    rand_offs = 20

    # Use CUDA if available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('Cuda used: ', args.cuda)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set up the data loaders
    cuda_kwargs = {'num_workers': args.workers,
                   'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        tttoe_data.Tttoe_dataset(
            dataset_size, image_size, text_size, rand_offs, noise_alpha=0.1, img_transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **cuda_kwargs)
    val_loader = torch.utils.data.DataLoader(
        tttoe_data.Tttoe_dataset(
            dataset_size, image_size, text_size, 1, img_transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **cuda_kwargs)

    # Set up the model, optimizer and loss function
    model = custom.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()
    start_epoch = 1

    test_accs = []
    test_losses = []
    val_accs = []
    val_losses = []

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_precision = float(checkpoint['best_precision'])
            print('Best prediction: ', best_precision)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            test_accs = checkpoint['testAcc']
            test_losses = checkpoint['testLoss']
            val_accs = checkpoint['valAcc']
            val_losses = checkpoint['valLoss']

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    # Train
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        exp_lr_scheduler(optimizer, epoch, args.lr, args.decay_epochs)
        start_time = time.clock()
        test_acc, test_loss = train(train_loader, model, criterion,
                                    optimizer, epoch, args.cuda, args.log_interval)
        end_time = time.clock()
        print('Time used training for epoch: ', (end_time - start_time))
        val_acc, val_loss = validate(val_loader, model, criterion, args.cuda)
        is_best = False
        if val_acc > best_precision:
            best_precision = val_acc
            is_best = True
        print('Precision:', val_acc)
        print('Best precision:', best_precision)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_precision': best_precision,
            'optimizer': optimizer.state_dict(),
            'testAcc': test_accs,
            'testLoss': test_losses,
            'valAcc': val_accs,
            'valLoss': val_losses,
        }
        model_fname = os.path.join(os.path.dirname(
            __file__), 'checkpoints', 'checkpoint.pth.tar')
        save_checkpoint(state, is_best, model_fname)
        print()


def train(train_loader, model, criterion, optimizer, epoch, cuda, log_interval):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.data[0]
        # get the index of the max probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    # loss function already averages over batch size
    train_loss /= len(train_loader)
    return correct / len(train_loader.dataset), train_loss


def validate(val_loader, model, criterion, use_cuda):
    model.eval()
    val_loss = 0
    correct = 0
    for (data, target) in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data_v, target_v = Variable(data, volatile=True), Variable(target)
        output = model(data_v)
        val_loss += criterion(output, target_v).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target_v.data).cpu().sum()
    # loss function already averages over batch size
    val_loss /= len(val_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    imshow(utils.make_grid(data), [float(x) for x in pred.numpy()])
    return correct / len(val_loader.dataset), val_loss


def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch=20):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'checkpoints/model_best.pth.tar')


def imshow(img, labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(labels)
    plt.show()


if __name__ == '__main__':
    main()
