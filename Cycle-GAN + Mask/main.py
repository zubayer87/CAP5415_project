from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.trainers import Trainer, CamStyleTrainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import torch.nn.functional as F


def get_data(dataname, data_dir, height, width, batch_size, camstyle=0, re=0, workers=8):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir, dataset.train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    
    if camstyle <= 0:
        camstyle_loader = None
    else:
        camstyle_loader = DataLoader(
            Preprocessor(dataset.camstyle, root=osp.join(dataset.images_dir, dataset.camstyle_path),
                         transform=train_transformer),
            batch_size=camstyle, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader, camstyle_loader


def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, num_classes, train_loader, query_loader, gallery_loader, camstyle_loader = \
        get_data(args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.camstyle, args.re, args.workers)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
        return

    # Criterion
    #criterion = nn.CrossEntropyLoss().cuda()

    class LSROloss(nn.Module):
        def __init__(self):     # change target to range(0,750)
            super(LSROloss,self).__init__()
                                            #input means the prediction score(torch Variable) 32*752,target means the corresponding label,   
        def forward(self,input,target,flg): # while flg means the flag(=0 for true data and 1 for generated data)  batchsize*1
           # print(type(input))
            if input.dim()>2:                   # N defines the number of images, C defines channels,  K class in total
                input=input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
                input=input.transpose(1,2)            # N,C,H*W => N,H*W,C
                input=input.contiguous().view(-1,input.size(2))  # N,H*W,C => N*H*W,C
           
           # normalize input
            maxRow, _ = torch.max(input.data, 1)   # outputs.data  return the index of the biggest value in each row
            maxRow=maxRow.unsqueeze(1)
            input.data=input.data-maxRow
            
            target=target.view(-1,1)      # batchsize*1
            flg=flg.view(-1,1) 
            #len=flg.size()[0]
            flos=F.log_softmax(input)    # N*K?      batchsize*751
            flos=torch.sum(flos,1)/flos.size(1)       # N*1  get average      gan loss    
            logpt=F.log_softmax(input)   # size: batchsize*751
            #print("logpt",logpt.size())
            #print("taarget", target.size())
            logpt=logpt.gather(1,target)   # here is a problem 
            logpt=logpt.view(-1)           # N*1     original loss   
            flg=flg.view(-1) 
            flg=flg.type(torch.cuda.FloatTensor)
            #print("logpt",logpt.size())
            #print("flg", flg.size())
            #print("flos", flos.size())
            loss=-1*logpt*(1-flg)-flos*flg
            return loss.mean()
    criterion = LSROloss()

    # Optimizer
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    if args.camstyle == 0:
        trainer = Trainer(model, criterion)
    else:
        trainer = CamStyleTrainer(model, criterion, camstyle_loader)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d} \n'.
              format(epoch))

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CamStyle")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    # camstyle batchsize
    parser.add_argument('--camstyle', type=int, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
