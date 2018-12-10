from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    


    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class CamStyleTrainer(object):
    def __init__(self, model, criterion, camstyle_loader):
        super(CamStyleTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.camstyle_loader = camstyle_loader
        self.camstyle_loader_iter = iter(self.camstyle_loader)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            try:
                camstyle_inputs = next(self.camstyle_loader_iter)
            except:
                self.camstyle_loader_iter = iter(self.camstyle_loader)
                camstyle_inputs = next(self.camstyle_loader_iter)
            inputs, targets = self._parse_data(inputs)
            camstyle_inputs, camstyle_targets = self._parse_data(camstyle_inputs)
            loss, prec1 = self._forward(inputs, targets, camstyle_inputs, camstyle_targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs.cuda())
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, camstyle_inputs, camstyle_targets):
        random_list =torch.randperm(72) #batch size + generated images
        combined_inputs = torch.cat((inputs, camstyle_inputs), 0)
        combined_inputs = combined_inputs[random_list]
        combined_outputs = self.model(combined_inputs)
        flags = torch.zeros(64).cuda()
        camstyle_flags = torch.ones(8).cuda()
        combined_flags = torch.cat((flags, camstyle_flags), 0)
        combined_flags = combined_flags[random_list]
        camstyle_targets = torch.zeros(8).cuda()
        camstyle_targets = camstyle_targets.type(torch.cuda.LongTensor)
        combined_targets = torch.cat((targets, camstyle_targets), 0)
        combined_targets = combined_targets[random_list]
        
        outputs = self.model(inputs)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]
        loss = self.criterion(combined_outputs, combined_targets, combined_flags)
        #print("combined loss", loss)
        #print("outputs",outputs.size())
        #print("targets", targets.size())
        '''
        camstyle_outputs = self.model(camstyle_inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            flags = torch.zeros(64).cuda()
            loss = self.criterion(outputs, targets, flags)
            print("real loss", loss)

            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
            #print("real loss", loss)
        else:
            flags = torch.zeros(64).cuda()
            loss = self.criterion(outputs, targets, flags)
            print("real loss", loss)

            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        
        camstyle_targets = camstyle_targets.type(torch.cuda.LongTensor)
        flags = torch.ones(16).cuda()
        camstyle_loss = self.criterion(camstyle_outputs, camstyle_targets, flags)
        print("camstyle_loss", camstyle_loss)
        loss += camstyle_loss
        '''
        return loss, prec


    def _lsr_loss(self, outputs, targets):
        num_class = outputs.size()[1]
        print(num_class)
        
        print("mean loss", loss)
        return loss



    



