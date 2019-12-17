import torch
import torch.nn.functional as F
from utils import *

import config
from tqdm import *


def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def VolleyballEpoch1(mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

    actions_meter = AverageMeter()
    actions_each_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()

    def base(data_loader, model, device, cfg=None):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B*N)
        actions_in = torch.cat(batch_data[2],dim=0)
        actions_in = actions_in.reshape(-1).to(device=device)

        # forward
        actions_scores = model((batch_data[0], batch_data[3]))  # tensor(B*N, actions_num)

        # Predict actions
        actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
        actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        actions_labels = torch.argmax(actions_scores, dim=1)
        actions_correct = torch.eq(actions_labels.int(), actions_in.int()).float()
        actions_correct_sum = torch.sum(actions_correct)
        #actions_each_meter.update(actions_correct, batch_size)

        # Get accuracy
        actions_accuracy = actions_correct_sum.item() / actions_scores.shape[0]

        actions_meter.update(actions_accuracy, batch_size)

        # Total loss
        total_loss = cfg.actions_loss_weight * actions_loss
        loss_meter.update(total_loss.item(), batch_size)

    if mode == 'train':
        print("Training in epoch %s" % epoch)
        for batch_data in tqdm(data_loader):
            base(data_loader, model,device,cfg)

            # Optim
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        info = {
            'mode': mode,
            'time': epoch_timer.timeit(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'actions_acc': actions_meter.avg * 100,
            #'actions_each _acc': actions_each_meter.avg * 100
        }
    elif mode == 'test':
        print("Testing in test dataset")
        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                # model.apply(set_bn_eval)

                batch_size = len(batch_data[0])

                # reshape the action label into tensor(B*N)
                actions_in = torch.cat(batch_data[2], dim=0)
                actions_in = actions_in.reshape(-1).to(device=device)

                # forward
                actions_scores = model((batch_data[0], batch_data[3]))  # tensor(B*N, actions_num)

                # Predict actions
                actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
                actions_labels = torch.argmax(actions_scores, dim=1)
                actions_correct = torch.eq(actions_labels.int(), actions_in.int()).float()
                actions_correct_sum = torch.sum(actions_correct)
                #actions_each_meter.update(actions_correct, batch_size)

                # Get accuracy
                actions_accuracy = actions_correct_sum.item() / actions_scores.shape[0]

                actions_meter.update(actions_accuracy, batch_size)

                # Total loss
                total_loss = cfg.actions_loss_weight * actions_loss
                loss_meter.update(total_loss.item(), batch_size)

        info = {
            'mode': mode,
            'time': epoch_timer.timeit(),
            'loss': loss_meter.avg,
            'actions_acc': actions_meter.avg * 100,
            #'actions_each _acc': actions_each_meter.avg * 100
        }

    return info

class VolleyballEpoch():

    def __init__(self, mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

        self.mode = mode
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch

        self.actions_meter = AverageMeter()
        self.actions_each_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.epoch_timer = Timer()

        self.total_loss = None

        self.main(mode)

    def main(self,mode):
        if mode == 'train':
            print("Training in epoch %s" % self.epoch)
            for batch_data in self.data_loader:
                self.base(batch_data)

                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            info = {
                'mode': mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.avg * 100,
                # 'actions_each _acc': actions_each_meter.avg * 100
            }
        elif mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in self.data_loader:
                    self.base(batch_data)
            info = {
                'mode': mode,
                'time': self.epoch_timer.timeit(),
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.avg * 100,
                # 'actions_each _acc': actions_each_meter.avg * 100
            }
        else:
            assert False, "mode name incorrect"

        return info

    def base(self,batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B*N)
        actions_in = torch.cat(batch_data[2],dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)

        # forward
        actions_scores = self.model((batch_data[0], batch_data[3]))  # tensor(B*N, actions_num)

        # Predict actions
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        actions_labels = torch.argmax(actions_scores, dim=1)
        actions_correct = torch.eq(actions_labels.int(), actions_in.int()).float()
        actions_correct_sum = torch.sum(actions_correct)
        #actions_each_meter.update(actions_correct, batch_size)

        # Get accuracy
        actions_accuracy = actions_correct_sum.item() / actions_scores.shape[0]

        self.actions_meter.update(actions_accuracy, batch_size)

        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss
        self.loss_meter.update(self.total_loss.item(), batch_size)
