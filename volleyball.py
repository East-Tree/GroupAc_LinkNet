import torch
import torch.nn.functional as F
from utils import *

import config
from tqdm import *


def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class VolleyballEpoch():

    def __init__(self, mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

        self.mode = mode
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch

        self.actions_meter = AverageMeterTensor(cfg.actions_num)
        self.loss_meter = AverageMeter()
        self.epoch_timer = Timer()

        self.total_loss = None



    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            print(self.cfg.actions_weights)
            for batch_data in tqdm(self.data_loader):
                self.base(batch_data)

                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            # renew the action loss weight by accuracy
            if self.cfg.renew_weight:
                new_weight = torch.nn.functional.softmin(self.actions_meter.correct_rate_each, dim=0)
                new_weight = new_weight * 9.
                old_weight = torch.tensor(self.cfg.actions_weights)
                new_weight = old_weight * (1-self.cfg.weight_renew_rate) + self.cfg.weight_renew_rate * new_weight
                self.cfg.actions_weights = new_weight.tolist()
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each
            }
        elif self.mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in tqdm(self.data_loader):
                    self.base(batch_data)
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each
            }
        else:
            assert False, "mode name incorrect"

        return info

    def base(self, batch_data):

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
        actions_result = torch.argmax(actions_scores, dim=1).int()

        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)


        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss
        self.loss_meter.update(self.total_loss.item(), batch_size)
