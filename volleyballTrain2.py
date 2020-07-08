from typing import Dict, Any, Callable, Tuple

import volleyballDataset
from basemodel import *
import config
import utils
import torch.nn.functional as F
from utils import *
from tqdm import *

from torch.utils import data
from torch import optim
import random
from tensorboardX import SummaryWriter


class Focalloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, device0=None, weight=None, attenuate=2.0):
        """
            this is a multi focal loss function base on F.nll_loss
            :param input: [N,p]
            :param target: [N]  ground truth
            :param weight:[p]
            :return:
            """
        if device0 is None:
            device1 = torch.device('cpu')
        else:
            device1 = device0
        input_soft = F.softmax(input, dim=1)
        input_logsoft = F.log_softmax(input, dim=1)
        batch = target.size()[0]
        target_mask = target.reshape(-1, 1)
        input_soft = torch.gather(input_soft, 1, target_mask)
        input_logsoft = torch.gather(input_logsoft, 1, target_mask)
        if weight is None:
            weight_tensor = torch.tensor([1] * batch, device=device1)
        else:
            weight_tensor = weight.repeat(batch, 1).to(device=device1)
            weight_tensor = torch.gather(weight_tensor, 1, target_mask)
        weight_tensor = weight_tensor.reshape(-1, 1)
        focal_weight = weight_tensor * torch.pow(1.0 - input_soft, attenuate)
        # print('focal loss coeff:' + str(focal_weight))
        loss = (-1) * focal_weight * input_logsoft
        loss = torch.mean(loss, dim=0)


        return loss, focal_weight


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
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            print(self.cfg.actions_weights)
            for batch_data in tqdm(self.data_loader):
                self.baseprocess(batch_data)

                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            # renew the action loss weight by accuracy
            if self.cfg.renew_weight:
                new_weight = torch.nn.functional.softmin(self.actions_meter.correct_rate_each, dim=0)
                new_weight = new_weight * 9.
                old_weight = torch.tensor(self.cfg.actions_weights)
                new_weight = old_weight * (1 - self.cfg.weight_renew_rate) + self.cfg.weight_renew_rate * new_weight
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
                    self.baseprocess(batch_data)
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B*N)
        actions_in = torch.cat(batch_data[2], dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)

        # forward
        actions_scores = self.model((batch_data[0], batch_data[3]))  # tensor(B#N, actions_num)

        # Predict actions
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        # loss
        #   cross entropy
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        focal_loss = Focalloss()
        actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, self.device, weight=actions_weights)
        actions_result = torch.argmax(actions_scores, dim=1).int()

        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss

        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)
        self.loss_meter.update(self.total_loss.item(), batch_size)
        self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)

class VolleyballEpoch2():

    def __init__(self, mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

        self.mode = mode
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch

        self.actions_meter = AverageMeterTensor(cfg.actions_num)
        self.activities_meter = AverageMeterTensor(cfg.activities_num)
        self.loss_meter = AverageMeter()
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)
        self.activities_loss_weight = GeneralAverageMeterTensor(cfg.activities_num)
        self.label_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.result_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.coef_var_meter = AverageMeter()
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            i = 0
            for batch_data in tqdm(self.data_loader):
                i = i + 1
                self.baseprocess(batch_data)
                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            # renew the action loss weight by accuracy
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_loss_weight': self.actions_loss_weight.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().tolist(),
                'activities_loss_weight': self.activities_loss_weight.correct_rate_each.numpy().tolist(),
                'activities_each_num': self.activities_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'result_coef_meter': self.result_coef_meter.coef_rate,
                'coef_var':self.coef_var_meter.avg
            }

        elif self.mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in tqdm(self.data_loader):
                    self.baseprocess(batch_data)
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().tolist(),
                'activities_each_num': self.activities_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'result_coef_meter': self.result_coef_meter.coef_rate,
                'coef_var': self.coef_var_meter.avg
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B#N)
        actions_in = torch.cat(batch_data[2], dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)
        # reshape the activity label into tensor(B)
        activities_in = torch.cat(batch_data[1], dim=0)
        activities_in = activities_in.reshape(-1).to(device=self.device)

        # forward
        actions_scores, activities_scores, coef0 = self.model(
            (batch_data[0], batch_data[3]))  # tensor(B#N, actions_num) & tensor(B)

        # Predict
        actions_result = torch.argmax(actions_scores, dim=1).int().detach()
        activities_result = torch.argmax(activities_scores, dim=1).int().detach()
        # loss
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        activities_weights = torch.tensor(self.cfg.activities_weights).to(device=self.device)
        #   cross entropy
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        focal_loss = Focalloss()
        actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, device0=self.device,
                                                 weight=actions_weights, attenuate=cfg.focal_loss_factor)
        activities_loss, activi_loss_w = focal_loss(activities_scores, activities_in, device0=self.device,
                                                    weight=activities_weights, attenuate=cfg.focal_loss_factor)
        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss + self.cfg.activities_loss_weight * activities_loss
        if np.any(np.isnan(self.total_loss.cpu().detach().numpy())):
            a = 1
            pass
        self.loss_meter.update(self.total_loss.item(), batch_size)
        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)
        self.activities_meter.update(activities_result, activities_in)
        self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)
        self.activities_loss_weight.update(activi_loss_w.squeeze(1), activities_in)

        #  get the coefficient matrix log
        self.label_coef_meter.update(coef0, actions_in)
        self.result_coef_meter.update(coef0, actions_result)
        for _ in coef0:
            coef_var = torch.var(_, dim=-1)
            self.coef_var_meter.update(torch.mean(coef_var),n=coef_var.size()[0])

class VolleyballEpoch3():

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
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)
        self.label_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.result_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.coef_var_meter = AverageMeter()
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            i = 0
            if self.epoch % 5 == 0:
                a = 1
            for batch_data in tqdm(self.data_loader):
                i = i + 1

                self.baseprocess(batch_data)
                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            # renew the action loss weight by accuracy
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_loss_weight': self.actions_loss_weight.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'result_coef_meter': self.result_coef_meter.coef_rate,
                'coef_var':self.coef_var_meter.avg
            }

        elif self.mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in tqdm(self.data_loader):
                    self.baseprocess(batch_data)
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'result_coef_meter': self.result_coef_meter.coef_rate,
                'coef_var': self.coef_var_meter.avg
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B#N)
        actions_in = torch.cat(batch_data[2], dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)

        # forward
        actions_scores,  coef0, kl_loss = self.model(
            (batch_data[0], batch_data[3]))  # tensor(B#N, actions_num) & tensor(B)

        # Predict
        actions_result = torch.argmax(actions_scores, dim=1).int().detach()
        # loss
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        #   cross entropy
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        focal_loss = Focalloss()
        actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, device0=self.device,
                                                 weight=actions_weights, attenuate=cfg.focal_loss_factor)

        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss + self.cfg.kl_loss_weight * kl_loss
        if np.any(np.isnan(self.total_loss.cpu().detach().numpy())):
            a = 1
            pass
        self.loss_meter.update(self.total_loss.item(), batch_size)
        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)

        self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)


        #  get the coefficient matrix log
        self.label_coef_meter.update(coef0, actions_in)
        self.result_coef_meter.update(coef0, actions_result)
        for _ in coef0:
            coef_var = torch.var(_, dim=-1)
            self.coef_var_meter.update(torch.mean(coef_var).item(),n=coef_var.size()[0])

class VolleyballEpoch4():

    def __init__(self, mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

        self.mode = mode
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch

        self.actions_meter = AverageMeterTensor(cfg.actions_num)
        self.activities_meter = AverageMeterTensor(cfg.activities_num)
        self.loss_meter = AverageMeter()
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)
        self.activities_loss_weight = GeneralAverageMeterTensor(cfg.activities_num)
        self.label_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.cross_coef_meter = CoefMatrixMeter(cfg.actions_num, activity_num=cfg.activities_num)
        self.action_rela_meter = CorelationMeter(cfg.actions_num)
        self.activity_rela_meter = CorelationMeter(cfg.activities_num)
        self.coef_var_meter = AverageMeter()
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            i = 0
            for batch_data in tqdm(self.data_loader):
                i = i + 1
                self.baseprocess(batch_data)
                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            # renew the action loss weight by accuracy
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_loss_weight': self.actions_loss_weight.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().tolist(),
                'activities_loss_weight': self.activities_loss_weight.correct_rate_each.numpy().tolist(),
                'activities_each_num': self.activities_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'cross_coef_meter': self.cross_coef_meter.coef_rate,
                'acton_relation': self.action_rela_meter.class_acc,
                'activity_relation': self.activity_rela_meter.class_acc,
                'coef_var':self.coef_var_meter.avg
            }

        elif self.mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in tqdm(self.data_loader):
                    self.baseprocess(batch_data)
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().tolist(),
                'activities_each_num': self.activities_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'cross_coef_meter': self.cross_coef_meter.coef_rate,
                'acton_relation': self.action_rela_meter.class_acc,
                'activity_relation': self.activity_rela_meter.class_acc,
                'coef_var': self.coef_var_meter.avg
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B#N)
        actions_in = torch.cat(batch_data[2], dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)
        # reshape the activity label into tensor(B)
        activities_in = torch.cat(batch_data[1], dim=0)
        activities_in = activities_in.reshape(-1).to(device=self.device)

        # forward
        actions_scores, activities_scores, coef0, kl_loss = self.model(
            (batch_data[0], batch_data[3]))  # tensor(B#N, actions_num) & tensor(B)

        # Predict
        actions_result = torch.argmax(actions_scores, dim=1).int().detach()
        activities_result = torch.argmax(activities_scores, dim=1).int().detach()
        # loss
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        activities_weights = torch.tensor(self.cfg.activities_weights).to(device=self.device)
        #   cross entropy
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        focal_loss = Focalloss()
        actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, device0=self.device,
                                                 weight=actions_weights, attenuate=cfg.focal_loss_factor)
        activities_loss, activi_loss_w = focal_loss(activities_scores, activities_in, device0=self.device,
                                                    weight=activities_weights, attenuate=cfg.focal_loss_factor)
        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss + self.cfg.activities_loss_weight * activities_loss + self.cfg.kl_loss_weight * kl_loss
        if np.any(np.isnan(self.total_loss.cpu().detach().numpy())):
            a = 1
            pass
        self.loss_meter.update(self.total_loss.item(), batch_size)
        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)
        self.activities_meter.update(activities_result, activities_in)
        self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)
        self.activities_loss_weight.update(activi_loss_w.squeeze(1), activities_in)

        #  get the coefficient matrix log
        self.label_coef_meter.update(coef0, actions_in)
        self.cross_coef_meter.update(coef0, actions_result, activity_in0=activities_in, mode = 1)
        self.action_rela_meter.update(actions_in,actions_result)
        self.activity_rela_meter.update(activities_in,activities_result)
        for _ in coef0:
            coef_var = torch.var(_, dim=-1)
            self.coef_var_meter.update(torch.mean(coef_var),n=coef_var.size()[0])

class VolleyballEpoch5():
    """
    this is the processing of pre-train of the action linknet. In this processing, each direction inference of linknet
    will be used to calculate loss with GT individually.
    """

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
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)

        self.label_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.result_coef_meter = CoefMatrixMeter(cfg.actions_num)
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            i = 0
            for batch_data in tqdm(self.data_loader):
                i = i + 1
                self.baseprocess(batch_data)
                # Optim
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
            # renew the action loss weight by accuracy
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_loss_weight': self.actions_loss_weight.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'result_coef_meter': self.result_coef_meter.coef_rate
            }

        elif self.mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in tqdm(self.data_loader):
                    self.baseprocess(batch_data)
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().tolist(),
                'actions_each_num': self.actions_meter.all_num_each,
                'label_coef_meter': self.label_coef_meter.coef_rate,
                'result_coef_meter': self.result_coef_meter.coef_rate
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])
        person_num = [i.size()[0] for i in batch_data[2]]

        # reshape the action label into tensor(B#N)
        actions_in = torch.cat(batch_data[2], dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)
        # build the other action label
        other_action_in = [batch_data[2][i].repeat(1, person_num[i]-1).reshape(-1) for i in range(batch_size)]
        other_action_in = torch.cat(other_action_in, dim=0).to(device=self.device)

        # forward
        actions_scores, other_actions_scores = self.model(
            (batch_data[0], batch_data[3]), mode=0)  # tensor(B#N, actions_num) & tensor(B)

        # Predict
        actions_result = torch.argmax(actions_scores, dim=1).int().detach()  # (batch#num)
        other_action_result = torch.argmax(other_actions_scores, dim=1).int().detach()  # (batch#num*num-1)
        # loss
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)

        #   cross entropy
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        focal_loss = Focalloss()
        actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, device0=self.device,
                                                 weight=actions_weights, attenuate=cfg.focal_loss_factor)
        other_action_loss = F.cross_entropy(other_actions_scores, other_action_in, weight=actions_weights)
        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss + self.cfg.other_actions_loss_weight * other_action_loss
        if np.any(np.isnan(self.total_loss.cpu().detach().numpy())):
            a = 1
            pass
        self.loss_meter.update(self.total_loss.item(), batch_size)
        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)
        self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)
        # calculate the acc matrix
        coef0=[]
        coef1=[]
        datum1 = 0
        datum2 = 0
        for i in range(batch_size):
            num = person_num[i]
            action_mat_in = batch_data[2][i].repeat(1, num).reshape(num, num)  # (num, num)
            action_mat_result = other_action_result[datum1:datum1+num*(num-1)].reshape(num-1,num)
            action_mat_result = torch.cat((actions_result[datum2:datum2+num-1].unsqueeze(1),action_mat_result), dim=1)
            action_mat_result = torch.cat((action_mat_result.reshape(-1),actions_result[datum2+num-1].unsqueeze(0)))
            action_mat_result = action_mat_result.reshape(num, num)
            action_mat_result = (action_mat_result == action_mat_in.to(device=self.device, dtype=action_mat_result.dtype)).to(dtype=torch.float)
            coef0.append(action_mat_result)
            # coef1 the kl divergence between action score and other action score
            action_tar_score = actions_scores[datum2:datum2+num, :]  # (num, dim)
            action_cal_score = other_actions_scores[datum1:datum1+num*(num-1), :].reshape(num-1,num,-1)  # (num-1,num,dim)
            action_cal_score = torch.cat((action_tar_score[0:num-1,:].unsqueeze(1),action_cal_score), dim=1)
            action_cal_score = torch.cat((action_cal_score.reshape((num+1)*(num-1),-1), action_tar_score[num-1,:].unsqueeze(0)), dim=0)
            action_cal_score = action_cal_score.reshape(num, num, -1)  # (num, num, dim)
            action_tar_score = action_tar_score.unsqueeze(1)  # (num, 1, dim)
            action_cal_score = torch.pow(math.e,torch.sum(((action_cal_score/(action_tar_score+1e-12)+1e-12).log())*action_tar_score, dim=-1))
            coef1.append(action_cal_score)
            datum1 += num*(num-1)
            datum2 += num
        #  get the coefficient matrix log
        self.label_coef_meter.update(coef0, actions_in)  # product
        self.result_coef_meter.update(coef1, actions_in)  # kl loss

if __name__ == '__main__':
    introduce = "base self model renew weight 1e-4"
    print(introduce)
    cfg = config.Config2()
    para_path = None
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
    # create logger object
    log = utils.Logger(cfg.outputPath)
    log.fPrint(introduce)
    for item in cfg.__dict__:
        log.fPrint('%s:%s' % (str(item), str(cfg.__dict__[item])))
    # create tensorboard writer
    TBWriter = SummaryWriter(logdir=cfg.outputPath, comment='tensorboard')
    # device state
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # generate the volleyball dataset object
    full_dataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, mode=cfg.dataset_mode)
    # get the object information(object categories count)
    cfg.actions_num, cfg.activities_num = full_dataset.classCount()

    # divide the whole dataset into train and test
    full_dataset_len = full_dataset.__len__()
    if cfg.split_mode == 3:
        train_len = int(full_dataset_len * cfg.dataset_splitrate)
        test_len = full_dataset_len - train_len
        trainDataset, testDataset = data.random_split(full_dataset, [train_len, test_len])
    elif cfg.split_mode == 2:
        random_seed = 137  # set the seed
        random.seed(random_seed)
        indices = list(range(full_dataset_len))
        random.shuffle(indices)
        split = int(cfg.dataset_splitrate * full_dataset_len)
        train_indices = indices[:split]
        test_indices = indices[split:]
        trainDataset = data.Subset(full_dataset, train_indices)
        testDataset = data.Subset(full_dataset, test_indices)
    else:  # split_mode = 1
        trainDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.train_seqs,
                                                            mode=cfg.dataset_mode)
        testDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.test_seqs,
                                                           mode=cfg.dataset_mode)
    # begin model train in
    #   dataloader implement
    if cfg.train_mode == 0:
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        #    build model
        model = SelfNet2(cfg.imageSize, cfg.crop_size, cfg.actions_num, device, **cfg.model_para)  # type: SelfNet2
        model.to(device=device)
        model.train()
        #    optimizer implement
        optimizer = optim.Adam(
            [
                {"params": model.baselayer.backbone_net.parameters()},
                {"params": model.baselayer.mod_embed.parameters()},
                {"params": model.read_actions.parameters()}
            ],
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        # continue work
        if cfg.goon is True:
            model.loadmodel(cfg.goon_path1)
            optimizer.load_state_dict(torch.load(cfg.goon_path2))
        #    begin training
        start_epoch = cfg.start_epoch
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
            if epoch in cfg.lr_plan:
                adjust_lr(optimizer, cfg.lr_plan[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                                epoch=epoch).main()
            for each_info in train_result_info:
                log.fPrint('%s:%s' % (str(each_info), str(train_result_info[each_info])))
            all_info.append(train_result_info)
            TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
            TBWriter.add_scalar('train1_acc', train_result_info['actions_acc'], epoch)
            TBWriter.add_scalars('train1_acc_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
            #  test in each interval times
            if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
                model.train(False)
                test_result_info = VolleyballEpoch('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_acc', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalars('test1_acc_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    1, epoch, test_result_info['activities_acc'])
                model.savemodel(filepath)
                para_path = filepath

            if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    elif cfg.train_mode==1 or cfg.train_mode==2:
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        #    build model
        model = LinkNet1(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.activities_num, device=device,
                         **cfg.model_para)  # type: LinkNet1
        model.to(device=device)
        model.train()
        # load model parameter from the path in mode 2
        if cfg.train_mode == 2:
            para_path = cfg.para_load_path
            model.loadmodel(para_path, mode=1)
        #    optimizer implement
        params_group = [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.linklayer.parameters()},
            #{"params": model.receilayer.parameters()},
            {"params": model.read_actions.parameters()},
            {"params": model.read_activities.parameters()}
        ]
        optimizer = optim.Adam(
            params_group,
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        # continue work
        if cfg.goon is True:
            model.loadmodel(cfg.goon_path1, 0)
            optimizer.load_state_dict(torch.load(cfg.goon_path2))
        #    begin training
        start_epoch = cfg.start_epoch
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
            if epoch in cfg.lr_plan:
                adjust_lr(optimizer, cfg.lr_plan[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch2('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                                 epoch=epoch).main()
            for each_info in train_result_info:
                log.fPrint('%s:%s' % (str(each_info), str(train_result_info[each_info])))
            all_info.append(train_result_info)
            TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
            TBWriter.add_scalar('train1_action', train_result_info['actions_acc'], epoch)
            TBWriter.add_scalar('train1_activity', train_result_info['activities_acc'], epoch)
            TBWriter.add_scalars('train1_action_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
            #  test in each interval times
            if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
                model.train(False)
                test_result_info = VolleyballEpoch2('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_action', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalar('test1_activity', test_result_info['activities_acc'], epoch)
                TBWriter.add_scalars('test1_action_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    1, epoch, test_result_info['activities_acc'])
                model.savemodel(filepath)
                filepath = cfg.outputPath + '/model/stage%d_optimizer_epoch%d.pth' % (
                    1, epoch)
                torch.save(optimizer.state_dict(), filepath)

            if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    elif cfg.train_mode==3 :
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        #    build model
        model = LinkNet1(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.activities_num, device=device,
                         **cfg.model_para)  # type: LinkNet
        model.to(device=device)
        model.train()
        # load model parameter from the path in mode 2
        if cfg.pre_train == 1:
            para_path = cfg.para_load_path
            model.loadmodel(para_path, 0)
        #    optimizer implement
        params_group = [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.linklayer.parameters()},
            {"params": model.read_actions.parameters()},
            {"params": model.read_activities.parameters()}
        ]
        optimizer = optim.Adam(
            params_group,
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        # continue work
        if cfg.goon is True:
            model.loadmodel(cfg.goon_path1, 0)
            optimizer.load_state_dict(torch.load(cfg.goon_path2))
        #    begin training
        start_epoch = cfg.start_epoch
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
            if epoch in cfg.lr_plan:
                adjust_lr(optimizer, cfg.lr_plan[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch4('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                                 epoch=epoch).main()
            for each_info in train_result_info:
                log.fPrint('%s:%s' % (str(each_info), str(train_result_info[each_info])))
            all_info.append(train_result_info)
            TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
            TBWriter.add_scalar('train1_action', train_result_info['actions_acc'], epoch)
            TBWriter.add_scalar('train1_activity', train_result_info['activities_acc'], epoch)
            TBWriter.add_scalars('train1_action_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
            #  test in each interval times
            if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
                model.train(False)
                test_result_info = VolleyballEpoch4('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_action', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalar('test_activity', test_result_info['activities_acc'], epoch)
                TBWriter.add_scalars('test1_action_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    3, epoch, test_result_info['activities_acc'])
                model.savemodel(filepath)
                filepath = cfg.outputPath + '/model/stage%d_optimizer_epoch%d.pth' % (
                    3, epoch)
                torch.save(optimizer.state_dict(), filepath)

            if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    elif cfg.train_mode==4 :
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        #    build model
        model = GCN_link(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.activities_num, device=device,
                         **cfg.model_para)  # type: GCN_link
        model.to(device=device)
        model.train()
        # load model parameter from the path in mode 2
        para_path = cfg.para_load_path
        model.loadmodel(para_path, 1)
        #    optimizer implement
        params_group = [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.linklayer.parameters()},
            {"params": model.read_actions.parameters()},
            {"params": model.GCN_embedding.parameters()},
            {"params": model.read_activities.parameters()}
        ]
        optimizer = optim.Adam(
            params_group,
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        # continue work
        if cfg.goon is True:
            model.loadmodel(cfg.goon_path1, 0)
            optimizer.load_state_dict(torch.load(cfg.goon_path2))
        #    begin training
        start_epoch = cfg.start_epoch
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
            if epoch in cfg.lr_plan:
                adjust_lr(optimizer, cfg.lr_plan[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch2('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                                 epoch=epoch).main()
            for each_info in train_result_info:
                log.fPrint('%s:%s' % (str(each_info), str(train_result_info[each_info])))
            all_info.append(train_result_info)
            TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
            TBWriter.add_scalar('train1_action', train_result_info['actions_acc'], epoch)
            TBWriter.add_scalars('train1_action_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
            TBWriter.add_scalar('train1_activity', train_result_info['activities_acc'], epoch)
            #  test in each interval times
            if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
                model.train(False)
                test_result_info = VolleyballEpoch2('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_action', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalars('test1_action_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                TBWriter.add_scalar('test1_activity', test_result_info['activities_acc'], epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    1, epoch, test_result_info['activities_acc'])
                model.savemodel(filepath)
                filepath = cfg.outputPath + '/model/stage%d_optimizer_epoch%d.pth' % (
                    1, epoch)
                torch.save(optimizer.state_dict(), filepath)

            if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    elif cfg.train_mode==5:
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        #    build model
        model = LinkNet4(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.activities_num, device=device,
                         **cfg.model_para)  # type: LinkNet4
        model.to(device=device)
        model.train()
        # load model parameter from the path in mode 2
        para_path = cfg.para_load_path
        model.loadmodel(para_path, 1)
        #    optimizer implement
        params_group = [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.self_view.parameters()},
            {"params": model.linklayer.parameters()},
            {"params": model.read_actions.parameters()},
            {"params": model.read_activities.parameters()}
        ]
        optimizer = optim.Adam(
            params_group,
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        # continue work
        if cfg.goon is True:
            model.loadmodel(cfg.goon_path1, 0)
            optimizer.load_state_dict(torch.load(cfg.goon_path2))
        #    begin training
        start_epoch = cfg.start_epoch
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
            if epoch in cfg.lr_plan:
                adjust_lr(optimizer, cfg.lr_plan[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch4('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                                 epoch=epoch).main()
            for each_info in train_result_info:
                log.fPrint('%s:%s' % (str(each_info), str(train_result_info[each_info])))
            all_info.append(train_result_info)
            TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
            TBWriter.add_scalar('train1_action', train_result_info['actions_acc'], epoch)
            TBWriter.add_scalar('train1_activity', train_result_info['activities_acc'], epoch)
            TBWriter.add_scalars('train1_action_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
            #  test in each interval times
            if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
                model.train(False)
                test_result_info = VolleyballEpoch4('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_action', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalar('test1_activity', test_result_info['activities_acc'], epoch)
                TBWriter.add_scalars('test1_action_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    1, epoch, test_result_info['activities_acc'])
                model.savemodel(filepath)
                filepath = cfg.outputPath + '/model/stage%d_optimizer_epoch%d.pth' % (
                    1, epoch)
                torch.save(optimizer.state_dict(), filepath)

            if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    elif cfg.train_mode==6 :
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        #    build model
        model = LinkNet3(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.activities_num, device=device,
                         **cfg.model_para)  # type: LinkNet3
        model.to(device=device)
        model.train()
        # load model parameter from the path in mode 2
        if cfg.pre_train == 1:
            para_path = cfg.para_load_path
            model.loadmodel(para_path, 1)
        #    optimizer implement
        params_group = [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.linklayer.parameters()},
            {"params": model.read_actions.parameters()}
        ]
        optimizer = optim.Adam(
            params_group,
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        # continue work
        if cfg.goon is True:
            model.loadmodel(cfg.goon_path1, 0)
            optimizer.load_state_dict(torch.load(cfg.goon_path2))
        #    begin training
        start_epoch = cfg.start_epoch
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
            if epoch in cfg.lr_plan:
                adjust_lr(optimizer, cfg.lr_plan[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch5('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                                 epoch=epoch).main()
            for each_info in train_result_info:
                log.fPrint('%s:%s' % (str(each_info), str(train_result_info[each_info])))
            all_info.append(train_result_info)
            TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
            TBWriter.add_scalar('train1_action', train_result_info['actions_acc'], epoch)
            TBWriter.add_scalars('train1_action_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
            #  test in each interval times
            if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
                model.train(False)
                test_result_info = VolleyballEpoch5('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_action', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalars('test1_action_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    2, epoch, test_result_info['actions_acc'])
                model.savemodel(filepath)
                filepath = cfg.outputPath + '/model/stage%d_optimizer_epoch%d.pth' % (
                    2, epoch)
                torch.save(optimizer.state_dict(), filepath)

            if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    if cfg.train_mode == 'T':
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        action_num = cfg.actions_num
        activity_num = cfg.activities_num
        activity_each = torch.zeros((activity_num, 1), dtype=torch.float)+1e-12
        action_each = torch.zeros((activity_num, action_num), dtype=torch.float)
        acc_each = action_each/activity_each
        for batch_data in tqdm(train_loader):
            batch_size = len(batch_data[0])
            for batch in range(batch_size):
                num = batch_data[2][batch].size()[0]
                activity_each[batch_data[1][batch][0],0] += num
                for per in range(num):
                    action_each[batch_data[1][batch][0], batch_data[2][batch][per, 0]] += 1
        acc_each = action_each/activity_each
        log.fPrint(acc_each)

    TBWriter.close()
