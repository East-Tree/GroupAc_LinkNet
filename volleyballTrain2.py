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
            :param target: [p']  ground truth
            :param weight:
            :return:
            """
        if device0 is None:
            device1 = torch.device('cpu')
        else:
            device1 = device0
        input_soft = F.softmax(input, dim=1)
        input_logsoft = F.log_softmax(input, dim=1)
        batch = target.size()[0]
        target_mask = target.reshape(-1,1)
        input_soft = torch.gather(input_soft, 1, target_mask)
        input_logsoft = torch.gather(input_logsoft, 1, target_mask)
        if weight is None:
            weight_tensor = torch.tensor([1]*batch, device=device1)
        else:
            weight_tensor = weight.repeat(batch,1).to(device=device1)
            weight_tensor = torch.gather(weight_tensor, 1, target_mask)
        weight_tensor = weight_tensor.reshape(-1,1)
        loss = (-1)*weight_tensor*torch.pow(1.0-input_soft,attenuate)*input_logsoft
        loss = torch.mean(loss,dim=0)

        return loss

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
        actions_loss = focal_loss(actions_scores, actions_in, self.device, weight=actions_weights)
        actions_result = torch.argmax(actions_scores, dim=1).int()

        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)

        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss
        self.loss_meter.update(self.total_loss.item(), batch_size)

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
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            for batch_data in tqdm(self.data_loader):
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
                'actions_each_num': self.actions_meter.all_num_each,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().tolist(),
                'activities_each_num': self.activities_meter.all_num_each
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
                'activities_each_num': self.activities_meter.all_num_each
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
        actions_scores, activities_scores = self.model((batch_data[0], batch_data[3]))  # tensor(B#N, actions_num) & tensor(B)

        # Predict
        actions_result = torch.argmax(actions_scores, dim=1).int()
        activities_result = torch.argmax(activities_scores, dim=1).int()
        # loss
        actions_weights = torch.tensor(self.cfg.actions_weights2).to(device=self.device)
        activities_weights = torch.tensor(self.cfg.activities_weights2).to(device=self.device)
        #   cross entropy
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        focal_loss = Focalloss()
        actions_loss = focal_loss(actions_scores, actions_in, device0=self.device, weight=actions_weights)
        activities_loss = focal_loss(activities_scores, activities_in, device0=self.device, weight=activities_weights)
        # Total loss
        self.total_loss = self.cfg.actions_loss_weight2 * actions_loss + self.cfg.activities_loss_weight2 * activities_loss
        self.loss_meter.update(self.total_loss.item(), batch_size)
        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)
        self.activities_meter.update(activities_result, activities_in)

if __name__ == '__main__':
    introduce = "base self model renew weight 1e-4"
    print(introduce)
    cfg = config.Config1()
    para_path = None
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
    # create logger object
    log = utils.Logger(cfg.outputPath)
    log.fPrint(introduce)
    # create tensorboard writer
    TBWriter = SummaryWriter(logdir=cfg.outputPath, comment='tensorboard')
    # device state
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # generate the volleyball dataset object
    full_dataset = volleyballDataset.VolleyballDataset(cfg.dataPath, cfg.imageSize)
    # get the object information(object categories count)
    cfg.actions_num, cfg.activities_num = full_dataset.classCount()

    # divide the whole dataset into train and test
    full_dataset_len = full_dataset.__len__()
    if cfg.random_split:
        train_len = int(full_dataset_len * cfg.dataset_splitrate)
        test_len = full_dataset_len - train_len
        trainDataset, testDataset = data.random_split(full_dataset, [train_len, test_len])
    else:
        random_seed = 137  # set the seed
        random.seed(random_seed)
        indices = list(range(full_dataset_len))
        random.shuffle(indices)
        split = int(cfg.dataset_splitrate * full_dataset_len)
        train_indices = indices[:split]
        test_indices = indices[split:]
        trainDataset = data.Subset(full_dataset, train_indices)
        testDataset = data.Subset(full_dataset, test_indices)

    # begin model train in
    #   dataloader implement
    if cfg.train_mode == 0 or cfg.train_mode == 2:
        params = {
            'batch_size': cfg.batch_size1,
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
        #    begin training
        start_epoch = 1
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
                    1, epoch, test_result_info['actions_acc'])
                model.savemodel(filepath)
                para_path = filepath

            if epoch > 10:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    if cfg.train_mode == 1 or cfg.train_mode == 2:
        params = {
            'batch_size': cfg.batch_size2,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        # confirm the model parameter loading path from stage1
        if cfg.train_mode == 1:
            para_path = cfg.para_load_path
        #    build model
        model = LinkNet(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.activities_num, device=device, **cfg.model_para2)  # type: LinkNet
        model.to(device=device)
        model.train()
        #    optimizer implement
        optimizer = optim.Adam(
            [
                {"params": model.baselayer.backbone_net.parameters()},
                {"params": model.baselayer.mod_embed.parameters()},
                {"params": model.read_actions.parameters()},
                {"params": model.read_activities.parameters()},
                {"params": model.linklayer.biaslayer.parameters()}
            ],
            lr=cfg.train_learning_rate,
            weight_decay=cfg.weight_decay)
        #    begin training
        start_epoch = 1
        all_info = []
        for epoch in range(start_epoch, start_epoch + cfg.max_epoch2):
            if epoch in cfg.lr_plan2:
                adjust_lr(optimizer, cfg.lr_plan2[epoch], log)

            #  each epoch in the iteration
            model.train()
            train_result_info = VolleyballEpoch2('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
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
                test_result_info = VolleyballEpoch2('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
                for each_info in test_result_info:
                    log.fPrint('%s:%s' % (str(each_info), str(test_result_info[each_info])))
                TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
                TBWriter.add_scalar('test1_acc', test_result_info['actions_acc'], epoch)
                TBWriter.add_scalars('test1_acc_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                     epoch)
                filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                    1, epoch, test_result_info['actions_acc'])
                model.savemodel(filepath)

            if epoch > 10:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
    if cfg.train_mode == 'T':
        a = torch.tensor([[0.2,0.8],[0.3,0.7]])
        b = torch.tensor([0,1])
        w = torch.tensor([1.0,2.0])
        focal_loss = Focalloss()
        print(focal_loss(a,b,weight=w))
    TBWriter.close()
