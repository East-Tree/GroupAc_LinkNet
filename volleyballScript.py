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
        train_len = int(full_dataset.__len__() * cfg.dataset_splitrate)
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
    if cfg.train_mode == 'T1':
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
    elif cfg.train_mode == 'T2':
        params = {
            'batch_size': cfg.batch_size,
            'shuffle': True
        }
        train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
        test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
        action_num = cfg.actions_num
        activity_num = cfg.activities_num
        action_dic = {i: [] for i in range(action_num)}

        cood0 = index.reshape(-1, 4)  # (ob_num, 4[x1,y1,x2,y2])
        cood0.requires_grad_(False)

        op1 = torch.tensor([[0.5, 0], [0, 0.5], [0.5, 0], [0, 0.5]], device=self.device)

        cood0 = torch.mm(cood0, op1.to(dtype=cood0.dtype))  # (ob_num, 2[x,y])
        cood0 = cood0.repeat(object_num, 1, 1)  # (ob_num(repeat), ob_num, 2)
        cood0 = torch.transpose(cood0, 0, 1) - cood0  # (ob_num(source),ob_num(sink),2[Xsi-Xso,Ysi-Yso]))


    TBWriter.close()
