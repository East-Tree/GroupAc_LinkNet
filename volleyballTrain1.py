from typing import Dict, Any, Callable, Tuple

import volleyballDataset
import volleyball
from basemodel import *
import config
import utils

from torch.utils import data
from torch import optim
import random

if __name__ == '__main__':
    introduce = "This program is the first model of Link Net model in volleyball dataset"
    print(introduce)
    cfg = config.Config1()
    # create logger object
    log = utils.Logger(cfg.outputPath)
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
        random_seed = 777  # set the seed
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
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True
    }
    train_loader = data.DataLoader(trainDataset,collate_fn=volleyballDataset.new_collate, **params)
    test_loader = data.DataLoader(testDataset,collate_fn=volleyballDataset.new_collate, **params)
    #    build model
    model = SelfNet(cfg.imageSize, cfg.crop_size, cfg.actions_num, device)  # type: SelfNet
    model.to(device=device)
    model.train()

    a = 1
    #    optimizer implement
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)
    #    begin training
    start_epoch = 1
    for epoch in range(start_epoch + cfg.max_epoch):
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch], log)

        #  each epoch in the iteration
        model.train()
        train_result_info = volleyball.VolleyballEpoch('train',train_loader,model,device,cfg,optimizer,epoch)
        log.fPrint(train_result_info)

        #  test in each interval times
        if epoch % cfg.test_interval_epoch == 0:
            model.train(False)
            test_result_info = volleyball.VolleyballEpoch('test',test_loader,model,device,cfg)
            log.fPrint(test_result_info)

