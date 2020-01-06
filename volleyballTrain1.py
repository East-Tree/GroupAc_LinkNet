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
    introduce = "base self model renew weight 1e-4"
    print(introduce)
    cfg = config.Config1()
    # create logger object
    log = utils.Logger(cfg.outputPath)
    log.fPrint(introduce)
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
        random_seed = 7 # set the seed
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
    arch_para = {
        'person_fea_dim': cfg.individual_dim,
        'state_fea_dim': cfg.state_dim,
        'dropout_prob': cfg.train_dropout_prob
    }
    model = SelfNet2(cfg.imageSize, cfg.crop_size, cfg.actions_num, device,**arch_para)  # type: SelfNet2
    model.to(device=device)
    model.train()
    log.fPrint(torch.sum)
    a = 1
    #    optimizer implement
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)
    #    begin training
    start_epoch = 1
    all_info = []
    for epoch in range(start_epoch + cfg.max_epoch + 1):
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch], log)

        #  each epoch in the iteration
        model.train()
        train_result_info = volleyball.VolleyballEpoch('train',train_loader,model,device,cfg,optimizer,epoch).main()
        log.fPrint(train_result_info)
        all_info.append(train_result_info)


        #  test in each interval times
        if epoch % cfg.test_interval_epoch == 0:
            model.train(False)
            test_result_info = volleyball.VolleyballEpoch('test',test_loader,model,device,cfg).main()
            log.fPrint(test_result_info)
            filepath = cfg.outputPath + '/stage%d_epoch%d_%.2f%%.pth' % (1, epoch, test_result_info['actions_acc'])
            model.savemodel(filepath)

        if epoch > 10:
            if abs(all_info[epoch]['loss'] - all_info[epoch-1]['loss']) < cfg.break_line:
                break
