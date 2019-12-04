import volleyballDataset
import volleyball
import config
import utils
from torch.utils import data
import torch
import random
import basemodel

if __name__ == '__main__':
    introduce = "This program is the first model of Link Net model in volleyball dataset"
    print(introduce)
    cfg = config.Config1()
    # create logger object
    log = utils.Logger(cfg.outputPath)

    # generate the volleyball dataset object
    full_dataset = volleyballDataset.VolleyballDataset(cfg)
    # get the object information(object categories count)
    cfg.actions_num, cfg.activities_num = full_dataset.classCount()

    # divide the whole dataset into train and test
    full_dataset_len = full_dataset.__len__()
    if cfg.random_split == True:
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
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True
    }
    train_loader = data.DataLoader(trainDataset, **params)

    #    device state
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #    build model
    model = basemodel.selfNet(cfg)
    model = model.to(device=device)

    model.train()

    a=1
