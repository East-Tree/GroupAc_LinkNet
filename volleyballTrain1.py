import volleyballDataset
import volleyball
from basemodel import *
import config
import utils

from torch.utils import data
from torch import optim
import random


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
full_dataset = volleyballDataset.VolleyballDataset(cfg.dataPath,cfg.imageSize)
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
train_loader = data.DataLoader(trainDataset, **params)
#    build model
model = SelfNet(cfg.imageSize, cfg.crop_size, cfg.actions_num, device)
model.to(device=device)
model.train()
a = 1
#    optimizer implement
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)