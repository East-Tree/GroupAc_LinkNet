import volleyballDataset
import config
import pickle
import math
from torch.utils import data

cfg = config.Config1()


volleyballSet = volleyballDataset.VolleyballDataset(cfg)

c = data.DataLoader(volleyballSet, batch_size=4)
for da in c:
    print(da[3].size())