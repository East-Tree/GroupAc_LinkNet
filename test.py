import volleyballDataset
import config
import pickle
import math
from torch.utils import data
import basemodel
import backbone

cfg = config.Config1()

a = {
    'a':1,
    'b':2
}
for c in a:
    print(c)

volleyballSet = volleyballDataset.VolleyballDataset(cfg.dataPath, cfg.imageSize)

c = data.DataLoader(volleyballSet, batch_size=4, collate_fn=volleyballDataset.new_collate)
mod = backbone.MyInception_v3()