import volleyballDataset
import config
import pickle
import math


cfg = config.Config()

a= pickle.load(open(cfg.dataPath + '/volleyball/tracks_normalized.pkl', 'rb'))

volleyballSet = volleyballDataset.VolleyballDataset(cfg)

c = volleyballSet.__getitem__(1)
print(c[0][1][200])