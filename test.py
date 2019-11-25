import volleyballDataset
import config
cfg = config.Config()

volleyballSet = volleyballDataset.VolleyballDataset(cfg)

c = volleyballSet.__getitem__(1)
print(c[0][1][200])