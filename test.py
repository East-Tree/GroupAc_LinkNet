import volleyballDataset
import config
cfg = config.Config()

volleyballSet = volleyballDataset.VolleyballDataset(cfg)

print(volleyballSet.__getitem__(1))