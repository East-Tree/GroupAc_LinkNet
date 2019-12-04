import volleyballDataset
import volleyball
import config
import utils

if __name__ == '__main__':
    introduce = "This program is the first model of Link Net model in volleyball dataset"
    print(introduce)
    cfg = config.Config1
    # generate the volleyball dataset object
    dataset = volleyballDataset.VolleyballDataset(cfg)
    # get the object information(object categories count)
    cfg.actions_num, cfg.activities_num = dataset.classCount()
