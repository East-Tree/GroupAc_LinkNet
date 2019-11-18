import config

import torch
import torchvision.transforms as transforms
from torch.utils import data

from PIL import Image

cfg = config.Config()

print("The work directory is %s" % cfg.workPath)
print("From %s import volleyball dataset" % cfg.dataPath)

# define the volleyball dataset class
class VolleyballDataset(data.Dataset):
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.datasetPath = cfg.dataPath + '/volleyball'
        self.frameList = list(range(50)) #generate reading list for volleyball dataset
        self.annotation = self.readAnnotation()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    @staticmethod
    def idTextChange():

        ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                      'l_set', 'l-spike', 'l-pass', 'l_winpoint']
        NUM_ACTIVITIES = 8
        ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
                   'moving', 'setting', 'spiking', 'standing',
                   'waiting']
        NUM_ACTIONS = 9

        gact_to_id = {name: i}

    def readAnnotation(self):
        data = {}
        for sid in self.frameList:
            # read annotation from each directory
            annotationPath = self.datasetPath + '/%d/annotations.txt' % sid
            annotation = {}
            with open(annotationPath) as f:
                for line in f.readlines():

