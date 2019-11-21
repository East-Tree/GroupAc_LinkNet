import config

import numpy as np
import torch
import torchvision.transforms.functional as Tfunc
from torch.utils import data

from PIL import Image


# define the volleyball dataset class
class VolleyballDataset(data.Dataset):
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.datasetPath = cfg.dataPath + '/volleyball'
        self.frameList = list(range(50)) #generate reading list for volleyball dataset
        self.annotationData = self.readAnnotation()
        self.allFrames = self.readAllFrames()

    def __len__(self):

        return len(self.allFrames)

    def __getitem__(self, index):

        frameItem = self.allFrames[index]
        return self.readSpecificFrame(frameItem)


    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
    NUM_ACTIVITIES = 8
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    NUM_ACTIONS = 9
    activity2id = {name:i for i, name in enumerate(ACTIVITIES)} # mode 1
    action2id = {name:i for i, name in enumerate(ACTIONS)} # mode 2
    id2activity = {i:name for i, name in enumerate(ACTIVITIES)} # mode 3
    id2action = {i:name for i, name in enumerate(ACTIONS)}

    @classmethod
    def activityToId(cls, index):
        assert index in cls.ACTIVITIES, 'not in activity list'
        return cls.activity2id[index]

    @classmethod
    def actionToId(cls,index):
        assert index in cls.ACTIONS, 'not in action list'
        return cls.action2id[index]

    @classmethod
    def idToActivity(cls,index):
        assert index in list(range(cls.NUM_ACTIVITIES)), 'not in activity range'
        return cls.id2activity[index]

    @classmethod
    def idToAction(cls,index):
        assert index in list(range(cls.NUM_ACTIONS)), 'not in action range'
        return  cls.id2action[index]


    def readAnnotation(self):
        """
        read all annotation according to self.frameList
        :return: {sid: {fid: {dict for frame information}, ...}, ...}
        """
        data0 = {}
        for sid in self.frameList:
            # read annotation from each directory
            annotationPath = self.datasetPath + '/%d/annotations.txt' % sid
            annotation = {}
            with open(annotationPath) as f:
                for line in f.readlines():
                    """
                    begin the data processing from meta annotation data, the meta formation like:
                    48075.jpg r_winpoint 372 442 86 130 falling 712 426 73 124 falling 338 346 45 159....
                    """
                    value = line[:-1].split(' ')
                    fileName = value[0]
                    activity = value[1]
                    activity = VolleyballDataset.activityToId(activity)  # convert the activity to id number
                    value = value[2:]
                    induvidualNum = len(value) / 5.0
                    assert type(induvidualNum) == int, 'the error occurs in %s at %s' % (fileName, annotationPath)
                    induvidualNum=int(induvidualNum)   # here is the bbox count in a frame
                    action = []
                    bbox = []
                    for i in range(induvidualNum):
                        x, y, w, h = map(int, value[i:i+4])
                        x2 = x + w
                        y2 = y + h
                        bbox.append((x, y, x2, y2))
                        action.append(VolleyballDataset.actionToId(value[i+5]))
                    fid = int(fileName.split('.')[0])
                    annotation[fid] = {
                        'file_name': fileName,  # the file name of a single frame, like 00000.jpg
                        'group_activity': activity,  # the index(int) of group activity
                        'people_num': induvidualNum,  # the count of people in a frame
                        'action': action,  # the index(int) of each people in a frame
                        'bounding_box': bbox  # the bbox of each people, formatted as [(x1,y1,x2,y2), ...]
                    }
            data0[sid] = annotation
        return data0

    def readAllFrames(self):
        frames = []
        for sid, anno in self.annotationData.items():
            for fid, subAnno in anno.items():
                frames.append((sid, fid))
        return frames

    def readSpecificFrame(self,frameIndex: tuple):

        sid, fid = frameIndex
        framePath = self.datasetPath + '/%d/%d/%d.jpg' % (sid, fid, fid)
        img = Image.open(framePath)
        img = Tfunc.resize(img,self.cfg.imageSize)
        img = np.array(img)

        # H, W, 3 -> 3, H, W
        img.transpose(2,0,1)

        activity = self.annotationData[sid][fid]['group_activity']
        action = self.annotationData[sid][fid]['action']
        bbox = self.annotationData[sid][fid]['bounding_box']

        return img, activity, action, bbox