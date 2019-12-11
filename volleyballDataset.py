import numpy as np
import torch
import torchvision.transforms.functional as Tfunc
from torch.utils import data

from PIL import Image


# a new a collate fun for this dataset, because each sampling contain different shape tensor
def new_collate(batch):
    # [sample1, sample2, ...]
    # (img, activities, actions, bbox)
    image = [item[0] for item in batch]
    activities = [item[1] for item in batch]
    actions = [item[2] for item in batch]
    bbox = [item[3] for item in batch]

    # return:
    # [image, activities, actions, bbox]
    # image:[sample1, sample2, ...]
    return [image, activities, actions, bbox]

# define the volleyball dataset class
class VolleyballDataset(data.Dataset):
    def __init__(self, cfg_dataPath, cfg_imagesize=(720,1280)):
        self.datasetPath = cfg_dataPath + '/volleyball'
        self.frameList = list(range(50)) #generate reading list for volleyball dataset

        # according to official document, the label bbox is corresponding to (720,1280)
        self.scaleW = float(cfg_imagesize[1]/1280)
        self.scaleH = float(cfg_imagesize[0]/720)
        self.imagesize = cfg_imagesize

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
        assert index in cls.ACTIONS, '%s not in action list' % index
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
                    activity = [VolleyballDataset.activityToId(activity)]  # convert the activity to id number
                    value = value[2:]
                    #assert len(value)%5 == 0, 'the error occurs in %s at %s' % (fileName, annotationPath)
                    if len(value)%5 != 0:
                        print('the error occurs in %s at %s' % (fileName, annotationPath))
                    induvidualNum = int(len(value) / 5)   # here is the bbox count in a frame
                    action = []
                    bbox = []
                    for i in range(induvidualNum):
                        x1, y1, w, h = map(int,value[i*5:i*5+4])
                        x2 = x1 + w
                        y2 = y1 + h
                        x1, x2 = map(lambda x: int(x * self.scaleW), [x1, x2])
                        y1, y2 = map(lambda x: int(x * self.scaleH), [y1, y2])
                        bbox.append([x1, y1, x2, y2])
                        action.append([VolleyballDataset.actionToId(value[i*5+4])])
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
        img = Tfunc.resize(img, self.imagesize)
        img = np.array(img)

        # H, W, 3 -> 3, H, W
        img = img.transpose((2, 0, 1))

        # transform all annotation data into numpy
        activity = np.array(self.annotationData[sid][fid]['group_activity'])
        action = np.array(self.annotationData[sid][fid]['action'])
        bbox = np.array(self.annotationData[sid][fid]['bounding_box'])

        # transform all data into torch
        img = torch.from_numpy(img)
        activity = torch.from_numpy(activity)
        action = torch.from_numpy(action)
        bbox = torch.from_numpy(bbox)

        return img, activity, action, bbox

    def classCount(self):

        return self.NUM_ACTIONS, self.NUM_ACTIVITIES
