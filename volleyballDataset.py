import numpy as np
import torch
import torchvision.transforms.functional as Tfunc
from torch.utils import data
import random

from PIL import Image
import pickle

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

def seq_collate(batch):

    activities = [item[1] for item in batch]
    actions = [item[2] for item in batch]

    batch_size = len(batch)
    seq_len = len(batch[0][0])

    image =[]
    bbox =[]
    for j in range(seq_len):
        for i in range(batch_size):
            image.append(batch[i][0][j])
            bbox.append(batch[i][3][j])

    return [image, activities, actions, bbox]        

def randTimes(pro:float):
    proInst = int(pro)
    proFrac = pro - proInst
    seed = random.random()
    if seed<=proFrac:
        return proInst+1
    else:
        return proInst


# define the volleyball dataset class
class VolleyballDataset(data.Dataset):
    def __init__(self, cfg_dataPath, cfg_imagesize=(720, 1280), frameList = None, augment=None):
        self.datasetPath = cfg_dataPath + '/volleyball'
        if frameList is None:
            self.frameList = list(range(55))  # generate reading list for volleyball dataset
        else:
            self.frameList = frameList
        # according to official document, the label bbox is corresponding to (720,1280)
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
    activity2id = {name: i for i, name in enumerate(ACTIVITIES)}  # mode 1
    action2id = {name: i for i, name in enumerate(ACTIONS)}  # mode 2
    id2activity = {i: name for i, name in enumerate(ACTIVITIES)}  # mode 3
    id2action = {i: name for i, name in enumerate(ACTIONS)}

    @classmethod
    def activityToId(cls, index):
        assert index in cls.ACTIVITIES, 'not in activity list'
        return cls.activity2id[index]

    @classmethod
    def actionToId(cls, index):
        assert index in cls.ACTIONS, '%s not in action list' % index
        return cls.action2id[index]

    @classmethod
    def idToActivity(cls, index):
        assert index in list(range(cls.NUM_ACTIVITIES)), 'not in activity range'
        return cls.id2activity[index]

    @classmethod
    def idToAction(cls, index):
        assert index in list(range(cls.NUM_ACTIONS)), 'not in action range'
        return cls.id2action[index]

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
                    # assert len(value)%5 == 0, 'the error occurs in %s at %s' % (fileName, annotationPath)
                    if len(value) % 5 != 0:
                        print('the error occurs in %s at %s' % (fileName, annotationPath))
                    induvidualNum = int(len(value) / 5)  # here is the bbox count in a frame
                    action = []
                    bbox = []
                    for i in range(induvidualNum):
                        x1, y1, w, h = map(float, value[i * 5:i * 5 + 4])
                        x2 = x1 + w
                        y2 = y1 + h
                        x1, x2 = map(lambda x: x / 1280, [x1, x2])
                        y1, y2 = map(lambda x: x / 720, [y1, y2])
                        bbox.append([y1, x1, y2, x2])
                        action.append([VolleyballDataset.actionToId(value[i * 5 + 4])])
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

    def readSpecificFrame(self, frameIndex: tuple):

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

class VolleyballDatasetS(data.Dataset):
    def __init__(self, cfg_dataPath, cfg_imagesize=(720, 1280), frameList=None, mode=0, dataagument=None, seq_num=1):
        self.datasetPath = cfg_dataPath + '/volleyball'
        if frameList is None:
            self.frameList = list(range(55))  # generate reading list for volleyball dataset
        else:
            self.frameList = frameList
        # according to official document, the label bbox is corresponding to (720,1280)
        self.imagesize = cfg_imagesize
        self.dataCate = [.0] * 9
        self.annotationData = self.readAnnotation()
        self.trackData = self.readTrack()
        self.allFrames = self.readAllFrames()

        # sequence valid length
        self.preframe = 4
        self.postframe = 4
        self.seqlength = self.preframe+self.postframe+1
        self.seq_num = seq_num

        # data sampling mode
        """
        0. read the central frame from the sequence 
        1. read all frames from the sequence
        2. randomly read a frame from the sequence
        3. read the image in sequence format
        """
        self.mode = mode

    def __len__(self):

        if self.mode==0 or self.mode==2 or self.mode==3:
            return len(self.allFrames)
        else:
            return len(self.allFrames)*self.seqlength

    def __getitem__(self, index):

        if self.mode == 0:
            frameItem = self.allFrames[index]
            return self.readSpecificFrame(frameItem)
        elif self.mode == 1:
            sid = int(index / self.seqlength)
            fidIn = index % self.seqlength
            frameItem = self.allFrames[sid]
            return self.readSpecificFrameS(frameItem, fidIn)
        elif self.mode == 2:
            frameItem = self.allFrames[index]
            fidIn = random.randint(0, self.seqlength-1)
            return self.readSpecificFrameS(frameItem, fidIn)
        elif self.mode == 3:
            frameItem = self.allFrames[index]
            return self.readSpecificSeq(frameItem)
        else:
            assert False

    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
    NUM_ACTIVITIES = 8
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    NUM_ACTIONS = 9
    activity2id = {name: i for i, name in enumerate(ACTIVITIES)}  # mode 1
    action2id = {name: i for i, name in enumerate(ACTIONS)}  # mode 2
    id2activity = {i: name for i, name in enumerate(ACTIVITIES)}  # mode 3
    id2action = {i: name for i, name in enumerate(ACTIONS)}

    @classmethod
    def activityToId(cls, index):
        assert index in cls.ACTIVITIES, 'not in activity list'
        return cls.activity2id[index]

    @classmethod
    def actionToId(cls, index):
        assert index in cls.ACTIONS, '%s not in action list' % index
        return cls.action2id[index]

    @classmethod
    def idToActivity(cls, index):
        assert index in list(range(cls.NUM_ACTIVITIES)), 'not in activity range'
        return cls.id2activity[index]

    @classmethod
    def idToAction(cls, index):
        assert index in list(range(cls.NUM_ACTIONS)), 'not in action range'
        return cls.id2action[index]

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
                    self.dataCate[activity] += 1
                    activity = [activity]

                    value = value[2:]
                    # assert len(value)%5 == 0, 'the error occurs in %s at %s' % (fileName, annotationPath)
                    if len(value) % 5 != 0:
                        print('the error occurs in %s at %s' % (fileName, annotationPath))
                    induvidualNum = int(len(value) / 5)  # here is the bbox count in a frame
                    action = []
                    bbox = []
                    for i in range(induvidualNum):
                        x1, y1, w, h = map(float, value[i * 5:i * 5 + 4])
                        x2 = x1 + w
                        y2 = y1 + h
                        x1, x2 = map(lambda x: x / 1280, [x1, x2])
                        y1, y2 = map(lambda x: x / 720, [y1, y2])
                        bbox.append([x1, y1, x2, y2])
                        action.append([VolleyballDataset.actionToId(value[i * 5 + 4])])
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

    def readTrack(self):
        trackPath = self.datasetPath + '/tracks_normalized.pkl'
        return pickle.load(open(trackPath, 'rb'))

    def readAllFrames(self):
        frames = []
        # calculate proportion for each categories
        max = torch.max(torch.tensor(self.dataCate).float())
        pro = max/torch.tensor(self.dataCate).float()

        for sid, anno in self.annotationData.items():
            for fid, subAnno in anno.items():
                n = randTimes(float(pro[subAnno['group_activity']]))
                frames.extend([(sid, fid)] * n)
        return frames

    def readSpecificFrame(self, frameIndex: tuple):

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

    def readSpecificFrameS(self, frameIndex: tuple, fidIn):

        sid, fid0 = frameIndex
        fid = fid0 - self.preframe + fidIn
        framePath = self.datasetPath + '/%d/%d/%d.jpg' % (sid, fid0, fid)
        img = Image.open(framePath)
        img = Tfunc.resize(img, self.imagesize)
        img = np.array(img)

        # H, W, 3 -> 3, H, W
        img = img.transpose((2, 0, 1))

        # transform all annotation data into numpy
        activity = np.array(self.annotationData[sid][fid0]['group_activity'])
        action = np.array(self.annotationData[sid][fid0]['action'])

        # read bbox coordinate from track file
        bbox = self.trackData[(sid, fid0)][fid]  # np(y1,x1,y2,x2)
        bbox = bbox[:, (1,0,3,2)]

        # transform all data into torch
        img = torch.from_numpy(img)
        activity = torch.from_numpy(activity)
        action = torch.from_numpy(action)
        bbox = torch.from_numpy(bbox)

        return img, activity, action, bbox

    def readSpecificSeq(self, frameIndex: tuple):

        # calculate the fid number
        if self.seq_num % 2 == 0: # even
            inter = self.seqlength // (self.seq_num+1)
            fidIn = [i for i in range(1,self.seqlength_1) if i % inter==0]
        else: # odd
            inter =  self.seqlength // (self.seq_num-1)
            fidIn = [i for i in range(self.seqlength) if i % inter==0]
        
        imgList = []
        bboxList = []
        for id in fidIn:
            sid, fid0 = frameIndex
            fid = fid0 - self.preframe + id
            framePath = self.datasetPath + '/%d/%d/%d.jpg' % (sid, fid0, fid)
            img = Image.open(framePath)
            img = Tfunc.resize(img, self.imagesize)
            img = np.array(img)

            # H, W, 3 -> 3, H, W
            img = img.transpose((2, 0, 1))
            imgList.append(img)

            # read bbox coordinate from track file
            bbox = self.trackData[(sid, fid0)][fid]  # np(y1,x1,y2,x2)
            bbox = bbox[:, (1,0,3,2)]

            bboxList.append(bbox)

        # transform all annotation data into numpy
        activity = np.array(self.annotationData[sid][fid0]['group_activity'])
        action = np.array(self.annotationData[sid][fid0]['action'])

        

        # transform all data into torch
        
        imgList = [torch.from_numpy(img) for img in imgList]
        bboxList = [torch.from_numpy(bbox) for bbox in bboxList]
        activity = torch.from_numpy(activity)
        action = torch.from_numpy(action)
        

        return imgList, activity, action, bboxList


    def classCount(self):

        return self.NUM_ACTIONS, self.NUM_ACTIVITIES

# this class only used to draw bounding box in origin images
class VolleyballDatasetDraw(data.Dataset):
    def __init__(self, cfg_dataPath, cfg_imagesize=(720, 1280), frameList=None, mode=0, dataagument=None):
        self.datasetPath = cfg_dataPath + '/volleyball'
        if frameList is None:
            self.frameList = list(range(55))  # generate reading list for volleyball dataset
        else:
            self.frameList = frameList
        # according to official document, the label bbox is corresponding to (720,1280)
        self.imagesize = cfg_imagesize
        self.dataCate = [.0] * 9
        self.annotationData = self.readAnnotation()
        self.trackData = self.readTrack()
        self.allFrames = self.readAllFrames()

        # sequence valid length
        self.preframe = 4
        self.postframe = 4
        self.seqlength = self.preframe+self.postframe+1

        # data sampling mode
        """
        0. read the central frame from the sequence 
        1. read all frames from the sequence
        2. randomly read a frame from the sequence
        """
        self.mode = mode

    def __len__(self):

        if self.mode==0 or self.mode==2:
            return len(self.allFrames)
        else:
            return len(self.allFrames)*self.seqlength

    def __getitem__(self, index):

        if self.mode == 0:
            frameItem = self.allFrames[index]
            return self.readSpecificFrame(frameItem)
        elif self.mode == 1:
            sid = int(index / self.seqlength)
            fidIn = index % self.seqlength
            frameItem = self.allFrames[sid]
            return self.readSpecificFrameS(frameItem, fidIn)
        elif self.mode == 2:
            frameItem = self.allFrames[index]
            fidIn = random.randint(0, self.seqlength-1)
            return self.readSpecificFrameS(frameItem, fidIn)
        else:
            assert False

    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
    NUM_ACTIVITIES = 8
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    NUM_ACTIONS = 9
    activity2id = {name: i for i, name in enumerate(ACTIVITIES)}  # mode 1
    action2id = {name: i for i, name in enumerate(ACTIONS)}  # mode 2
    id2activity = {i: name for i, name in enumerate(ACTIVITIES)}  # mode 3
    id2action = {i: name for i, name in enumerate(ACTIONS)}

    @classmethod
    def activityToId(cls, index):
        assert index in cls.ACTIVITIES, 'not in activity list'
        return cls.activity2id[index]

    @classmethod
    def actionToId(cls, index):
        assert index in cls.ACTIONS, '%s not in action list' % index
        return cls.action2id[index]

    @classmethod
    def idToActivity(cls, index):
        assert index in list(range(cls.NUM_ACTIVITIES)), 'not in activity range'
        return cls.id2activity[index]

    @classmethod
    def idToAction(cls, index):
        assert index in list(range(cls.NUM_ACTIONS)), 'not in action range'
        return cls.id2action[index]

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
                    self.dataCate[activity] += 1
                    activity = [activity]

                    value = value[2:]
                    # assert len(value)%5 == 0, 'the error occurs in %s at %s' % (fileName, annotationPath)
                    if len(value) % 5 != 0:
                        print('the error occurs in %s at %s' % (fileName, annotationPath))
                    induvidualNum = int(len(value) / 5)  # here is the bbox count in a frame
                    action = []
                    bbox = []
                    for i in range(induvidualNum):
                        x1, y1, w, h = map(float, value[i * 5:i * 5 + 4])
                        x2 = x1 + w
                        y2 = y1 + h
                        x1, x2 = map(lambda x: x / 1280, [x1, x2])
                        y1, y2 = map(lambda x: x / 720, [y1, y2])
                        bbox.append([x1, y1, x2, y2])
                        action.append([VolleyballDataset.actionToId(value[i * 5 + 4])])
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

    def readTrack(self):
        trackPath = self.datasetPath + '/tracks_normalized.pkl'
        return pickle.load(open(trackPath, 'rb'))

    def readAllFrames(self):
        frames = []
        # calculate proportion for each categories
        max = torch.max(torch.tensor(self.dataCate).float())
        pro = max/torch.tensor(self.dataCate).float()

        for sid, anno in self.annotationData.items():
            for fid, subAnno in anno.items():
                frames.extend([(sid, fid)])
        return frames

    def readSpecificFrame(self, frameIndex: tuple):

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

    def readSpecificFrameS(self, frameIndex: tuple, fidIn):

        sid, fid0 = frameIndex
        fid = fid0 - self.preframe + fidIn
        framePath = self.datasetPath + '/%d/%d/%d.jpg' % (sid, fid0, fid)

        # transform all annotation data into numpy
        activity = np.array(self.annotationData[sid][fid0]['group_activity'])
        action = np.array(self.annotationData[sid][fid0]['action'])

        # read bbox coordinate from track file
        bbox = self.trackData[(sid, fid0)][fid]  # np(y1,x1,y2,x2)
        bbox = bbox[:, (1,0,3,2)] # np(x1,y1,x2,y2)


        return framePath, activity, action, bbox, sid, fid0, fid

    def classCount(self):

        return self.NUM_ACTIONS, self.NUM_ACTIVITIES
