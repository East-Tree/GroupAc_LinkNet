from typing import Dict, Any, Callable, Tuple

import volleyballDataset
from utils import *
import cv2 as cv2
from tqdm import *

if __name__ == '__main__':

    workPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN'
    dataPath = workPath + '/data'
    resultPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/data/volleyballDraw'

    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']

    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    # build folder for each activity
    for acti in ACTIVITIES:
        folderPath  = resultPath + '/%s' %acti
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
    # generate the volleyball dataset object
    full_dataset = volleyballDataset.VolleyballDatasetDraw(dataPath, (720, 1280), mode=1)
    # get the object information(object categories count)

    imgNum = full_dataset.__len__()

    for i in tqdm(range(imgNum)):
        framePath, activityid, actionid, bbox, sid, fid0, fid = full_dataset.__getitem__(i)
        sidPath = resultPath + '/%d' % sid
        fid0Path = sidPath + '/%d' % fid0
        activityPath = resultPath + '/%s/%d-%d' %(ACTIVITIES[int(activityid)], sid, fid0)
        if not os.path.exists(sidPath):
            os.mkdir(sidPath)
        if not os.path.exists(fid0Path):
            os.mkdir(fid0Path)
        if not os.path.exists(activityPath):
            os.mkdir(activityPath)

        # read img and draw the bbox
        img = cv2.imread(framePath)
        img = cv2.resize(img, (1280, 720))
        # draw the activity label
        cv2.putText(img, ACTIVITIES[int(activityid)], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # draw each action bbox
        for act0 in range(actionid.size):
            act = int(actionid[act0,:])
            x1, x2 = (bbox[act0,(0,2)]*1280).astype(np.int)
            y1, y2 = (bbox[act0,(1,3)]*720).astype(np.int)
            cv2.putText(img, ACTIONS[act], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        imgSavePath_seq = fid0Path + '/%d.jpg' %fid
        cv2.imwrite(imgSavePath_seq, img)
        imgSavePath_act = activityPath + '/%d.jpg' %fid
        cv2.imwrite(imgSavePath_act, img)

