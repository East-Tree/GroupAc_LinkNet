from typing import Dict, Any, Callable, Tuple

import volleyballDataset
from utils import *
import cv2 as cv2
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    workPath = '/media/hpc/ssd960/chenduyu'
    dataPath = workPath + '/data'
    resultPath = '/media/hpc/ssd960/chenduyu/data/volleyballLabel'

    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']

    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    
    # generate the volleyball dataset object
    full_dataset = volleyballDataset.VolleyballDatasetDraw(dataPath, (720, 1280), mode=0)
    # get the object information(object categories count)

    #frameList = full_dataset.output_allFrame()
    #np.save(resultPath+'/frameList.npy',frameList)
    frameList = np.load(resultPath+'/frameList.npy')
    frameList = frameList.tolist()
    full_dataset.set_allFrame(frameList)
    imgNum = full_dataset.__len__()

    orien_map = {
        '8':'up',
        '5':'down',
        '2':'down',
        '4':'left',
        '6':'right',
        '7':'up-left',
        '9':'up-right',
        '1':'down-left',
        '3':'down-right',
        'w':'up',
        's':'down',
        'x':'down-right',
        'a':'left',
        'd':'right',
        'q':'up-left',
        'e':'up-right',
        'z':'down-left',
        'c':'down-right'
    }

    with open(resultPath+'/index_log.txt','r') as f:
        begini = int(f.read())
    
    for i in range(begini,imgNum):
        print('continue the label in frame %d, tatal %d' % (i, imgNum))
        framePath, activityid, actionid, bbox, sid, fid = full_dataset.__getitem__(i)

        # orgenize the list format
        info = []
        for act0 in range(actionid.size):
            action = ACTIONS[int(actionid[act0,:])]
            x1, x2 = (bbox[act0,(0,2)]*1280).astype(np.int)
            y1, y2 = (bbox[act0,(1,3)]*720).astype(np.int)
            cx = (x1 + x2)/2
            di = {
                'index': act0,
                'x1':x1,
                'x2':x2,
                'y1':y1,
                'y2':y2,
                'cx':cx,
                'w':x2-x1,
                'h':y2-y1,
                'action':action
            }
            info.append(di)
        def take(elem):
            return elem['cx']
        info.sort(key=take)

        all_bia = 0

        while True:
            # read img and draw the bbox
            img = cv2.imread(framePath)
            img = cv2.resize(img, (1280, 720))
            # draw the activity label
            cv2.putText(img, ACTIVITIES[int(activityid)], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            for id in range(len(info)):
                item = info[id]
                x1 = item['x1']
                x2 = item['x2']
                y1 = item['y1']
                y2 = item['y2']
                cv2.putText(img, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            plt.figure(dpi=250)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.draw()
            plt.pause(0.1)

            # receive orientation message
            while True:
                ori_mes = input('orientation:')
                if ori_mes == 'end':
                    break
                if ori_mes == 'adjust':
                    break
                if len(ori_mes) != len(info):
                    print('unmatch number')
                else:
                    break
                    
            if ori_mes == 'end':
                print('program shut down in frame %d' %i)
                exit()
            
            
            # receive area message
            while True:
                area_mes = input('area:')
                if area_mes == 'end':
                    break
                if len(area_mes) != 3:
                    print('unmatch number')
                else:
                    break
            
            if area_mes == 'end':
                print('program shut down in frame %d' %i)
                exit()
            
            plt.close('all')

            # adjust the bbox coordinate
            if ori_mes == 'adjust':
                bia = int(input('please input shift x coordinate: '))
                for item in info:
                    item['x1'] += bia
                    item['x2'] += bia
                all_bia += bia
                print('x coordinate shift: %d, in %d-%d' % (bia, sid, fid))
                continue

            # process the order
            for id in range(len(info)):
                info[id]['orientation'] = orien_map[ori_mes[id]]
                index = id
                area0 = int(area_mes[0])
                area1 = int(area_mes[1])
                if int(area_mes[2])<5:
                    area2 = int(area_mes[2])+10
                else:
                    area2 = int(area_mes[2])

                if index < area0:
                    info[id]['area'] = 0
                elif index < area1:
                    info[id]['area'] = 1
                elif index < area2:
                    info[id]['area'] = 2
                else:
                    info[id]['area'] = 3


            # save annotation and log

            # writing format: fid.jpg activity x1 y1 w h action orientation area
            line = '%d.jpg %s' % (fid, ACTIVITIES[int(activityid)])
            for item in info:
                line = line + ' %d %d %d %d %s %s %d' % (item['x1'],item['y1'],item['w'],item['h'],item['action'],item['orientation'],item['area'])
            line = line + '\n'
            with open(resultPath+'/annotations%d.txt'%sid,'a') as f:
                f.write(line)

            with open(resultPath+'/index_log.txt','w') as f:
                f.write(str(i+1))

            if all_bia != 0:
                with open(resultPath+'/coordinate_modify.txt','a') as f:
                    f.write('%d %d x %d\n' % (sid, fid, all_bia))

            break

        


