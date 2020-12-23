import pickle
import numpy as np
trackPath =  '/media/hpc/ssd960/chenduyu/data/' + '/tracks_normalized.pkl'

f = pickle.load(open(trackPath, 'rb'))
newf = {}
for key in f:
    subf = {}
    for id in f[key]:
        bbox = f[key][id]
        new_bbox = []
        for i in range(bbox.shape[0]):
            new_bbox.append(bbox[i,:])
        def take(item):
            # y1, x1, y2, x2
            return (item[1]+item[3])/2
        new_bbox.sort(key=take)
        new_bbox = np.stack(new_bbox)
        subf[id] = new_bbox
    newf[key] = subf

with open('/media/hpc/ssd960/chenduyu/data/tracks_normalized_new.pkl', 'wb') as k:
    pickle.dump(newf,k)

f2 = pickle.load(open('/media/hpc/ssd960/chenduyu/data/tracks_normalized_new.pkl', 'rb'))
pass