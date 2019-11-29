import os
import sys

class Config(object):
    """a class to storage the basic parameter"""

    def __init__(self):
        #work path
        self.workPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN'
        self.dataPath = self.workPath + '/data'

        # data parameter
        self.imageSize = 720, 1280

        # dataset parameter
        self.actions_num = 0
        self.activities_num = 0

        # Backbone
        self.backbone = 'inv3'
        self.crop_size = 5, 5  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056

        self.num_features_boxes = 1024

        # training parameter
        self.train_dropout_prob = 0.3