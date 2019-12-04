import os
import time


class Config1(object):
    """a class to storage the basic parameter"""

    def __init__(self):
        # work path
        self.workPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN'
        self.dataPath = self.workPath + '/data'
        self.resultPath = self.workPath + '/result'
        self.outputPath = None
        self.logPath = None

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

        self.individual_dim = 1024

        # training parameter
        self.train_dropout_prob = 0.3
        self.actions_weights = None  # weight for each actions categories

        self.action_loss_weight = 1  # weight for actions in loss fuction

        self.initial()

    def initial(self):

        # check the existence of result dir
        if not os.path.exists(self.resultPath):
            os.mkdir(self.resultPath)

        # build output dir
        date = time.strftime("%Y%m%d", time.localtime())
        date = date[2:]
        for i in range(100):
            outputPath0 = self.resultPath + "/" + date + str(i)
            if os.path.exists(outputPath0):
                self.outputPath = outputPath0
                print("The output path is %s" % self.outputPath)
                break

            assert False, "not enough dir index today, you silly B"
        print("The output result will be saved in %s" % self.resultPath)
        self.logPath = self.resultPath + "/logger.txt"
