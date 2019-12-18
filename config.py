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

        # data parameter
        self.imageSize = 720, 1280

        # dataset parameter
        self.actions_num = 0
        self.activities_num = 0
        self.dataset_splitrate = 0.8
        self.random_split = False

        # Backbone
        self.backbone = 'inv3'
        self.crop_size = (5, 5)  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056

        # model parameter
        self.individual_dim = 1024
        self.state_dim = 512

        # training parameter
        self.use_gpu = True
        self.batch_size = 16
        self.train_dropout_prob = 0.5
        #self.actions_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1.]  # weight for each actions categories
        self.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
        self.actions_loss_weight = 1  # weight for actions in loss function
        self.max_epoch = 20
        self.lr_plan = {5: 1e-4, 10: 2e-5, 20: 1e-5}
        self.train_learning_rate = 2e-4
        self.weight_decay = 0.05
        self.break_line = 1e-4

        # testing parameter
        self.test_interval_epoch = 5
        self.test_batch_size = 4



        self.initial()

    def initial(self):

        # check the existence of result dir
        if not os.path.exists(self.resultPath):
            os.mkdir(self.resultPath)

        # build output dir
        date = time.strftime("%Y%m%d", time.localtime())
        date = date[2:]
        for i in range(101):
            outputPath0 = self.resultPath + "/" + date + '-' + "%02d" % i
            if not os.path.exists(outputPath0):
                self.outputPath = outputPath0
                os.mkdir(self.outputPath)
                print("The output path is %s" % self.outputPath)
                break
            assert i != 100, "not enough dir index today, you silly B"
        print("The output result will be saved in %s" % self.outputPath)
