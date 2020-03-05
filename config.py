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

        # model parameter

        # Backbone
        self.backbone = 'inv3'
        self.crop_size = (5, 5)  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056

        self.model_para = {
            'person_fea_dim': 1024,
            'state_fea_dim': 256,
            'dropout_prob': 0.3
        }
        # linkModel
        self.model_para2 = {
            'person_fea_dim': 1024,
            'state_fea_dim': 256,
            'dropout_prob': 0.3,
            'feature1_renew_rate': 0.2,
            'biasNet_channel': 8,
            'iterative_times': 1,
            'routing_times': 3,
            'pooling_method': 'ave'
        }
        # training parameter
        self.train_mode = 1
        """
        the relative parameter for stage1
        stage1 is a pre-train for self action feature output and readout 
        """
        # training
        self.use_gpu = True
        self.renew_weight = False
        self.batch_size1 = 8
        #self.actions_weights = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
        self.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
        self.actions_loss_weight = 1  # weight for actions in loss function
        self.max_epoch = 120
        self.lr_plan = {
            1: {
                1: 2e-4, 2: 2e-4, 3: 2e-4
            },
            25: {
                1: 1e-4, 2: 1e-4, 3: 1e-4
            },
            60: {
                1: 2e-5, 2: 5e-5, 3: 5e-5
            },
            80: {
                1: 1e-5, 2: 2e-5, 3: 2e-5
            }
        }
        self.train_learning_rate = 2e-4
        self.weight_decay = 0.05
        self.break_line = 1e-5

        # testing parameter
        self.test_interval_epoch = 5
        self.test_batch_size1 = 4

        """
        the relative parameter for stage2
        in stage2, the network parameter in stage1 will be fixed, then train link layer and activities readout 
        """
        self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200221-00/model/stage1_epoch55_29.64%.pth'
        self.batch_size2 = 4
        self.actions_weights2 = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
        # self.actions_weights = [1., 3., 2., 5., 1., 2., 2., 0.2, 1.]
        self.actions_loss_weight2 = 0.5  # weight for actions in loss function
        self.activities_weights2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.activities_loss_weight2 = 0.5
        self.max_epoch2 = 30
        self.lr_plan2 = {
            1: {
                1: 2e-4, 2: 2e-4, 3: 2e-4
            },
            15: {
                1: 2e-5, 2: 1e-4, 3: 1e-4
            },
            30: {
                1: 1e-5, 2: 2e-5, 3: 2e-5
            },
            50: {
                1: 0, 2: 1e-5, 3: 1e-5
            }
        }
        self.loss_plan2 = {
            1:{
                1: 0, 2: 1.0
            },
            2:{
                1: 1.0, 2: 1.0
            }
        }
        """
        the relative parameter for stage3
        in stage3, the all the parameter will be adjusted in low learning rate
        """


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
                outputPathM = outputPath0 + "/model"
                os.mkdir(outputPathM)
                break
            assert i != 100, "not enough dir index today, you silly B"
        print("The output result will be saved in %s" % self.outputPath)


class Config2(object):
    """
    a class to storage the basic parameter
    this configure is for linknet1 model trainning
    """

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

        # model parameter

        # Backbone
        self.backbone = 'inv3'
        self.crop_size = (5, 5)  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056
        self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200221-00/model/stage1_epoch55_29.64%.pth'
        self.model_para = {
            'person_fea_dim': 1024,
            'relation_fea_dim': 512,
            'dropout_prob': 0.3,
            'feature1_renew_rate': 0.2,
            'biasNet_channel': 8,
            'iterative_times': 1,
            'routing_times': 3,
            'pooling_method': 'ave'
        }
        # training parameter
        self.train_mode = 1
        """
        the relative parameter for stage1
        stage1 is a pre-train for self action feature output and readout 
        """
        # training
        self.use_gpu = True
        self.renew_weight = False
        self.batch_size = 8
        self.train_learning_rate = 2e-4
        self.weight_decay = 0.05
        self.break_line = 1e-5
        self.max_epoch = 30
        # testing parameter
        self.test_batch_size = 4
        self.test_interval_epoch = 5

        # loss function parameter
        self.actions_weights = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
        # self.actions_weights = [1., 3., 2., 5., 1., 2., 2., 0.2, 1.]
        self.actions_loss_weight = 0.5  # weight for actions in loss function
        self.activities_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.activities_loss_weight = 0.5

        self.lr_plan = {
            1: {
                1: 2e-4, 2: 2e-4, 3: 2e-4
            },
            25: {
                1: 1e-4, 2: 1e-4, 3: 1e-4
            },
            60: {
                1: 2e-5, 2: 5e-5, 3: 5e-5
            },
            80: {
                1: 1e-5, 2: 2e-5, 3: 2e-5
            }
        }
        self.loss_plan = {
            1:{
                1: 0, 2: 1.0
            },
            2:{
                1: 1.0, 2: 1.0
            }
        }

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
                outputPathM = outputPath0 + "/model"
                os.mkdir(outputPathM)
                break
            assert i != 100, "not enough dir index today, you silly B"
        print("The output result will be saved in %s" % self.outputPath)