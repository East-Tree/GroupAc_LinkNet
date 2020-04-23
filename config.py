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
        # self.actions_weights = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
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
        self.actions_weights2 = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352,
                                 0.3830]  # weight for each actions categories
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
            1: {
                1: 0, 2: 1.0
            },
            2: {
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
        # self.outputPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200310-00'

        # data parameter
        self.imageSize = 720, 1280

        # dataset parameter
        self.actions_num = 0
        self.activities_num = 0
        self.dataset_splitrate = 0.8
        """
        1: manually split 2: random split with stable seed 3: random split with random seed
        """
        self.split_mode = 2
        """
        0. read the central frame from the sequence 
        1. read all frames from the sequence
        2. randomly read a frame from the sequence
        """
        self.dataset_mode = 0

        #self.train_seqs = [1,2,3]
        #self.test_seqs = [4,5]
        self.train_seqs = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]# video id list of train set
        self.test_seqs = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]  # video id list of test set

        # model parameter
        # Backbone
        self.backbone = 'inv3'
        self.crop_size = (5, 5)  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056

        self.model_para = {
            'person_fea_dim': 1024,
            'relation_fea_dim': 256,
            'general_fea_dim': 32,
            'dropout_prob': 0.5,
            'feature1_renew_rate': 0.2,
            'biasNet_channel_pos': 8,
            'biasNet_channel_dis': [0, 0.15,0.3,0.5],
            'iterative_times': 1,
            'routing_times': 3,
            'pooling_method': 'ave',
            'readout_max_num': 1
        }
        # training parameter
        """
        0: train the network with base backbone
        1: fully linknet model train
        2: train linknet with pre-train backbone part
        3: train the linknet action version with pre-train backbone part
        4: train the GCN with a pretrained action linknet 
        5: silly version of linknet, generate intermediate variable -- state
        6: pretrain of mode3
        """
        self.train_mode = 6
        self.pre_train = 1
        self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200417-01/model/stage2_epoch38_67.36%.pth'
        # self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200327-00/model/stage1_epoch25_75.13%new.pth'
        # self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200412-00/model/stage1_epoch24_71.11%.pth'
        self.goon = False
        self.goon_path1 = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200325-01/model/stage1_epoch160_68.34%.pth'
        self.goon_path2 = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200310-00/model/stage1_optimizer_epoch160.pth'
        """
        the relative parameter for stage1
        stage1 is a pre-train for self action feature output and readout 
        """
        # training
        self.use_gpu = True
        self.renew_weight = False
        self.batch_size = 8
        self.train_learning_rate = 5e-5
        self.weight_decay = 0.005
        self.break_line = 1e-5
        self.start_epoch = 1
        self.max_epoch = 200
        # testing parameter
        self.test_batch_size = 4
        self.test_interval_epoch = 2

        # loss function parameter
        # self.actions_weights = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
        self.actions_weights = [1., 1., 2., 3., 1., 1., 2., 0.1, 1.]
        self.actions_loss_weight = 1.  # weight for actions in loss function
        self.activities_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.activities_loss_weight = 1.
        self.focal_loss_factor = 0.3
        self.kl_loss_weight = 0
        self.other_actions_loss_weight = 0.2

        self.lr_plan = None
        self.loss_plan = None

        if self.outputPath is None:
            self.initial()

        self.loss_apply()
        self.lr_apply()

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

    def loss_apply(self):
        self.loss_plan = {
            1: {
                1: 2.0, 2: 2.0
            },
            2: {
                1: 1.0, 2: 1.0
            }
        }
        if self.train_mode in self.loss_plan:
            mode = self.train_mode
            self.actions_loss_weight = self.loss_plan[mode][1]
            self.activities_loss_weight = self.loss_plan[mode][2]

    def lr_apply(self):
        lr_plan1 = {
            1: {
                1: 2e-5, 2: 2e-5, 3: 1e-5, 4: 2e-5
            },
            40:{
                1: 1e-5, 2: 1e-5, 3: 1e-5, 4: 1e-5
            }
        }
        lr_plan2 = {
            1: {
                1: 0, 2: 0, 3: 2e-4, 4: 2e-4, 5: 2e-4
            },
            40: {
                3: 5e-5, 4: 5e-5, 5: 5e-5
            },
            80: {
                3: 2e-5, 4: 2e-5, 5: 2e-5
            },
            120: {
                3: 1e-5, 4: 1e-5, 5: 1e-5
            }
        }
        lr_plan3 = {
            1: {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 2e-5
            },
            6: {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 2e-4
            },
            80: {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 1e-4
            },
            200: {
                1: 0, 2: 0, 3: 2e-6, 4: 5e-6, 5: 5e-5
            },
            300: {
                1: 0, 2: 0, 3: 1e-5, 4: 2e-5, 5: 2e-5
            }
        }
        lr_plan4 = {
            1: {
                0: 2e-5
            }
        }
        lr_plan5 = {
            1: {
                1: 0, 2: 2e-5, 3: 2e-5, 4: 2e-5
            },
            6: {
                1: 0, 2: 2e-4, 3: 2e-4, 4: 2e-4
            },
            80: {
                1: 0, 2: 1e-4, 3: 1e-4, 4: 1e-4
            },
            200: {
                1: 0, 2: 5e-5, 3: 5e-5, 4: 5e-5
            },
            300: {
                1: 0, 2: 2e-5, 3: 2e-5, 4: 2e-5
            }
        }
        lr_plan6 = {
            1: {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 2e-5, 6: 2e-5
            },
            6: {
                1: 0, 2: 2e-6, 3: 2e-6, 4: 2e-6, 5: 1e-4, 6: 1e-4
            },
            40: {
                1: 0, 2: 2e-6, 3: 2e-6, 4: 2e-6, 5: 5e-5, 6: 5e-5
            },
            80: {
                1: 0, 2: 2e-6, 3: 2e-6, 4: 2e-6, 5: 2e-5, 6: 2e-5
            },
            120: {
                1: 0, 2: 2e-6, 3: 2e-6, 4: 2e-6, 5: 1e-5, 6: 1e-5
            }
        }
        lr_plan7 = {
            1: {
                1: 1e-6, 2: 2e-5, 3: 2e-5, 4: 1e-5, 5: 2e-5, 6: 2e-5
            },
            6: {
                1: 1e-6, 2: 1e-4, 3: 1e-4, 4: 5e-5, 5: 1e-4, 6: 1e-4
            },
            40: {
                1: 1e-6, 2: 5e-5, 3: 5e-5, 4: 2-5, 5: 5e-5, 6: 5e-5
            },
            80: {
                1: 1e-7, 2: 2e-5, 3: 2e-5, 4: 1e-5, 5: 2e-5, 6: 2e-5
            },
            120: {
                1: 1e-7, 2: 1e-5, 3: 1e-5, 4: 5e-6, 5: 1e-5, 6: 1e-5
            }
        }
        if self.train_mode == 0:
            self.lr_plan = lr_plan4
        elif self.train_mode == 1:
            self.lr_plan = lr_plan4
        elif self.train_mode == 2:
            self.lr_plan = lr_plan3
        elif self.train_mode == 3:
            self.lr_plan = lr_plan3
        elif self.train_mode == 4:
            self.lr_plan = lr_plan6
        elif self.train_mode == 5:
            self.lr_plan = lr_plan7
        elif self.train_mode == 6:
            self.lr_plan = lr_plan1
