from typing import Dict, Any, Callable, Tuple

import volleyballDataset
from basemodel import *
import explicitMPNN
import loss_lab
import utils
import torch.nn.functional as F
from utils import *
from tqdm import *

from torch.utils import data
from torch import optim
import random
from tensorboardX import SummaryWriter

class Config(object):
    """
    a class to storage the basic parameter
    this configure is for linknet1 model trainning
    """

    def __init__(self):
        # work path
        self.workPath = '/media/hpc/ssd960/chenduyu'
        self.dataPath = self.workPath + '/data'
        self.resultPath = '/media/hpc/ssd960/chenduyu/result'
        self.outputPath = None
        # self.outputPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200310-00'

        # data parameter
        self.imageSize = 720, 1280

        # dataset parameter
        self.actions_num = 0
        self.activities_num = 0
        self.dataset_splitrate = 0.8
        """
        1: manually split 2: random split with stable seed 3: random split with random seed 4： for train set read all frames, for test set read central frame
        """
        self.split_mode = 2
        """
        0. read the central frame from the sequence 
        1. read all frames from the sequence
        2. randomly read a frame from the sequence
        3. read the image in sequence format
        """
        self.dataset_mode = 3

        self.seq_len=3

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
            'fea_decoup' : True,
            'person_fea_dim': 1024,
            'GCN_embed_fea': 256,
            'metric_dim': 100,
            'general_fea_dim': 32,
            'dropout_prob': 0.5,
            'pooling_method': 'ave',
            'readout_max_num': 6,
            'readout_mode': 'con'
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
        self.train_mode = 0
        self.load_para = True
        # self.para_load_path = '/media/hpc/ssd960/chenduyu/result/201230-01/model/stage1_epoch44_78.92%.pth'
        self.para_load_path = '/media/hpc/ssd960/chenduyu/result/201230-02/model/stage1_epoch62_79.87%.pth'
        self.goon = False
        self.goon_path1 = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200325-01/model/stage1_epoch160_68.34%.pth'
        self.goon_path2 = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200310-00/model/stage1_optimizer_epoch160.pth'
        """
        the relative parameter for stage1
        stage1 is a pre-train for self action feature output and readout 
        """
        # training
        self.cata_balance = False
        self.use_gpu = True
        self.renew_weight = False
        self.batch_size = 2
        self.train_learning_rate = 5e-5
        self.weight_decay = 1e-4
        self.break_line = 1e-5
        self.start_epoch = 1
        self.max_epoch = 120
        # testing parameter
        self.test_batch_size = 4
        self.test_interval_epoch = 2

        # loss function parameter
        #self.actions_weights = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
        #self.actions_weights = [1., 1., 2., 3., 1., 1., 2., 0.1, 1.]
        self.actions_weights = [1., 1., 1., 3., 1., 1., 1., 1., 1.]
        self.actions_loss_weight = 1.  # weight for actions in loss function
        self.activities_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.activities_loss_weight = 1.
        self.oriens_weights = [2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
        self.oriens_loss_weight = 1.
        self.center_loss_weight = 0
        self.focal_loss_use = False
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
        """
        1:actions_loss_weight
        2:oriens_loss_weight
        3:activities_loss_weight
        4:center_loss_weight
        """
        loss_plan1 = {
            1: {
                1: 0, 2: 2.0, 3: 0.1
            },
            21: {
                1: 1.0, 2: 1.0, 3: 0.1
            }
        }
        loss_plan2 = {
            1: {
                1: 0, 2: 0, 3: 1., 4: 0
            }
        }
        self.loss_plan = loss_plan2
        
    def lr_apply(self):
        lr_plan1 = {
            1: {
                0: 2e-6
            }
        }
        lr_plan2 = {
            1: {
                1: 0, 2: 5e-5, 3: 5e-5
            },
            21: {
                1: 0, 2: 2e-5, 3: 1e-5
            },
            41: {
                1: 0, 2: 2e-6, 3: 2e-5
            },
            250: {
                1: 0, 2: 5e-5, 3: 5e-5
            },
            300: {
                1: 0, 2: 2e-5, 3: 2e-5
            }
        }
        self.lr_plan = lr_plan2
        

'''
new backbone training
1. use pre-train inception v3
2. add confusion matrix
3. add average acc of each class 
4. contains center loss
5. use LSTM model
6. use new label orientation and area
'''

class VolleyballEpoch():

    def __init__(self, mode, data_loader, model, device, cfg=None, optimizer=None,lossmodel=None,optimizer2=None,epoch=0):

        self.mode = mode
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch
        self.centerlossModel = lossmodel
        self.lossOpti = optimizer2

        self.loss_meter = AverageMeter()
        # action meter
        self.activities_meter = AverageMeterTensor(cfg.activities_num)
        self.activities_loss_weight = GeneralAverageMeterTensor(cfg.activities_num)
        self.confuMatrix = CorelationMeter(cfg.activities_num)
        
        self.epoch_timer = Timer()

        self.total_loss = None

    def main(self):

        if self.mode == 'train':
            print("Training in epoch %s" % self.epoch)
            print(self.cfg.actions_weights)
            for batch_data in tqdm(self.data_loader):

                self.baseprocess(batch_data)

                # Optim
                self.optimizer.zero_grad()
                if self.cfg.center_loss_weight > 0:
                    self.lossOpti.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
                # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                if self.cfg.center_loss_weight > 0:
                    for param in self.centerlossModel.parameters():
                        param.grad.data *= (1./self.cfg.center_loss_weight)
                        lossOpti.step()
                
            # renew the action loss weight by accuracy
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_ave_acc': self.activities_meter.ave_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().round(3),
                'activities_each_num': self.activities_meter.all_num_each,
                'activities_loss_weights': self.activities_loss_weight.correct_rate_each.numpy().round(3),
                'activities_confusion': self.confuMatrix.class_acc.numpy().round(3)
            }
        elif self.mode == 'test':
            print("Testing in test dataset")
            with torch.no_grad():
                for batch_data in tqdm(self.data_loader):
                    self.baseprocess(batch_data)

            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'activities_acc': self.activities_meter.correct_rate,
                'activities_ave_acc': self.activities_meter.ave_rate,
                'activities_each_acc': self.activities_meter.correct_rate_each.numpy().round(3),
                'activities_each_num': self.activities_meter.all_num_each,
                'activities_confusion': self.confuMatrix.class_acc.numpy().round(3)
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[1])

        # reshape the action label into tensor(B*N)
        #actions_in = torch.cat(batch_data[2], dim=0)
        #actions_in = actions_in.reshape(-1).to(device=self.device)
        # reshape the orientation label into tensor(B*N)
        #oriens_in = torch.cat(batch_data[4], dim=0)
        #oriens_in = oriens_in.reshape(-1).to(device=self.device)

        activities_in = torch.cat(batch_data[1], dim=0)
        activities_in = activities_in.reshape(-1).to(device=self.device)

        # forward
        if self.mode == 'train':
            activities_scores = self.model(batch_data, mode='train', seq_len = cfg.seq_len)
        else:
            activities_scores = self.model(batch_data, mode='test', seq_len = cfg.seq_len)

        # Predict actions
        #actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        #oriens_weights = torch.tensor(self.cfg.oriens_weights).to(device=self.device)
        activities_weights = torch.tensor(self.cfg.activities_weights).to(device=self.device)

        # loss
        if self.cfg.focal_loss_use:
            #   focal loss
            focal_loss = loss_lab.Focalloss()
            activities_loss, activities_loss_w = focal_loss(activities_scores, activities_in, self.device, weight=activities_weights, attenuate=4.)
            
        else:
            #   cross entropy
            activities_loss = F.cross_entropy(activities_scores, activities_in, weight=activities_weights)
            activities_loss_w = torch.tensor([[1.]*batch_size])
            
        
        
        activities_result = torch.argmax(activities_scores, dim=1).int()

        # Total loss
        self.total_loss = self.cfg.activities_loss_weight * activities_loss

        # add center loss
        #if self.cfg.center_loss_weight>0 and self.mode == 'train':
        #    self.total_loss += self.cfg.center_loss_weight * self.#centerlossModel(activities_fea, activities_in)

        # Get accuracy
        self.loss_meter.update(self.total_loss.item(), batch_size)

        self.activities_meter.update(activities_result, activities_in)
        self.confuMatrix.update(activities_in,activities_result)
        self.activities_loss_weight.update(activities_loss_w.reshape(-1), activities_in)

"""
1.this program use the sequential LSTM to processing image 
2.use center loss
"""

if __name__ == '__main__':
    introduce = "the base model(LSTM include) train with orientation label"
    cfg = Config()
    np.set_printoptions(precision=3)
    para_path = None
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
    ORIENTATION = ['up','down','left','right','up-left',
                   'up-right','down-left','down-right']
    # create logger object
    log = utils.Logger(cfg.outputPath)
    log.fPrint(introduce)
    for item in cfg.__dict__:
        log.fPrint('%s:%s' % (str(item), str(cfg.__dict__[item])))
    # create tensorboard writer
    TBWriter = SummaryWriter(logdir=cfg.outputPath, comment='tensorboard')
    # device state
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # generate the volleyball dataset object
    full_dataset = volleyballDataset.VolleyballDatasetNew(cfg.dataPath, cfg.imageSize, frameList=list(range(17)) ,mode=cfg.dataset_mode, dataagument=True ,seq_num=cfg.seq_len)
    # get the object information(object categories count)
    cfg.actions_num, cfg.activities_num, cfg.orientations_num = full_dataset.classCount()

    # divide the whole dataset into train and test
    full_dataset_len = full_dataset.__len__()
    if cfg.split_mode == 3:
        train_len = int(full_dataset_len * cfg.dataset_splitrate)
        test_len = full_dataset_len - train_len
        trainDataset, testDataset = data.random_split(full_dataset, [train_len, test_len])
    elif cfg.split_mode == 2:
        random_seed = 137  # set the seed
        random.seed(random_seed)
        indices = list(range(full_dataset_len))
        random.shuffle(indices)
        split = int(cfg.dataset_splitrate * full_dataset_len)
        train_indices = indices[:split]
        test_indices = indices[split:]
        trainDataset = data.Subset(full_dataset, train_indices)
        testDataset = data.Subset(full_dataset, test_indices)
    elif cfg.split_mode == 4:
        trainDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.train_seqs,
                                                            mode=1)
        testDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.test_seqs,
                                                           mode=0)
    else:  # split_mode = 1
        trainDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.train_seqs,
                                                            mode=cfg.dataset_mode,seq_num=cfg.seq_len)
        testDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.test_seqs,
                                                           mode=cfg.dataset_mode,seq_num=cfg.seq_len)
    # begin model train in
    #   dataloader implement
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True
    }
    train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.seq_collate_new, **params)
    test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.seq_collate_new, **params)
    #    build model
    model = explicitMPNN.imp_GCN(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.orientations_num,cfg.activities_num,coor_use=True,area_use=True,device=device, **cfg.model_para)  # type: SelfNet2
    model.to(device=device)
    model.train()
    #    optimizer implement
    optimizer = optim.Adam(
        [
            {"params": model.baselayer.parameters()},
            {"params": model.GCN_layer.parameters()},
            {"params": model.read_activity.parameters()}
        ],
        lr=cfg.train_learning_rate,
        weight_decay=cfg.weight_decay)
    # continue work
    if cfg.goon:
        model.loadmodel(cfg.goon_path1)
        optimizer.load_state_dict(torch.load(cfg.goon_path2))
    # load para
    if cfg.load_para:
        model.loadmodel(cfg.para_load_path, mode=1)
    #    begin training
    start_epoch = cfg.start_epoch
    all_info = []
    best_result_acv = MaxItem()
    best_ave_acv = MaxItem()
    
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch], log)
        if epoch in cfg.loss_plan:
            adjust_loss(cfg, cfg.loss_plan[epoch], log)

        #  each epoch in the iteration
        model.train()
        # check use center loss or not
        train_result_info = VolleyballEpoch('train', train_loader, model, device, cfg=cfg, optimizer=optimizer, epoch=epoch).main()
        
        for each_info in train_result_info:
            log.fPrint('%s:\n%s\n' % (str(each_info), str(train_result_info[each_info])))
        all_info.append(train_result_info)
        TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
        TBWriter.add_scalar('train1_acc', train_result_info['activities_acc'], epoch)
        TBWriter.add_scalars('train1_acc_each', dict(zip(ACTIONS, train_result_info['activities_each_acc'])), epoch)
        
        #  test in each interval times
        if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
            model.train(False)
            test_result_info = VolleyballEpoch('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
            for each_info in test_result_info:
                log.fPrint('%s:\n%s\n' % (str(each_info), str(test_result_info[each_info])))
            TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
            TBWriter.add_scalar('test1_acc', test_result_info['activities_acc'], epoch)
            TBWriter.add_scalars('test1_acc_each', dict(zip(ACTIONS, test_result_info['activities_each_acc'])), epoch)
            filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                2, epoch, test_result_info['activities_acc'])
            model.savemodel(filepath)
            para_path = filepath

            # log the best result
            best_result_acv.update(test_result_info['activities_acc'], test_result_info['epoch'])
            best_ave_acv.update(test_result_info['activities_ave_acc'], test_result_info['epoch'])
            log.fPrint('best result activity: %.4f in epoch %d' % (best_result_acv.maxitem, best_result_acv.maxnum))
            log.fPrint('best ave activity: %.4f in epoch %d' % (best_ave_acv.maxitem, best_ave_acv.maxnum))


        if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
        log.fPrint('*'*100+'\n')

    TBWriter.close()
