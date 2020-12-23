from typing import Dict, Any, Callable, Tuple

import volleyballDataset
from basemodel import *
import improvemodel
import BBonemodel
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
            'person_fea_dim': 1024,
            'relation_fea_dim': 256,
            'general_fea_dim': 32,
            'dropout_prob': 0.5,
            'feature1_renew_rate': 0.2,
            'biasNet_channel_pos': 8,
            'biasNet_channel_dis': [0, 0.15, 0.3, 0.5],
            'iterative_times': 1,
            'routing_times': 3,
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
        self.pre_train = 0
        #self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200417-01/model/stage2_epoch38_67.36%.pth'
        self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200327-00/model/stage1_epoch25_75.13%new.pth'
        #self.para_load_path = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN/result/200412-00/model/stage1_epoch24_71.11%.pth'
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
        self.actions_weights = [1., 1., 2., 3., 1., 1., 2., 0.1, 1.]
        #self.actions_weights = [1., 1., 1., 3., 1., 1., 1., 1., 1.]
        self.actions_loss_weight = 1.  # weight for actions in loss function
        self.activities_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.activities_loss_weight = 1.
        self.oriens_weights = [2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
        self.oriens_loss_weight = 1.
        self.center_loss_weight = 1. 
        self.focal_loss_use = True
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
                1: 1, 2: 1.0, 3:1.,4:0.01
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
                1: 0, 2: 1e-4, 3: 0
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
        lr_plan3 = {
            1: {
                1: 0, 2: 5e-6, 3: 5e-6, 4: 5e-6, 5: 5e-6
            }
        }
        self.lr_plan = lr_plan3
        

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
        self.actions_meter = AverageMeterTensor(cfg.actions_num)
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)
        self.confuMatrix = CorelationMeter(cfg.actions_num)
        # orientation meter
        self.oriens_meter = AverageMeterTensor(cfg.orientations_num)
        self.confuMatrix2 = CorelationMeter(cfg.orientations_num)

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
            if self.cfg.renew_weight:
                new_weight = torch.nn.functional.softmin(self.actions_meter.correct_rate_each, dim=0)
                new_weight = new_weight * 9.
                old_weight = torch.tensor(self.cfg.actions_weights)
                new_weight = old_weight * (1 - self.cfg.weight_renew_rate) + self.cfg.weight_renew_rate * new_weight
                self.cfg.actions_weights = new_weight.tolist()
            info = {
                'mode': self.mode,
                'time': self.epoch_timer.timeit(),
                'epoch': self.epoch,
                'loss': self.loss_meter.avg,
                'actions_acc': self.actions_meter.correct_rate,
                'actions_ave_acc': self.actions_meter.ave_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().round(3),
                'actions_each_num': self.actions_meter.all_num_each,
                'actions_confusion': self.confuMatrix.class_acc.numpy().round(3),
                'oriens_acc': self.oriens_meter.correct_rate,
                'oriens_ave_acc': self.oriens_meter.ave_rate,
                'oriens_each_acc': self.oriens_meter.correct_rate_each.numpy().round(3),
                'oriens_each_num': self.oriens_meter.all_num_each,
                'oriens_confusion': self.confuMatrix2.class_acc.numpy().round(3)
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
                'actions_acc': self.actions_meter.correct_rate,
                'actions_ave_acc': self.actions_meter.ave_rate,
                'actions_each_acc': self.actions_meter.correct_rate_each.numpy().round(3),
                'actions_each_num': self.actions_meter.all_num_each,
                'actions_confusion': self.confuMatrix.class_acc.numpy().round(3),
                'oriens_acc': self.oriens_meter.correct_rate,
                'oriens_ave_acc': self.oriens_meter.ave_rate,
                'oriens_each_acc': self.oriens_meter.correct_rate_each.numpy().round(3),
                'oriens_each_num': self.oriens_meter.all_num_each,
                'oriens_confusion': self.confuMatrix2.class_acc.numpy().round(3)
            }
        else:
            assert False, "mode name incorrect"

        return info

    def baseprocess(self, batch_data):

        # model.apply(set_bn_eval)

        batch_size = len(batch_data[0])

        # reshape the action label into tensor(B*N)
        actions_in = torch.cat(batch_data[2], dim=0)
        actions_in = actions_in.reshape(-1).to(device=self.device)

        # reshape the orientation label into tensor(B*N)
        oriens_in = torch.cat(batch_data[4], dim=0)
        oriens_in = oriens_in.reshape(-1).to(device=self.device)

        # forward
        if self.mode == 'train':
            if self.cfg.center_loss_weight > 0:
                actions_scores,oriens_scores,actions_fea,actions_in = self.model((batch_data[0], batch_data[3]),mode='train',return_fea=True,cata_balance=self.cfg.cata_balance,label=actions_in,seq_len=cfg.seq_len)
            else:
                actions_scores,oriens_scores,actions_in = self.model((batch_data[0], batch_data[3]),mode='train',cata_balance=self.cfg.cata_balance,label=actions_in,seq_len=cfg.seq_len)  # tensor(B#N, actions_num)
        else:
            actions_scores,oriens_scores = self.model((batch_data[0], batch_data[3]),seq_len=cfg.seq_len)

        # Predict actions
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        oriens_weights = torch.tensor(self.cfg.oriens_weights).to(device=self.device)

        # loss
        if self.cfg.focal_loss_use:
            #   focal loss
            focal_loss = loss_lab.Focalloss()
            actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, self.device, weight=actions_weights)
            oriens_loss, oriens_loss_w = focal_loss(oriens_scores, oriens_in, self.device, weight=oriens_weights)
        else:
            #   cross entropy
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
            oriens_loss = F.cross_entropy(oriens_scores, oriens_in, weight=oriens_weights)
        
        
        actions_result = torch.argmax(actions_scores, dim=1).int()
        oriens_result = torch.argmax(oriens_scores, dim=1).int()

        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss + self.cfg.oriens_loss_weight * oriens_loss

        # add center loss
        if self.cfg.center_loss_weight>0 and self.mode == 'train':
            self.total_loss += self.cfg.center_loss_weight * self.centerlossModel(actions_fea, actions_in)

        # Get accuracy
        self.loss_meter.update(self.total_loss.item(), batch_size)

        self.actions_meter.update(actions_result, actions_in)
        self.confuMatrix.update(actions_in,actions_result)

        self.oriens_meter.update(oriens_result, oriens_in)
        self.confuMatrix2.update(oriens_in, oriens_result)
        # self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)

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
    full_dataset = volleyballDataset.VolleyballDatasetNew(cfg.dataPath, cfg.imageSize, frameList=list(range(17)) ,mode=cfg.dataset_mode, seq_num=cfg.seq_len)
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
    model = BBonemodel.SelfNetSN(cfg.imageSize, cfg.crop_size, cfg.actions_num, cfg.orientations_num, device, **cfg.model_para)  # type: SelfNet2
    model.to(device=device)
    model.train()
    #    optimizer implement
    optimizer = optim.Adam(
        [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.fea_lstm.parameters()},
            {"params": model.read_actions.parameters()},
            {"params": model.read_orientations.parameters()}
        ],
        lr=cfg.train_learning_rate,
        weight_decay=cfg.weight_decay)
    # continue work
    if cfg.goon is True:
        model.loadmodel(cfg.goon_path1)
        optimizer.load_state_dict(torch.load(cfg.goon_path2))
    # initial the center loss
    if cfg.center_loss_weight > 0:
        center_loss = loss_lab.CenterLoss(num_classes=cfg.actions_num, feat_dim=cfg.model_para['person_fea_dim'], use_gpu=cfg.use_gpu)
        lossOpti = optim.SGD(center_loss.parameters(), lr=0.5)
    #    begin training
    start_epoch = cfg.start_epoch
    all_info = []
    best_result_ac = MaxItem()
    best_ave_ac = MaxItem()
    best_result_or = MaxItem()
    best_ave_or = MaxItem()
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch], log)
        if epoch in cfg.loss_plan:
            adjust_loss(cfg, cfg.loss_plan[epoch], log)

        #  each epoch in the iteration
        model.train()
        # check use center loss or not
        if cfg.center_loss_weight > 0:
            train_result_info = VolleyballEpoch('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,lossmodel=center_loss,optimizer2=lossOpti, epoch=epoch).main()
        else:
            train_result_info = VolleyballEpoch('train', train_loader, model, device, cfg=cfg, optimizer=optimizer, epoch=epoch).main()
        for each_info in train_result_info:
            log.fPrint('%s:\n%s\n' % (str(each_info), str(train_result_info[each_info])))
        all_info.append(train_result_info)
        TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
        TBWriter.add_scalar('train1_acc', train_result_info['actions_acc'], epoch)
        TBWriter.add_scalars('train1_acc_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
        TBWriter.add_scalar('train1_acc', train_result_info['oriens_acc'], epoch)
        TBWriter.add_scalars('train1_acc_each', dict(zip(ORIENTATION, train_result_info['oriens_each_acc'])), epoch)
        #  test in each interval times
        if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
            model.train(False)
            test_result_info = VolleyballEpoch('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
            for each_info in test_result_info:
                log.fPrint('%s:\n%s\n' % (str(each_info), str(test_result_info[each_info])))
            TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
            TBWriter.add_scalar('test1_acc', test_result_info['actions_acc'], epoch)
            TBWriter.add_scalars('test1_acc_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])), epoch)
            TBWriter.add_scalar('test1_acc', test_result_info['oriens_acc'], epoch)
            TBWriter.add_scalars('test1_acc_each', dict(zip(ORIENTATION, test_result_info['oriens_each_acc'])), epoch)
            filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                1, epoch, test_result_info['actions_acc'])
            model.savemodel(filepath)
            para_path = filepath
            # log the best result
            best_result_ac.update(test_result_info['actions_acc'], test_result_info['epoch'])
            best_ave_ac.update(test_result_info['actions_ave_acc'], test_result_info['epoch'])
            log.fPrint('best result action: %.4f in epoch %d' % (best_result_ac.maxitem, best_result_ac.maxnum))
            log.fPrint('best ave action: %.4f in epoch %d' % (best_ave_ac.maxitem, best_ave_ac.maxnum))

            best_result_or.update(test_result_info['oriens_acc'], test_result_info['epoch'])
            best_ave_or.update(test_result_info['oriens_ave_acc'], test_result_info['epoch'])
            log.fPrint('best result orientation: %.4f in epoch %d' % (best_result_or.maxitem, best_result_or.maxnum))
            log.fPrint('best ave orientation: %.4f in epoch %d' % (best_ave_or.maxitem, best_ave_or.maxnum))

        if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
        log.fPrint('*'*100+'\n')

    TBWriter.close()
