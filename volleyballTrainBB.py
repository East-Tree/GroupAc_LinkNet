from typing import Dict, Any, Callable, Tuple

import volleyballDataset
from basemodel import *
import improvemodel
import BBonemodel
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
        self.workPath = '/kmj-labmen-007/Data1/Project/Code/HyperReco/groupActivity_GCN'
        self.dataPath = self.workPath + '/data'
        self.resultPath = '/home/kmj-labmen-007/Data1/Project/Code/HyperReco/result'
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
        self.split_mode = 1
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
        self.use_gpu = True
        self.renew_weight = False
        self.batch_size = 8
        self.train_learning_rate = 5e-5
        self.weight_decay = 0.005
        self.break_line = 1e-5
        self.start_epoch = 1
        self.max_epoch = 120
        # testing parameter
        self.test_batch_size = 4
        self.test_interval_epoch = 2

        # loss function parameter
        #self.actions_weights = [0.5453, 0.5881, 1.1592, 3.9106, 0.2717, 1.0050, 1.1020, 0.0352, 0.3830]  # weight for each actions categories
        #self.actions_weights = [1., 1., 2., 3., 1., 1., 2., 0.1, 1.]
        self.actions_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
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
                1: 5e-5, 2: 2e-4, 3: 2e-4, 4: 2e-4, 5: 2e-4
            },
            40: {
                1: 2e-5, 2: 1e-4, 3: 1e-4, 4: 1e-4, 5: 1e-4
            },
            80: {
                1: 1e-5, 2: 5e-5, 3: 5e-5, 4: 5e-5, 5: 5e-5
            },
            120: {
                1: 1e-5, 2: 2e-5, 3: 2e-5, 4: 2e-5, 5: 2e-5
            },
            300: {
                1: 0, 2: 0, 3: 1e-5, 4: 2e-5, 5: 2e-5
            }
        }
        lr_plan4 = {
            1: {
                0: 2e-6
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


class Focalloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, device0=None, weight=None, attenuate=2.0):
        """
            this is a multi focal loss function base on F.nll_loss
            :param input: [N,p]
            :param target: [N]  ground truth
            :param weight:[p]
            :return:
            """
        if device0 is None:
            device1 = torch.device('cpu')
        else:
            device1 = device0
        input_soft = F.softmax(input, dim=1)
        input_logsoft = F.log_softmax(input, dim=1)
        batch = target.size()[0]
        target_mask = target.reshape(-1, 1)
        input_soft = torch.gather(input_soft, 1, target_mask)
        input_logsoft = torch.gather(input_logsoft, 1, target_mask)
        if weight is None:
            weight_tensor = torch.tensor([1] * batch, device=device1)
        else:
            weight_tensor = weight.repeat(batch, 1).to(device=device1)
            weight_tensor = torch.gather(weight_tensor, 1, target_mask)
        weight_tensor = weight_tensor.reshape(-1, 1)
        focal_weight = weight_tensor * torch.pow(1.0 - input_soft, attenuate)
        # print('focal loss coeff:' + str(focal_weight))
        loss = (-1) * focal_weight * input_logsoft
        loss = torch.mean(loss, dim=0)


        return loss, focal_weight

'''
new backbone training
1. use pre-train inception v3
2. add confusion matrix
3. add average acc of each class 
'''

class VolleyballEpoch():

    def __init__(self, mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

        self.mode = mode
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch

        self.actions_meter = AverageMeterTensor(cfg.actions_num)
        self.loss_meter = AverageMeter()
        self.actions_loss_weight = GeneralAverageMeterTensor(cfg.actions_num)
        self.confuMatrix = CorelationMeter(cfg.actions_num)
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
                self.total_loss.backward()
                self.optimizer.step()
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
                'actions_confusion': self.confuMatrix.class_acc.numpy().round(3)
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
                'actions_confusion': self.confuMatrix.class_acc.numpy().round(3)
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

        # forward
        if self.mode == 'train':
            actions_scores, actions_in = self.model((batch_data[0], batch_data[3]),mode='train',label=actions_in)  # tensor(B#N, actions_num)
        else:
            actions_scores = self.model((batch_data[0], batch_data[3]))

        # Predict actions
        actions_weights = torch.tensor(self.cfg.actions_weights).to(device=self.device)
        # loss
        #   cross entropy
        actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        #   focal loss
        #focal_loss = Focalloss()
        #actions_loss, action_loss_w = focal_loss(actions_scores, actions_in, self.device, weight=actions_weights)
        actions_result = torch.argmax(actions_scores, dim=1).int()

        # Total loss
        self.total_loss = self.cfg.actions_loss_weight * actions_loss

        # Get accuracy
        self.actions_meter.update(actions_result, actions_in)
        self.loss_meter.update(self.total_loss.item(), batch_size)
        self.confuMatrix.update(actions_in,actions_result)
        # self.actions_loss_weight.update(action_loss_w.squeeze(1), actions_in)

if __name__ == '__main__':
    introduce = "base self model renew weight 1e-4"
    print(introduce)
    cfg = Config()
    np.set_printoptions(precision=3)
    para_path = None
    ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
               'moving', 'setting', 'spiking', 'standing',
               'waiting']
    ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                  'l_set', 'l-spike', 'l-pass', 'l_winpoint']
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
    full_dataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, mode=cfg.dataset_mode)
    # get the object information(object categories count)
    cfg.actions_num, cfg.activities_num = full_dataset.classCount()

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
                                                            mode=cfg.dataset_mode)
        testDataset = volleyballDataset.VolleyballDatasetS(cfg.dataPath, cfg.imageSize, cfg.test_seqs,
                                                           mode=cfg.dataset_mode)
    # begin model train in
    #   dataloader implement
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True
    }
    train_loader = data.DataLoader(trainDataset, collate_fn=volleyballDataset.new_collate, **params)
    test_loader = data.DataLoader(testDataset, collate_fn=volleyballDataset.new_collate, **params)
    #    build model
    model = BBonemodel.SelfNet2(cfg.imageSize, cfg.crop_size, cfg.actions_num, device, **cfg.model_para)  # type: SelfNet2
    model.to(device=device)
    model.train()
    #    optimizer implement
    optimizer = optim.Adam(
        [
            {"params": model.baselayer.backbone_net.parameters()},
            {"params": model.baselayer.mod_embed.parameters()},
            {"params": model.read_actions.parameters()}
        ],
        lr=cfg.train_learning_rate,
        weight_decay=cfg.weight_decay)
    # continue work
    if cfg.goon is True:
        model.loadmodel(cfg.goon_path1)
        optimizer.load_state_dict(torch.load(cfg.goon_path2))
    #    begin training
    start_epoch = cfg.start_epoch
    all_info = []
    best_result = MaxItem()
    best_ave = MaxItem()
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch], log)

        #  each epoch in the iteration
        model.train()
        train_result_info = VolleyballEpoch('train', train_loader, model, device, cfg=cfg, optimizer=optimizer,
                                            epoch=epoch).main()
        for each_info in train_result_info:
            log.fPrint('%s:\n%s\n' % (str(each_info), str(train_result_info[each_info])))
        all_info.append(train_result_info)
        TBWriter.add_scalar('train1_loss', train_result_info['loss'], epoch)
        TBWriter.add_scalar('train1_acc', train_result_info['actions_acc'], epoch)
        TBWriter.add_scalars('train1_acc_each', dict(zip(ACTIONS, train_result_info['actions_each_acc'])), epoch)
        #  test in each interval times
        if epoch % cfg.test_interval_epoch == 0 or epoch == start_epoch:
            model.train(False)
            test_result_info = VolleyballEpoch('test', test_loader, model, device, cfg=cfg, epoch=epoch).main()
            for each_info in test_result_info:
                log.fPrint('%s:\n%s\n' % (str(each_info), str(test_result_info[each_info])))
            TBWriter.add_scalar('test1_loss', test_result_info['loss'], epoch)
            TBWriter.add_scalar('test1_acc', test_result_info['actions_acc'], epoch)
            TBWriter.add_scalars('test1_acc_each', dict(zip(ACTIONS, test_result_info['actions_each_acc'])),
                                 epoch)
            filepath = cfg.outputPath + '/model/stage%d_epoch%d_%.2f%%.pth' % (
                1, epoch, test_result_info['actions_acc'])
            model.savemodel(filepath)
            para_path = filepath
            # log the best result
            best_result.update(test_result_info['actions_acc'], test_result_info['epoch'])
            best_ave.update(test_result_info['actions_ave_acc'], test_result_info['epoch'])
            log.fPrint('best result: %.4f in epoch %d' % (best_result.maxitem, best_result.maxnum))
            log.fPrint('best ave: %.4f in epoch %d' % (best_ave.maxitem, best_ave.maxnum))

        if epoch > 10 + start_epoch:
                if abs(all_info[epoch - start_epoch]['loss'] - all_info[epoch - start_epoch - 1][
                    'loss']) < cfg.break_line:
                    break
        log.fPrint('*'*100+'\n')

    TBWriter.close()
