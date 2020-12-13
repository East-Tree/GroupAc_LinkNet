from backbone import *
from utils import *

from torchvision import ops  # RoIAlign module

import BBonemodel


# link net
class LinkNet(nn.Module):
    """
    the link net using other individual's feature
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, cfg_activities_num, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_align(arch_feature)
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.activities_num = cfg_activities_num
        self.device = device

        # network layers
        #   self awareness
        self.baselayer = BBonemodel.SelfNetS(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)
        #   action sequence


        # initial network parameter
        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'feature1_renew_rate': 0.2,
            'dropout_prob': 0.3,
            'biasNet_channel': 8,
            'iterative_times': 3,
            'pooling_method': 'ave'
        }
        for i in arch_para:
            if i not in para:
                para[i] = arch_para[i]
        return para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.state_dict(),
            'link_layer_dict': self.linklayer.state_dict(),
            'read_actions_dict': self.read_actions.state_dict(),
            'read_activities_dict': self.read_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath, mode=0):
        state = torch.load(filepath)

        if mode == 0:
            self.baselayer.load_state_dict(state['base_state_dict'])
            self.linklayer.load_state_dict(state['link_layer_dict'])
            self.read_actions.load_state_dict(state['read_actions_dict'])
            self.read_activities.load_state_dict(state['read_activities_dict'])
            print('Load model states from: ', filepath)
        elif mode == 1:
            self.baselayer.load_state_dict(state['base_state_dict'])
            self.read_actions.load_state_dict(state['read_actions_dict'])
            print('Load model states from: ', filepath)
        else:
            assert False, "mode pattern error, you silly B"

    def forward(self, batch_data):
        """
        :param batch_cood: tensor [person num, 4]
        :param batch_state: tensor [person num, feature_dim]
        :return: [new]
        """
        # image_in is a list containing image batch data(tensor(c,h,w))
        # boxes_in is a list containing bbox batch data(tensor(num,4))
        _, boxes_in = batch_data
        batch_num = len(boxes_in)
        person_num = [boxes_in[i].size()[0] for i in range(batch_num)]
        # first stage , backbone layer
        person_fea = self.baselayer(batch_data)  # (batch#num, feature)
        # second stage, link layer
        action_scores = []
        activity_scores = []
        datum = 0
        coef0 = None
        for i in range(batch_num):
            vec0 = person_fea[datum:datum + person_num[i]]
            for j in range(self.arch_para['iterative_times']):
                vec1, coef0 = self.linklayer(vec0, boxes_in[i])
                # renew the personal feature
                rate = self.arch_para['feature1_renew_rate']
                vec0 = torch.add(rate * vec1, (1 - rate) * vec0)
            # pooling for group activity(feature2)
            # coef0 (source, sink, 1)
            coef0, _ = torch.max(coef0, dim=1)  # (source ,1)
            coef0 = torch.softmax(coef0, dim=0)  # (source, 1)
            group_fea = pooling_func(vec0, method=self.arch_para['pooling_method'], other=coef0).unsqueeze(dim=0)
            datum = datum + person_num[i]
            # scores result output
            action_scores.append(vec0)  # (batch#num, actions_num)
            activity_scores.append(group_fea)  # (batch#num, activities_num)

        action_scores = torch.cat(action_scores, dim=0)
        activity_scores = torch.cat(activity_scores, dim=0)

        activity_scores = self.read_activities(activity_scores)
        action_scores = self.read_actions(action_scores)

        return action_scores, activity_scores


# linknet1, this model combine basic GCN process and two modified module
# in this model, we designed different functions for different group-activity readout methods,
# sum, concatenate, rnn
class LinkNet1(nn.Module):
    """
    the link net using other individual's feature
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, cfg_activities_num, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_align(arch_feature)
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.activities_num = cfg_activities_num
        self.device = device
        self.readout_max_n = self.arch_para['readout_max_num']
        self.readout_mode = self.arch_para['readout_mode']

        # network layers
        #   self awareness
        self.baselayer = SelfNet01(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)

        #   action sequence
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'] + self.arch_para['relation_fea_dim'], self.actions_num),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.actions_num)
        )
        #   activity sequence
        if self.readout_mode == 'sum':
            self.read_activities = nn.Sequential(
                nn.Linear(self.arch_para['relation_fea_dim'], self.activities_num),
                nn.Sigmoid(),
                nn.BatchNorm1d(self.activities_num)
            )
        elif self.readout_mode == 'rnn':
            self.read_activities = nn.Sequential(
                Rnn_S(self.arch_para['relation_fea_dim'], self.activities_num),
                nn.BatchNorm1d(self.activities_num)
            )
        else:
            """default mode : concatenate"""
            self.read_activities = nn.Sequential(
                nn.Linear(self.arch_para['relation_fea_dim']*self.readout_max_n, self.activities_num),
                nn.Sigmoid(),
                nn.BatchNorm1d(self.activities_num)
            )
        #  posi-bias convolution model group
        self.linklayer = []
        self.linklayer.append(PosiBiasNet2(self.arch_para['person_fea_dim'], self.arch_para['relation_fea_dim'],
                                           device=self.device, inter_num=self.arch_para['biasNet_channel_pos'],
                                           inter_dis=self.arch_para['biasNet_channel_dis']))
        if self.arch_para['iterative_times'] > 1:
            i = 1
            while i < self.arch_para['iterative_times']:
                self.linklayer.append(
                    PosiBiasNet(self.arch_para['relation_fea_dim'], self.arch_para['relation_fea_dim'],
                                device=self.device, inter_num=self.arch_para['biasNet_channel']))
        self.linklayer = nn.ModuleList(self.linklayer)
        # initial network parameter
        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'feature1_renew_rate': 0.2,
            'dropout_prob': 0.3,
            'biasNet_channel_pos': 8,
            'iterative_times': 3,
            'pooling_method': 'ave',
            'routing_times': 3,
            'readout_mode': 'con'
        }
        for i in arch_para:
            if i not in para:
                para[i] = arch_para[i]
        return para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.backbone_net.state_dict(),
            'mod_embed_state_dict': self.baselayer.mod_embed.state_dict(),
            'link_layer_dict': self.linklayer.state_dict(),
            'read_actions_dict': self.read_actions.state_dict(),
            'read_activities_dict': self.read_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath, mode=0):
        state = torch.load(filepath)

        if mode == 0:
            self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
            self.baselayer.mod_embed.load_state_dict(state['mod_embed_state_dict'])
            self.linklayer.load_state_dict(state['link_layer_dict'])
            self.read_actions.load_state_dict(state['read_actions_dict'])
            self.read_activities.load_state_dict(state['read_activities_dict'])
            print('Load model states from: ', filepath)
        elif mode == 1:
            self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
            print('Load model states from: ', filepath)
        else:
            assert False, "mode pattern error, you silly B"

    def forward(self, batch_data):
        """
        :param batch_data: [image_in, boxes_in]]
        image_in is a list containing image batch data(tensor(c,h,w))
        boxes_in is a list containing bbox batch data(tensor(num,4[0,1]))
        :return: [new]
        """
        ###################################
        # first layer, the backbone
        ###################################
        batch_num = len(batch_data[1])
        person_num = [batch_data[1][i].size()[0] for i in range(batch_num)]
        person_fea0 = self.baselayer(batch_data)  # (batch#num, feature)

        ###################################
        # second layer, feature embedding and feature gathering
        ###################################
        action_scores = []
        activity_scores = []
        datum = 0
        coef0 = []
        for i in range(batch_num):
            vec0 = person_fea0[datum:datum + person_num[i]]
            for j in range(self.arch_para['iterative_times']):
                vec0 = self.linklayer[j](vec0, batch_data[1][i])  # (ob_num(sink),ob_num(source),out_dim)
                vec0, coef1 = routing(vec0, times=self.arch_para['routing_times'])
            coef0.append(coef1)
            # coef0 (sink, source, 1)
            # coef0, _ = torch.max(coef0, dim=0)  # (source, 1)
            # concatenate final person feature and group feature
            person_fea = torch.cat((person_fea0[datum:datum + person_num[i]], vec0), dim=1)
            if self.readout_mode == 'sum':
                group_fea = group_max(vec0, coef1, self.readout_max_n).reshape(self.readout_max_n,-1)
                group_fea = (torch.sum(group_fea,0)).reshape(1, -1)
            else:
                group_fea = group_max(vec0, coef1, self.readout_max_n).reshape(1, -1)
            datum = datum + person_num[i]
            # scores result output
            action_scores.append(person_fea)  # (batch#num, actions_num)
            activity_scores.append(group_fea)  # (batch#num, activities_num)

        action_scores = self.read_actions(torch.cat(action_scores, dim=0))
        activity_scores = self.read_activities(torch.cat(activity_scores, dim=0))

        return action_scores, activity_scores, coef0

# in linknet2, we use group routing method, means the gathering feature will be divided into several group, also means using several tiny linknet instead of big one
class LinkNet2(nn.Module):
    """
    the link net using other individual's feature
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, cfg_activities_num, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_align(arch_feature)
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.activities_num = cfg_activities_num
        self.device = device
        self.readout_max_n = self.arch_para['readout_max_num']
        self.readout_mode = self.arch_para['readout_mode']

        # network layers
        #   self awareness
        self.baselayer = SelfNet01(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)

        #   action sequence
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'] + self.arch_para['relation_fea_dim'], self.actions_num),
            nn.Sigmoid(),
            nn.BatchNorm1d(self.actions_num)
        )
        #   activity sequence
        if self.readout_mode == 'sum':
            self.read_activities = nn.Sequential(
                nn.Linear(self.arch_para['relation_fea_dim'], self.activities_num),
                nn.Sigmoid(),
                nn.BatchNorm1d(self.activities_num)
            )
        elif self.readout_mode == 'rnn':
            self.read_activities = nn.Sequential(
                Rnn_S(self.arch_para['relation_fea_dim'], self.activities_num),
                nn.BatchNorm1d(self.activities_num)
            )
        else:
            """default mode : concatenate"""
            self.read_activities = nn.Sequential(
                nn.Linear(self.arch_para['relation_fea_dim']*self.readout_max_n, self.activities_num),
                nn.Sigmoid(),
                nn.BatchNorm1d(self.activities_num)
            )
        #  posi-bias convolution model group
        self.linklayer = []
        self.linklayer.append(PosiBiasNet2(self.arch_para['person_fea_dim'], self.arch_para['relation_fea_dim'],
                                           device=self.device, inter_num=self.arch_para['biasNet_channel_pos'],
                                           inter_dis=self.arch_para['biasNet_channel_dis']))
        if self.arch_para['iterative_times'] > 1:
            i = 1
            while i < self.arch_para['iterative_times']:
                self.linklayer.append(
                    PosiBiasNet(self.arch_para['relation_fea_dim'], self.arch_para['relation_fea_dim'],
                                device=self.device, inter_num=self.arch_para['biasNet_channel']))
        self.linklayer = nn.ModuleList(self.linklayer)
        # initial network parameter
        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'feature1_renew_rate': 0.2,
            'dropout_prob': 0.3,
            'biasNet_channel_pos': 8,
            'iterative_times': 3,
            'pooling_method': 'ave',
            'routing_times': 3,
            'readout_mode': 'con'
        }
        for i in arch_para:
            if i not in para:
                para[i] = arch_para[i]
        return para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.backbone_net.state_dict(),
            'mod_embed_state_dict': self.baselayer.mod_embed.state_dict(),
            'link_layer_dict': self.linklayer.state_dict(),
            'read_actions_dict': self.read_actions.state_dict(),
            'read_activities_dict': self.read_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath, mode=0):
        state = torch.load(filepath)

        if mode == 0:
            self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
            self.baselayer.mod_embed.load_state_dict(state['mod_embed_state_dict'])
            self.linklayer.load_state_dict(state['link_layer_dict'])
            self.read_actions.load_state_dict(state['read_actions_dict'])
            self.read_activities.load_state_dict(state['read_activities_dict'])
            print('Load model states from: ', filepath)
        elif mode == 1:
            self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
            print('Load model states from: ', filepath)
        else:
            assert False, "mode pattern error, you silly B"

    def forward(self, batch_data):
        """
        :param batch_data: [image_in, boxes_in]]
        image_in is a list containing image batch data(tensor(c,h,w))
        boxes_in is a list containing bbox batch data(tensor(num,4[0,1]))
        :return: [new]
        """
        ###################################
        # first layer, the backbone
        ###################################
        batch_num = len(batch_data[1])
        person_num = [batch_data[1][i].size()[0] for i in range(batch_num)]
        person_fea0 = self.baselayer(batch_data)  # (batch#num, feature)

        ###################################
        # second layer, feature embedding and feature gathering
        ###################################
        action_scores = []
        activity_scores = []
        datum = 0
        coef0 = []
        for i in range(batch_num):
            vec0 = person_fea0[datum:datum + person_num[i]]
            for j in range(self.arch_para['iterative_times']):
                vec0 = self.linklayer[j](vec0, batch_data[1][i])  # (ob_num(sink),ob_num(source),out_dim)
                vec0, coef1 = routing(vec0, times=self.arch_para['routing_times'])
            coef0.append(coef1)
            # coef0 (sink, source, 1)
            # coef0, _ = torch.max(coef0, dim=0)  # (source, 1)
            # concatenate final person feature and group feature
            person_fea = torch.cat((person_fea0[datum:datum + person_num[i]], vec0), dim=1)
            if self.readout_mode == 'sum':
                group_fea = group_max(vec0, coef1, self.readout_max_n).reshape(self.readout_max_n,-1)
                group_fea = (torch.sum(group_fea,0)).reshape(1, -1)
            else:
                group_fea = group_max(vec0, coef1, self.readout_max_n).reshape(1, -1)
            datum = datum + person_num[i]
            # scores result output
            action_scores.append(person_fea)  # (batch#num, actions_num)
            activity_scores.append(group_fea)  # (batch#num, activities_num)

        action_scores = self.read_actions(torch.cat(action_scores, dim=0))
        activity_scores = self.read_activities(torch.cat(activity_scores, dim=0))

        return action_scores, activity_scores, coef0

class SpectatorNet(nn.Module):
    """
    generate sideline aware by a group of feature and the objective index
    """

    def __init__(self, cfg_self_dim, cfg_object_dim, cfg_index_dim, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_align(arch_feature)
        self.self_dim = cfg_self_dim
        self.object_dim = cfg_object_dim
        self.index_dim = cfg_index_dim
        self.device = device

        # spectator feature pooling methods library
        self.poolLab = {
            'average': self.ave_pooling
        }

        # network layers
        self.biaslayer = PosiBiasNet(self.self_dim, self.self_dim, device=self.device,
                                     inter_num=self.arch_para['biasNet_channel'])

    @staticmethod
    def para_align(para):
        arch_para = {
            'biasNet_channel': 8,
            'routing_times': 3,
            'pooling_method': 'average'
        }
        for i in arch_para:
            if i not in para:
                para[i] = arch_para[i]
        return para

    @staticmethod
    def ave_pooling(*args, **kwargs):
        def method(tensor):
            """
            :param tensor: (k,feature_dim)
            :return: (feature_dim)
            """
            return torch.sum(tensor, dim=0)

        return method

    def forward(self, feature, coordinate):
        """
        the receive data only contains one frame
        feature: tensor(num, fea_dim)
        coor: tensor(num, coordinate[x1,y1,x2,y2] )
        """
        num = feature.size()[0]
        inter = self.biaslayer(feature, coordinate)  # (num(source), num(sink), fea_dim)
        eye = torch.eye(num).to(device=self.device, dtype=torch.float)  # (num,num)
        eye = torch.unsqueeze(eye, 2)  # (num, num, 1) unit matrix
        eye0 = (1.0 - eye)
        inter = torch.mul(inter, eye)  # eliminate the element in diagonal
        coef = eye0.to(device=self.device)

        # from here, start the dynamic routing algorithm
        inter, coef = self.routing(inter, coef, self.arch_para['routing_times'])  # (num(source), num(sink), fea_dim)

        # from here, doing pooling from source to sink
        result = torch.mean(inter, dim=0)  # (num(sink), fea_dim)

        return result, coef

    def routing(self, input, coef0, times):
        """
        the base processing in routing
        :param input: (num(source), num(sink), fea_dim)
        :param coef0: eye0 (num,num,1)
        :return: (num(source), num(sink), fea_dim)
        """
        coef = coef0
        for i in range(times):
            coef = torch.softmax(coef, dim=0)  # (num(source), num(sink),1)
            inter = torch.mul(input, coef)  # (num(source), num(sink), fea_dim)
            inter = torch.sum(inter, dim=0)  # (num(sink), fea_dim)
            # calculate new coefficient matrix
            inter = torch.unsqueeze(inter, 0)  # (1(source), num(sink), fea_dim)
            inter = torch.mul(input, inter)  # (num(source), num(sink), fea_dim)
            coef = torch.sum(inter, dim=2).unsqueeze(dim=2)  # (num(source), num(sink),1)
        coef = torch.softmax(coef, dim=0)  # (num(source), num(sink),1)
        inter = torch.mul(input, coef)  # (num(source), num(sink), fea_dim)
        return inter, coef

class GCN_link(nn.Module):
    """
    the link net using other individual's feature
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, cfg_activities_num, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_align(arch_feature)
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.activities_num = cfg_activities_num
        self.device = device
        self.readout_max_n = self.arch_para['readout_max_num']

        # network layers
        #   self awareness
        self.baselayer = SelfNet01(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)

        #   action sequence
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.actions_num),
            nn.Sigmoid()
        )
        #  posi-bias convolution model group
        self.linklayer = PosiBiasNet3(self.arch_para['person_fea_dim'], self.actions_num,
                                      device=self.device, inter_num=self.arch_para['biasNet_channel_pos'],
                                      inter_dis=self.arch_para['biasNet_channel_dis'])
        #   activity sequence
        self.read_activities = nn.Sequential(
            nn.Linear(self.arch_para['relation_fea_dim']*self.readout_max_n, self.activities_num),
            nn.Sigmoid()
        )
        #  GCN linear
        self.GCN_embedding = []
        self.GCN_embedding.append(
            nn.Linear(self.arch_para['person_fea_dim'], self.arch_para['relation_fea_dim'])
        )
        if self.arch_para['iterative_times'] > 1:
            i = 1
            while i < self.arch_para['iterative_times']:
                self.GCN_embedding.append(
                    nn.Linear(self.arch_para['relation_fea_dim'], self.arch_para['relation_fea_dim']))
        self.GCN_embedding = nn.ModuleList(self.GCN_embedding)
        # initial network parameter
        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'feature1_renew_rate': 0.2,
            'dropout_prob': 0.3,
            'biasNet_channel_pos': 8,
            'iterative_times': 3,
            'pooling_method': 'ave',
            'routing_times': 3
        }
        for i in arch_para:
            if i not in para:
                para[i] = arch_para[i]
        return para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.backbone_net.state_dict(),
            'mod_embed_state_dict': self.baselayer.mod_embed.state_dict(),
            'link_layer_dict': self.linklayer.state_dict(),
            'read_actions_dict': self.read_actions.state_dict(),
            'GCN_embedding_dict': self.GCN_embedding.state_dict(),
            'read_activities_dict': self.read_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath, mode=0):
        state = torch.load(filepath)

        if mode == 0:
            self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
            self.baselayer.mod_embed.load_state_dict(state['mod_embed_state_dict'])
            self.GCN_embedding.load_state_dict(state['link_layer_dict'])
            self.read_actions.load_state_dict(state['read_actions_dict'])
            self.read_activities.load_state_dict(state['read_activities_dict'])
            print('Load model states from: ', filepath)
        elif mode == 1:
            self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
            self.baselayer.mod_embed.load_state_dict(state['mod_embed_state_dict'])
            self.linklayer.load_state_dict(state['link_layer_dict'])
            self.read_actions.load_state_dict(state['read_actions_dict'])
            print('Load model states from: ', filepath)
        else:
            assert False, "mode pattern error, you silly B"

    def forward(self, batch_data):
        """
        :param batch_data: [image_in, boxes_in]]
        image_in is a list containing image batch data(tensor(c,h,w))
        boxes_in is a list containing bbox batch data(tensor(num,4[0,1]))
        :return: [new]
        """
        ###################################
        # first layer, the backbone
        ###################################
        batch_num = len(batch_data[1])
        person_num = [batch_data[1][i].size()[0] for i in range(batch_num)]
        person_fea0 = self.baselayer(batch_data)  # (batch#num, feature)

        ###################################
        # second layer, the action linklayer
        ###################################
        action_scores = []
        activities_scores = []
        datum = 0
        coef0 = []
        kl_loss_all = torch.tensor(0., device=self.device)
        for i in range(batch_num):
            """
            generate self-see action score
            """
            vec1 = self.read_actions(person_fea0[datum:datum + person_num[i]])
            vec1 = F.normalize(vec1, p=1, dim=-1)
            """
            generate the side-see action score from other node
            """
            vec0 = self.linklayer(person_fea0[datum:datum + person_num[i]],
                                  batch_data[1][i])  # (ob_num(sink),ob_num(source),out_dim)
            vec0 = F.normalize(vec0, p=1, dim=-1)
            # before routing, delete the diagonal vectors
            nodia = torch.tensor([j for j in range(person_num[i] ** 2) if j % (person_num[i] + 1) != 0],
                                 device=self.device)
            vec0 = torch.index_select(vec0.reshape(person_num[i] ** 2, -1), 0, nodia)
            if np.any(np.isnan(vec0.cpu().detach().numpy())):
                a = 1
                pass
            vec0 = vec0.reshape(person_num[i], person_num[i] - 1, -1)
            vec0, coef1, kl_loss = routing_link3(vec0, vec1, times=self.arch_para['routing_times'])  # rounting
            # scores result output
            action_scores.append(vec0)  # (batch#num, actions_num)
            kl_loss_all = kl_loss_all + kl_loss

            ###################################
            # third layer, the GCN net
            ###################################
            # coef1  (batch(sink), batch(source))
            fea0 = person_fea0[datum:datum + person_num[i]]
            for j in range(self.arch_para['iterative_times']):
                fea0 = self.GCN_embedding[j](fea0)  # (batch, fea)
                fea0 = torch.mm(coef1, fea0)  # (batch, fea)
                fea0 = torch.sigmoid(fea0)
            # readout activity
            group_fea = group_max(fea0, coef1, max_n=self.readout_max_n).reshape(-1)
            activities_scores.append(self.read_activities(group_fea))
            coef0.append(coef1)

            datum = datum + person_num[i]

        action_scores = torch.cat(action_scores, dim=0)
        activities_scores = torch.cat(activities_scores, dim=0)
        a = torch.sum(torch.tensor(person_num))
        kl_loss_all1 = kl_loss_all / a

        return action_scores, activities_scores, coef0

class Rnn_S(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn=nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True,
               nonlinearity='relu')
    def forward(self, input):
        output, _ = self.rnn(input) #(batch,seq,dim)
        output = torch.squeeze(output[:,-1:,:]) #(batch,dim)

        return output