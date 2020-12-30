from backbone import *
from utils import *

from torchvision import ops  # RoIAlign module

import BBonemodel as BB


class GCN(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_feature_dim, cfg_output_dim, device=None, **arch_feature):

        super().__init__()
        self.fea_dim = cfg_feature_dim
        self.output_dim = cfg_output_dim
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # metric embeding layer
        self.metric_embed = nn.Sequential(
            nn.Linear(self.fea_dim, self.arch_para['metric_dim']),
            nn.Sigmoid()
        )

        # GCN processing
        self.GCN_embed = nn.Linear(self.fea_dim, self.output_dim)
            

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'metric_dim': 100,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath):
        state = {
            'metric_embed_state_dict': self.metric_embed.state_dict(),
            'GCN_embed_state_dict': self.GCN_embed.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.metric_embed.load_state_dict(state['metric_embed_state_dict'])
        self.GCN_embed.load_state_dict(state['GCN_embed_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, data):
        # this is the forward function for GCN model
        # data (num*fea_dim)

        # the metric embedding
        metric = self.metric_embed(data)  # (num*metric dim)
        relation = torch.mm(metric, metric.transpose(0,1)) # (num * num)
        relation = F.softmax(relation,dim=1)

        # GCN layer
        GCN_fea = self.GCN_embed(data)  # (num*GCN dim)
        out  = torch.mm(relation, GCN_fea)
        out = torch.sigmoid(out)

        return out

# use the explicit state and the simple GCN 

class exp_GCN(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, cfg_orien_num,cfg_activi_num,coor_use=True,area_use=False,device=None, **arch_feature):

        super().__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.activities_num = cfg_activi_num
        self.orien_num = cfg_orien_num
        self.device = device
        self.use_coor = coor_use
        self.use_area = area_use

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # use the selfnetSN to out put the personal state
        self.selfnet = BB.SelfNetSN(cfg_imagesize, cfg_roisize, cfg_actions_num,cfg_orien_num,device=device,**self.arch_para)

        # calculate the state dim
        self.state_dim = 9 + 8
        if self.use_coor:
            self.state_dim += 6
        if self.use_coor:
            self.state_dim += 4

        # use the simple GCN model
        self.GCN_layer = GCN(self.state_dim,self.arch_para['GCN_embed_fea'],device=self.device,**self.arch_para)

        # read out activity 
        self.read_activity = nn.Sequential(
            nn.Linear(self.arch_para['GCN_embed_fea'], self.activities_num),
            nn.Sigmoid()
        )

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'GCN_embed_fea': 100,
            'person_fea_dim': 2048,
            'state_fea_dim': 512,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath, mode=0):
        if mode == 0:
            state = {
                'GCN_state_dict': self.GCN_layer.state_dict(),
                'read_activities_dict': self.read_activity.state_dict()
            }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath, mode=1, filepath2=None):
        state = torch.load(filepath)
        if mode == 0:
            self.selfnet.loadmodel(filepath2)
            self.GCN_layer.load_state_dict(state['GCN_state_dict'])
            self.read_activity.load_state_dict(state['read_activities_dict'])
        elif mode == 1:
            self.selfnet.loadmodel(filepath)
        print('Load model states from: ', filepath)

    def forward(self, batch_data, mode=None,seq_len=1):

        batch_size = len(batch_data[1])
        person_list = [i.size()[0] for i in batch_data[2]]
        
        # generate the state for action and orientation
        actions_scores,oriens_scores = self.selfnet((batch_data[0], batch_data[3]),seq_len=seq_len)
        state = [actions_scores,oriens_scores]

        # calculate the position state
        if self.use_coor:
            coor = batch_data[3]  #  b00,b10,b01,b11,b02,b12 
            coor = torch.cat(coor,dim=0)  # seq*batch_sum, 4(x1,y1,x2,y2)
            coor = coor.reshape(seq_len,-1,4) # seq , batch_sum , 4
            coor = coor.transpose(0,1)  # batch_sum , seq , 4
            coor = coor. reshape(-1,seq_len*4)  # batch_sum ,seq(3) *4
            op = torch.tensor(
                [
                    [0,0,0,0,0.5,0,0.5,0,0,0,0,0],  # cx = 0.5x1+0.5x2
                    [0,0,0,0,0,0.5,0,0.5,0,0,0,0],  # cy = 0.5y1+0.5y2
                    [0,0,0,0,-1,0,1,0,0,0,0,0],  # w = x2-x1
                    [0,0,0,0,0,-1,0,1,0,0,0,0],  # h = y2-y1
                    [-0.5,0,-0.5,0,0,0,0,0,0.5,0,0.5,0],   # delta cx
                    [0,-0.5,0,-0.5,0,0,0,0,0,0.5,0,0.5]  # delta cy
                ]
            )
            op = op.transpose(0,1).to(device=coor.device,dtype=coor.dtype) # (12*6)

            coor_state = torch.mm(coor,op)
            state.append(coor_state.to(device=self.device))
        
        # calculate the area state
        if self.use_area:
            area_index = torch.cat(batch_data[5], dim=0)
            area_state = torch.zeros(area_index.size()[0], 4).scatter_(1, area_index, 1)
            state.append(area_state.to(device=self.device))

        # concatnate the state
        person_state = torch.cat(state,dim=1)
        
        #  GCN processing
        j = 0
        activity_fea = []
        for i in person_list:
            activity_fea.append(self.GCN_layer(person_state[j:j+i]))
            j += i

        # read the activity label
        global_fea = [torch.mean(i,dim=0) for i in activity_fea]
        global_fea = torch.stack(global_fea, dim=0)

        activity_score = self.read_activity(global_fea)

        return activity_score