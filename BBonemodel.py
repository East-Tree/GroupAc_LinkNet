from backbone import *
from utils import *

from torchvision import ops  # RoIAlign module


# add a state layer from model 1
class SelfNet2(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, device=None, **arch_feature):

        super().__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # here determine the backbone net and embedding layer
        self.baselayer = SelfNet0(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)
        """
        # state sequence
        self.mod_state = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.arch_para['state_fea_dim']),
            nn.LeakyReLU(),
            nn.Dropout(p=self.arch_para['dropout_prob']),
            nn.BatchNorm1d(self.arch_para['state_fea_dim'])
        )
        """

        # action sequence
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.actions_num),
            nn.LeakyReLU()
        )

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'person_fea_dim': 2048,
            'state_fea_dim': 512,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.backbone_net.state_dict(),
            'mod_embed_state_dict': self.baselayer.mod_embed.state_dict(),
            'read_actions_dict': self.read_actions.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.baselayer.load_state_dict(state['base_state_dict'])
        self.read_actions.load_state_dict(state['read_actions_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data, mode=None, return_fea=False,cata_balance=False,label=None):
        # image_in is a list containing image batch data(tensor(c,h,w))
        # boxes_in is a list containing bbox batch data(tensor(num,4))
        self_features = self.baselayer(batch_data)  # (B*N, NFB)
        feature_label = label

        """
        # self states
        self_states = self.mod_state(self_features)  # B*N, state_feature_dim
        """

        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB
        if mode == 'train' and cata_balance:
            self_features, feature_label = category_balance(self_features,feature_label)
        actions_scores = self.read_actions(self_features)  # B*N, actions_num

        if mode == 'train':
            if return_fea:
                return actions_scores, self_features, feature_label
            else:
                return actions_scores, feature_label
        else:
            return actions_scores


# only backbone and embedding layer
class SelfNet0(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, device=None, **arch_feature):

        super().__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # here determine the backbone net
        self.backbone_net = MyInception_v3(transform_input=False, pretrained= True)  # type: MyInception_v3
        self.backbone_dim = MyInception_v3.outputDim()
        self.backbone_size = MyInception_v3.outputSize(*self.imagesize)

        # embedding sequence
        self.mod_embed = nn.Sequential(
            nn.Linear(self.RoI_crop_size[0] * self.RoI_crop_size[0] * self.backbone_dim,
                      self.arch_para['person_fea_dim']),
            nn.LeakyReLU(),
            nn.LayerNorm(self.arch_para['person_fea_dim']),
            nn.Dropout(0.3)
        )


        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'person_fea_dim': 2048,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone_net.state_dict(),
            'mod_embed_state_dict': self.mod_embed.state_dict(),
            # 'mod_state_state_dict': self.mod_state.state_dict(),
            'mod_actions_state_dict': self.mod_actions.state_dict(),
            # 'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone_net.load_state_dict(state['backbone_state_dict'])
        self.mod_emb.load_state_dict(state['mod_embed_state_dict'])
        self.mod_actions.state_dict(state['mod_actions_state_dict'])
        # self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # image_in is a list containing image batch data(tensor(c,h,w))
        # boxes_in is a list containing bbox batch data(tensor(num,4))
        images_in, boxes_in = batch_data

        # read config parameters
        B = len(images_in)  # batch size
        K = self.RoI_crop_size[0]
        D = self.backbone_dim
        OH, OW = self.backbone_size
        NFB = self.arch_para['person_fea_dim']

        # Reshape the input data
        images_in_flat = torch.stack(images_in).to(device=self.device)  # B, 3, H, W

        #    reshape the bboxs coordinates into [K,(index,x1,y1,x2,y2)]
        boxes_in_flat = boxes_in
        boxes_in_index = []
        for i in range(B):
            box_num = boxes_in_flat[i].size()[0]
            boxes_in_index.append(torch.tensor([[i]] * box_num))
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0).float()
        boxes_in_index = torch.cat(boxes_in_index, dim=0).float()
        #    cat flat and index together
        boxes_in_flat = torch.cat([boxes_in_index, boxes_in_flat], dim=1).to(device=self.device)
        #    convert the origin coordinate(rate) into backbone feature scale coordinate(absolute int)
        operator = torch.tensor([1, OW, OH, OW, OH], device=self.device).float()
        boxes_in_flat = boxes_in_flat * operator
        boxes_in_flat = boxes_in_flat.int().to(device=self.device)
        boxes_in_flat = boxes_in_flat.float()

        # Use backbone to extract features of images_in
        # Pre-precess first  normalized to [-1,1]
        images_in_flat = prep_images(images_in_flat.float())

        outputs = self.backbone_net(images_in_flat)

        # Build multiscale features
        # normalize all feature map into same scale
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1).to(device=self.device)  # B*T, D, OH, OW

        # RoI Align
        boxes_features = ops.roi_align(features_multiscale, boxes_in_flat, (K, K))  # B*N, D, K, K,

        boxes_features = boxes_features.reshape(-1, D * K * K)  # B*N, D*K*K
        boxes_features.to(device=self.device)

        # Embedding to feature
        self_features = self.mod_embed(boxes_features)  # (B*N, NFB)


        return self_features

# randomly organize the output individual feature
def category_balance(input0, label0, batch0=None, class_num=9):

    """
    :param input0: tensor(batch,fea) or list[tensor(batch,fea), ...]
    :param label0: rensor(batch)
    :return: tensor(batch,fea)
    """
    cate_num = class_num
    if isinstance(input0,list):
        for item in input0:
            assert item.size()[0] == label0.size()[0], 'input tensor should be same with index tensor in 0 dim'
    else:
        assert input0.size()[0] == label0.size()[0], 'input tensor should be same with index tensor in 0 dim'
    if batch0 == None:
        batch = label0.size()[0]
    else:
        batch = batch0
    # calculate each category num in label0
    labelInfo = []
    for i in range(cate_num):
        item = (label0 == i).nonzero().reshape(-1)
        labelInfo.append({
            'index': item.tolist(),
            'num': item.size()[0]
        })
    # rebuild output tensor's index
    index = []
    for i in range(batch):
        num = 0
        cate = 0
        while num == 0:
            rand = torch.rand(1)
            cate = int(rand * (cate_num))
            num = labelInfo[cate]['num']
        rand = torch.rand(1)
        instance = int(rand * (num))
        index.append(labelInfo[cate]['index'][instance])

    if isinstance(input0,list):
        outputTensor = [torch.index_select(i, 0, torch.tensor(index).to(device=i.device)) for i in input0]
    else:
        outputTensor = torch.index_select(input0, 0, torch.tensor(index).to(device=input0.device))
    outputLabel = torch.index_select(label0, 0, torch.tensor(index).to(device=label0.device))

    return outputTensor, outputLabel


# the sequential mode of selfnet, add a LSTM layer , the input should be (B*(realB*frame num),C,W,H)
class SelfNetS(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, device=None, **arch_feature):

        super().__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # here determine the backbone net and embedding layer
        self.baselayer = SelfNet0(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)
        
        
        # LSTM model to analyze the personal feature
        self.fea_lstm = nn.LSTM(self.arch_para['person_fea_dim'],self.arch_para['person_fea_dim'])

        # action sequence
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.actions_num),
            nn.LeakyReLU()
        )

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'person_fea_dim': 2048,
            'state_fea_dim': 512,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.backbone_net.state_dict(),
            'mod_embed_state_dict': self.baselayer.mod_embed.state_dict(),
            'fea_lstm_state_dict':self.fea_lstm.state_dict(),
            'read_actions_dict': self.read_actions.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.baselayer.load_state_dict(state['base_state_dict'])
        self.read_actions.load_state_dict(state['read_actions_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data, mode=None, return_fea=False,cata_balance=False,label=None, seq_len=1):
        # image_in is a list containing image batch data()tensor(c,h,w)
        # boxes_in is a list containing bbox batch data()tensor(num,4)
        # in order to apply the lstm after feature embeding, the fitted data is orgnized as [f00,f10,f01,f11,f02,f12]
        
        self_features = self.baselayer(batch_data)  # ([f00,f10,f01,f11,f02,f12]**N, NFB)
        feature_label = label

        # LSTM layer, the input should be resize as (seq_len, batch, input_size)
        # [[f00N,f10N],
        #  [f01N,f11N],
        #  [f02N,f12N]]
        if seq_len>1:
            self_features = self_features.reshape(seq_len,-1,self_features.size()[-1])
            self_features,_ = self.fea_lstm(self_features) #(seq_len,node_num, input_size)
            self_features = self_features[-1,:].squeeze()
        
        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB
        if mode == 'train' and cata_balance:
            self_features, feature_label = category_balance(self_features,feature_label)
        actions_scores = self.read_actions(self_features)  # B*N, actions_num

        if mode == 'train':
            if return_fea:
                return actions_scores, self_features, feature_label
            else:
                return actions_scores, feature_label
        else:
            return actions_scores

# the model from SelfNetS, but add new label informtion orientation and area
class SelfNetSN(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, cfg_orien_num, device=None, **arch_feature):

        super().__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.orien_num = cfg_orien_num
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # here determine the backbone net and embedding layer
        self.baselayer = SelfNet0(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)
        
        
        # LSTM model to analyze the personal feature
        self.fea_lstm = nn.LSTM(self.arch_para['person_fea_dim'],self.arch_para['person_fea_dim'])

        if self.arch_para['fea_decoup']:
            op = 2
        else:
            op = 1
        # action model
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim']//op, self.actions_num),
            nn.LeakyReLU()
        )

        # orientation model
        self.read_orientations = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim']//op, self.orien_num),
            nn.LeakyReLU()
        )

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'fea_decoup' : False,
            'person_fea_dim': 2048,
            'state_fea_dim': 512,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath):
        state = {
            'base_state_dict': self.baselayer.backbone_net.state_dict(),
            'mod_embed_state_dict': self.baselayer.mod_embed.state_dict(),
            'fea_lstm_state_dict':self.fea_lstm.state_dict(),
            'read_actions_dict': self.read_actions.state_dict(),
            'read_orien_dict': self.read_orientations.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.baselayer.backbone_net.load_state_dict(state['base_state_dict'])
        self.baselayer.mod_embed.load_state_dict(state['mod_embed_state_dict'])
        self.fea_lstm.load_state_dict(state['fea_lstm_state_dict'])
        self.read_actions.load_state_dict(state['read_actions_dict'])
        self.read_orientations.load_state_dict(state['read_orien_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data, mode=None, return_fea=False,cata_balance=False,label=None, addi_label=None, seq_len=1):
        # image_in is a list containing image batch data()tensor(c,h,w)
        # boxes_in is a list containing bbox batch data()tensor(num,4)
        # in order to apply the lstm after feature embeding, the fitted data is orgnized as [f00,f10,f01,f11,f02,f12]
        
        self_features = self.baselayer(batch_data)  # ([f00,f10,f01,f11,f02,f12]**N, NFB)
        feature_label = label

        # LSTM layer, the input should be resize as (seq_len, batch, input_size)
        # [[f00N,f10N],
        #  [f01N,f11N],
        #  [f02N,f12N]]
        if seq_len>1:
            self_features = self_features.reshape(seq_len,-1,self_features.size()[-1])
            self_features,_ = self.fea_lstm(self_features) #(seq_len,node_num, input_size)
            self_features = self_features[-1,:].squeeze()
        
        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB
        if mode == 'train' and cata_balance:
            if addi_label is None:
                self_features, feature_label = category_balance(self_features,feature_label)
            else:
                feature_list, feature_label = category_balance([self_features,addi_label], feature_label)
                self_features = feature_list[0]
                feature_label = [feature_label]
                feature_label.extend(feature_list[1:])
        else:
            feature_label = [label, addi_label]
        
        if self.arch_para['fea_decoup']:
            inte = self.arch_para['person_fea_dim']
            inte2 = self.arch_para['person_fea_dim']//2
            actions_scores = self.read_actions(self_features[:,0:inte2])
            orien_scores = self.read_orientations(self_features[:,inte2:inte])
        else:
            actions_scores = self.read_actions(self_features)  # B*N, actions_num
            orien_scores = self.read_orientations(self_features) # B*N, orien_num

        if mode == 'train':
            if return_fea:
                return actions_scores, orien_scores, self_features, feature_label
            else:
                return actions_scores, orien_scores, feature_label
        else:
            return actions_scores, orien_scores
