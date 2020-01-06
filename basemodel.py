from backbone import *
from utils import *

from torchvision import ops  # RoIAlign module


# only self actions recognition similar to GCN work
class SelfNet(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, device=None, **arch_feature):

        super(SelfNet, self).__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # here determine the backbone net
        self.backbone_net = MyInception_v3(transform_input=False, pretrained=False)  # type: MyInception_v3
        self.backbone_dim = MyInception_v3.outputDim()
        self.backbone_size = MyInception_v3.outputSize(*self.imagesize)

        self.fc_emb = nn.Linear(self.RoI_crop_size[0] * self.RoI_crop_size[0] * self.backbone_dim,
                                self.arch_para['person_fea_dim'])
        self.dropout_emb = nn.Dropout(p=self.arch_para['dropout_prob'])

        self.fc_actions = nn.Linear(self.arch_para['person_fea_dim'], self.actions_num)

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def para_align(self, para):
        arch_para = {
            'person_fea_dim': 1024,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone_net.state_dict(),
            'fc_emb_state_dict': self.fc_emb.state_dict(),
            'fc_actions_state_dict': self.fc_actions.state_dict(),
            # 'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # image_in is a list containing image batch data(tensor(c,h,w))
        # boxes_in is a list containing bbox batch data(tensor(num,4))
        images_in, boxes_in = batch_data

        # read config parameters
        B = len(images_in)  # batch size
        K = self.RoI_crop_size[0]
        D = self.backbone_dim
        H, W = self.imagesize
        OH, OW = self.backbone_size
        Hr = OH / H
        Wr = OW / W
        NFB = self.arch_para['person_fea_dim']

        # Reshape the input data
        images_in_flat = torch.stack(images_in).to(device=self.device)  # B, 3, H, W

        #    reshape the bboxs coordinates into [K,(index,x1,y1,x2,y2)]
        boxes_in_flat = boxes_in
        boxes_in_index = []
        for i in range(B):
            box_num = boxes_in_flat[i].size()[0]
            boxes_in_index.append(torch.tensor([[i]] * box_num))
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0)
        boxes_in_index = torch.cat(boxes_in_index, dim=0)
        #    cat flat and index together
        boxes_in_flat = torch.cat([boxes_in_index, boxes_in_flat], dim=1)
        #    convert the origin coordinate(rate) into backbone feature scale coordinate(absolute int)
        operator = torch.tensor([1, Wr, Hr, Wr, Hr])
        boxes_in_flat = boxes_in_flat.float() * operator
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

        # ActNet
        boxes_in_flat.requires_grad = False
        #  features_multiscale.requires_grad=False

        # RoI Align
        boxes_features = ops.roi_align(features_multiscale, boxes_in_flat, (K, K))  # B*N, D, K, K,

        boxes_features = boxes_features.reshape(-1, D * K * K)  # B*N, D*K*K
        boxes_features.to(device=self.device)

        # Embedding to hidden state
        boxes_features = self.fc_emb(boxes_features)  # B*N, NFB
        boxes_features = F.relu(boxes_features)
        boxes_features = self.dropout_emb(boxes_features)

        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB

        actions_scores = self.fc_actions(boxes_features)  # B*N, actions_num

        return actions_scores


# add a state layer from model 1
class SelfNet2(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg_imagesize, cfg_roisize, cfg_actions_num, device=None, **arch_feature):

        super(SelfNet2, self).__init__()
        self.imagesize = cfg_imagesize
        self.RoI_crop_size = cfg_roisize
        self.actions_num = cfg_actions_num
        self.device = device

        # define architecture parameter
        self.arch_para = self.para_align(arch_feature)

        # here determine the backbone net
        self.backbone_net = MyInception_v3(transform_input=False, pretrained=False)  # type: MyInception_v3
        self.backbone_dim = MyInception_v3.outputDim()
        self.backbone_size = MyInception_v3.outputSize(*self.imagesize)

        # embedding sequence
        self.mod_embed = nn.Sequential(
            nn.Linear(self.RoI_crop_size[0] * self.RoI_crop_size[0] * self.backbone_dim,
                      self.arch_para['person_fea_dim']),
            nn.LeakyReLU(),
            nn.Dropout(p=self.arch_para['dropout_prob']),
            nn.BatchNorm1d(self.arch_para['person_fea_dim'])
        )

        # state sequence
        self.mod_state = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.arch_para['state_fea_dim']),
            nn.LeakyReLU(),
            nn.Dropout(p=self.arch_para['dropout_prob']),
            nn.BatchNorm1d(self.arch_para['state_fea_dim'])
        )

        # action sequence
        self.mod_actions = nn.Sequential(
            nn.Linear(self.arch_para['state_fea_dim'], self.actions_num),
            nn.LeakyReLU(),
            nn.Dropout(p=self.arch_para['dropout_prob'])
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
            'backbone_state_dict': self.backbone_net.state_dict(),
            'mod_embed_state_dict': self.mod_embed.state_dict(),
            'mod_state_state_dict': self.mod_state.state_dict(),
            'mod_actions_state_dict': self.mod_actions.state_dict(),
            # 'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # image_in is a list containing image batch data(tensor(c,h,w))
        # boxes_in is a list containing bbox batch data(tensor(num,4))
        images_in, boxes_in = batch_data

        # read config parameters
        B = len(images_in)  # batch size
        K = self.RoI_crop_size[0]
        D = self.backbone_dim
        H, W = self.imagesize
        OH, OW = self.backbone_size
        Hr = OH / H
        Wr = OW / W
        NFB = self.arch_para['person_fea_dim']

        # Reshape the input data
        images_in_flat = torch.stack(images_in).to(device=self.device)  # B, 3, H, W

        #    reshape the bboxs coordinates into [K,(index,x1,y1,x2,y2)]
        boxes_in_flat = boxes_in
        boxes_in_index = []
        for i in range(B):
            box_num = boxes_in_flat[i].size()[0]
            boxes_in_index.append(torch.tensor([[i]] * box_num))
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0)
        boxes_in_index = torch.cat(boxes_in_index, dim=0)
        #    cat flat and index together
        boxes_in_flat = torch.cat([boxes_in_index, boxes_in_flat], dim=1)
        #    convert the origin coordinate(rate) into backbone feature scale coordinate(absolute int)
        operator = torch.tensor([1, Wr, Hr, Wr, Hr])
        boxes_in_flat = boxes_in_flat.float() * operator
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

        # ActNet
        boxes_in_flat.requires_grad = False
        #  features_multiscale.requires_grad=False

        # RoI Align
        boxes_features = ops.roi_align(features_multiscale, boxes_in_flat, (K, K))  # B*N, D, K, K,

        boxes_features = boxes_features.reshape(-1, D * K * K)  # B*N, D*K*K
        boxes_features.to(device=self.device)

        # Embedding to feature
        self_features = self.mod_embed(boxes_features)  # B*N, NFB
        # self states
        self_states = self.mod_state(self_features)  # B*N, state_feature_dim

        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB

        actions_scores = self.mod_actions(self_states)  # B*N, actions_num

        return actions_scores


# add link actions recognition
class SelfNet3(nn.Module):
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

        # here determine the backbone net
        self.backbone_net = MyInception_v3(transform_input=False, pretrained=False)  # type: MyInception_v3
        self.backbone_dim = MyInception_v3.outputDim()
        self.backbone_size = MyInception_v3.outputSize(*self.imagesize)

        # embedding sequence
        self.mod_embed = nn.Sequential(
            nn.Linear(self.RoI_crop_size[0] * self.RoI_crop_size[0] * self.backbone_dim,
                      self.arch_para['person_fea_dim']),
            nn.LeakyReLU(),
            nn.Dropout(p=self.arch_para['dropout_prob']),
            nn.BatchNorm1d(self.arch_para['person_fea_dim']),
        )

        # state sequence
        self.mod_state = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.arch_para['state_fea_dim']),
            nn.LeakyReLU(),
            nn.Dropout(p=self.arch_para['dropout_prob'])
        )

        # action sequence
        self.mod_actions = nn.Sequential(
            nn.Linear(self.arch_para['state_fea_dim'], self.actions_num),
            nn.LeakyReLU(),
            nn.Softmax(dim=1)
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
            'backbone_state_dict': self.backbone_net.state_dict(),
            'mod_embed_state_dict': self.mod_embed.state_dict(),
            'mod_state_state_dict': self.mod_state.state_dict(),
            'mod_actions_state_dict': self.mod_actions.state_dict(),
            # 'fc_activities_state_dict': self.fc_activities.state_dict()
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        # image_in is a list containing image batch data(tensor(c,h,w))
        # boxes_in is a list containing bbox batch data(tensor(num,4))
        images_in, boxes_in = batch_data

        # read config parameters
        B = len(images_in)  # batch size
        K = self.RoI_crop_size[0]
        D = self.backbone_dim
        H, W = self.imagesize
        OH, OW = self.backbone_size
        Hr = 1 / H
        Wr = 1 / W
        NFB = self.arch_para['person_fea_dim']

        # Reshape the input data
        images_in_flat = torch.stack(images_in).to(device=self.device)  # B, 3, H, W

        #    reshape the bboxs coordinates into [K,(index,x1,y1,x2,y2)]
        boxes_in_flat = boxes_in
        boxes_in_index = []
        for i in range(B):
            box_num = boxes_in_flat[i].size()[0]
            boxes_in_index.append(torch.tensor([[i]] * box_num))
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0)
        boxes_in_index = torch.cat(boxes_in_index, dim=0)
        #    cat flat and index together
        boxes_in_flat = torch.cat([boxes_in_index, boxes_in_flat], dim=1)
        #    convert the origin coordinate(rate) into backbone feature scale coordinate(absolute int)
        operator1 = torch.tensor([1, Wr, Hr, Wr, Hr])
        operator2 = torch.tensor([1, OW, OH, OW, OH])  # rate coordinate
        boxes_in_flat_rate = boxes_in_flat.float() * operator1
        boxes_in_flat = boxes_in_flat_rate * operator2
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

        # ActNet
        boxes_in_flat.requires_grad = False
        #  features_multiscale.requires_grad=False

        # RoI Align
        boxes_features = ops.roi_align(features_multiscale, boxes_in_flat, (K, K))  # B*N, D, K, K,

        boxes_features = boxes_features.reshape(-1, D * K * K)  # B*N, D*K*K
        boxes_features.to(device=self.device)

        # Embedding to feature
        self_features = self.mod_embed(boxes_features)  # B*N, NFB
        # self states
        self_states = self.mod_state(self_features)  # B*N, state_feature_dim

        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB

        actions_scores = self.mod_actions(self_states)  # B*N, actions_num

        return actions_scores


# link net
class LinkNet(nn.Module):
    """
    the link net using other individual's feature
    """

    def __init__(self, cfg_state_dim, cfg_cood_dim, cfg_out_dim, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_align(arch_feature)
        self.state_dim = cfg_state_dim
        self.cood_dim = cfg_cood_dim
        self.coodmap_dim = int(self.state_dim * self.arch_para['map_dim_rate'])
        self.out_dim = cfg_out_dim
        self.device = device

        # network layers

    def para_align(self, para):
        arch_para = {
            'map_dim_rate': 0.5,
            'dropout_prob': 0.3
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para

    def forward(self, batch_cood, batch_state):
        """
        :param batch_cood: tensor [person num, 4]
        :param batch_state: tensor [person num, feature_dim]
        :return: [new]
        """
        person_num = batch_cood.size()[0]


class SpectatorNet(nn.Module):
    """
    generate sideline aware by a group of feature and the objective index
    """

    def __init__(self, cfg_self_dim, cfg_object_dim, cfg_index_dim, device=None, **arch_feature):
        super().__init__()
        self.arch_para = self.para_aling(arch_feature)
        self.self_dim = cfg_self_dim
        self.object_dim = cfg_object_dim
        self.index_dim = cfg_index_dim
        self.device = device

        # network layers
        self.biaslayer = BiasNet(self.self_dim,self.index_dim,self.object_dim,device=self.device,inter_num=self.arch_para['biasNet_channel'])

    def para_align(self, para):
        arch_para = {
            'biasNet_channel': 8
        }
        for i in arch_para:
            if i in para:
                arch_para[i] = para[i]
        return arch_para


# tiny attention network. This layer works on situation that doing linear connect from one feature to another while it higtly depends on one part of it
class BiasNet(nn.Module):
    def __init__(self, input_dim, index_dim, output_dim, device=None, inter_num=8):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.output_dim = output_dim
        self.device = device
        self.inter_num = inter_num

        # network layer
        self.layer1 = nn.Linear(self.input_dim, self.output_dim * self.inter_num)
        self.layer2 = nn.Linear(self.input_dim, self.index_dim * self.inter_num)

    def forward(self, input, index, parallel=False):
        """
        :param parallel:
        :type index: torch.tensor
        :type input: torch.tensor
        """
        batch_size = input.size()[0]
        if not parallel:
            # network forward
            intern1 = self.layer1(input)  # (batch_num, inter_num*out_dim)
            index1 = self.layer2(input)  # (batch_num, inter_num*index_dim)

            # calculate the dot product with index(out) and index(label)
            index0 = index.reshape(-1, 1, self.index_dim).to(device=self.device)  # (batch_num,1,index_dim)
            index1 = index1.reshape(-1, self.inter_num, self.index_dim)  # (batch_num, inter_num, index_dim)
            index1 = torch.mul(index1, index0)  # (batch_num, inter_num, index_dim)
            coefficient = torch.sum(index1, dim=2)  # (batch_num, inter_num), this is the similar factor with each channel and label
            coefficient = F.softmax(coefficient, dim=1)  # (batch_num, inter_num)

            # calculate the final result by coefficient
            intern1 = intern1.reshape(-1, self.inter_num, self.output_dim)  # (batch_num, inter_num, out_dim)
            coefficient = coefficient.reshape(-1, self.inter_num, 1)  # (batch_num, inter_num,1)
            intern1 = torch.mul(intern1, coefficient)  # (batch_num, inter_num, out_dim)
            intern1 = torch.sum(intern1, dim=1)  # (batch_num, out_dim)

            return intern1
        else:
            """
            in parallel mode, one batch only contains a group of feature,  the result will generated between each pairs of features
            """
            # network forward
            intern1 = self.layer1(input)  # (batch_num, inter_num*out_dim)
            index1 = self.layer2(input)  # (batch_num, inter_num*index_dim)

            # dot product with (each feature channel(batch*inter_num)) and (each label(batch))
            index1 = index1.reshape(-1, self.index_dim)  # (batch*inter_num, index_dim)
            index0t = torch.t(index)  # (index_dim, batch)
            coefficient = torch.mm(index1, index0t)  # (batch(source)*inter_num, batch(sink))
            coefficient = coefficient.reshape(-1, self.inter_num, batch_size)  # (batch, inter_num, batch)
            coefficient = torch.transpose(coefficient, 1, 2)  # (batch,batch,inter_num)
            coefficient = torch.softmax(coefficient, dim=2)  # (batch(source),batch(sink),inter_num)
            coefficient = torch.transpose(coefficient, 0, 1)  # (batch(sink),batch(source),inter_num)

            # calculate result for each pairs of feature
            intern1 = intern1.repeat(batch_size, 1)  # (batch_num(sink), batch_num(source), inter_num*out_dim)
            intern1 = intern1.reshape(-1, batch_size, self.inter_num, self.output_dim)  # (batch_num, batch_num, inter_num, out_dim)
            coefficient = coefficient.unsqueeze(3)  # (batch,batch, inter_num,1)
            intern1 = torch.mul(intern1, coefficient)  # (batch_num, batch_num, inter_num, out_dim)
            intern1 = torch.sum(intern1, dim=2)  # (batch_num(sink), batch_num(source), out_dim)

            return intern1

