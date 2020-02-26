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
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.actions_num)
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
            'base_state_dict': self.baselayer.state_dict(),
            'read_actions_dict': self.read_actions.state_dict()
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
        self_features = self.baselayer(batch_data)  # (B*N, NFB)

        """
        # self states
        self_states = self.mod_state(self_features)  # B*N, state_feature_dim
        """

        # Predict actions
        # boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB

        actions_scores = self.read_actions(self_features)  # B*N, actions_num

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
        self.backbone_net = MyInception_v3(transform_input=False, pretrained=False)  # type: MyInception_v3
        self.backbone_dim = MyInception_v3.outputDim()
        self.backbone_size = MyInception_v3.outputSize(*self.imagesize)

        # embedding sequence
        self.mod_embed = nn.Sequential(
            nn.Linear(self.RoI_crop_size[0] * self.RoI_crop_size[0] * self.backbone_dim,
                      self.arch_para['person_fea_dim']),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.arch_para['person_fea_dim'])
            # nn.Dropout(p=self.arch_para['dropout_prob'])
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
        self.baselayer = SelfNet0(self.imagesize, self.RoI_crop_size, device=self.device, **self.arch_para)
        #   action sequence
        self.read_actions = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.actions_num),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.actions_num)
        )
        #   activity sequence
        self.read_activities = nn.Sequential(
            nn.Linear(self.arch_para['person_fea_dim'], self.activities_num),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.activities_num)
        )
        #  side awareness
        self.linklayer = SpectatorNet(self.arch_para['person_fea_dim'], self.arch_para['person_fea_dim'], 2,
                                      device=self.device, **self.arch_para)

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
        person_fea = self.baselayer(batch_data)    # (batch#num, feature)
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
                vec0 = torch.add(rate*vec1,(1-rate)*vec0)
            # pooling for group activity(feature2)
            # coef0 (source, sink, 1)
            coef0, _ = torch.max(coef0, dim=1)  # (source ,1)
            coef0 = torch.softmax(coef0, dim=0)  # (source, 1)
            group_fea = pooling_func(vec0, method=self.arch_para['pooling_method'],other=coef0).unsqueeze(dim=0)
            datum = datum + person_num[i]
            # scores result output
            action_scores.append(vec0)  # (batch#num, actions_num)
            activity_scores.append(group_fea)  # (batch#num, activities_num)

        action_scores = torch.cat(action_scores, dim=0)
        activity_scores = torch.cat(activity_scores, dim=0)

        activity_scores = self.read_activities(activity_scores)
        action_scores = self.read_actions(action_scores)

        return action_scores, activity_scores

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
        :type index: torch.tensor (batch, index_dim)
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
            coefficient = torch.sum(index1,
                                    dim=2)  # (batch_num, inter_num), this is the similar factor with each channel and label
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
            index0 = index.repeat(self.inter_num, 1)  # (batch,inter_num,index_dim)
            index0 = index.reshape(-1, self.index_dim)  # (batch*inter_num, index_dim)
            index1 = index0 + index1
            index0t = torch.t(index)  # (index_dim, batch)
            coefficient = torch.mm(index1, index0t)  # (batch(source)*inter_num, batch(sink))
            coefficient = coefficient.reshape(-1, self.inter_num, batch_size)  # (batch, inter_num, batch)
            coefficient = torch.transpose(coefficient, 1, 2)  # (batch,batch,inter_num)
            coefficient = torch.softmax(coefficient, dim=2)  # (batch(source),batch(sink),inter_num)
            coefficient = torch.transpose(coefficient, 0, 1)  # (batch(sink),batch(source),inter_num)

            # calculate result for each pairs of feature
            intern1 = intern1.repeat(batch_size, 1)  # (batch_num(sink), batch_num(source), inter_num*out_dim)
            intern1 = intern1.reshape(-1, batch_size, self.inter_num,
                                      self.output_dim)  # (batch_num, batch_num, inter_num, out_dim)
            coefficient = coefficient.unsqueeze(3)  # (batch,batch, inter_num,1)
            intern1 = torch.mul(intern1, coefficient)  # (batch_num, batch_num, inter_num, out_dim)
            intern1 = torch.sum(intern1, dim=2)  # (batch_num(sink), batch_num(source), out_dim)

            return intern1


# an isologue with biasnet, here the index part is the special position
class PosiBiasNet(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, index_dim=4, inter_num=8):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim  # absolute coordinate (x,y) [0,1]^2
        self.output_dim = output_dim
        self.device = device
        self.inter_num = inter_num

        # index vector generate
        baseAngel = math.pi / self.inter_num
        self.index_vector = torch.tensor(
            [[math.sin(baseAngel * i), math.cos(baseAngel * i)] for i in range(self.inter_num)]).to(device=device)

        # network layer
        self.layer1 = nn.Linear(self.input_dim, self.output_dim * self.inter_num)

    def forward(self, input, index):
        """
        :param parallel:
        :type index: torch.tensor (num, index_dim)
        :type input: torch.tensor
        """
        object_num = input.size()[0]
        """
        in parallel mode, one batch only contains a group of feature,  the result will generated between each pairs of features
        """
        # network forward
        intern1 = self.layer1(input)  # (ob_num, out_dim*inter_num)
        intern1 = intern1.to(device=self.device)

        # calculate the relative coordinate between each objects
        cood0 = index.reshape(-1, self.index_dim)  # (ob_num, 4[x1,y1,x2,y2])
        cood0 = cood0.to(device=self.device)

        op1 = torch.tensor([[0.5, 0], [0, 0.5], [0.5, 0], [0, 0.5]], device=self.device)

        cood0 = torch.mm(cood0, op1.to(dtype=cood0.dtype))  # (ob_num, 2[x,y])
        cood0 = cood0.repeat(object_num, 1, 1)  # (ob_num(repeat), ob_num, 2)
        cood0 = torch.transpose(cood0, 0, 1) - cood0  # (ob_num(source),ob_num(sink),2[Xsi-Xso,Ysi-Yso]))
        cood0 = F.normalize(cood0, p=2, dim=2)  # normalize cood0 with norm2

        # calculate the relative coefficient between relative coordinate and index vector
        coef0 = torch.mm(cood0.reshape(-1, 2),
                         torch.transpose(self.index_vector, 0, 1).to(dtype=cood0.dtype))  # (ob_num(so)*ob_num(si),inter_num)
        coef0 = F.softmax(coef0, dim=1)  # softmax in dim 2
        coef0 = coef0.reshape(object_num, object_num, -1)  # (ob_num(so),ob_num(si),inter_num)

        # doing batch matrix multiple
        intern1 = intern1.reshape(object_num, self.output_dim, -1)  # (ob_num(source), out_dim, inter_num)
        coef0 = torch.transpose(coef0, 1, 2)  # (ob_num(so),inter_num,ob_num(si))
        intern1 = torch.matmul(intern1, coef0.to(dtype=intern1.dtype))  # (ob_num(source),out_dim,ob_num(sink))
        intern1 = torch.transpose(intern1, 1, 2)  # (ob_num(source),ob_num(sink),out_dim)

        return intern1
