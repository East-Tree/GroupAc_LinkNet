from backbone import *
from utils import *

import backbone
from torchvision import ops  # RoIAlign module


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
        self.backbone_net = backbone.MyInception_v3(transform_input=False, pretrained=True)
        self.backbone_dim = backbone.MyInception_v3.outputDim()
        self.backbone_size = backbone.MyInception_v3.outputSize(*self.imagesize)

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
        NFB = self.arch_para['person_fea_dim']

        # Reshape the input data
        images_in_flat = torch.stack(images_in).to(device=self.device)  # B, 3, H, W

        #    reshape the bboxs coordinates into [K,(index,x1,y1,x2,y2)]
        boxes_in_flat = boxes_in
        boxes_in_index = []
        for i in range(B):
            box_num = boxes_in_flat[i].size()[0]
            boxes_in_index.append([[i]] * box_num)
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0)
        boxes_in_index = torch.tensor(boxes_in_index)
        #    cat flat and index together
        boxes_in_flat = torch.cat([boxes_in_index, boxes_in_flat], dim=1)
        #    convert the origin coordinate(rate) into backbone feature scale coordinate(absolute int)
        operator = torch.tensor([1, OW, OH, OW, OH])
        boxes_in_flat = boxes_in_flat * operator
        boxes_in_flat = boxes_in_flat.int().to(device=self.device)

        # Use backbone to extract features of images_in
        # Pre-precess first  normalized to [-1,1]
        images_in_flat = prep_images(images_in_flat)

        outputs = self.backbone(images_in_flat)

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
        boxes_features = ops.roi_align(features_multiscale, boxes_in_flat, K)  # B*N, D, K, K,

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
