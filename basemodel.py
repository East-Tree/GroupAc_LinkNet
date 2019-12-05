import torch.nn as nn
import torch.nn.functional as F
import torch
import config
import utils

import backbone
from torchvision import ops   # RoIAlign module

class selfNet(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg: config.Config1, device = None):
        super(selfNet, self).__init__()
        self.cfg = cfg
        self.person_feature_dim = self.cfg.individual_dim
        self.RoI_crop_size = self.cfg.crop_size[0]
        self.actions_num = self.cfg.actions_num
        self.device = device

        # here determine the backbone net
        self.backbone_net = backbone.MyInception_v3(transform_input=False, pretrained=True)
        self.backbone_dim = backbone.MyInception_v3.outputDim()
        self.backbone_size = backbone.MyInception_v3.outputSize(*self.cfg.imageSize)


        self.fc_emb = nn.Linear(self.RoI_crop_size * self.RoI_crop_size * self.backbone_dim, self.person_feature_dim)
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_actions = nn.Linear(self.person_feature_dim, self.actions_num)

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
        images_in, boxes_in = batch_data

        # read config parameters
        B = len(images_in) # batch size
        K = self.RoI_crop_size
        D = self.backbone_dim
        H, W = self.cfg.imageSize
        OH, OW = self.backbone_size
        NFB = self.cfg.individual_dim

        # Reshape the input data
        images_in_flat = torch.stack(images_in)  # B, 3, H, W

        #    reshape the bboxs coordinates into [K,(index,x1,y1,x2,y2)]
        boxes_in_flat = boxes_in
        boxes_in_index = []
        for i in range(B):
            box_num = boxes_in_flat[i].size()[0]
            boxes_in_index.append([[i]]*box_num)
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0)
        boxes_in_index = torch.tensor(boxes_in_index)
        #    cat flat and index together
        boxes_in_flat = torch.cat([boxes_in_index, boxes_in_flat], dim=1)
        #    convert the origin coordinate(rate) into backbone feature scale coordinate(absolute int)
        operator = torch.tensor([1,OW,OH,OW,OH])
        boxes_in_flat = boxes_in_flat * operator
        boxes_in_flat = boxes_in_flat.int()

        # Use backbone to extract features of images_in
        # Pre-precess first  normalized to [-1,1]
        images_in_flat = utils.prep_images(images_in_flat)

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

        boxes_features = boxes_features.reshape(-1, D*K*K)  # B*N, D*K*K
        boxes_features.to(device=self.device)

        # Embedding to hidden state
        boxes_features = self.fc_emb(boxes_features)  # B*N, NFB
        boxes_features = F.relu(boxes_features)
        boxes_features = self.dropout_emb(boxes_features)


        # Predict actions
        boxes_states_flat = boxes_features.reshape(-1, NFB)  # B*N, NFB

        actions_scores = self.fc_actions(boxes_states_flat)  # B*N, actions_num

        return actions_scores
