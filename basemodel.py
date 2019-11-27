import torch
import torch.nn as nn
import torch.nn.functional as F
import config

import backbone
from roi_align.roi_align import RoIAlign      # RoIAlign module

class selfNet(nn.Module):
    """
    main module of base model for the volleyball
    """

    def __init__(self, cfg: config.Config):
        super(selfNet, self).__init__()
        self.cfg = cfg
        NFB = self.cfg.num_features_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb = nn.Linear(K * K * D, NFB)
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB, self.cfg.num_activities)

        for m in self.modules():  # network initial for linear layer
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict': self.fc_emb.state_dict(),
            'fc_actions_state_dict': self.fc_actions.state_dict(),
            'fc_activities_state_dict': self.fc_activities.state_dict()
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
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)

        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # ActNet
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        #         features_multiscale.requires_grad=False

        # RoI Align
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B * T * N, -1)  # B*T*N, D*K*K

        # Embedding to hidden state
        boxes_features = self.fc_emb(boxes_features)  # B*T*N, NFB
        boxes_features = F.relu(boxes_features)
        boxes_features = self.dropout_emb(boxes_features)

        boxes_states = boxes_features.reshape(B, T, N, NFB)

        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFB)  # B*T*N, NFB

        actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)  # B, T, NFB
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFB)  # B*T, NFB

        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        if T != 1:
            actions_scores = actions_scores.reshape(B, T, N, -1).mean(dim=1).reshape(B * N, -1)
            activities_scores = activities_scores.reshape(B, T, -1).mean(dim=1)

        return actions_scores, activities_scores