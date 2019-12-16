import torch
import torch.nn.functional as F
from utils import *

import config
from tqdm import *


def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def VolleyballEpoch(mode, data_loader, model, device, cfg=None, optimizer=None, epoch=0):

    actions_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()

    if mode == 'train':
        print("Training in epoch %s" % epoch)
        for batch_data in tqdm(data_loader):

            # model.apply(set_bn_eval)

            batch_size = len(batch_data[0])

            # reshape the action label into tensor(B*N)
            actions_in = torch.cat(batch_data[2],dim=0)
            actions_in = actions_in.reshape(-1).to(device=device)

            # forward
            actions_scores = model((batch_data[0], batch_data[3]))  # tensor(B*N, actions_num)

            # Predict actions
            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
            actions_labels = torch.argmax(actions_scores, dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

            # Get accuracy
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]

            actions_meter.update(actions_accuracy, batch_size)

            # Total loss
            total_loss = cfg.actions_loss_weight * actions_loss
            loss_meter.update(total_loss.item(), batch_size)

            # Optim
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        info = {
            'mode': mode,
            'time': epoch_timer.timeit(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'actions_acc': actions_meter.avg * 100
        }
    elif mode == 'test':
        print("Testing in test dataset")
        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                # model.apply(set_bn_eval)

                batch_size = len(batch_data[0])

                # reshape the action label into tensor(B*N)
                actions_in = torch.cat(batch_data[2], dim=0)
                actions_in = actions_in.reshape(-1).to(device=device)

                # forward
                actions_scores = model((batch_data[0], batch_data[3]))  # tensor(B*N, actions_num)

                # Predict actions
                actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
                actions_labels = torch.argmax(actions_scores, dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

                # Get accuracy
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]

                actions_meter.update(actions_accuracy, batch_size)

                # Total loss
                total_loss = cfg.actions_loss_weight * actions_loss
                loss_meter.update(total_loss.item(), batch_size)

        info = {
            'mode': mode,
            'time': epoch_timer.timeit(),
            'loss': loss_meter.avg,
            'actions_acc': actions_meter.avg * 100
        }

    return info
