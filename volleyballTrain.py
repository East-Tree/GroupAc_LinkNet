import torch
import torch.nn.functional as F

import volleyballDataset
import config

cfg = config.Config()
dataset = volleyballDataset.VolleyballDataset(cfg)
cfg.actions_num, cfg.activities_num = dataset.classCount()

def train1volleyball(data_loader, model, device, optimizer, epoch=0, cfg=None):

    for batch_data in data_loader:
        model.train()
        # model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size, num_frames))

        actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
        activities_in = activities_in[:, 0].reshape((batch_size,))

        # forward
        actions_scores, activities_scores = model((batch_data[0], batch_data[1]))

        # Predict actions
        actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
        actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        actions_labels = torch.argmax(actions_scores, dim=1)
        actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

        # Predict activities
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        activities_labels = torch.argmax(activities_scores, dim=1)
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

        # Get accuracy
        actions_accuracy = actions_correct.item() / actions_scores.shape[0]
        activities_accuracy = activities_correct.item() / activities_scores.shape[0]

        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss = activities_loss + cfg.actions_loss_weight * actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return train_info
