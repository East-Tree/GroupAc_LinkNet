import torch
import time
import os


def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images, 0.5)
    images = torch.mul(images, 2.0)

    return images


def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx = X.pow(2).sum(dim=1).reshape((-1, 1))
    ry = Y.pow(2).sum(dim=1).reshape((-1, 1))
    dist = rx - 2.0 * X.matmul(Y.t()) + ry.t()
    return torch.sqrt(dist)


def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    return torch.sqrt(dist)


def sincos_encoding_2d(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N = positions.shape[0]

    d = d_emb // 2

    idxs = [np.power(1000, 2 * (idx // 2) / d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)

    idxs = idxs.repeat(N, 2)  # N, d_emb

    pos = torch.cat([positions[:, 0].reshape(-1, 1).repeat(1, d), positions[:, 1].reshape(-1, 1).repeat(1, d)], dim=1)

    embeddings = pos / idxs

    embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])  # dim 2i
    embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])  # dim 2i+1

    return embeddings


def print_log(file_path, *args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, k, ': ', v)
    print_log(cfg.log_path, '======================End=======================')


def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d' % (phase, info['epoch']))

    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
        info['activities_acc'], info['loss'], info['time']))


def adjust_lr(optimizer, new_lr, logger):
    logger.fPrint('change learning rate: %s' % new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def label_gather(cate_size, obj_tensor, res_tensor):
    ob = torch.zeros(cate_size)
    tensor = obj_tensor * res_tensor


class Logger(object):
    def __init__(self, path):
        if os.path.exists(path):
            self.logPath = path + '/Logger.txt'
            with open(self.logPath, 'w') as f:
                f.write("the logger file have been created" + '\n')
        else:
            assert False, "can not find logger, you silly B"

    def fPrint(self, message):
        with open(self.logPath, 'a') as f:
            f.write(str(message) + '\n')
        print(message)

    def getPath(self):
        return self.logPath


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterTensor(object):
    """
    Computes the average value
    """

    def __init__(self, actions_num):
        self.actions_num = actions_num

        self.correct_num_each = torch.zeros(actions_num, dtype=torch.float)
        self.all_num_each = torch.zeros(actions_num, dtype=torch.float) + 1e-10
        self.correct_rate_each = self.correct_num_each / self.all_num_each

        self.correct_num = 0
        self.all_num = 0
        self.correct_rate = 0

    def reset(self, actions_num=None):
        if actions_num is None:
            this_actions_num = self.actions_num
        else:
            this_actions_num = actions_num
        self.correct_num_each = torch.zeros(actions_num, dtype=torch.float)
        self.all_num_each = torch.zeros(actions_num, dtype=torch.float) + 1e-10
        self.correct_rate_each = self.correct_num_each / self.all_num_each
        self.correct_num = 0
        self.all_num = 0
        self.correct_rate = 0

    def update(self, result_tensor0, label_tensor0):
        """

        :param result_tensor: actions mark for predicted result
        :param label_tensor:  actions mark for ground truth
        :result: renew each variable
        """
        result_tensor = result_tensor0.int()
        label_tensor = label_tensor0.int()
        correct_tensor = torch.eq(result_tensor, label_tensor)  # bool type
        for i in range(correct_tensor.size()[0]):
            self.all_num_each[label_tensor[i]] += 1.0
            if correct_tensor[i]:
                self.correct_num_each[result_tensor[i]] += 1.0
        self.correct_rate_each = ((self.correct_num_each / self.all_num_each).cpu().numpy()*100)

        self.correct_num = int(torch.sum(self.correct_num_each))
        self.all_num = int(torch.sum(self.all_num_each))
        self.correct_rate = (self.correct_num / self.all_num) * 100



class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time
