import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import math

# vector normalization
def vec_norm(input0):
    x = torch.pow(input0, 2.0)
    x = torch.sum(x, dim=-1, keepdim=True)
    # to prevent NaN appearing, replace all Inf and 0 with 1.
    x = torch.where(x==0,torch.ones(size=x.size(),dtype=x.dtype,device=x.device),x)
    x = torch.where(x==float('Inf'), torch.ones(size=x.size(), dtype=x.dtype, device=x.device), x)
    x = torch.sqrt(x)
    if np.any(np.isnan(x.cpu().detach().numpy())):
        a = 1
        pass
    return torch.div(input0, x)

def vec_squash(input0):
    input_norm = torch.norm(input0, dim=-1, keepdim=True)
    return input0 * input_norm / (1+input_norm**2)

# a softmax funtion with changeable factor
def softmax0(input0, power=10.0, dim=0):
    out = torch.pow(power, input0)
    summ = torch.sum(out, dim=dim, keepdim=True)
    divv = torch.div(out, summ)

    if np.any(np.isnan(divv.cpu().detach().numpy())):
        a = 1
        pass
    return divv

def softmaxS(input0, dim=0):
    x = torch.softmax(input0, dim=dim)
    ave = 1. / input0.size()[dim]
    x = torch.where(x<ave, torch.zeros(input0.size(),device=input0.device,dtype=input0.dtype),x)

    summa = x.sum(dim=dim, keepdim=True)
    x = torch.div(x, summa)

    return x

def noNegmax(input0, dim=-1):
    x = torch.where(input0>0, input0, torch.zeros(input0.size(), device=input0.device,dtype=input0.dtype))
    x = F.normalize(x, p=1, dim=dim)

    return  x
# dynamic routing algorithm
def routing(input0, weight0=None, times=3):
    """
    :param input0: (batch, num, fea)
    :param weight0: (batch, num)
    :param times:
    :return:
    """
    input0d = input0.detach()
    if weight0 is None:
        weight = torch.ones(input0.size()[0:2], device=input0.device, dtype=input0.dtype)
    else:
        weight = weight0.to(device=input0.device, dtype=input0.dtype)
    weight = F.softmax(weight * 10., dim=1)
    for i in range(times):
        weight1 = F.normalize(weight, p=1, dim=-1)
        vec = torch.mul(input0d, weight1.unsqueeze(2))  # (batch, num, fea)
        vec = torch.sum(vec, dim=1, keepdim=True)  # (batch, 1, fea)
        weight1 = torch.mul(input0d, vec)  # (batch, num, fea)
        weight1 = torch.sum(weight1, dim=-1)  # (batch,num)
        weight1 = F.softmax(weight1 * 10, dim=-1)
        weight = torch.add(weight, weight1)

    weight = F.normalize(weight, p=1, dim=-1)
    vec = torch.mul(input0, weight.unsqueeze(2))  # (batch, num, fea)
    vec1 = torch.sum(vec, dim=1)  # (batch, fea)

    # normalize output by 2d-norm
    vec1 = F.normalize(vec1, p=2, dim=-1)

    return vec1, weight

# dynamic routing algorithm for linkNet3
def routing_link3(input0, vec0, weight0=None, times=3):
    """
    :param input0: (batch, num, fea)
    :param weight0: (batch, num)
    :param times:
    :return:
    """
    input0d = input0.detach()
    vec0d = vec0.detach()
    if weight0 is None:
        weight = torch.mul(input0d, vec0d.unsqueeze(1))
        weight = torch.sum(weight, dim=-1)
    else:
        weight = weight0.to(device=input0.device, dtype=input0.dtype)
    weight = F.softmax(weight*10., dim=1)
    for i in range(times):
        weight1 = F.normalize(weight, p=1, dim=-1)
        vec = torch.mul(input0d, weight1.unsqueeze(2))    # (batch, num, fea)
        vec = torch.sum(vec, dim=1, keepdim=True)  # (batch, 1, fea)
        weight1 = torch.mul(input0d, vec)    # (batch, num, fea)
        weight1 = torch.sum(weight1, dim=-1)  # (batch,num)
        weight1 = F.softmax(weight1*10, dim=-1)
        weight = torch.add(weight, weight1)

    weight = F.normalize(weight, p=1, dim=-1)
    vec = torch.mul(input0, weight.unsqueeze(2))  # (batch, num, fea)
    vec1 = torch.sum(vec, dim=1)  # (batch, fea)

    #vec2 = vec0  # (batch, fea)
    vec2 = (vec0 + vec1)/2.  # (batch, fea)
    # calculate the loss KL divergence
    loss = F.kl_div((vec1+1e-12).log(), vec0, reduction='sum')
    # calculate the coefficient matrix
    coef0 = torch.mul(vec0d, vec2)
    coef0 = torch.sum(coef0, dim=-1).detach()  # (batch)
    coef1 = torch.mul(input0d, vec2.unsqueeze(1))
    coef1 = torch.sum(coef1, dim=-1).detach()  # (batch,num(batch-1))
    batch = vec0.size()[0]
    coef1 = torch.cat((coef0[:batch-1].unsqueeze(1), coef1.reshape(batch-1,batch)), dim=1)  # (batch-1, batch+1)
    coef1 = torch.cat((coef1.reshape(-1),coef0[-1].unsqueeze(0)), dim=0)  # (batch^2,1)

    return vec2, coef1.reshape(batch,batch), loss

def rela_entropy(input0):
    """
    :param input0: [(batch, weight),...]
    :return:
    """
    result = []
    for _ in input0:
        weight = F.normalize(_,p=1,dim=-1)
        num = _.size()[-1]
        max_entropy = torch.tensor([-(math.log(1/num))]).to(device=_.device)
        real_entropy = torch.sum(-(weight * ((weight+1e-9).log())),dim=-1)
        real_entropy = torch.mean(real_entropy,dim=0,keepdim=True)
        result.append( real_entropy / max_entropy)
    result = torch.cat(result)
    return result

def group_max(input0, coef0, max_n=3):
    """
    :param input0:  (batch, fea)
    :param cooef0:  (batch, batch)
    :param max_n:
    :return:  (n, fea)
    """
    batch_num = input0.size()[0]
    dia_index = torch.tensor([i * (batch_num + 1) for i in range(batch_num)], requires_grad=False, device=coef0.device)
    dia_element = torch.index_select(coef0.reshape(-1,1), 0, dia_index)
    coef_sum = torch.sum(coef0, dim=0)
    coef_sum = coef_sum - dia_element.reshape(-1)
    sort, index = torch.sort(coef_sum, dim=-1, descending=True)
    max_index = index.reshape(-1)[0:max_n]
    out = torch.index_select(input0, 0, max_index)

    return out

def thread_max(input0, factor=0.1):
    new = torch.where(input0 > factor, input0, torch.zeros(input0.size(), dtype=input0.dtype, device=input0.device))
    # normalize the value's sum into 1


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

def pooling_func(input, method='ave', other=None):
    """
    this is a pooling lib including lot of pooling method
    :param input: torch.tensor [num, input_fea_dim]
    :param method: 'ave' 'max' 'coef'
    :param other:
    :return: input feature dim
    """
    if method == 'ave':
        # doing average pooling
        num = input.size()[0]
        return torch.sum(input,dim=0) / num
    elif method == 'max':
        # doing maximum pooling
        return torch.max(input,dim=0)[0]
    elif method == 'coef':
        """
        input: [num, dim]
        other: [num,1]
        """
        return torch.sum(torch.mul(input, other), dim=0)
    else:
        assert False, 'not this pooling method found, you silly B'


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


def adjust_lr(optimizer, lr_plan, logger):
    logger.fPrint('change learning rate:' + str(lr_plan))
    if 0 in lr_plan:
        for i in optimizer.param_groups:
            i['lr'] = lr_plan[0]
            if lr_plan[0] == 0:
                for each_para in i["params"]:
                    each_para.requires_grad = False
            else:
                for each_para in i["params"]:
                    each_para.requires_grad = True

    else:
        for param_group in lr_plan:
            optimizer.param_groups[param_group-1]['lr'] = lr_plan[param_group]
            if lr_plan[param_group] == 0:
                for each_para in optimizer.param_groups[param_group-1]["params"]:
                    each_para.requires_grad = False
            else:
                for each_para in optimizer.param_groups[param_group-1]["params"]:
                    each_para.requires_grad = True


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

class MaxItem(object):
    def __init__(self):
        self.maxitem = 0
        self.maxnum = None
    def update(self, item, num):
        if self.maxitem < item:
            self.maxitem = item
            self.maxnum = num

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
            self.all_num_each[label_tensor[i]] = self.all_num_each[label_tensor[i]]+1.0
            if correct_tensor[i]:
                self.correct_num_each[result_tensor[i]] += 1.0
        self.correct_rate_each = ((self.correct_num_each / self.all_num_each).cpu()*100)

        self.correct_num = int(torch.sum(self.correct_num_each))
        self.all_num = int(torch.sum(self.all_num_each))
        self.correct_rate = (self.correct_num / self.all_num) * 100

class GeneralAverageMeterTensor(object):
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
        label_tensor = label_tensor0.int()
        result_tensor = result_tensor0.detach()
        for i in range(label_tensor.size()[0]):

            self.all_num_each[label_tensor[i]] = self.all_num_each[label_tensor[i]]+1.0
            self.correct_num_each[label_tensor[i]] += result_tensor[i]

        self.correct_rate_each = (self.correct_num_each / self.all_num_each).cpu()

        self.correct_num = (torch.sum(self.correct_num_each))
        self.all_num = (torch.sum(self.all_num_each))
        self.correct_rate = (self.correct_num / self.all_num)

class CoefMatrixMeter(object):

    def __init__(self, actions_num, activity_num=None):
        self.actions_num = actions_num
        if activity_num == None:
            self.activity_num = self.actions_num
        else:
            self.activity_num = activity_num
        self.coef_value = torch.zeros([self.activity_num, self.actions_num],
                                      dtype=torch.float)
        self.coef_num = self.coef_value + 1e-10
        self.coef_rate = self.coef_value / self.coef_num

    def update(self, coef0, action_in0, activity_in0=None, mode = 0):
        # coef0 list[(target, source), ...]
        action_in = action_in0.detach().cpu()
        if mode == 0:
            activity_in = action_in
        else:
            activity_in = activity_in0.detach().cpu()
        datum = 0
        batch_datum = 0
        for batch in coef0:
            batch = batch.detach().cpu()
            numi = batch.size()[0]
            numj = batch.size()[1]
            action_batch = action_in[datum:datum+numi]
            if mode == 0:
                activity_batch = action_batch
            else:
                activity_batch = activity_in[batch_datum].reshape(1).repeat(numi)
            for i in range(numi):
                for j in range(numj):
                    self.coef_value[activity_batch[i]][action_batch[j]] += batch[i][j]
                    self.coef_num[activity_batch[i]][action_batch[j]] += 1
            datum += numi
            batch_datum += 1
        # return (source, target)
        self.coef_rate = self.coef_value / self.coef_num

class CorelationMeter(object):

    def __init__(self, num):
        self.class_all = torch.zeros(num, 1, dtype=torch.float)
        self.class_all += 1e-12
        self.class_each = torch.zeros(num,num, dtype=torch.float)

        self.class_acc = self.class_each / self.class_all

    def update(self, label, predict):
        batch = label.size()[0]
        for i in range(batch):
            self.class_all[label[i], 0] += 1
            self.class_each[label[i], predict[i]] += 1

        self.class_acc = self.class_each / self.class_all

class CrossrelaMeter(object):

    def __init__(self, num1, num2):
        self.activity_each = torch.zeros((num1, 1), dtype=torch.float) + 1e-12
        self.action_each = torch.zeros((num1, num2), dtype=torch.float)
        self.acc_each = self.action_each / self.activity_each

    def update(self, tensor1, tensor2):
        pass

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
