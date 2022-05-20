import os
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer


### IO
def check_dir(d):
    if not os.path.exists(d):
        print("Directory {} does not exist. Exit.".format(d))
        exit(1)

def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)



def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adagrad':
        # use my own adagrad to allow for init accumulator value
        return torch.optim.Adagrad(parameters, lr=lr, initial_accumulator_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2, betas=(0.9, 0.98), eps=1e-9) # use default lr
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var

def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)

    if mask is not None: 
        scores = scores.masked_fill(mask == 0, -1e10)

    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


def make_src_mask(src, vocab, device):
    src_mask = (src != vocab.word2id['<pad>']).unsqueeze(1).unsqueeze(2).to(device)
    return src_mask

def make_trg_mask(trg, vocab, device):

    trg_pad_mask =  (trg != vocab.word2id['<pad>']).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)
