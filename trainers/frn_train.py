import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss, BCEWithLogitsLoss, BCELoss
from center_loss import CenterLoss

def auxrank(support):
    way = support.size(0)
    shot = support.size(1)
    support = support/support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2-way)//2).long().cuda()
    L2 = torch.zeros((way**2-way)//2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(0, L1) # s^2-s, s, d
    s2 = support.index_select(0, L2) # s^2-s, s, d
    dists = s1.matmul(s2.permute(0,2,1)) # s^2-s, s, s
    assert dists.size(-1)==shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum().mul(.03)


def default_train(train_loader, model, optimizer, iter_counter):

    way = model.way
    query_shot = model.shots[-1]

    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()
    center_loss = CenterLoss(num_classes=10, feat_dim=2, use_gpu=True)
    avg_acc = 0
    avg_loss = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1
        inp = inp.cuda()
        log_prediction, s, features = model(inp)
        s = s[:way]
        frn_loss = criterion(log_prediction, target)
        aux_loss = auxrank(s)
        centerLoss = center_loss(features, target) * 0.05

        loss = frn_loss + aux_loss + centerLoss 
        avg_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way
        
        avg_acc += acc

    avg_loss = avg_loss / (i + 1)
    avg_acc = avg_acc / (i + 1)

    return iter_counter, avg_acc, avg_loss