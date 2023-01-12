# -*- coding: utf-8 -*-
import numpy as np
import random
import torch as tr
import torch.nn as nn


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy * 100


def cal_acc_multi(loader, netF_list, netC_list, args, weight_epoch=None, netG_list=None):
    num_src = len(netF_list)
    for i in range(len(netF_list)): netF_list[i].eval()

    if args.use_weight:
        if args.method == 'msdt':
            domain_weight = weight_epoch.detach()
            # tmp_weight = np.round(tr.squeeze(domain_weight, 0).t().cpu().detach().numpy().flatten(), 3)
            # print('\ntest domain weight: ', tmp_weight)
    else:
        domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()

    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs, labels = data[0].cuda(), data[1]
            outputs_all = tr.cat([netC_list[i](netF_list[i](inputs)).unsqueeze(1) for i in range(num_src)], 1).cuda()
            preds = tr.softmax(outputs_all, dim=2)
            outputs_all_w = (preds * domain_weight).sum(dim=1).cuda()

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    for i in range(len(netF_list)): netF_list[i].train()

    return accuracy * 100

