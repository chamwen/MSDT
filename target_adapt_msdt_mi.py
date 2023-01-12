# -*- coding: utf-8 -*-
import numpy as np
import argparse
import time
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import os.path as osp
import torch.utils.data as Data

from utils import network, loss
from utils.dataloader import read_mi_test
from utils.utils import lr_scheduler, fix_random_seed, op_copy, cal_acc_multi


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = tr.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def estimate_source_weight_epoch(batch_tar, netF_list, netC_list, args):
    loss_all = tr.ones(len(args.src), )
    for s in range(len(args.src)):
        features_test = netF_list[s](batch_tar)
        outputs_test = netC_list[s](features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        im_loss = tr.mean(loss.Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = tr.sum(-msoftmax * tr.log(msoftmax + args.epsilon))
        loss_all[s] = im_loss - gentropy_loss
    weights_domain = loss_all / tr.sum(loss_all)

    return weights_domain


def train_target(args):
    X_tar, y_tar = read_mi_test(args)
    dset_loaders = data_load(X_tar, y_tar, args)
    num_src = len(args.src)

    # base network feature extract
    netF_list, netC_list = [], []
    for i in range(num_src):
        if args.bottleneck == 50:
            netF, netC = network.backbone_net(args, 100, return_type='y')
        if args.bottleneck == 64:
            netF, netC = network.backbone_net(args, 128, return_type='y')
        netF_list.append(netF)
        netC_list.append(netC)

    param_group = []
    for i in range(num_src):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        netF_list[i].load_state_dict(tr.load(modelpath))
        netF_list[i].eval()
        modelpath = args.output_dir_src[i] + '/source_C.pt'
        netC_list[i].load_state_dict(tr.load(modelpath))
        netC_list[i].eval()

        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

    ###################################################################################
    loss1 = loss.InstanceEntropyLoss().cuda()
    loss2 = loss.BatchEntropyLoss().cuda()
    loss3 = loss.source_inconsistency_loss().cuda()
    losses = (loss1, loss2, loss3)

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    for epoch in range(1, args.max_epoch + 1):
        t1 = time.time()
        weights_domain = estimate_source_weight_epoch(X_tar.cuda(), netF_list, netC_list, args)
        # print('\n\n', weights_domain)
        weights_domain = weights_domain.reshape([1, num_src, 1]).cuda()

        update_msdt(dset_loaders, netF_list, netC_list, optimizer, losses, weights_domain)
        test_acc = cal_acc_multi(dset_loaders['Target'], netF_list, netC_list, args, weights_domain, None)
        duration = time.time() - t1
        print(f'Epoch:{epoch:2d}/{args.max_epoch:2d} [{duration:5.2f}], Acc: {test_acc:.2f}')

    return test_acc


def update_msdt(dset_loaders, netF_list, netC_list, optimizer, losses, weight_epoch):
    instance_entropy, batch_entropy, si_variance = losses
    max_iter = len(dset_loaders["target"])
    num_src = len(args.src)

    iter_num = 0
    while iter_num < max_iter:
        try:
            inputs_target, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders['target'])
            inputs_target, _, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue
        inputs_target = inputs_target.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # output, domain weight, weighted output
        if args.use_weight:
            domain_weight = weight_epoch.detach()
        else:
            domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()
        # weights_domain = np.round(tr.squeeze(domain_weight, 0).t().flatten().cpu().detach(), 3)
        # print(type(domain_weight), type(weights_domain))  # [1, 3, 1], [3]

        outputs_all = tr.cat([netC_list[i](netF_list[i](inputs_target)).unsqueeze(1) for i in range(num_src)], 1).cuda()
        preds = tr.softmax(outputs_all, dim=2)
        outputs_all_w = (preds * domain_weight).sum(dim=1).cuda()
        # print(outputs_all.shape, preds.shape, domain_weight.shape, outputs_all_w.shape)
        # [4, 8, 4], [4, 8, 4], [1, 8, 1], [4, 4]

        # loss1: instance entropy loss
        instance_entropy_loss, _ = instance_entropy(outputs_all)

        # loss2: batch entropy loss
        batch_entropy_loss, _ = batch_entropy(outputs_all)

        # loss3: source models inconsistency loss
        si_loss = si_variance(outputs_all)

        loss_all = 0.1 * si_loss + instance_entropy_loss + batch_entropy_loss

        if args.use_mix:
            alpha = 0.3  # raw 0.3/0.2
            lam = np.random.beta(alpha, alpha)
            index = tr.randperm(inputs_target.size()[0]).cuda()
            mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
            mixed_output = (lam * outputs_all_w + (1 - lam) * outputs_all_w[index, :]).detach()

            for src in range(num_src):
                src_net_tmp = nn.Sequential(netF_list[src], netC_list[src]).cuda()
                outputs_target_m = src_net_tmp(mixed_input)
                outputs_target_m = tr.nn.Softmax(dim=1)(outputs_target_m)
                mixup_loss = nn.CrossEntropyLoss()(outputs_target_m.log(), mixed_output.argmax(dim=1))
                mixup_loss.backward()

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


if __name__ == '__main__':
    # dataset can be downloaded and processed with MOABB in http://moabb.neurotechx.com/docs/datasets.html
    data_name_list = ['001-2014_2', '001-2014']
    data_name = data_name_list[0]
    if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288
    if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144

    args = argparse.Namespace(N=N, trial=trial_num, chn=chn, class_num=class_num,
                              lr=0.01, lr_decay1=0.1, lr_decay2=1.0, ent_par=1.0, epsilon=1e-05, layer='wn',
                              bottleneck=50, tempt=1, eps_th=0.9, use_weight=1, use_mix=1, cov_type='oas')

    args.data = data_name
    args.method = 'msdt'
    args.backbone = 'Net_ln2'
    args.batch_size = 4  # 4
    args.max_epoch = 10  # 10
    args.input_dim = int(args.chn * (args.chn + 1) / 2)
    args.validation = 'last'

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
    args.SEED = 2020
    fix_random_seed(args.SEED)
    tr.backends.cudnn.deterministic = True

    mdl_path = 'ckps/'
    args.output_src = mdl_path + args.data + '/source/'
    print(args)

    sub_acc_all = np.zeros(N)
    duration_all = np.zeros(N)
    for t in range(N):
        args.idt = t
        target_str = 'S' + str(t + 1)
        info_str = '\n========================== Transfer to ' + target_str + ' =========================='
        print(info_str)

        args.src = ['S' + str(i + 1) for i in range(N)]
        args.src.remove(target_str)

        args.output_dir_src = []
        for i in range(len(args.src)):
            args.output_dir_src.append(osp.join(args.output_src, args.src[i]))

        t1 = time.time()
        sub_acc_all[t] = train_target(args)
        duration_all[t] = time.time() - t1
        print(f'Sub:{t:2d}, [{duration_all[t]:5.2f}], Acc: {sub_acc_all[t]:.4f}')
    print('Sub acc: ', np.round(sub_acc_all, 3))
    print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
    print('Avg duration: ', np.round(np.mean(duration_all), 3))
