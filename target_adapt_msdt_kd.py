# -*- coding: utf-8 -*-
import time

import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import os.path as osp
import torch.utils.data as Data

from utils import network, loss
from utils.dataloader import read_mi_test
from utils.utils import lr_scheduler, fix_random_seed, op_copy, cal_acc


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = tr.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def estimate_source_weight_epoch(outputs_test, args):
    loss_all = tr.ones(len(args.src), )
    for s in range(len(args.src)):
        softmax_out = nn.Softmax(dim=1)(tr.squeeze(outputs_test[:, s, :]))
        im_loss = tr.mean(loss.Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = tr.sum(-msoftmax * tr.log(msoftmax + args.epsilon))
        loss_all[s] = im_loss - gentropy_loss
    weights_domain = loss_all / tr.sum(loss_all)

    return weights_domain


def train_target_kd(args):
    num_src = len(args.src)
    # load pre-trained source models
    source_api_list = source_models_query(args)
    X_tar, y_tar = read_mi_test(args)
    dset_loaders = data_load(X_tar, y_tar, args)

    # define new student network
    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='y')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='y')

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    # STE weights
    outputs_all = tr.cat([source_api_list[s](X_tar.cuda()).unsqueeze(1) for s in range(num_src)], 1).cuda()
    if args.use_weight:
        domain_weight = estimate_source_weight_epoch(outputs_all, args)
        domain_weight = domain_weight.reshape([1, num_src, 1]).cuda()
    else:
        domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()

    max_iter = args.max_epoch * len(dset_loaders['target'])
    interval_iter = max_iter // 10
    iter_num = 0

    student_model = nn.Sequential(netF, netC).cuda()
    student_model.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["Target"])
        for i in range(len(dset_loaders["Target"])):
            data = iter_test.next()
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()

            # [4,8,2]
            outputs_all = tr.cat([source_api_list[s](inputs).unsqueeze(1) for s in range(num_src)], 1).cuda()
            preds_softmax = tr.softmax(outputs_all, dim=2)
            outputs = (preds_softmax * domain_weight).sum(dim=1).cuda()

            _, src_idx = torch.sort(outputs, 1, descending=True)
            if start_test:
                all_output = outputs.float()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels), 0)
        mem_P = all_output.detach()

    student_model.train()
    while iter_num < max_iter:
        try:
            inputs_target, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders['target'])
            inputs_target, _, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)

        # kd loss
        inputs_target = inputs_target.cuda()
        with torch.no_grad():
            outputs_target_by_source = mem_P[tar_idx, :]
            _, src_idx = torch.sort(outputs_target_by_source, 1, descending=True)
        outputs_target = student_model(inputs_target)
        softmax_out = torch.nn.Softmax(dim=1)(outputs_target)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(softmax_out.log(), outputs_target_by_source)

        # IM loss
        # for diverse preds H(Y|X)
        ent_loss = tr.mean(loss.Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        # for class balance H(Y)
        gentropy_loss = - tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))
        im_loss = ent_loss - gentropy_loss
        loss_all = kd_loss + im_loss

        if args.use_mix > 0:
            alpha = 0.3  # raw 0.3/0.2
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs_target.size()[0]).cuda()
            mixed_input = lam * inputs_target + (1 - lam) * inputs_target[index, :]
            mixed_output = (lam * outputs_target + (1 - lam) * outputs_target[index, :]).detach()

            # just offers the values of ft but needs no gradient optimization
            outputs_target_m = student_model(mixed_input)
            outputs_target_m = torch.nn.Softmax(dim=1)(outputs_target_m)
            mixup_loss = nn.CrossEntropyLoss()(outputs_target_m.log(), mixed_output.argmax(dim=1))
            mixup_loss.backward()

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            student_model.eval()
            test_acc = cal_acc(dset_loaders['Target'], netF, netC)
            student_model.train()
            print('Iter:{}/{}; Acc = {:.2f}%'.format(iter_num, max_iter, test_acc))

    return test_acc


def source_models_query(args):
    num_src = len(args.src)

    # load pre-trained source models, only for eval, without parameter sharing
    source_api_list = []
    # base network feature extract
    for i in range(num_src):
        if args.bottleneck == 50:
            netF, netC = network.backbone_net(args, 100, return_type='y')
        if args.bottleneck == 64:
            netF, netC = network.backbone_net(args, 128, return_type='y')

        modelpath = args.output_dir_src[i] + '/source_F.pt'
        netF.load_state_dict(tr.load(modelpath))
        modelpath = args.output_dir_src[i] + '/source_C.pt'
        netC.load_state_dict(tr.load(modelpath))
        single_mdl = nn.Sequential(netF, netC).cuda()
        single_mdl.eval()
        source_api_list.append(single_mdl)

    return source_api_list


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
    args.max_epoch = 30  # 30
    args.input_dim = int(args.chn * (args.chn + 1) / 2)
    args.validation = 'last'

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
    args.SEED = 2020
    fix_random_seed(args.SEED)
    tr.backends.cudnn.deterministic = True

    mdl_path = 'ckps/'
    args.output_src = mdl_path + args.data + '/source/'
    args.local_dir = r'/Users/xxx/MSDT_demo/'
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
        sub_acc_all[t] = train_target_kd(args)
        duration_all[t] = time.time() - t1
        print(f'Sub:{t:2d}, [{duration_all[t]:5.2f}], Acc: {sub_acc_all[t]:.4f}')
    print('Sub acc: ', np.round(sub_acc_all, 3))
    print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
    print('Avg duration: ', np.round(np.mean(duration_all), 3))
