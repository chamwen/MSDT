# -*- coding: utf-8 -*-
import numpy as np
import argparse
import random
import torch as tr
import torch.optim as optim
import torch.onnx
import torch.nn as nn
import os.path as osp
import os
import torch.utils.data as Data
from utils import network
from utils.loss import CELabelSmooth
from utils.dataloader import read_mi_train
from utils.utils import lr_scheduler_full, cal_acc


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size
    tr.manual_seed(args.SEED)
    src_idx = range(y.shape[0])
    num_all = args.trial

    num_train = int(0.9 * num_all)
    id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
    id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

    source_tr = Data.TensorDataset(X[id_train, :], y[id_train])
    dset_loaders['source_tr'] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)

    source_te = Data.TensorDataset(X[id_val, :], y[id_val])
    dset_loaders['source_te'] = Data.DataLoader(source_te, batch_size=train_bs * 2, shuffle=False, drop_last=True)

    return dset_loaders


def train_source(args):
    X_src, y_src = read_mi_train(args)
    dset_loaders = data_load(X_src, y_src, args)

    if args.bottleneck == 50:
        netF, netC = network.backbone_net(args, 100, return_type='y')
    if args.bottleneck == 64:
        netF, netC = network.backbone_net(args, 128, return_type='y')
    base_network = nn.Sequential(netF, netC)
    optimizer = optim.SGD(base_network.parameters(), lr=args.lr)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders['source_tr'])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer, init_lr=args.lr, iter_num=iter_num, max_iter=max_iter)
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        # print(inputs_source.shape, labels_source.shape)  # [4, 253] [4]
        labels_source = tr.zeros(args.batch_size, args.class_num).cuda().scatter_(1, labels_source.reshape(-1, 1), 1)
        outputs_source = netC(netF(inputs_source))
        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_s_te = cal_acc(dset_loaders['source_te'], netF, netC)
            print('Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te))

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return acc_s_te


if __name__ == '__main__':
    # dataset can be downloaded and processed with MOABB in http://moabb.neurotechx.com/docs/datasets.html
    data_name_list = ['001-2014_2', '001-2014']
    data_name = data_name_list[0]
    if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288
    if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144

    args = argparse.Namespace(N=N, trial=trial_num, chn=chn, class_num=class_num,
                              batch_size=4, lr=0.01, epsilon=1e-05, layer='wn', max_epoch=100,
                              bottleneck=50, smooth=0.1, cov_type='oas')

    args.aug = 1
    args.data = data_name
    args.method = 'multiple'
    args.backbone = 'Net_ln2'
    args.validation = 'last'

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args.data_env = 'gpu' if tr.cuda.device_count() != 0 else 'local'
    args.SEED = 2020
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    args.input_dim = int(args.chn * (args.chn + 1) / 2)
    mdl_path = 'ckps/'
    args.output = mdl_path + args.data + '/source/'
    args.local_dir = r'/Users/xxx/MSDT_demo/'

    sub_acc_all = []
    for s in range(N):
        args.ids = s
        source_str = 'S' + str(s + 1)
        info_str = '\n========================== Within subject ' + source_str + ' =========================='
        print(info_str)

        args.name_src = source_str
        args.output_dir_src = osp.join(args.output, args.name_src)

        if not osp.exists(args.output_dir_src):
            os.system('mkdir -p ' + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            if args.data_env == 'gpu':
                os.mkdir(args.output_dir_src)
            elif args.data_env == 'local':
                os.makedirs(args.local_dir + args.output_dir_src)

        acc_sub = train_source(args)
        sub_acc_all.append(acc_sub)
    print(np.round(sub_acc_all, 3))
    print(np.round(np.mean(sub_acc_all), 3))
