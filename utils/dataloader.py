# -*- coding: utf-8 -*-
import torch as tr
import numpy as np
from torch.autograd import Variable
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from utils.data_augment import data_aug


def read_mi_train(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    src_data = np.squeeze(Data_raw[args.ids, :, :, :])
    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (288, 22, 750)

    if args.aug:
        sample_size = src_data.shape[2]  # (288, 22, 750)
        # mult_flag, noise_flag, neg_flag, freq_mod_flag
        flag_aug = [True, True, True, True]
        src_data = np.transpose(src_data, (0, 2, 1))
        src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
        src_data = np.transpose(src_data, (0, 2, 1))
        src_label = tr.from_numpy(src_label).long()
    # print(src_data.shape, src_label.shape)  # (288*7, 22, 750)

    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, #feas)
    print(fea_tsm.shape, src_label.shape)

    return fea_tsm, src_label


def read_mi_test(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()

    # 288 * 22 * 750
    covar_src = Covariances(estimator=args.cov_type).transform(tar_data)
    fea_tsm = TangentSpace().fit_transform(covar_src)
    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, #feas)
    print(fea_tsm.shape, tar_label.shape)
    return fea_tsm, tar_label



