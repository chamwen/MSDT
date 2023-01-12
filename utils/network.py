# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net_ln2(nn.Module):
    def __init__(self, n_feature, n_hidden, bottleneck_dim):
        super(Net_ln2, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, bottleneck_dim)
        self.fc2.apply(init_weights)
        self.ln2 = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        x = x.view(x.size(0), -1)
        return x


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        y = self.fc(x)
        return y


class feat_classifier_xy(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier_xy, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        y = self.fc(x)
        return x, y


def backbone_net(args, n_hidden, return_type='y'):
    if args.backbone == 'Net_ln2':
        netF = Net_ln2(args.input_dim, n_hidden, args.bottleneck).cuda()

    if return_type == 'y':
        netC = feat_classifier(class_num=args.class_num, bottleneck_dim=args.bottleneck, type=args.layer).cuda()
    if return_type == 'xy':
        netC = feat_classifier_xy(class_num=args.class_num, bottleneck_dim=args.bottleneck, type=args.layer).cuda()

    return netF, netC

