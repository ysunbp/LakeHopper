import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import json
import sys
import numpy as np
import random
import torch

from sudowoodo_dataset_semtab import CADataset, SupCADataset

from dataset import SupDataset

from torch.utils import data
from train_script_stru import train # remember to modify the line 119 of train_script_stru

csv_all_path = './sudowoodo_semtab_all.csv'
csv_path = './sudowoodo_semtab_'

if __name__ == '__main__':
    for fold in range(1):
        default_lr = 2e-5
        default_f_lr = 5e-5
        parser = argparse.ArgumentParser()
        parser.add_argument("--max_len", type=int, default=128)
        parser.add_argument("--lm", type=str, default='bert') # bert
        parser.add_argument("--da", type=str, default='empty')
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=default_lr)
        parser.add_argument("--f_lr", type=float, default=default_f_lr)
        parser.add_argument("--fp16", dest="fp16", action="store_true")
        parser.add_argument('--projector', default='768', type=str,
                            metavar='MLP', help='projector MLP')
        parser.add_argument("--n_classes", type=int, default=275)
        parser.add_argument("--n_epochs", type=int, default=5)
        parser.add_argument("--clustering", type=bool, default=True)
        parser.add_argument("--n_ssl_epochs", type=int, default=5) # default 5
        parser.add_argument("--save_path", type=str, default='LakeHopper/checkpoints/sudo-v2s-after_contrast_transfer_prior_sato.pkl')
        parser.add_argument("--switching", type=bool, default=False) # switch training

        hp = parser.parse_args()

        # TODO: change the size of datasets into a hyper-parameter

        trainset_nolabel = CADataset(csv_all_path,
                                    lm=hp.lm,
                                    size=None,
                                    max_len=hp.max_len,
                                    da=hp.da) # data augmentation

        train_set = SupCADataset(csv_path, folds=[fold], flag = 'train', max_len=hp.max_len,
                        size=56,
                        lm=hp.lm,
                        da=None)

        valid_set = SupCADataset(csv_path, folds=[fold],flag = 'val',max_len=hp.max_len,
                        size=None,
                        lm=hp.lm,
                        da=None)

        test_set = SupCADataset(csv_path,folds=[fold] , flag = 'test',max_len=hp.max_len,
                        size=None,
                        lm=hp.lm,
                        da=None)
        print(hp.save_path)
        train(trainset_nolabel, train_set, valid_set, test_set, hp)
        print('finish fold', fold)