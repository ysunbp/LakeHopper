# This file contains the structural alignment dataset classes
import os
import csv
import numpy as np
import json
import torch
import random
from torch.utils import data
from transformers import AutoTokenizer
import pandas as pd


lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class CADataset(data.Dataset): #copy from BTDataset class
    """Dataset for pre-training"""

    def __init__(self,
                    csv_dir,
                    max_len=512,
                    size=None,
                    lm='bert',
                    da='all'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.instances= []
        self.rel_instances = []
        self.max_len = max_len
        self.size = size # dataset size
        
        pd_list = []
        folds = [0,1,2,3,4]
        for fold in folds:
            csv_path = csv_dir
            cur_df = pd.read_csv(csv_path)
            cur_df = cur_df.dropna()
            cur_df = cur_df[['text','c_text']]
            pd_list.append(cur_df)

        merged_df = pd.concat(pd_list, axis = 0)
        self.instances = list(merged_df.text.values)
        self.rel_instances = list(merged_df.c_text.values)

        #print(self.instances)

        if size is not None:
            if size > len(self.instances):
                N = size // len(self.instances) + 1
                self.instances = (self.instances * N)[:size] # over sampling
                self.rel_instances = (self.rel_instances * N)[:size]
            else:
                indices = [i for i in range(len(self.instances))]
                selected_indices = random.sample(indices, size)
                out_instances = []
                out_rel = []
                for index in selected_indices:
                    out_instances.append(self.instances[index])
                    out_rel.append(self.rel_instances[index])
                self.instances = out_instances
                self.rel_instances = out_rel


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.instances)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

            Args:
                idx (int): the index of the item

            Returns:
                List of int: token ID's of the 1st entity
                List of int: token ID's of the 2nd entity
                List of int: token ID's of the two entities combined
                int: the label of the pair (0: unmatch, 1: match)
        """
        #print('A')
        A = self.instances[idx]
        # combine with the deletion operator
        #print('B')
        B = self.rel_instances[idx]
           
        # left

        yA = self.tokenizer.encode(text=A,
                                    max_length=self.max_len,
                                    truncation=True)
        yB = self.tokenizer.encode(text=B,
                                    max_length=self.max_len,
                                    truncation=True)
        return yA, yB

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

            Args:
                batch (list of tuple): a list of dataset items

            Returns:
                LongTensor: x1 of shape (batch_size, seq_len)
                LongTensor: x2 of shape (batch_size, seq_len).
                            Elements of x1 and x2 are padded to the same length
        """
        yA, yB = zip(*batch)

        maxlen = max([len(x) for x in yA])
        maxlen = max(maxlen,max([len(x) for x in yB]))

        yA = [xi + [0]*(maxlen - len(xi)) for xi in yA]
        yB = [xi + [0]*(maxlen - len(xi)) for xi in yB]

        return torch.LongTensor(yA), \
                torch.LongTensor(yB)


class SupCADataset(data.Dataset):
    """dataset for the evaluation"""

    def __init__(self,
                 csv_dir, folds, flag,
                 max_len=256,
                 size=None,
                 lm='bert',
                 da=None):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.labels = []
        self.instances = []
        self.max_len = max_len
        self.size = size
        
        pd_list = []
        for fold in folds:
            csv_path = csv_dir+flag+'_'+str(fold)+".csv"
            df = pd.read_csv(csv_path)
            df = df.dropna()
            pd_list.append(df)
        merged_dataset = pd.concat(pd_list, axis = 0)
        df = merged_dataset[['text','gt']]
        df = df.values

        
        for i in range(df.shape[0]):
            #print(df[i,1])
            if int(df[i,1]) >= 0:
                self.instances.append(df[i,0])
                self.labels.append(int(df[i,1])) #待修
        self.instances = list(self.instances[:size])
        self.labels = list(self.labels[:size])

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the column
            int: the label of the column 
        """
        # idx = random.randint(0, len(self.pairs)-1)
        col = self.instances[idx]
        x = self.tokenizer.encode(text=col,
                                      max_length=self.max_len,
                                      truncation=True)
        return x, self.labels[idx]

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        # cleaning
        x1, y = zip(*batch)
        maxlen = max([len(x) for x in x1])
        x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]

        return torch.LongTensor(x1), torch.LongTensor(y)


if __name__ == '__main__':
    csv_path = '/export/data/ysunbp/CCTA/SimTAB/semtab_data/sudowoodo_semtab_'
    fold = 0

    train_set = SupCADataset(csv_path, folds=[fold], flag = 'train', max_len=128,
                        size=None,
                        lm='bert',
                        da=None)