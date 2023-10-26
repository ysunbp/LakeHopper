from functools import reduce
import operator
import os
import csv
import numpy as np
import json
import torch
import random
from torch.utils import data
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import jsonlines
import itertools
from sklearn.model_selection import StratifiedKFold
import gc
import pickle


lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

class SupAnnDataset(data.Dataset):
    """dataset for the evaluation"""

    def __init__(self,
                    csv_file_path,
                    max_length=512,
                    size=None,
                    lm='bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_length = max_length
        self.size = size # dataset size
        pd_list = []   
        cur_df = pd.read_csv(csv_file_path)
        pd_list.append(cur_df)
        
        merged_dataset = pd.concat(pd_list, axis = 0)
        data_list = []

        for i, (index, group_df) in enumerate(tqdm(merged_dataset.groupby("table_id"))):

            token_ids_list = []
            
            for index in range(len(group_df["data"])):
                cur_text = group_df["data"].iloc[index]
                token_ids = self.tokenizer.encode(cur_text, add_special_tokens=True, max_length=max_length, truncation=True)
                token_ids_list.append(token_ids)

            filtered_token_ids = []
            # filter out the non-annotated columns
            cls_index_list = []
            for j, x in enumerate(token_ids_list):
                if not group_df['class_id'].iloc[j] == -1:
                    cls_index_list.append(0)
                    filtered_token_ids.append(x)

            token_ids_list = filtered_token_ids

            for idx, cls_index in enumerate(cls_index_list):
                assert token_ids_list[idx][
                    cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = cls_index_list
            class_ids = list(group_df[group_df["class_id"]>=0]["class_id"].values)
            raw_input_data = list(group_df[group_df["class_id"]>=0]["data"].values)

            for i, item in enumerate(token_ids_list):
                data_list.append(
                    [index,
                    len(group_df), item, [class_ids[i]], [cls_indexes[i]], raw_input_data[i]])
            

        if size is not None:
            if size > len(data_list):
                N = size // len(data_list) + 1
                data_list = (data_list * N)[:size] # over sampling
            else:
                indices = [i for i in range(len(data_list))]
                random.seed(42)
                selected_indices = random.sample(indices, size)
                out_instances = []
                for index in selected_indices:
                    out_instances.append(data_list[index])
                data_list = out_instances

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "raw_input"
                                     ])        

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.table_df)

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
        return self.table_df.iloc[idx]["data_tensor"], self.table_df.iloc[idx]["label_tensor"], self.table_df.iloc[idx]["cls_indexes"], self.table_df.iloc[idx]["raw_input"], self.table_df.iloc[idx]["table_id"]

    def shuffle_col(self, col_str):
        col_cells = col_str.split()
        random.shuffle(col_cells)
        out_str = ''.join(col_cells)
        return out_str

    @staticmethod
    def pad( batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        # cleaning
        x1, y, clsA, raw, table_id = zip(*batch)
        maxlen1 = 128
        maxlen2 = 1
        out_y = []
        x1 = [xi + [0]*(maxlen1 - len(xi)) for xi in x1]
        for yi in y:
            out_y += yi
        clsA = [ci + [0]*(maxlen2 - len(ci)) for ci in clsA] #fill with 0, will be counter off by cls 0 in scatter_ function
        
        return torch.LongTensor(x1), torch.LongTensor(out_y), torch.LongTensor(clsA), raw, table_id

class SupAnnDatasetIndex(data.Dataset):
    """dataset for the evaluation"""

    def __init__(self,
                    csv_file_path,
                    cur_selected_indices = None,
                    max_length=512,
                    size=None,
                    lm='bert', size_ratio=False, size_column=False):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_length = max_length
        self.size = size # dataset size
        pd_list = []
        
        cur_df = pd.read_csv(csv_file_path)
        pd_list.append(cur_df)
        
        merged_dataset = pd.concat(pd_list, axis = 0)
        data_list = []
        label_list = []
        

        for i, (index, group_df) in enumerate(tqdm(merged_dataset.groupby("table_id"))):
            token_ids_list = []
            
            for index in range(len(group_df["data"])):
                cur_text = group_df["data"].iloc[index]
                token_ids = self.tokenizer.encode(cur_text, add_special_tokens=True, max_length=max_length, truncation=True)
                token_ids_list.append(token_ids)

            filtered_token_ids = []
            # filter out the non-annotated columns
            cls_index_list = []
            for j, x in enumerate(token_ids_list):
                if not group_df['class_id'].iloc[j] == -1:
                    cls_index_list.append(0)
                    filtered_token_ids.append(x)

            for idx, cls_index in enumerate(cls_index_list):
                assert token_ids_list[idx][
                    cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = cls_index_list
            class_ids = list(group_df[group_df["class_id"]>=0]["class_id"].values)
            raw_input_data = list(group_df[group_df["class_id"]>=0]["data"].values)

            for k, item in enumerate(filtered_token_ids):
                data_list.append(
                    [index,
                    len(group_df), item, [class_ids[k]], [cls_indexes[k]], raw_input_data[k]])
            label_list += class_ids

        if (size is not None) and (not size_ratio):
            if size > len(data_list):
                N = size // len(data_list) + 1
                data_list = (data_list * N)[:size] # over sampling
            else:
                indices = [i for i in range(len(data_list))]
                random.seed(42)
                selected_indices = random.sample(indices, size)
                out_instances = []
                for index in selected_indices:
                    out_instances.append(data_list[index])
                data_list = out_instances
        
        if (size is not None) and (size_ratio): #for the construction of stratified validation set
            skf = StratifiedKFold(n_splits=size) # use 1/10 data for validation/testing; here we use size to represent the ratio
            for train_index, validation_index in skf.split(label_list, label_list):
                splitted_indices = validation_index
                break
            out_instances = []
            for cur_idx in splitted_indices:
                cur_instance = data_list[cur_idx]
                out_instances.append(cur_instance)
            data_list = out_instances
        
        if cur_selected_indices is not None: #这里没调整raw input因为好像不需要
            out_instances = []
            for index in cur_selected_indices:
                cur_instance = data_list[index]
                out_instances.append(cur_instance)
            data_list = out_instances

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "raw_input"])        

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.table_df)

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
        return self.table_df.iloc[idx]["data_tensor"], self.table_df.iloc[idx]["label_tensor"], self.table_df.iloc[idx]["cls_indexes"], self.table_df.iloc[idx]["raw_input"]

    def shuffle_col(self, col_str):
        col_cells = col_str.split()
        random.shuffle(col_cells)
        out_str = ''.join(col_cells)
        return out_str

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
        x1, y, clsA, raw = zip(*batch)

        maxlen1 = 128
        maxlen2 = 1
        out_y = []
        x1 = [xi + [0]*(maxlen1 - len(xi)) for xi in x1]
        for yi in y:
            out_y += yi
        clsA = [ci + [ci[-1]]*(maxlen2 - len(ci)) for ci in clsA] #fill with the last entry, will be counter off by cls 0 in scatter_ function
        return torch.LongTensor(x1), torch.LongTensor(out_y), torch.LongTensor(clsA), raw

if __name__ == '__main__':
    validation_path = 'xxxxx/semtab_train_0.csv'
    validation_set = SupAnnDatasetIndex(validation_path, lm='bert', size=20, max_length = 128, size_ratio=True)

    