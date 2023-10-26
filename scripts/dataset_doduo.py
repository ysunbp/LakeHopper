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

        rel_dict = {}
        
        cur_df = pd.read_csv(csv_file_path)
        pd_list.append(cur_df)
        
        merged_dataset = pd.concat(pd_list, axis = 0)
        data_list = []
        data_dict = {}

        for i, (index, group_df) in enumerate(tqdm(merged_dataset.groupby("table_id"))):
            token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))
            token_ids_up = reduce(operator.add, token_ids_list)
            while len(token_ids_up) > 512:
                max_length = int(max_length/2)
                token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length, truncation=True)).tolist()
                token_ids_up = reduce(operator.add, token_ids_list)
            token_ids = token_ids_up

            # filter out the non-annotated columns
            cls_index_list = []
            cur_length = 0
            for j, x in enumerate(token_ids_list):
                if not group_df['class_id'].iloc[j] == -1:
                    cls_index_list.append(cur_length)
                cur_length += len(x) 

            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = cls_index_list
            class_ids = list(group_df[group_df["class_id"]>=0]["class_id"].values)
            raw_input_data = group_df[group_df["class_id"]>=0]["data"]
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes, raw_input_data])
            data_dict[index] = [token_ids, cls_indexes]

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
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        # cleaning
        x1, y, clsA, raw, table_id = zip(*batch)
        maxlen1 = max([len(x) for x in x1])
        maxlen2 = max([len(c1) for c1 in clsA])
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
                    col_tab_map = None,
                    max_length=512,
                    size=None,
                    lm='bert', size_ratio=False, size_column=False):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_length = max_length
        self.size = size # dataset size
        pd_list = []

        rel_dict = {}
        
        cur_df = pd.read_csv(csv_file_path)
        pd_list.append(cur_df)
        
        merged_dataset = pd.concat(pd_list, axis = 0)
        data_list = []
        label_list = []
        col_2_table_idx = {}
        table_2_col_idx = {}
        cur_col_idx = 0

        for i, (index, group_df) in enumerate(tqdm(merged_dataset.groupby("table_id"))):
            token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))
            token_ids_up = reduce(operator.add, token_ids_list)
            while len(token_ids_up) > 512:
                max_length = int(max_length/2)
                token_ids_list = group_df["data"].apply(lambda x: self.tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length, truncation=True)).tolist()
                token_ids_up = reduce(operator.add, token_ids_list)
            token_ids = token_ids_up

            # filter out the non-annotated columns
            cls_index_list = []
            cur_length = 0
            for j, x in enumerate(token_ids_list):
                if not group_df['class_id'].iloc[j] == -1:
                    cls_index_list.append(cur_length)
                cur_length += len(x) 

            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = cls_index_list
            class_ids = list(group_df[group_df["class_id"]>=0]["class_id"].values)
            raw_input_data = group_df[group_df["class_id"]>=0]["data"]
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes, raw_input_data])
            #data_dict[index] = [token_ids, cls_indexes]
            label_list += class_ids
            for col_idx in range(cur_col_idx, cur_col_idx+len(class_ids)):
                col_2_table_idx[col_idx] = i
                if i in table_2_col_idx.keys():
                    table_2_col_idx[i].append(col_idx)
                else:
                    table_2_col_idx[i] = [col_idx]
            cur_col_idx += len(class_ids)
            #print(len(label_list))

        if (size is not None) and (not size_ratio) and (not size_column):
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

                table_idx = col_2_table_idx[cur_idx]

                cur_instance = data_list[table_idx].copy()
                col_indices = table_2_col_idx[table_idx]

                col_idx_in_table = col_indices.index(cur_idx)
                
                cur_instance[3] = [cur_instance[3][col_idx_in_table]]
                cur_instance[4] = [cur_instance[4][col_idx_in_table]]
                cur_instance[5] = [cur_instance[5].iloc[col_idx_in_table]]
                out_instances.append(cur_instance)
            data_list = out_instances

        if (size is not None) and (size_column):
            random.seed(42)
            splitted_indices = random.sample(range(len(label_list)), size)
            out_instances = []
            for cur_idx in splitted_indices:
                table_idx = col_2_table_idx[cur_idx]
                cur_instance = data_list[table_idx].copy()
                col_indices = table_2_col_idx[table_idx]
                col_idx_in_table = col_indices.index(cur_idx)
                cur_instance[3] = [cur_instance[3][col_idx_in_table]]
                cur_instance[4] = [cur_instance[4][col_idx_in_table]]
                cur_instance[5] = [cur_instance[5].iloc[col_idx_in_table]]
                out_instances.append(cur_instance)
            data_list = out_instances
        
        if cur_selected_indices is not None: 
            out_instances = []
            out_table_ids = {}
            for index in cur_selected_indices:
                cur_instance = data_list[col_tab_map[index][0]].copy()
                if cur_instance[0] in out_table_ids.keys():
                    out_instances[out_table_ids[cur_instance[0]]][1] += 1
                    insert_label = data_list[col_tab_map[index][0]][3][col_tab_map[index][1]]
                    insert_cls = data_list[col_tab_map[index][0]][4][col_tab_map[index][1]]
                    insert_idx = np.searchsorted(out_instances[out_table_ids[cur_instance[0]]][4], insert_cls)
                    out_instances[out_table_ids[cur_instance[0]]][3].insert(insert_idx, insert_label)
                    out_instances[out_table_ids[cur_instance[0]]][4].insert(insert_idx, insert_cls)
                else:
                    out_table_ids[cur_instance[0]] = len(out_instances) # record the table_id and its corresponding idx in out_indices
                    cur_instance[1] = 1
                    cur_instance[3] = [cur_instance[3][col_tab_map[index][1]]]
                    cur_instance[4] = [cur_instance[4][col_tab_map[index][1]]]
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
        maxlen1 = max([len(x) for x in x1])
        maxlen2 = max([len(c1) for c1 in clsA])
        out_y = []
        x1 = [xi + [0]*(maxlen1 - len(xi)) for xi in x1]
        for yi in y:
            out_y += yi
        clsA = [ci + [ci[-1]]*(maxlen2 - len(ci)) for ci in clsA] #fill with the last entry, will be counter off by cls 0 in scatter_ function
        
        return torch.LongTensor(x1), torch.LongTensor(out_y), torch.LongTensor(clsA), raw

if __name__ == '__main__':
    validation_path = 'xxx/semtab_train_0.csv'
    validation_set = SupAnnDatasetIndex(validation_path, lm='bert', size=20, max_length = 128, size_ratio=True)
    