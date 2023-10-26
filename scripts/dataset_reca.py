from functools import reduce
import operator
import os
import csv
import numpy as np
import json
import torch
import random
from torch.utils import data
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import jsonlines
import itertools
from sklearn.model_selection import StratifiedKFold
import gc
import pickle


class SupAnnDataset(data.Dataset):
    """dataset for the evaluation"""

    def __init__(self,
                    data_path,
                    label_path='../data/Semtab2019/semtab_labels.json', max_length=512, size=None, pickle_path = '../data/RECA/semtab-reca-train.pkl'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.size = size # dataset size
        self.labels = []
        self.data = []
        self.rel = []
        self.sub = []
        self.raw = []
        
        with open(label_path, 'r') as dict_in:
            label_dict = json.load(dict_in)
        
        labels = []
        out_data = []
        rel_cols = []
        sub_cols = []
        raw_column_content = []
        with open(data_path, "r+", encoding="utf8") as jl:
            for item in tqdm(jsonlines.Reader(jl)):
                label_idx = int(label_dict[item['label']])
                target_data = np.array(item['content'])[:,int(item['target'])]
                data = ""
                for i, cell in enumerate(target_data):
                    data+=cell
                    data+=' '
                cur_rel_cols = []
                cur_sub_rel_cols = []
                for rel_col in item['related_cols']:
                    cur_rel_cols.append(np.array(rel_col))
                for sub_rel_col in item['sub_related_cols']:
                    cur_sub_rel_cols.append(np.array(sub_rel_col))
                #
                out_col_string = data[:150]
                
                raw_column_content.append(out_col_string)
                sub_cols.append(cur_sub_rel_cols)
                rel_cols.append(cur_rel_cols)
                labels.append(label_idx)
                out_data.append(data)

        target_tokens = []
        rel_tokens = []
        sub_tokens = []
        if not os.path.exists(pickle_path):
            for i in trange(len(labels)):
                self.labels.append(torch.tensor(labels[i]))
                self.raw.append(raw_column_content[i])
                
                target_token_ids = self.tokenize(out_data[i])
                target_tokens.append(target_token_ids)
                #self.data.append(target_token_ids)
                if len(rel_cols[i]) == 0: # If there is no related tables, use the target column content
                    rel_token_ids = target_token_ids
                else:
                    rel_token_ids = self.tokenize_set_equal(rel_cols[i])
                rel_tokens.append(rel_token_ids)
                #self.rel.append(rel_token_ids)
                if len(sub_cols[i]) == 0: # If there is no sub-related tables, use the target column content
                    sub_token_ids = target_token_ids
                else:
                    sub_token_ids = self.tokenize_set_equal(sub_cols[i])
                #self.sub.append(sub_token_ids)
                sub_tokens.append(sub_token_ids)
            pickled_file = [target_tokens, rel_tokens, sub_tokens]
            with open(pickle_path, 'wb') as file:
                pickle.dump(pickled_file, file)
        else:
            with open(pickle_path, 'rb') as file:
                target_tokens, rel_tokens, sub_tokens = pickle.load(file)
            for i in trange(len(labels)):
                self.labels.append(torch.tensor(labels[i]))
                self.raw.append(raw_column_content[i])
        self.data = target_tokens
        self.rel = rel_tokens
        self.sub = sub_tokens

        if size is not None:
            if size > len(self.labels):
                N = size // len(self.labels) + 1
                self.labels = (self.labels * N)[:size]
                self.data = (self.data * N)[:size]
                self.rel = (self.rel * N)[:size]
                self.sub = (self.sub * N)[:size]
                self.raw = (self.raw * N)[:size]
            else:
                indices = [i for i in range(len(self.labels))]
                random.seed(42)
                selected_indices = random.sample(indices, size)
                out_instances = []
                updated_labels = []
                updated_data = []
                updated_rel = []
                updated_sub = []
                updated_raw = []
                for index in selected_indices:
                    updated_labels.append(self.labels[index])
                    updated_data.append(self.data[index])
                    updated_rel.append(self.rel[index])
                    updated_sub.append(self.sub[index])
                    updated_raw.append(self.raw[index])
                self.labels = updated_labels
                self.data = updated_data
                self.rel = updated_rel
                self.sub = updated_sub   
                self.raw = updated_raw 

    def tokenize(self, text): # Normal practice of tokenization
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids
    
    def tokenize_set_equal(self, cols): # Assigning the tokens equally to each identified column
        init_text = ''
        for i, col in enumerate(cols):
            for cell in col:
                init_text+=cell
                init_text+=' '
            if not i==len(cols)-1:
                init_text += '[SEP]'
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= self.max_length:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=self.max_length, padding = 'max_length', truncation=True)     
        else:
            ratio = self.max_length/total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col)*ratio:
                        break
                    text += cell
                    text += ' '
                if not i==len(cols)-1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)

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
        return self.data[idx], self.rel[idx], self.sub[idx], self.labels[idx], self.raw[idx]
    
    def shuffle_col(self, col_str):
        col_cells = col_str.split()
        random.shuffle(col_cells)
        out_str = ''.join(col_cells)
        return out_str

    def pad(self, batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        # cleaning
        _, _, _, _, raw = zip(*batch)
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        
        return token_ids, rel_ids, sub_ids, labels, raw


class SupAnnDatasetIndex(data.Dataset):
    """dataset for the evaluation"""

    def __init__(self,
                    data_path,
                    label_path='../data/Semtab2019/semtab_labels.json', 
                    max_length=512, 
                    size=None, 
                    cur_selected_indices=None,
                    size_ratio=False,
                    size_column=False, pickle_path = '../data/RECA/semtab-reca-train.pkl'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.size = size # dataset size
        self.labels = []
        self.data = []
        self.rel = []
        self.sub = []
        self.raw = []
        
        with open(label_path, 'r') as dict_in:
            label_dict = json.load(dict_in)
        
        labels = []
        out_data = []
        rel_cols = []
        sub_cols = []
        raw_column_content = []
        with open(data_path, "r+", encoding="utf8") as jl:
            for item in tqdm(jsonlines.Reader(jl)):
                label_idx = int(label_dict[item['label']])
                target_data = np.array(item['content'])[:,int(item['target'])]
                data = ""
                for i, cell in enumerate(target_data):
                    data+=cell
                    data+=' '
                cur_rel_cols = []
                cur_sub_rel_cols = []
                for rel_col in item['related_cols']:
                    cur_rel_cols.append(np.array(rel_col))
                for sub_rel_col in item['sub_related_cols']:
                    cur_sub_rel_cols.append(np.array(sub_rel_col))
                table = np.array(item['content'])
                col_idx = int(item['target'])
                column_list = table[:,col_idx]
                out_col_string = ''
                for cell in column_list:
                    out_col_string += cell
                out_col_string = out_col_string[:150]
                
                raw_column_content.append(out_col_string)
                sub_cols.append(cur_sub_rel_cols)
                rel_cols.append(cur_rel_cols)
                labels.append(label_idx)
                out_data.append(data)

        
        target_tokens = []
        rel_tokens = []
        sub_tokens = []
        if not os.path.exists(pickle_path):
            for i in trange(len(labels)):
                self.labels.append(torch.tensor(labels[i]))
                self.raw.append(raw_column_content[i])
                
                target_token_ids = self.tokenize(out_data[i])
                target_tokens.append(target_token_ids)
                #self.data.append(target_token_ids)
                if len(rel_cols[i]) == 0: # If there is no related tables, use the target column content
                    rel_token_ids = target_token_ids
                else:
                    rel_token_ids = self.tokenize_set_equal(rel_cols[i])
                rel_tokens.append(rel_token_ids)
                #self.rel.append(rel_token_ids)
                if len(sub_cols[i]) == 0: # If there is no sub-related tables, use the target column content
                    sub_token_ids = target_token_ids
                else:
                    sub_token_ids = self.tokenize_set_equal(sub_cols[i])
                #self.sub.append(sub_token_ids)
                sub_tokens.append(sub_token_ids)
            pickled_file = [target_tokens, rel_tokens, sub_tokens]
            with open(pickle_path, 'wb') as file:
                pickle.dump(pickled_file, file)
        else:
            with open(pickle_path, 'rb') as file:
                target_tokens, rel_tokens, sub_tokens = pickle.load(file)
            for i in trange(len(labels)):
                self.labels.append(torch.tensor(labels[i]))
                self.raw.append(raw_column_content[i])
        self.data = target_tokens
        self.rel = rel_tokens
        self.sub = sub_tokens

        if (size is not None) and (not size_ratio):
            if size > len(self.labels):
                N = size // len(self.labels) + 1
                self.labels = (self.labels * N)[:size]
                self.data = (self.data * N)[:size]
                self.rel = (self.rel * N)[:size]
                self.sub = (self.sub * N)[:size]
                self.raw = (self.raw * N)[:size]
            else:
                indices = [i for i in range(len(self.labels))]
                random.seed(42)
                selected_indices = random.sample(indices, size)
                out_instances = []
                updated_labels = []
                updated_data = []
                updated_rel = []
                updated_sub = []
                updated_raw = []
                for index in selected_indices:
                    updated_labels.append(self.labels[index])
                    updated_data.append(self.data[index])
                    updated_rel.append(self.rel[index])
                    updated_sub.append(self.sub[index])
                    updated_raw.append(self.raw[index])
                self.labels = updated_labels
                self.data = updated_data
                self.rel = updated_rel
                self.sub = updated_sub  
                self.raw = updated_raw 

        if (size is not None) and size_ratio:
            skf = StratifiedKFold(n_splits=size) # use 1/10 data for validation/testing; here we use size to represent the ratio
            for train_index, validation_index in skf.split(label_list, label_list):
                splitted_indices = validation_index
                break
            out_labels = []
            out_data = []
            out_rel = []
            out_sub = []
            out_raw = []
            for cur_idx in splitted_indices:
                out_labels.append(self.labels[cur_idx])
                out_data.append(self.data[cur_idx])
                out_rel.append(self.rel[cur_idx])
                out_sub.append(self.sub[cur_idx])
                out_raw.append(self.raw[cur_idx])
            self.labels = out_labels
            self.data = out_data
            self.rel = out_rel
            self.sub = out_sub  
            self.raw = out_raw

        if cur_selected_indices is not None: #这里没调整raw input因为好像不需要
            out_labels = []
            out_data = []
            out_rel = []
            out_sub = []
            out_raw = []
            for cur_idx in cur_selected_indices:
                out_labels.append(self.labels[cur_idx])
                out_data.append(self.data[cur_idx])
                out_rel.append(self.rel[cur_idx])
                out_sub.append(self.sub[cur_idx])
                out_raw.append(self.raw[cur_idx])
            self.labels = out_labels
            self.data = out_data
            self.rel = out_rel
            self.sub = out_sub  
            self.raw = out_raw

    def tokenize(self, text): # Normal practice of tokenization
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids
    
    def tokenize_set_equal(self, cols): # Assigning the tokens equally to each identified column
        init_text = ''
        for i, col in enumerate(cols):
            for cell in col:
                init_text+=cell
                init_text+=' '
            if not i==len(cols)-1:
                init_text += '[SEP]'
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= self.max_length:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=self.max_length, padding = 'max_length', truncation=True)     
        else:
            ratio = self.max_length/total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col)*ratio:
                        break
                    text += cell
                    text += ' '
                if not i==len(cols)-1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.labels)

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
        return self.data[idx], self.rel[idx], self.sub[idx], self.labels[idx]
    
    def shuffle_col(self, col_str):
        col_cells = col_str.split()
        random.shuffle(col_cells)
        out_str = ''.join(col_cells)
        return out_str

    def pad(self, batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: a batch of labels, (batch_size,)
        """
        # cleaning
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        #return token_ids, rel_ids, sub_ids, labels
        
        return token_ids, rel_ids, sub_ids, labels


if __name__ == '__main__':
    semtab_label_path = 'xxx/semtab_labels.json'
    semtab_training_path = 'xxx/train.jsonl'
    annotation_dataset = SupAnnDataset(semtab_training_path, semtab_label_path, size=None, max_length = 512)
    #SupAnnDataset(validation_path, lm='bert', max_length = 128)
    annotation_iter = data.DataLoader(dataset=annotation_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=annotation_dataset.pad)
    
    