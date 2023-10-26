import os

MAX_LEN = 128
SEP_TOKEN_ID = 102
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from torch import nn
import tqdm
import time
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from tqdm import trange
from math import sqrt



NERs = {'PERSON1':0, 'PERSON2':1, 'NORP':2, 'FAC':3, 'ORG':4, 'GPE':5, 'LOC':6, 'PRODUCT':7, 'EVENT':8, 'WORK_OF_ART':9, 'LAW':10, 'LANGUAGE':11, 'DATE1':12, 'DATE2':13, 'DATE3':14, 'DATE4':15, 'DATE5':16, 'TIME':17, 'PERCENT':18, 'MONEY':19, 'QUANTITY':20, 'ORDINAL':21, 'CARDINAL':22, 'EMPTY':23}

def setup_seed(seed): # Set up random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TableDataset(Dataset): # Generate tabular dataset
    def __init__(self, target_cols, tokenizer, rel_cols, sub_rel_cols, labels):
        self.labels = []
        self.target_cols = []
        self.tokenizer = tokenizer
        self.rel_cols = []
        self.sub_rel_cols = []
        for i in trange(len(labels)):
            self.labels.append(torch.tensor(labels[i]))
            target_token_ids = self.tokenize(target_cols[i])
            self.target_cols.append(target_token_ids)
            if len(rel_cols[i]) == 0: # If there is no related tables, use the target column content
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(rel_cols[i])
            self.rel_cols.append(rel_token_ids)
            if len(sub_rel_cols[i]) == 0: # If there is no sub-related tables, use the target column content
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(sub_rel_cols[i])
            self.sub_rel_cols.append(sub_token_ids)
        
    def tokenize(self, col): # Normal practice of tokenization
        text = ''
        for cell in col:
            text+=cell
            text+=' '
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
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
        if total_length <= MAX_LEN:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)     
        else:
            ratio = MAX_LEN/total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col)*ratio:
                        break
                    text += cell
                    text += ' '
                if not i==len(cols)-1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        return self.target_cols[idx], self.rel_cols[idx], self.sub_rel_cols[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels

def get_loader(path, batch_size, is_train): # Generate the dataloaders for the training process
    dataset = torch.load(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=dataset.collate_fn)
    loader.num = len(dataset)
    return loader

class KREL(torch.nn.Module):
    def __init__(self, n_classes=78):
        super(KREL, self).__init__()
        self.model_name = 'KREL'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.5)
        self.fcc_tar = torch.nn.Linear(768, n_classes)
        self.fcc_rel = torch.nn.Linear(768, n_classes)
        self.fcc_sub = torch.nn.Linear(768, n_classes)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(3)])

    def encode(self, target_ids, rel_ids, sub_ids):
        att_tar = (target_ids>0)
        _, tar = self.bert_model(input_ids=target_ids, attention_mask=att_tar, return_dict=False)
        att_rel = (rel_ids>0)
        _, rel = self.bert_model(input_ids=rel_ids, attention_mask=att_rel, return_dict=False)
        att_sub = (sub_ids>0)
        _, sub = self.bert_model(input_ids=sub_ids, attention_mask=att_sub, return_dict=False)

        return tar, rel, sub
    
    def forward(self,tar_ids,rel_ids, sub_ids):
        tar, rel, sub = self.encode(tar_ids, rel_ids, sub_ids)
        tar_out = self.dropout(tar)
        rel_out = self.dropout(rel)
        sub_out = self.dropout(sub)
        out_tar = self.fcc_tar(tar_out)
        out_rel = self.fcc_rel(rel_out)
        out_sub = self.fcc_sub(sub_out)
        res = self.weights[0]*out_tar+self.weights[1]*out_rel+self.weights[2]*out_sub
        return res

def metric_fn(preds, labels): # The Support-weighted F1 score and Macro Average F1 score
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def train_model(model,train_loaders,val_loader,test_loader,lr,model_save_path='.pkl',early_stop_epochs=5,epochs=20,size=None): # Training process
    no_improve_epochs = 0
    weight_decay = 0
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    cur_best = 100


    model.eval()
    bar2 = tqdm(val_loader)
    pred_labels = []
    true_labels = []
    v_epoch_loss = 0

    
    for j,(ids, rels, subs, labels) in enumerate(bar2):
        labels = labels.cuda()
        rels = rels.cuda()
        subs = subs.cuda()
        output = model(ids.cuda(), rels, subs)
        y_pred_prob = output
        y_pred_label = y_pred_prob.argmax(dim=1)
        vloss = loss_fn(y_pred_prob.view(-1, 78), labels.view(-1))
        pred_labels.append(y_pred_label.detach().cpu().numpy())
        true_labels.append(labels.detach().cpu().numpy())
        v_epoch_loss += vloss.item()
        v_length_label = len(labels)
        del ids, rels, subs
        torch.cuda.empty_cache()

    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    val_length = len(bar2)
    print("validation_loss:", v_epoch_loss / (val_length))
    
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])

    length = len(train_loaders[0])
    if not size == None:
        selected_indices = random.sample(range(length), size)
    else:
        selected_indices = range(length)

    for epoch in range(1,epochs+1):
        model.train()
        epoch_loss = 0
        v_epoch_loss = 0
        train_length = 0

        for train_loader in train_loaders:
            bar1 = tqdm(train_loader)
            for i,(ids, rels, subs, labels) in enumerate(bar1):
                if i in selected_indices:
                    labels = labels.cuda()
                    rels = rels.cuda()
                    subs = subs.cuda()
                    output = model(ids.cuda(), rels, subs)
                    y_pred_prob = output
                    y_pred_label = y_pred_prob.argmax(dim=1) # Argmax to find the prediction from the outputs
                    loss = loss_fn(y_pred_prob.view(-1, 78), labels.view(-1))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    length_label = len(labels)
                    del ids, rels, subs, labels
                    torch.cuda.empty_cache() # Release the memory
            train_length += len(selected_indices)
        print("Epoch:", epoch, "training_loss:", epoch_loss / (train_length))
        model.eval()
        bar2 = tqdm(val_loader)
        pred_labels = []
        true_labels = []


        for j,(ids, rels, subs, labels) in enumerate(bar2):
            labels = labels.cuda()
            rels = rels.cuda()
            subs = subs.cuda()
            output = model(ids.cuda(), rels, subs)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            vloss = loss_fn(y_pred_prob.view(-1, 78), labels.view(-1))
            pred_labels.append(y_pred_label.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
            v_epoch_loss += vloss.item()
            v_length_label = len(labels)
            del ids, rels, subs
            torch.cuda.empty_cache()

        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        val_length = len(bar2)
        print("validation_loss:", v_epoch_loss / (val_length))
        if v_epoch_loss / (val_length) < cur_best:
            print('model updated')
            cur_best = v_epoch_loss / (val_length)
            torch.save(model.state_dict(),model_save_path)
            cur_best_model = model
        f1_scores = metric_fn(pred_labels, true_labels)
        print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])

    model.load_state_dict(torch.load(model_save_path))
    
    model.eval()
    bar3 = tqdm(test_loader)
    pred_labels = []
    true_labels = []

    for j,(ids, rels, subs, labels) in enumerate(bar3):
        labels = labels.cuda()
        rels = rels.cuda()
        subs = subs.cuda()
        output = model(ids.cuda(), rels, subs)
        y_pred_prob = output
        y_pred_label = y_pred_prob.argmax(dim=1)
        vloss = loss_fn(y_pred_prob.view(-1, 78), labels.view(-1))
        pred_labels.append(y_pred_label.detach().cpu().numpy())
        true_labels.append(labels.detach().cpu().numpy())
        v_epoch_loss += vloss.item()
        v_length_label = len(labels)
        del ids, rels, subs
        torch.cuda.empty_cache()

    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    val_length = len(bar3)
    print("test_loss for best model:", v_epoch_loss / (val_length))
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])



if __name__ == '__main__':
    setup_seed(20)
    
    rounds = [0]
    
    BS = 8
    lrs = [7e-6]
    print('start loading data')
    
    for round in rounds:
        if True:
            for lr in lrs:
                print('start training fold', round+1, 'learning rate', lr, 'batch size', BS, 'max length', MAX_LEN)
                model = KREL().cuda()
    
                
                public_bi_path = 'LakeHopper/checkpoints/out_model_p2v
        
                print('Start pretraining on PublicBI')
                publicbi_train_loader = [get_loader(path='LakeHopper/data/RECA/'+str(MAX_LEN)+'/publicbi-train', batch_size=BS, is_train=True)]
                publicbi_test_loader = get_loader(path='LakeHopper/data/RECA/'+str(MAX_LEN)+'/publicbi-test', batch_size=BS, is_train=False)
                train_model(model, publicbi_train_loader, publicbi_test_loader, publicbi_test_loader, lr, model_save_path=public_bi_path, epochs = 5)
                