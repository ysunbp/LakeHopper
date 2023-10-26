# This file contains the class needed for contrastive learning 
import logging
import os
import sys

from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModel
import random

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}




class SimTAB(nn.Module):
    # the encoder is bert+projector
    def __init__(self, hp, device='cuda', lm='bert'):
        super().__init__()
        self.hp = hp
        self.n_classes = hp.n_classes
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768

        # projector
        sizes = [hidden_size] + list(map(int, hp.projector.split('-'))) # [768, 768]
        self.projector = nn.Linear(hidden_size, sizes[-1])

        # a fully connected layer for fine tuning
        self.fc = nn.Linear(hidden_size, self.n_classes)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)


    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        #print('mask', mask.shape)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        #print('simm',similarity_matrix.shape)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        #print('logits', logits, logits.shape)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        #print('labels', labels, labels.shape)
        logits = logits / temperature
        return logits, labels


    def forward(self, y1, y2, flag=True, da=None, cutoff_ratio=0.1):
        if flag:
            # pre-training
            # encode
            batch_size = len(y1)
            y1 = y1.to(self.device) # original
            y2 = y2.to(self.device) # augment
            if da == 'cutoff': #这里用了cutoff方法 我们之后可能需要修改
                #print('cut')
                seq_len = y2.size()[1]
                y1_word_embeds = self.bert.embeddings.word_embeddings(y1)
                y2_word_embeds = self.bert.embeddings.word_embeddings(y2)

                # modify the position embeddings of y2
                position_ids = torch.LongTensor([list(range(seq_len))]).to(self.device)
                # position_ids = self.bert.embeddings.position_ids[:, :seq_len]
                pos_embeds = self.bert.embeddings.position_embeddings(position_ids)

                # sample again
                l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                s = random.randint(0, seq_len - l - 1)
                y2_word_embeds[:, s:s+l, :] -= pos_embeds[:, s:s+l, :]

                # merge y1 and y2
                y_embeds = torch.cat((y1_word_embeds, y2_word_embeds))
                z = self.bert(inputs_embeds=y_embeds)[0][:, 0, :]
            else:
                #print('no-cut')
                # cat y1 and y2 for faster training
                #print(y1.shape, y2.shape)
                y = torch.cat((y1, y2))
                #print(y.shape)
                #print(self.bert(y)[0].shape)
                z = self.bert(y)[0][:, 0, :] # take the first CLS
            z = self.projector(z)

            #print('z', z.shape)
            # simclr
            logits, labels = self.info_nce_loss(z, batch_size, 2)
            loss = self.criterion(logits, labels)
            return loss
            
        else:
            # finetune
            x1 = y1
            x1 = x1.to(self.device) # (batch_size, seq_len)
            enc = self.projector(self.bert(x1)[0][:, 0, :]) # (batch_size, emb_size)
            return self.fc(enc)

class SimTAB_combine(nn.Module):
    # the encoder is bert+projector
    def __init__(self, hp, device='cuda', lm='bert'):
        super().__init__()
        self.hp = hp
        self.n_classes = hp.n_classes
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768

        # projector
        sizes = [hidden_size] + list(map(int, hp.projector.split('-'))) # [768, 768]
        self.projector = nn.Linear(hidden_size, sizes[-1])

        # seperate trainer
        self.fc_stru = nn.Linear(hidden_size, hidden_size)
        self.fc_cont = nn.Linear(hidden_size, hidden_size)

        # a fully connected layer for fine tuning
        self.fc = nn.Linear(hidden_size, self.n_classes)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)


    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        #print('logits', logits, logits.shape)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        #print('labels', labels, labels.shape)
        logits = logits / temperature
        return logits, labels


    def forward(self, y1, y2, flag=True, is_stru = True, da=None, cutoff_ratio=0.1):
        if flag:
            # pre-training
            # encode
            batch_size = len(y1)
            y1 = y1.to(self.device) # original
            y2 = y2.to(self.device) # augment
            if da == 'cutoff': #这里用了cutoff方法 我们之后可能需要修改
                #print('cut')
                seq_len = y2.size()[1]
                y1_word_embeds = self.bert.embeddings.word_embeddings(y1)
                y2_word_embeds = self.bert.embeddings.word_embeddings(y2)

                # modify the position embeddings of y2
                position_ids = torch.LongTensor([list(range(seq_len))]).to(self.device)
                # position_ids = self.bert.embeddings.position_ids[:, :seq_len]
                pos_embeds = self.bert.embeddings.position_embeddings(position_ids)

                # sample again
                l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                s = random.randint(0, seq_len - l - 1)
                y2_word_embeds[:, s:s+l, :] -= pos_embeds[:, s:s+l, :]

                # merge y1 and y2
                y_embeds = torch.cat((y1_word_embeds, y2_word_embeds))
                z = self.bert(inputs_embeds=y_embeds)[0][:, 0, :]
            else:
                #print('no-cut')
                # cat y1 and y2 for faster training
                y = torch.cat((y1, y2))
                z = self.bert(y)[0][:, 0, :]
            z = self.projector(z)

            if is_stru:
                z = self.fc_stru(z)
            else:
                z = self.fc_cont(z)
            
            # simclr
            logits, labels = self.info_nce_loss(z, batch_size, 2)
            loss = self.criterion(logits, labels)
            return loss
            
        else:
            # finetune
            x1 = y1
            x1 = x1.to(self.device) # (batch_size, seq_len)
            enc = self.projector(self.bert(x1)[0][:, 0, :]) # (batch_size, emb_size)
            return self.fc(enc)
