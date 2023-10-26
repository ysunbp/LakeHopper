from torch import nn
import torch
from transformers import AutoModel

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}


class base_model(nn.Module):
    def __init__(self, hp, device='cuda', lm='bert'):
        super().__init__()
        self.hp = hp
        self.n_classes = hp.n_classes
        self.bert = AutoModel.from_pretrained(lm_mp[lm]).to(device)
        self.device = device
        hidden_size = 768

        # projector
        sizes = [hidden_size] + list(map(int, hp.projector.split('-'))) # [768, 768]
        self.projector = nn.Linear(hidden_size, sizes[-1]).to(device)

        # a fully connected layer for fine tuning
        self.fc = nn.Linear(hidden_size, self.n_classes).to(device)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, y1, clsA):
        # finetune
        x1 = y1
        x1 = x1.to(self.device) # (batch_size, seq_len)
        clsA = clsA.to(self.device)
        indexA = torch.zeros(512, clsA.shape[0]).to(self.device)
        indexA.scatter_(0, clsA.transpose(0,1), 1)
        indexA = indexA.transpose(0,1)
        nonzero = indexA.nonzero()
        enc = self.projector(self.bert(x1)[0][nonzero[:, 0], nonzero[:, 1]]) # (batch_size, emb_size)
        out = self.fc(enc)
        return out
