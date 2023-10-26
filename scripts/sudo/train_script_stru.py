# This file contains the training script
import torch
from .simtab import SimTAB
from transformers import AdamW
#from apex import amp
from torch.utils import data
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import json


def finetune(train_iter, model, optimizer, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = torch.nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm(train_iter)):
        optimizer.zero_grad()
        x, y = batch
        
        prediction = model(x, None, flag=False)
        loss = criterion(prediction, y.to(model.device))
        #print(prediction.shape)
        #print('y',y.shape)
        # loss = criterion(prediction, y.float().to(model.device))
        
        loss.backward()
        optimizer.step()
        if i % 50 == 0: # monitoring
            print(f"    fine tune step: {i}, loss: {loss.item()}")
        del loss

def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def evaluation(model, iter, model_save_path, is_test = False, cur_best_loss=100):
    if is_test:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
    labels = []
    predicted_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()
    v_epoch_loss = 0
    for i, item in enumerate(tqdm(iter)):
        x, y = item
        prediction = model(x, None, flag=False)
        vloss = loss_fn(prediction, y.to(model.device))
        v_epoch_loss += vloss.item()
        predicted_label = prediction.argmax()
        labels.append(y.detach().numpy())
        predicted_labels.append(predicted_label.detach().cpu().numpy())
        del x, vloss
    v_length_label = len(labels)
    v_total_loss = v_epoch_loss / (v_length_label)
    print("loss:", v_total_loss)
    
    if v_total_loss < cur_best_loss:
        cur_best_loss = v_total_loss
        if not is_test:
            torch.save(model.state_dict(), model_save_path)
            print('model updated')
    
    pred_labels = np.concatenate([np.expand_dims(i, axis=0) for i in predicted_labels]) 
    true_labels = np.concatenate(labels)
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1'], cur_best_loss

def train(trainset_nolabel, train_set, valid_set, test_set, hp):
    print('=====================================================================')
    print('start training')
    print('lr ', hp.lr, ' f_lr', hp.f_lr)
    model_save_path = hp.save_path
    num_ssl_epochs = hp.n_ssl_epochs

    train_nolabel_iter = data.DataLoader(dataset=trainset_nolabel,
                                             batch_size=hp.batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=trainset_nolabel.pad)
    
    train_iter = data.DataLoader(dataset=train_set,
                                    batch_size=hp.batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=train_set.pad)
    
    valid_iter = data.DataLoader(dataset=valid_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=valid_set.pad)
    
    test_iter = data.DataLoader(dataset=test_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=test_set.pad)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimTAB(hp, device=device, lm=hp.lm)
    base_model_path = '../checkpoints/base-sudowoodo-p2v.pkl'
    #base_model_path = '../checkpoints/base-sudowoodo-v2s.pkl' # You need to uncomment this line if you are using the v2s
    model.load_state_dict(torch.load(base_model_path))

    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    optimizer_f = AdamW(model.parameters(), lr=hp.f_lr) 
    
    cur_best = 100
    
    if hp.switching == False:
        for epoch in range(hp.n_epochs):
            if epoch < num_ssl_epochs:
                for i, batch in enumerate(tqdm(train_nolabel_iter)):
                    yA, yB = batch
                    optimizer.zero_grad()
                    loss = model(yA, yB, da=hp.da)
                    
                    loss.backward()
                    optimizer.step()
                    if i % 50 == 0: # monitoring
                        print(f"    step: {i}, loss: {loss.item()}")
                    del loss
                cur_best_model = model
                
            else: # TODO: should we freeze the embedding after self-supervise
                model.train()
                finetune(train_iter, model, optimizer_f, hp)
                print('start validating epoch: ', epoch-num_ssl_epochs)
                model.eval()
                v_sw, v_ma, cur_best_loss = evaluation(model, valid_iter, model_save_path, is_test = False, cur_best_loss=cur_best)
                cur_best = cur_best_loss
                cur_best_model = model
    else:
        cur_ssl = 1
        for epoch in trange(hp.n_epochs):
            cur_epoch = epoch + 1
            if cur_epoch % 2 == 1 and cur_ssl <= num_ssl_epochs:
                for i, batch in enumerate(tqdm(train_nolabel_iter)):
                    yA, yB = batch
                    optimizer.zero_grad()
                    loss = model(yA, yB, da=hp.da)

                    loss.backward()
                    optimizer.step()
                    if i % 50 == 0: # monitoring
                        print(f"    step: {i}, loss: {loss.item()}")
                    del loss
                cur_ssl += 1
            else:
                model.train()
                finetune(train_iter, model, optimizer_f, hp)
                print('start validating epoch: ', epoch-cur_ssl+1)
                model.eval()
                v_sw, v_ma, cur_best_loss = evaluation(model, valid_iter, model_save_path, is_test = False, cur_best_loss=cur_best)
                cur_best = cur_best_loss
                cur_best_model = model

    print('start testing')
    model = cur_best_model
    model.eval()
    t_sw, t_ma, t_loss = evaluation(model, test_iter, model_save_path, is_test = True, cur_best_loss=100)


