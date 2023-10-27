import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'X'
import argparse
import torch
import numpy as np
from dataset_reca_sato import SupAnnDataset, SupAnnDatasetIndex
from transformers import BertModel, BertForSequenceClassification, BertConfig
from skeleton import base_model
from sklearn.metrics import f1_score
from torch.utils import data
from transformers import AdamW
import pickle
from tqdm import tqdm
import csv
import json
from sklearn.cluster import KMeans
import random
import openai
import gc
import time

openai.api_base = 'xxx'
openai.api_key = 'xxxx'


class KREL(torch.nn.Module): # KREL model structure
    def __init__(self, n_classes=78):
        super(KREL, self).__init__()
        self.device = 'cuda'
        self.model_name = 'KREL'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased") # BERT encoder
        self.dropout = torch.nn.Dropout(0.3) # Dropout layer
        self.fcc_tar = torch.nn.Linear(768, n_classes) # linear layer
        self.fcc_rel = torch.nn.Linear(768, n_classes)
        self.fcc_sub = torch.nn.Linear(768, n_classes)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(3)]) # Weighted combination

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generateResponse(prompt):
    
    '''
    Input:
        prompt (str): composed prompt.
    Output:
        response (str): response from GPT-4.

    This function takes composed prompt as input and returns the response from GPT.
    '''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user","content": prompt}]
    )
    return response['choices'][0]['message']['content']

def composeTemplate(intro, type_set, input_column, predicted_type):
    
    '''
    Input:
        intro (str): The introduction prompt of our task.
        type_set (str): The semantic type set considered.
        input_column (str): The flattened input column string.
        predicted_type (str): The predicted type output by the fine-tuned model.
    Output:
        template (str): The composed prompt.

    This function generates the prompt template in reference to the paper https://arxiv.org/pdf/2306.00745.pdf
    '''
    
    template = intro+type_set+'Column: '+input_column +'\n'+ 'The semantic type of the column is ' + predicted_type + " (Please answer with Yes/No, provide reasons and your most confident semantic type)?"
    return template

def sample_cur_dataset_idx(current_exploration_round, dataset_length, sample_size, sampled_data_path,random_state=42):

    '''
    Input:
        current_exploration_round (int): the current round number, use to identify the sampled idx files that we need to load currently.
        dataset_length (int): The length of the table dataset.
        sample_size (int): The number of tables sampled for query.
        random_state (optional int): The random seed.
    Output:
        out_sampled_idx (1-d tensor): The updated already sampled table indices starting from the current round.
        cur_round_idx (1-d tensor): The sampled table ids in the current round.
    
    This function sample the table ids for current round of query.
    '''
    remaining_indices = torch.arange(dataset_length)
    
    if current_exploration_round > 0:
        for r in range(current_exploration_round):
            with open(sampled_data_path+'sampled_tensor_'+str(r)+'.pkl', 'rb') as file:
                current_sampled = pickle.load(file)
            combined = torch.cat((remaining_indices,current_sampled),dim=0)
            uniques, counts = combined.unique(return_counts = True)
            remaining_indices = uniques[counts==1]
            del current_sampled
            gc.collect()
    remaining_indices = remaining_indices.tolist()
    random.seed(42)
    cur_round_idx = torch.tensor(random.sample(remaining_indices, sample_size))

    return cur_round_idx

def compute_infinity_norm(softmax_scores):

    '''
    Input:
        softmax_scores (tensor of size of label set): The outputed softmax scores from the current fine-tuned model.
    Output:
        max_score (int): The maximum of the softmax score which is the infinity norm.

    This function computes the infinity norm of the softmax scores.
    '''

    return max(softmax_scores).item()

def get_annotation(model, annotation_dataset, label_dict, round, cur_round_idx, out_path, confidence=0.9, mode='trust'):

    '''
    Input:
        model (skeleton.base_model): The fine-tuned model for annotation on the target domain.
        annotation_dataset (dataset): The sampled tables from the target domain.
        label_dict (dict): The label to label index.
        round (int): The current round index.
        cur_round_idx (1-d array): The list of table selected in the current round.
        out_path (str): output csv path.
        confidence (float): The confidence threshold for the trade-off between BERT and GPT-4.
        mode (str): The different function mode that decide the different functionality.
    Output:
        Write to the csv files.

    This function test the knowledge of the fine-tuned BERT with the help of GPT-4. It store the query results in the csv files.
    '''

    model.eval()
    annotation_iter = data.DataLoader(dataset=annotation_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=annotation_dataset.pad)
    inverted_dict = dict(map(reversed, label_dict.items()))
    soft_fn = torch.nn.Softmax(dim=0)

    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['table_id','ground_truth','gt_id','prediction','pred_id','should be','llm_decision','reasoning','data','col_idx'])  # 写入表头
        for i, item in enumerate(tqdm(annotation_iter)):
            if not i in cur_round_idx:
                continue
            predicted_labels = []
            label_list = []
            
            ids, rels, subs, labels, raw = item
            table_id = i
            prediction = model(ids.to(model.device), rels.to(model.device), subs.to(model.device))
            predicted_label = torch.argmax(prediction, dim=1)
            label_list.append(list(labels.detach().numpy()))
            
            predicted_labels += predicted_label.detach().cpu().numpy().tolist()
            del ids, rels, subs

            table_width = 1
            for col_idx in range(table_width):
                cur_col = raw[0]
                cur_label_idx = predicted_labels[col_idx]
                cur_label = inverted_dict[str(cur_label_idx)]
                truncated_col = cur_col.strip()[:150]

                softmax_score = compute_infinity_norm(soft_fn(prediction[col_idx]))
                if cur_label == inverted_dict[str(label_list[0][col_idx])]:
                    gt_decision = True
                else:
                    gt_decision = False
                
                if mode=='trust':
                    if softmax_score >= confidence:
                        response = 'LM is confident, no need to query GPT.'
                        llm_decision = 2
                        reasoning = response
                    else:
                        while True:
                            try:
                                response = generateResponse(composeTemplate(intro, type_set, truncated_col, cur_label))
                                break
                            except Exception as error:
                                print('current error is', error)
                                import time
                                time.sleep(10)
                        response = response.strip()
                        if response.split(',')[0] == 'Yes' or response.split('.')[0] == 'Yes':
                            llm_decision = 1
                            reasoning = response[5:]
                        elif response.split(',')[0] == 'No' or response.split('.')[0] == 'No':
                            llm_decision = 0
                            reasoning = response[4:]
                        else:
                            llm_decision = 3
                            reasoning = response
                    csv_writer.writerow([table_id, inverted_dict[str(label_list[0][col_idx])], label_list[0][col_idx], cur_label, cur_label_idx, gt_decision, llm_decision, reasoning, cur_col, i])

                else:
                    response = generateResponse(composeTemplate(intro, type_set, truncated_col, cur_label))
                    response = response.strip()
                    if response.split(',')[0] == 'Yes' or response.split('.')[0] == 'Yes':
                        llm_decision = 1
                        reasoning = response[5:]
                    elif response.split(',')[0] == 'No' or response.split('.')[0] == 'No':
                        llm_decision = 0
                        reasoning = response[4:]
                    else:
                        llm_decision = 3
                        reasoning = response
                    if softmax_score >= confidence:
                        llm_decision = 2
                    csv_writer.writerow([table_id, inverted_dict[str(label_list[0][col_idx])], label_list[0][col_idx], cur_label, cur_label_idx, gt_decision, llm_decision, reasoning, cur_col, i])


def identify_indices_from_csv(csv_path):

    '''
    Input:
        csv_path (str): The csv file that stores the GPT query results.
        table_col_mapping_dict (dict): The dict that stores the table_idx to a list of col_idx it contains.
    Output:
        identified_indices (1d-array): The list of weak column indices identified.
    
    This function identifies the weak columns based on the response from GPT-4.
    '''

    identified_indices = []
    with open(csv_path) as f:
        for i, row in enumerate(csv.reader(f, skipinitialspace=True)):
            if i == 0:
                continue
            else:
                llm_decision = int(row[6])
                if llm_decision == 0 or llm_decision == 3:
                    identified_indices.append(int(row[9]))
    
        if len(identified_indices) == 0: 
            row_idx = random.choice(range(1,len(csv.reader(f, skipinitialspace=True))))
            row = csv.reader(f, skipinitialspace=True)[row_idx]
            identified_indices.append(int(row[9]))
            print('No wrong answer')
    
    f.close()
    return identified_indices

def random_select_samples(cluster_values, to_be_sampled, identified):

    '''
    Input:
        cluster_values (2-d array): The array that contains the column indices in each cluster. e.g., [[1,2,3], [4,5,6]]
        to_be_sampled (int): The number of columns remained to be sampled.
        identified (1-d array): The already identified weak columns.
    Output:
        identified+select (1-d array): The resulting selected columns.

    This function samples the remaining columns randomly.
    '''

    flattened = []
    for item in cluster_values:
        flattened += list(item)
    init_size = len(identified)
    choices = list(set(flattened)-set(identified))
    select = random.sample(choices, to_be_sampled)
    return identified+select

def identify_weak_columns(cluster_values, weak_sample_size, identified_indices):

    '''
    Input:
        cluster_values (2-d array): The array that contains the column indices in each cluster. e.g., [[1,2,3], [4,5,6]]
        weak_sample_size (int): The number of columns we want to identify.
        identified_indices (1-d array): The weak columns identified by GPT-4.
    Output:
        out_indices (1-d array): The selected output identified columns.

    This function randomly selects the identified weak columns based on the identified columns selected by GPT-4 and the k-means clusters.
    '''

    num_clusters = len(cluster_values)
    if len(identified_indices) == 0:
        return random_select_samples(cluster_values, weak_sample_size, identified_indices)
    else:
        sample_ratio = int(weak_sample_size/len(identified_indices)) 
    if sample_ratio < 1: # identified weak columns is more than the sample size
        return random.sample(identified_indices, weak_sample_size)
    flattened_cluster_values = []

    for item in cluster_values:
        flattened_cluster_values += list(item)
    num_cluster_items = len(flattened_cluster_values)
    out_indices = identified_indices
    if num_cluster_items <= weak_sample_size:
        # If the total number of columns identified is less than the weak sample size
        return list(set(flattened_cluster_values))
    else:
        if sample_ratio == 1:
            to_be_sampled = weak_sample_size - len(identified_indices)
            return random_select_samples(cluster_values, to_be_sampled, identified_indices)
        else:
            sample_size_each_cluster = sample_ratio - 1
            for cluster_item in cluster_values:
                removed_identified = list(set(cluster_item)-set(identified_indices))
                num_identified_samples = len(cluster_item) - len(removed_identified)           
                num_sample_cur_cluster = num_identified_samples*sample_size_each_cluster
                if num_sample_cur_cluster >= len(removed_identified):
                    out_indices += removed_identified
                else:
                    select_cur_cluster = random.sample(removed_identified, num_sample_cur_cluster)
                    out_indices += select_cur_cluster
            to_be_sampled = weak_sample_size - len(out_indices)
            if to_be_sampled == 0:
                return out_indices
            else:
                return random_select_samples(cluster_values, to_be_sampled, out_indices)

def LLM_transfer_clustering(model, dataset, current_exploration_round, query_size, sample_size, csv_data_path, label_dict, round, sampled_data_path):
    
    '''
    Input: 
        model (skeleton.base_model): current round fine-tuned model.
        dataset (dataset): the target domain dataset used, where we can generate labeled training samples.
        current_exploration_round (int): the current round number, use to identify the sampled idx files that we need to load currently.
        query_size (int): an integer indicating the number of tables selected for querring LLM.
        sample_size (int): an integer indicating the number of sampled labeled columns.
        csv_data_path (str): the path to the input data file for the target domain training data.
    Output:
        indentified_cluster_col_idx (1-d array): the list of identified column indices.

    This function is called in rounds by the upper level function, it identify the tables to ask for annotation from model and GPT-4.
    Based on the feedback, it identify the unlabeled columns from the training data by clustering, and ask for ground truth labels.
    '''

    data_iter = data.DataLoader(dataset=dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=dataset.pad)

    model.eval()
    soft_fn = torch.nn.Softmax(dim=0)

    label_set = []

    softmax_scores_cluster = []

    dataset_size = len(dataset)

    cur_round_idx = sample_cur_dataset_idx(current_exploration_round, dataset_size, query_size, sampled_data_path, random_state=42)

    for i, item in enumerate(tqdm(data_iter)):
        ids, rels, subs, labels, _ = item
        prediction = model(ids.to(model.device), rels.to(model.device), subs.to(model.device))
        for cur_idx in range(len(prediction)):
            softmax_score = soft_fn(prediction[cur_idx]).cpu().detach().numpy().tolist()
            label_set.append(labels[cur_idx].item())
            softmax_scores_cluster.append(softmax_score)
            # maps table idx to the list of column idxs it contains for the use of calling query GPT-4 and transforming the table idx to column idx in the output
    csv_path = '../gpt-logs/reca-p2v-' + str(round)+'.csv'
    
    if not os.path.exists(csv_path):
        get_annotation(model, dataset, label_dict, round, cur_round_idx, csv_path, confidence=0.9, mode='trust')
    identified_indices = identify_indices_from_csv(csv_path)  # need to be updated based on the output of the GPT query results. Identified indices are the indices for columns.
    #######################################################################################################

    num_clusters = len(list(set(label_set)))

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(softmax_scores_cluster)

    kmeans_dict = {}
    for idx, kmeans_label in enumerate(kmeans.labels_):
        if not kmeans_label in kmeans_dict.keys():
            kmeans_dict[kmeans_label] = [idx]
        else:
            kmeans_dict[kmeans_label].append(idx)
    
    cluster_values = []
    cluster_values_str = []
    for identified_idx in identified_indices:
        cluster_key = kmeans.labels_[identified_idx]
        if not str(kmeans_dict[cluster_key]) in cluster_values_str:    
            cluster_values.append(kmeans_dict[cluster_key])
            cluster_values_str.append(str(kmeans_dict[cluster_key]))
    
       
    indentified_cluster_col_idx = identify_weak_columns(cluster_values, sample_size, identified_indices)


    return cur_round_idx, indentified_cluster_col_idx

def metric_fn(preds, labels):

    '''
    Input:
        preds (1d-array): The annotations provided by the model.
        labels (1d-array): The ground truth labels.
    Output:
        The support-weighted F1 scores and the Macro average F1 scores.

    This function computes the two metrics based on the annotations provided by the model.
    '''

    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def LLM_finetuning(model, dataset_iter, valid_iter, model_save_path, optimizer, epochs, cur_best=100, previous_dataloaders=[], finetune_flag=False):

    '''
    Input:
        model (skeleton.base_model): The fine-tuned model.
        dataset_iter (DataLoader): The dataloader used for fine-tuning.
        optimizer (Optimizer): The optimizer used to train the model.
        epochs (int): The training epochs for the current round of weak columns.
    Output:
        model (skeleton.base_model): The fine-tuned model.

    This function fine-tunes the model with the current round of training samples.
    '''

    criterion = torch.nn.CrossEntropyLoss()
    
    cur_best_model = model
    
    if len(previous_dataloaders) == 0: # pre-train mode
        previous_dataloaders.append(dataset_iter)
    print('---------------------------------------')
    times = []
    for epoch in range(epochs):
        start = time.time()
        model.train()
        label_list = []
        predicted_labels = []
        for cur_dataset_iter in previous_dataloaders:
            for i, item in enumerate(tqdm(cur_dataset_iter)):
                optimizer.zero_grad()
                ids, rels, subs, labels = item
                
                prediction = model(ids.to(model.device), rels.to(model.device), subs.to(model.device))
                loss = criterion(prediction, labels.to(model.device))
                predicted_label = torch.argmax(prediction, dim=1)
                label_list.append(labels.detach().numpy())
                predicted_labels += predicted_label.detach().cpu().numpy().tolist()
                loss.backward()
                optimizer.step()
                if i % 100 == 0: # monitoring
                    print(f"    fine tune step: {i}, loss: {loss.item()}")
                del loss
        pred_labels = np.concatenate([np.expand_dims(i, axis=0) for i in predicted_labels]) 
        true_labels = np.concatenate(label_list)
        print(len(true_labels))
        f1_scores = metric_fn(pred_labels, true_labels)
        print("Training set weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
        end = time.time()
        times.append(end-start)
        model.eval()

        v_sw, v_ma, cur_best_loss = LLM_evaluate(model, valid_iter, model_save_path, cur_best_loss=cur_best)
        if not finetune_flag:
            if cur_best_loss < cur_best:
                cur_best = cur_best_loss
                cur_best_model = model
        else:
            cur_best_model = model
    
    print('training times:', times)

    return cur_best_model, cur_best

def LLM_evaluate(model, iter, model_save_path, is_test=False, cur_best_loss=100):

    '''
    Input:
        model (skeleton.base_model): The current round of model for evaluation.
        iter (dataloader): The validation set dataloader.
        model_save_path (str): The path to the saved model.
        is_test (bool): The flag indicating whether the evaluate is test or validation.
        cur_best_loss (int): The loss for selecting the current best validation model.
    Output:
        f1_scores['weighted_f1'] (float): The support weighted F1 score.
        f1_scores['macro_f1'] (float): The macro average F1 score.
        cur_best_loss (float): The current best validation loss.
    
    This function validates/tests the current round fine-tuned model by selecting the model with the smallest validation loss and computes the F1 scores.
    '''
    if is_test:
        model.load_state_dict(torch.load(model_save_path))
    model.eval()
    label_list = []
    predicted_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()
    v_epoch_loss = 0

    start = time.time()
    for i, item in enumerate(tqdm(iter)):
        ids, rels, subs, labels = item
        prediction = model(ids.to(model.device), rels.to(model.device), subs.to(model.device))
        vloss = loss_fn(prediction, labels.to(model.device))
        v_epoch_loss += vloss.item()
        predicted_label = torch.argmax(prediction, dim=1)
        label_list.append(labels.detach().numpy())
        predicted_labels += predicted_label.detach().cpu().numpy().tolist()
        del ids, rels, subs, vloss
    v_length_label = len(label_list)
    v_total_loss = v_epoch_loss / (v_length_label)
    print("loss:", v_total_loss)
    end = time.time()
    print('validation time:', end-start)
    if v_total_loss < cur_best_loss:
        cur_best_loss = v_total_loss
        if not is_test:
            torch.save(model.state_dict(), model_save_path)
            print('model updated')
    
    pred_labels = np.concatenate([np.expand_dims(i, axis=0) for i in predicted_labels]) 
    true_labels = np.concatenate(label_list)
    f1_scores = metric_fn(pred_labels, true_labels)
    print("Validation / Testing weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1'], cur_best_loss

def transfer_training_process(hp, csv_data_path, validation_path, test_path, label_dict, model, dataset, exploring_rounds, total_finetune_epochs, previous_dataloaders = []):
    
    sampled_data_path = '../sampled_data/reca-p2v/'
    validation_set = SupAnnDatasetIndex(validation_path, max_length = 128, size = 10, size_ratio=True, pickle_path = '/export/data/ysunbp/LLM-veri/data/reca-pickle/K0.pkl')
    validation_data_iter = data.DataLoader(dataset=validation_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=validation_set.pad)
    
    test_set = SupAnnDatasetIndex(test_path, max_length = 128, size = 20, size_ratio=True, pickle_path = '/export/data/ysunbp/LLM-veri/data/reca-pickle/K2.pkl')
    test_data_iter = data.DataLoader(dataset=test_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=test_set.pad)
    model_save_path = hp.save_path
    sampled_idx = torch.tensor([])
    overall_identified_col_idx = []
    
    if hp.design == 3: # 每一轮identify weak然后fine-tune 5 epochs
        print('checking with lr = ', hp.w_lr)
        finetune_epochs = 0
        swf1, maf1, _ = LLM_evaluate(model, test_data_iter, model_save_path, is_test=True, cur_best_loss=100)
        print('Support Weighted F1 score is:', swf1, 'Macro Average F1 score is:', maf1)
        optimizer = AdamW(model.parameters(), lr=hp.w_lr)
        decayRate = 1.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
        early_stop = 5
        finetune_flag = False
        cur_best = 100
        no_improve = 0 ###
        
        for round in range(exploring_rounds): ###
            print('current round', round)
            overall_identified_col_idx = [] 
            current_round_idx, indentified_cluster_col_idx = LLM_transfer_clustering(model, dataset, round, hp.query_size, hp.sample_size, csv_data_path, label_dict, round, sampled_data_path)

            with open(sampled_data_path+'sampled_tensor_'+str(round)+'.pkl', 'wb') as file:
                pickle.dump(current_round_idx, file)
            with open(sampled_data_path+'sampled_tensor_'+str(round)+'.pkl', 'rb') as file:
                current_round_idx = pickle.load(file)

            overall_identified_col_idx += indentified_cluster_col_idx

            selected_dataset = SupAnnDatasetIndex(csv_data_path, size=None, max_length = 128, cur_selected_indices=overall_identified_col_idx, pickle_path = '/export/data/ysunbp/LLM-veri/data/reca-pickle/K1.pkl')
            selected_data_iter = data.DataLoader(dataset=selected_dataset,
                                            batch_size=8,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=selected_dataset.pad)

            with open(sampled_data_path+'round-'+str(round)+'-selected.pkl', 'wb') as file:
                pickle.dump(selected_data_iter, file)
            with open(sampled_data_path+'round-'+str(round)+'-selected.pkl', 'rb') as file:
                selected_data_iter = pickle.load(file)
            
            previous_dataloaders.append(selected_data_iter)
            
            model, cur_best_val_loss = LLM_finetuning(model, selected_data_iter, validation_data_iter, model_save_path, optimizer, epochs=total_finetune_epochs, cur_best = cur_best, previous_dataloaders=previous_dataloaders)
            

            swf1, maf1, _ = LLM_evaluate(model, test_data_iter, model_save_path, is_test=True, cur_best_loss=100)
            print('Support Weighted F1 score is:', swf1, 'Macro Average F1 score is:', maf1)
            lr_scheduler.step()
            
            if cur_best_val_loss < cur_best:
                cur_best = cur_best_val_loss
                no_improve = 0
                torch.save(model.state_dict(), '../checkpoints/reca-p2v-cur_best.pkl') # in case azure problem, for continue training
            else:
                no_improve += 1
            
            if no_improve >= early_stop:
                print('The LLM transfer knowledge is used up! Shift to fine-tuning mode.')
                torch.save(model.state_dict(), '../checkpoints/reca-p2v-stop-at-'+str(round)+'-'+'.pkl')
                break
        finetune_flag = True

        if finetune_flag:
            model.load_state_dict(torch.load('../checkpoints/doduo-p2v-cur_best.pkl'))  ###
            full_set = SupAnnDatasetIndex(csv_data_path, max_length = 128, pickle_path = '../data/RECA/K1.pkl')
            full_dataset_iter = data.DataLoader(dataset=full_set,
                                                batch_size=8,
                                                shuffle=False,
                                                num_workers=0,
                                                collate_fn=full_set.pad)
            
            model, _ = LLM_finetuning(model, full_dataset_iter, validation_data_iter, model_save_path, optimizer, epochs=20, cur_best=cur_best, previous_dataloaders=[full_dataset_iter], finetune_flag=finetune_flag)
            swf1, maf1, _ = LLM_evaluate(model, test_data_iter, model_save_path, is_test=True, cur_best_loss=100)
            print('Support Weighted F1 score is:', swf1, 'Macro Average F1 score is:', maf1)
        out_model = model
    elif hp.design == 2: 
        cur_best = 100
        optimizer = AdamW(model.parameters(), lr=hp.w_lr)
        previous_dataloaders = []
        current_training_set = SupAnnDatasetIndex(csv_data_path, size=None, max_length = 128, size_column=True, pickle_path = '../data/RECA/K1.pkl')
        current_dataset_iter = data.DataLoader(dataset=current_training_set,
                                                batch_size=8,
                                                shuffle=False,
                                                num_workers=0,
                                                collate_fn=current_training_set.pad)
        for i in range(50):
            with open(sampled_data_path+'round-'+str(i)+'-selected.pkl', 'rb') as file:
                selected_data_iter = pickle.load(file)
            previous_dataloaders.append(selected_data_iter)
        previous_dataloaders.append(current_dataset_iter)
        model_path = '../checkpoints/doduo-p2v-cur_best.pkl'
        model.load_state_dict(torch.load(model_path))
        finetune_flag = True
        model, _ = LLM_finetuning(model, current_dataset_iter, validation_data_iter, model_save_path, optimizer, epochs=20, cur_best=cur_best, previous_dataloaders=previous_dataloaders, finetune_flag=finetune_flag)
        swf1, maf1, _ = LLM_evaluate(model, test_data_iter, model_save_path, is_test=True, cur_best_loss=100)
        print('Support Weighted F1 score is:', swf1, 'Macro Average F1 score is:', maf1)
        out_model = model
    
    return out_model

def adjust_label_diff(source_label_path, target_label_path, pretrained_model_path):
    print('Use the adjusted knowledge from the source')
    with open(source_label_path, 'r', encoding='utf-8') as file:
        source_label_dict = json.load(file)
    with open(target_label_path, 'r', encoding='utf-8') as file:
        label_dict = json.load(file)
    hp.n_classes = len(list(label_dict.keys()))
    model = base_model(hp, device=device, lm=hp.lm) # reload the model on the target data lake
    source_labels = list(source_label_dict.keys())
    target_labels = list(label_dict.keys())
    updated_target_labels = [s.lower() for s in target_labels]
    overlapping = list(set(updated_target_labels).intersection(set(source_labels)))
    source2target_mapping = []
    for item in overlapping:
        source2target_mapping.append((int(source_label_dict[item]), int(label_dict[target_labels[updated_target_labels.index(item)]])))
    
    pretrained_model_dict = torch.load(pretrained_model_path)
    current_model_dict = model.state_dict()
    state_dict = {k:v for k, v in pretrained_model_dict.items() if k in list(pretrained_model_dict.keys())[:-2]}
    current_model_dict.update(state_dict)
    for s,t in source2target_mapping:
        current_model_dict[list(pretrained_model_dict.keys())[-2]][t] = pretrained_model_dict[list(pretrained_model_dict.keys())[-2]][s]
        current_model_dict[list(pretrained_model_dict.keys())[-1]][t] = pretrained_model_dict[list(pretrained_model_dict.keys())[-1]][s]
    model.load_state_dict(current_model_dict)
    print('number of overlapping type adjusted: ', len(source2target_mapping))
    return model, label_dict

if __name__ == '__main__':
    setup_seed(1998)
    type_set = 'Here are the set of semantic types we consider: [name, rank, city, year, status, age, club, class, team, result, company, symbol, artist, location, notes, weight, type, day, state, description, address, category, position, person, code, plays, gender, service, album, duration, format, area, range, elevation, currency, credit, depth, filesize, component, country, county, industry, product, teamname, birthdate, sex, jockey, owner, publisher, language, nationality, affiliation, origin, creator, order, affiliate, collection, family, capacity, classification, grades, birthplace, requirement, species, ranking, region, isbn, genre, brand, religion, manufacturer, continent, command, operator, education, director, organisation, sales] \n'
    intro = 'Use your domain knowledge to verify the semantic types. If you don\'t know, response \'I don\'t know\'\n'
    cur_design = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lm", type=str, default='bert') # bert
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--w_lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--design", type=int, default=cur_design)
    parser.add_argument("--query_size", type=int, default=int(50*2.3))
    parser.add_argument("--sample_size", type=int, default=25)
        
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument('--projector', default='768', type=str,
                            metavar='MLP', help='projector MLP')
    parser.add_argument("--n_classes", type=int, default=78)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--exploring_rounds", type=int, default=50)
    
    parser.add_argument("--save_path", type=str, default='../checkpoints/reca-p2v.pkl')
    
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = KREL()
    model = model.to(model.device)
    
    dict_path = '../data/VizNet/label.json'
    with open(dict_path, 'r', encoding='utf-8') as file:
        label_dict = json.load(file)

    public_bi_path = '../checkpoints/out_model_p2v' # need to run ./reca/train-publicbi.py if you want to retrain on publicbi
    model.load_state_dict(torch.load(public_bi_path))
    sato_test_path = '../data/RECA/2.jsonl'
    testing_set = SupAnnDatasetIndex(sato_test_path, max_length = 128, size=500, pickle_path = '../data/RECA/K2.pkl')
    
    test_iter = data.DataLoader(dataset=testing_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=testing_set.pad)
    LLM_evaluate(model, test_iter,  hp.save_path, is_test=True, cur_best_loss=100)
                                       

    sato_train_path = '../data/RECA/1.jsonl'
    sato_valid_path = '../data/RECA/0.jsonl'
    sato_test_path = '../data/RECA/2.jsonl'
    transfer_finetune_dataset = SupAnnDatasetIndex(sato_train_path, size=None, max_length = 128, pickle_path = '../data/RECA/K1.pkl')
    transfer_training_dataloader = data.DataLoader(dataset=transfer_finetune_dataset,
                                        batch_size=8,
                                        shuffle=True,
                                        num_workers=0,
                                        collate_fn=transfer_finetune_dataset.pad)
    validation_set = SupAnnDatasetIndex(sato_valid_path, max_length = 128, size=500, pickle_path = '../data/RECA/K0.pkl')
        
    valid_iter = data.DataLoader(dataset=validation_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=validation_set.pad)
    testing_set = SupAnnDatasetIndex(sato_test_path, max_length = 128, size=500, pickle_path = '../data/RECA/K2.pkl')
    test_iter = data.DataLoader(dataset=testing_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=testing_set.pad)
    if not os.path.exists('../checkpoints/base-reca-p2v-finetuned.pkl'):
        print('Start initial fine-tuning on VizNet')
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        model, _ = LLM_finetuning(model, transfer_training_dataloader, valid_iter, hp.save_path, optimizer, hp.n_epochs)
        LLM_evaluate(model, test_iter, hp.save_path, is_test=True, cur_best_loss=100)
        torch.save(model.state_dict(),'../checkpoints/base-reca-p2v-finetuned.pkl')
    else:
        model.load_state_dict(torch.load('../checkpoints/base-reca-p2v-finetuned.pkl'))
        
    ### transfering knowledge on VizNet ###
    print('Storing model at: ', hp.save_path)
    torch.save(model.state_dict(), hp.save_path)
    annotation_dataset = SupAnnDataset(sato_train_path, dict_path, size=None, max_length = hp.max_len, pickle_path = '/export/data/ysunbp/LLM-veri/data/reca-pickle/K1.pkl') # used for sample queries from GPT
    previous_dataloaders = [transfer_training_dataloader]
    transfer_training_process(hp, sato_train_path, sato_valid_path, sato_test_path, label_dict, model, annotation_dataset, exploring_rounds=hp.exploring_rounds, total_finetune_epochs=5, previous_dataloaders=previous_dataloaders)