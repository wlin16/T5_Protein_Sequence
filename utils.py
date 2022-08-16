import config

import csv
from tqdm import tqdm
import pickle
import os
from termcolor import colored
import dataframe_image as dfi

import numpy as np
import pandas as pd
import re
import torch

import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from transformers import T5Tokenizer
from transformers import T5EncoderModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def dataset_split(data_X, data_y):
    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                        test_size=0.2,
                                                        stratify=data_y,
                                                        random_state=config.SEED)
    # 切分出valid数据集
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                        test_size=0.7,
                                                        shuffle=True,
                                                        stratify=y_test,
                                                        random_state=config.SEED)
    return X_train, X_test, y_train, y_test, X_valid, y_valid
    # return X_train, X_test, y_train, y_test


def tokenize(data):
    tokenizer = config.tokenizer
    token_data = tokenizer.batch_encode_plus(
        data.tolist(),
        max_length=config.max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    return token_data


def token_to_tensor(token_data, label):
    id = torch.tensor(token_data['input_ids'])
    mask = torch.tensor(token_data['attention_mask'])
    labels = torch.tensor(label.tolist())

    return id, mask, labels


def data_loader(id, mask, labels, batch_size=config.batch_size, type='eval'):
    batch_size = batch_size
    tensor_data = TensorDataset(id, mask, labels)

    if type == 'train':
        train_sampler = RandomSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=train_sampler, batch_size=batch_size)
    else:
        val_sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=val_sampler, batch_size=batch_size)

    return dataloader


# For model testing
def predict(test_loader, model, device=config.device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for batch in tqdm(test_loader):
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(b_input_ids, b_attn_mask)
            preds.append(pred.detach().cpu())  # 放入cpu计算
    preds = torch.cat(preds, dim=0)
    # print(preds)
    preds = torch.argmax(preds, dim=1)
    return preds


# Save model prediction results
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# Define a get embedding function for T5

def get_embedding_large_GPU(protein_seq, save_path, start=None, stop=None, device = config.device):
    tokenizer = config.tokenizer
    model = config.model
    
    xs = []
    result = None
    count = 0
    embed_error_count = 0
    protein_seq = protein_seq[start:stop]
    data_len = len(protein_seq)

    for index, seq in tqdm(protein_seq.iterrows(), total=protein_seq.shape[0]):
        s_len = len(seq['wt_seq'].replace(" ",'')) + 1
        aa_index = seq['aa_index']
        label = seq['label']
        wt_aa = seq['wt']
        mt_aa = seq['mt']
        wt_seq = seq['wt_seq']
        mt_seq = seq['mt_seq']
        # AF_DB = seq['AlphaFoldDB']
        gene_id = seq['gene_id']
        
        # add_special_tokens adds extra token at the end of each sequence
        token_encoding = tokenizer.batch_encode_plus([seq['wt_seq'], seq['mt_seq']], add_special_tokens=True, padding="longest")
        input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        with torch.no_grad():
            # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
            embedding_repr = model(input_ids, attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state[:, :s_len]
            # emb = emb[:, aa_index, :]
            try:
                emb = emb[:, aa_index, :]
            except:
                embed_error_count += 1
                print(f'embedding error: index: {index}, aa_index:{aa_index}, aa_length: {s_len} , error_count:{embed_error_count}')
                
            # print(aa_index)
            x = emb.detach().cpu().numpy().squeeze()
           
            temp = pd.DataFrame({
                'gene_id':gene_id,
                'mutant_index': aa_index,
                'wt_aa': wt_aa,
                't_aa': mt_aa,
                'wt_seq': wt_seq,
                'mt_seq': mt_seq,
                'wt_emb': [x[0, :]],
                'mt_emb':[x[1,:]],
                'label':label
                # 'AF_DB': AF_DB,
                # 'PDB_ID': PDB
               
            })
            
            if result is None:
                result=temp
            else:
                result = pd.concat([result,temp])

            xs.append({'x':x.reshape(1,-1), 'label':label})
            
    # Save results
    if not os.path.isdir(f'{save_path}'):
        os.mkdir(f'{save_path}')
            
    if start is None:
        result.to_csv(f'{save_path}/sequence_embeddings({data_len}).csv', index=False)
        with open(f'{save_path}/emb({data_len}).pkl', 'wb') as f:
            pickle.dump(xs, f)
    else:
        result.to_csv(f'{save_path}/sequence_{stop}_embeddings.csv', index=False)
        with open(f'{save_path}/emb_{stop}.pkl', 'wb') as f:
            pickle.dump(xs, f)

# generate embeddings in batch:
def embed_in_batch_for_large(protein_seq, amount, path):
    value_input = amount
    data_len = len(protein_seq)
    fold = data_len // value_input
    remainder = data_len - data_len % value_input

    for i in range(fold):
        get_embedding_large_GPU(protein_seq,  path, start = i* value_input, stop = (i+1)*value_input)
    
    get_embedding_large_GPU(protein_seq, path, start = remainder, stop = data_len)


# get embeddings for downstream tasks
# def embeds_for_tasks(path=config.embed_path):
#     with open(path, 'rb') as file:
#         y = pickle.load(file)

#     data_y = []
#     data_X = []

#     for i in range(len(y)):
#         data_X.append(y[i]['x'][0])
#         data_y.append(y[i]['label'])

#     data_X = np.array(data_X)

#     return data_X, data_y


# get embeddings in batch for downstream tasks
def data_for_downstream_for_large():
    path = config.model_2_embed_path + '/'
    concat = []
    for pkl in os.listdir(path):
        if ".pkl" in pkl:
            file_path = path + pkl
            with open(file_path, 'rb') as file:
                y = pickle.load(file)
                concat += y
    data_y = []
    data_X = []
    for i in range(len(concat)):
        data_X.append(concat[i]['x'][0])
        data_y.append(int(concat[i]['label']))
    data_X = np.array(data_X)

    return data_X, data_y





def get_embedding_for_small_GPU(protein_seq,  save_path, start=None, stop=None, device=config.device):
    tokenizer = config.tokenizer
    model = config.model
    
    xs = []
    result = None
    count = 0
    embed_error_count = 0
    protein_seq = protein_seq[start:stop]
    data_len = len(protein_seq)

    for index, seq in tqdm(protein_seq.iterrows(),total=protein_seq.shape[0]):
        s_len = len(seq['wt_seq'].replace(" ",'')) + 1
        aa_index = seq['aa_index']
        label = seq['label']
        wt_aa = seq['wt']
        mt_aa = seq['mt']
        wt_seq = seq['wt_seq']
        mt_seq = seq['mt_seq']
        # AF_DB = seq['AlphaFoldDB']
        # PDB = seq['PDB']
        # pathogenicity = seq['pathogenicity']
        
        # add_special_tokens adds extra token at the end of each sequence
        # token_encoding = tokenizer.batch_encode_plus([seq['wt_seq'], seq['mt_seq']], add_special_tokens=True, padding="longest")
        wt_token_encoding = tokenizer.batch_encode_plus([seq['wt_seq']], add_special_tokens=True, padding="longest")
        wt_input_ids      = torch.tensor(wt_token_encoding['input_ids']).to(device)
        wt_attention_mask = torch.tensor(wt_token_encoding['attention_mask']).to(device)
        
        mt_token_encoding = tokenizer.batch_encode_plus([seq['mt_seq']], add_special_tokens=True, padding="longest")
        mt_input_ids      = torch.tensor(mt_token_encoding['input_ids']).to(device)
        mt_attention_mask = torch.tensor(mt_token_encoding['attention_mask']).to(device)

        with torch.no_grad():
            # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
            wt_embedding_repr =model(wt_input_ids, attention_mask=wt_attention_mask)
            wt_emb = wt_embedding_repr.last_hidden_state[:, :s_len]
            wt_emb = wt_emb[:, aa_index, :]
            wt = wt_emb.detach().cpu().numpy().squeeze()
            
            mt_embedding_repr =model(mt_input_ids, attention_mask=mt_attention_mask)
            mt_emb = mt_embedding_repr.last_hidden_state[:, :s_len]
            mt_emb = mt_emb[:, aa_index, :]
            mt = mt_emb.detach().cpu().numpy().squeeze()

            xs.append({'wt':wt.reshape(1,-1),'mt':mt.reshape(1,-1), 'label':label})
            
    # Save results
    if not os.path.isdir(f'{save_path}'):
        os.mkdir(f'{save_path}')
            
    if start is None:
        # result.to_csv(f'{save_path}/emb_({data_len}).csv', index=False)
        with open(f'{save_path}/emb({data_len}).pkl', 'wb') as f:
            pickle.dump(xs, f)
    else:
        # result.to_csv(f'{save_path}/emb_{stop}.pkl.csv', index=False)
        with open(f'{save_path}/emb_{stop}.pkl', 'wb') as f:
            pickle.dump(xs, f)
    
def embed_in_batch_for_small(protein_seq, amount, path):
    value_input = amount
    data_len = len(protein_seq)
    fold = data_len // value_input
    remainder = data_len - data_len % value_input

    try:
        for i in range(fold):
            get_embedding_for_small_GPU(protein_seq,  save_path = path, start = i* value_input, stop = (i+1)*value_input)
        
        get_embedding_for_small_GPU(protein_seq, save_path = path, start = remainder, stop = data_len)
    except Exception as e:
        print(e)
        os.system('tput bel')
    
def data_for_downstream_for_small():
    path = config.model_2_embed__1080_path + '/'
    concat = []
    for pkl in os.listdir(path):
        if(".pkl" in pkl):
            file_path = path + pkl
            with open(file_path, 'rb') as file:
                y = pickle.load(file)
                concat += y
    data_y = []
    data_wt = []
    data_mt = []
    for i in range(len(concat)):
        data_wt.append(concat[i]['wt'][0])
        data_mt.append(concat[i]['mt'][0])
        data_y.append(int(concat[i]['label']))
    data_wt = np.array(data_wt)
    data_mt = np.array(data_mt)
    data_X = np.hstack((data_wt,data_mt))
    return data_X, data_y



# Traditional ML training & Save the corresponded results
def traditional_model(name, X_train, y_train, X_test, y_test, early_stopping_rounds=None, eval_set=None, verbose=None):
    model_name = re.search(r"(.*)(\(.*)", str(name))
    model_name = model_name.group(1)

    name.fit(X_train, y_train)
    y_pred = name.predict(X_test)

    report_save(y_test, y_pred, model_name)
    


def report_save(y_true, y_pred, name, result_path = config.model_2_result_path, label_names=None, *args, **kv):
    # print the classification report here
    report = classification_report(y_true, y_pred, target_names=label_names)
    print(colored(
        f'\n\t\t\t\t *** {name}_report ***:\n\n\n', 'blue', attrs=['bold']), report)
    MCC = matthews_corrcoef(y_true, y_pred)
    print(f"{name} MCC:", MCC)
    # create report dataframe
    report_for_save = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True)
    report_csv = pd.DataFrame(report_for_save)
    report_csv = pd.DataFrame(report_for_save).transpose()
    report_csv['MCC'] = MCC

    # style.background_gradient or highlight_max
    report_styled = report_csv.style.background_gradient(
        subset=['precision', 'recall', 'f1-score'])

    # create folder to store results
    if not os.path.isdir(f'{result_path}'):
        os.mkdir(f'{result_path}')

        # export dataframe to .png
    # dfi.export(report_styled, f'{config.result_path}/{name}_report.png')

    report_csv.to_csv(f'{result_path}/{name}_report_save.csv')
    
    
    # Bayesian Optimization implementation based on HYPEROPT
