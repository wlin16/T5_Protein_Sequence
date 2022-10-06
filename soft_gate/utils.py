import os
import pickle
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

import sys

import config

# soft gate

# tokenizer = config.tokenizer
# model = config.model
def get_embedding(protein_test, model, tokenizer, device=config.device):
    aa_emb = []
    seq_emb = []
    result = None
    count = 0
    embed_error_count = 0
    label = []
    # protein_seq = protein_seq[start:stop]
    data_len = len(protein_test)

    for index, seq in tqdm(protein_test.iterrows(),total=protein_test.shape[0]):
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
            wt_seq = wt_emb.detach().cpu().numpy()
            # print(wt_seq.shape)


            mt_embedding_repr =model(mt_input_ids, attention_mask=mt_attention_mask)
            mt_emb = mt_embedding_repr.last_hidden_state[:, :s_len]
            mt_seq = mt_emb.detach().cpu().numpy()
        
            embedding_dict = {'wt_seq':wt_seq,
                            'mt_seq':mt_seq, 
                            'aa_index':aa_index,
                            'label':label
                            }
            seq_emb.append(embedding_dict)
    return seq_emb
    # Save results
    #     if not os.path.isdir(f'{save_path}'):
    #         os.mkdir(f'{save_path}')

    #     if start is None:
    #         # result.to_csv(f'{save_path}/emb_({data_len}).csv', index=False)
    #         with open(f'{save_path}/emb({data_len}).pkl', 'wb') as f:
    #             pickle.dump(xs, f)
    #     else:
    #         # result.to_csv(f'{save_path}/emb_{stop}.pkl.csv', index=False)
    #         with open(f'{save_path}/emb_{stop}.pkl', 'wb') as f:
    #             pickle.dump(xs, f)
    




def data_for_downstream(embed_path):
    path = embed_path + '/'
    concat = []
    for pkl in os.listdir(path):
        if(".pkl" in pkl):
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


def train_test_dataset(situation, gene_constrain, mode):
    if 'gene_balance' in situation:
        mode_train = np.load(f'../../data/{gene_constrain}/gene_balance/For_ML/{mode}_train.npy', allow_pickle=True)
        mode_test = np.load(f'../../data/{gene_constrain}/gene_balance/For_ML/{mode}_test.npy', allow_pickle=True)
        
        X_train = mode_train[:,:-1]
        y_train = mode_train[:,-1].astype(int).tolist()
        X_test = mode_test[:,:-1]
        y_test = mode_test[:,-1].astype(int).tolist()
        
        return X_train, y_train, X_test, y_test
    
    elif 'gene_not_balance' in situation:
        mode_train = np.load(f'../../data/{gene_constrain}/gene_not_balance/For_ML/{mode}_train.npy')
        mode_test = np.load(f'../../data/{gene_constrain}/gene_not_balance/For_ML/{mode}_test.npy')
        
        X_train = mode_train[:,:-1]
        y_train = mode_train[:,-1].astype(int).tolist()
        X_test = mode_test[:,:-1]
        y_test = mode_test[:,-1].astype(int).tolist()
        
        return X_train, y_train, X_test, y_test

    elif 'imbalance' in situation:
        mode_train = np.load(f'../../data/{gene_constrain}/{situation}/For_ML/{mode}_train.npy')
        mode_test = np.load(f'../../data/{gene_constrain}/{situation}/For_ML/{mode}_test.npy')

        X_train = mode_train[:,:-1]
        y_train = mode_train[:,-1].astype(int).tolist()
        X_test = mode_test[:,:-1]
        y_test = mode_test[:,-1].astype(int).tolist()
        
        return X_train, y_train, X_test, y_test
    
    else:
        mode_train1 = np.load(f'../../data/{gene_constrain}/{situation}/For_ML/{mode}_train_1.npy')
        mode_train2 = np.load(f'../../data/{gene_constrain}/{situation}/For_ML/{mode}_train_2.npy')
        mode_train3 = np.load(f'../../data/{gene_constrain}/{situation}/For_ML/{mode}_train_2.npy')
        mode_test = np.load(f'../../data/{gene_constrain}/{situation}/For_ML/{mode}_test.npy')
        
        X_train1, X_train2, X_train3 = mode_train1[:,:-1], mode_train2[:,:-1], mode_train3[:,:-1]
        y_train1, y_train2, y_train3 = mode_train1[:,-1].astype(int).tolist(), mode_train2[:,-1].astype(int).tolist(), mode_train3[:,-1].astype(int).tolist()
        X_test, y_test = mode_test[:,:-1], mode_test[:,-1].astype(int).tolist()
        
        
        return X_train1, X_train2, X_train3, y_train1, y_train2, y_train3, X_test, y_test



## Prepare datasets for models 
class TestDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.seq = torch.tensor(X)
        self.label = torch.tensor(y)
    
    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index], self.label[index]


## Set up models     
class MLPClassifier(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """
    def __init__(self,num_input,num_hidden,num_output):
        super(MLPClassifier, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden=nn.Linear(num_input,num_hidden)
        # self.dropout=nn.Dropout(0.1,inplace= False)
        self.predict = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_hidden, num_output)
            # nn.Linear(num_hidden, 768),
            # # nn.ReLU(inplace = True),
            # nn.Linear(768, 512),
            # nn.Linear(512, 256),
            # # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.Linear(64, num_output)
        )
           
    def forward(self,x):
        x=self.hidden(x)
        # x = self.dropout(x)
        x=self.predict(x)
        return x

class MLPClassifier2(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """
    def __init__(self,num_input,num_hidden,num_output):
        super(MLPClassifier2, self).__init__()
 
        # Instantiate an one-layer feed-forward classifier
        self.hidden=nn.Linear(num_input,num_hidden)
        # self.dropout=nn.Dropout(0.1,inplace= False)
        self.linear = nn.Linear(25,16)
        

        self.predict = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16, num_output)
            # nn.Linear(num_hidden, 768),
            # # nn.ReLU(inplace = True),
            # nn.Linear(768, 512),
            # nn.Linear(512, 256),
            # # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.Linear(64, num_output)
        )
    
    def forward(self,wt_emb, mt_emb,aa_index, label):
        window_size = 5
        
        wt_emb =torch.from_numpy(wt_emb) # (batch_size, seq_length, 1024)
        mt_emb = torch.from_numpy(mt_emb) # (batch_size, seq_length, 1024)
        
        wt_aa_emb = wt_emb[:,aa_index,:] # 1,1024
        mt_aa_emb = mt_emb[:,aa_index,:] # 1,1024
        
        wt_emb = self.hidden(wt_emb) # (batch_size, seq_length, 1024)
        mt_emb = self.hidden(mt_emb)

        wt_aa_emb = wt_aa_emb.unsqueeze(1).permute(0,2,1) #(k transpose)
        # print('wt_emb shape: ', wt_emb.shape)
        # print('wt_aa_emb shape: ', wt_aa_emb.shape)
        
        wt_attn_weights = torch.bmm(wt_emb,wt_aa_emb) # (bs,length,embedding_1024) , (embedding_1024, bs) = (bs,length)
        # print('attn_weights shape: ',attn_weights.shape)
        wt_attn_weights = wt_attn_weights.squeeze(2)
        wt_attention = F.softmax(wt_attn_weights, 1) # softmax(q,k)  # (bs, length, 5)
        wt_index = torch.topk(wt_attention, window_size)

        # print('attention shape: ',attention.shape) # (bs, length)
        # print('wt_emb shape: ',wt_emb.shape)
        wt_attn_out = torch.bmm(wt_emb.permute(0,2,1), wt_attention.unsqueeze(2))# softmax(q,k) * v # (bs, embedding_1024, seq_length) , (bs, seq_length,seq_length)
        print('wt_index: ', wt_index)
        # print(attn_out.shape) #(bs, embedding_1024, 1)
        # print(attn_out)
        # print(index)
        # print(wt_aa_emb)
        # print(mt_aa_emb)

        mt_aa_emb = mt_aa_emb.unsqueeze(1).permute(0,2,1) #(k transpose word embedding)
        mt_attn_weights = torch.bmm(mt_emb,mt_aa_emb) # (sequence embedding * word embedding)
        mt_attn_weights = mt_attn_weights.squeeze(2)
        mt_attention = F.softmax(mt_attn_weights, 1) # softmax(q,k)  # (bs, length, 5)
        mt_index = torch.topk(mt_attention, window_size)
        mt_attn_out = torch.bmm(mt_emb.permute(0,2,1), mt_attention.unsqueeze(2))# softmax(q,k) * v # (bs, embedding_1024, seq_length) , (bs, seq_length,seq_length)
        print('mt_index: ', mt_index)

        # get the top 5 significant aa embeds
        wt_top = []
        mt_top = []
        for i in wt_index[1][0].tolist():
            a = wt_emb[:,i,:]
            wt_top.append(a)

        for i in mt_index[1][0].tolist():
            a = mt_emb[:,i,:]
            mt_top.append(a)

        wt_top_emb = torch.cat(wt_top, dim=0)
        # wt_top_emb = torch.transpose(wt_top_emb,0,1)
        mt_top_emb = torch.cat(mt_top, dim=0)
        

        # mt_top_emb = torch.transpose(mt_top_emb,0,1)

        # new_wt_aa = wt_attn_out.squeeze(2)
        # new_mt_aa = mt_attn_out.squeeze(2)

        wt_aa_emb = wt_aa_emb.squeeze(2)
        mt_aa_emb = mt_aa_emb.squeeze(2)

        print('wt_aa_emb shape: ', wt_aa_emb.shape)
        print('mt_aa_emb shape: ', mt_aa_emb.shape)
        print('wt_top_emb shape: ', wt_top_emb.shape)
        print('mt_top_emb shape: ', mt_top_emb.shape)

        wt_top_emb = torch.cat((wt_top_emb, wt_aa_emb), dim=0)
        mt_top_emb = torch.cat((mt_top_emb, mt_aa_emb), dim=0)

        mt_top_emb = mt_top_emb.permute(1,0)

        print('wt_top_emb shape: ', wt_top_emb.shape)
        print('mt_top_emb shape: ', mt_top_emb.shape)

        dot_product = (wt_top_emb @ mt_top_emb).reshape(1,-1)
        print('dot_product shape: ', dot_product.shape)

        # print('new_wt_aa shape: ', new_wt_aa.shape)

        # feed = torch.cat((wt_top_emb, new_wt_aa, mt_top_emb, new_mt_aa), dim=0)
        # print('feed shape: ', feed.shape)

        # cos_sim = F.cosine_similarity(wt_top_emb, mt_top_emb, dim=0)
        # print(cos_sim.shape)
        # print('label: ', label)



        # normal_point = []
        sys.exit(0)
        x=self.linear(dot_product)
        # x = self.dropout(x)
        x=self.predict(x)
        return x
    

class MLPT5Classifier(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """
    def __init__(self,num_input,num_hidden,num_output):
        super(MLPT5Classifier, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden=nn.Linear(num_input,num_hidden)
        # self.dropout=nn.Dropout(0.1,inplace= False)
        self.predict = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_hidden, num_output)
            # nn.Linear(num_hidden, 768),
            # # nn.ReLU(inplace = True),
            # nn.Linear(768, 512),
            # nn.Linear(512, 256),
            # # nn.ReLU(),
            # nn.Linear(256, 64),
            # nn.Linear(64, num_output)
        )
           
    def forward(self,x):
        x=self.hidden(x)
        # x = self.dropout(x)
        x=self.predict(x)
        return x



def flat_accuracy(preds, labels):
    preds = preds.detach().cpu().numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

    
def trainer(train_loader, val_loader, model, device = config.device, early_stop = config.early_stop, n_epochs = config.n_epochs):

    criterion = nn.CrossEntropyLoss() # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay = 0.0000) 
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps= 0,
                                            num_training_steps= len(train_loader)*n_epochs)
    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = n_epochs, math.inf, 0, early_stop

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for batch in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            b_seq, b_labels = tuple(t.to(device) for t in batch) # Move your data to device. 
            print('b_seq', b_seq.shape)
            pred = model(b_seq.float())  
            loss = criterion(pred, b_labels)
            
            loss.backward()                     # Compute gradient(backpropagation).
            
            optimizer.step()                    # Update parameters.
            scheduler.step()
            
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            # print(model.classifier[3].bias)

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        wandb.log({'epoch': n_epochs, 'loss': mean_train_loss, 'step': step})

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        total_eval_accuracy = 0
        
        for batch in val_loader:
            
            b_seq, b_labels = tuple(t.to(device) for t in batch) # Move your data to device. 
            # pred = model(b_input_ids, b_attn_mask) 
            with torch.no_grad():
                pred = model(b_seq.float())
                loss = criterion(pred, b_labels)

            loss_record.append(loss.item())
            total_eval_accuracy += flat_accuracy(pred, b_labels)
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        avg_val_accuracy = total_eval_accuracy / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        wandb.log({'epoch': n_epochs,
            'val_loss': mean_valid_loss,
            'step': step,
            'accuracy': avg_val_accuracy})

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), './models/model.ckpt') # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')
            
            return

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    labels = []
    for batch in tqdm(test_loader):
        b_seq, b_labels = tuple(t.to(device) for t in batch)                   
        with torch.no_grad():                   
            pred = model(b_seq.float())                     
            preds.append(pred.detach().cpu()) # 放入cpu计算
            labels.append(b_labels.detach().cpu())
            
    preds = torch.cat(preds, dim=0)
    # print(preds)
    preds = torch.argmax(preds, dim=1)
    labels = torch.cat(labels, dim = 0)

    return preds, labels

def predict_results(y_true, preds, mode, name, gene_constrain, situation,path = config.save_path):
    label_names = {'0':0, '1':1}
    result_path = path + '/' + gene_constrain + '/' + situation + '/' + f'ML_predicted_results/{mode}_results'
    
    report = classification_report(y_true, preds,target_names=label_names)
    report_for_save = classification_report(y_true, preds,target_names=label_names, output_dict=True)
    print(report)

    
    MCC = matthews_corrcoef(y_true, preds)
    print('MCC: ', MCC)
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    print(pd.DataFrame([{'True Positive':tp,'True Negative':tn,'False Positive':fp,'False Negative':fn}]))
    
    report_csv = pd.DataFrame(report_for_save).transpose()
    report_csv['MCC'] = MCC
    report_csv['True Positive'] = tp
    report_csv['True Negative'] = tn
    report_csv['False Positive'] = fp
    report_csv['False Negative'] = fn
    
    # Save results
    if not os.path.isdir(f'{result_path}'):
        os.mkdir(f'{result_path}')
    
    report_csv.to_csv(f'{result_path}/{name}_{mode}_report_save.csv')
    
    