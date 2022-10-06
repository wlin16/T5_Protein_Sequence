
import pandas as pd
import re
import os
import wandb
import pdb

from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
import torch

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

import utils
import config
from utils import MLPClassifier2

# def run(gene_constrain, situation, mode):
# ==================== Data for Downstream Task ====================================
# seq_emb, aa_emb = utils.get_embedding()

# # read embeddings for downstream tasks


# X_train, y_train, X_test, y_test = utils.train_test_dataset(situation, gene_constrain, mode)

# # 切分数据集
# # X_train, X_test, y_train, y_test= train_test_split(data_X, data_y,
# #                                                     test_size=0.2,
# #                                                     stratify=data_y,
# #                                                    random_state=SEED)
# # 切分出valid数据集
# X_valid, X_test, y_valid, y_test = utils.train_test_split(X_test,y_test,
#                                             test_size=0.3,
#                                             shuffle=True,
#                                             stratify=y_test,
#                                             random_state=config.SEED)

# train_dataset = utils.TestDataset(X_train, y_train)
# test_dataset = utils.TestDataset(X_test, y_test)
# val_dataset = utils.TestDataset(X_valid, y_valid)

# train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
# val_loader = DataLoader(val_dataset, batch_size=config.batch_size)


# #
# print('\n\n\n\n')
# print('=============== MLP model training start ===============')
# # # ==================== MLP classifier ====================================
# if gene_constrain == 'ohe_constrain':
#     input_size = 2088
    
# elif gene_constrain == 'gene_constrain':
#     input_size = 2048
# else:
#     print('constrain_error')
    
# model = utils.MLPClassifier(num_input = input_size, num_hidden = 1024, num_output = config.label_num).to(config.device)


# wandb.init(
#     project="t5-lstm",
#     config={
#         "epochs": config.n_epochs,
#         "batch_size": config.batch_size,
#         "lr": config.learning_rate,
#         "dropout": 0.1
#         })
    
# utils.trainer(train_loader, val_loader, model, config.device)
    
    
# wandb.finish()


# print('\n\n\n\n')
# print('=============== Predicting & Evaluating the trained model ===============')

# model = utils.MLPClassifier(num_input = input_size, num_hidden = 1024, num_output = config.label_num).to(config.device)
# model.load_state_dict(torch.load('./models/model.ckpt'))
# preds, y_true = utils.predict(test_loader, model, config.device)

# utils.predict_results(y_true, preds, gene_constrain = gene_constrain, mode = mode, name = 'MLP', situation=situation)


# print('\n\n\n\n')
# print('=============== No Bug No Error, Finished!!! ===============')
# os.system('tput bel')

# print('\n\n\n\n')




if __name__ == '__main__':
    
    # # run(gene_constrain = 'ohe_constrain', situation = 'gene_not_balance', mode = 'mode1')
    # # run(gene_constrain = 'ohe_constrain', situation = 'gene_not_balance', mode = 'mode2')
    # # run(gene_constrain = 'ohe_constrain', situation = 'gene_balance', mode = 'mode1')
    # # run(gene_constrain = 'ohe_constrain', situation = 'gene_balance', mode = 'mode2')
    
    # # run(gene_constrain = 'gene_constrain', situation = 'gene_not_balance', mode = 'mode1')
    # run(gene_constrain = 'gene_constrain', situation = 'gene_not_balance', mode = 'mode2')
    # # run(gene_constrain = 'gene_constrain', situation = 'gene_balance', mode = 'mode1')
    # # run(gene_constrain = 'gene_constrain', situation = 'gene_balance', mode = 'mode2')


    # ==================== Data for Downstream Task ====================================
    from transformers import T5Tokenizer, T5EncoderModel
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(config.device)
    model = model.eval()

#     pdb.set_trace()
    
    protein_seq = pd.read_csv('../data/mode1_for_embed.csv')
    # add space between each amino aicds
    protein_seq['wt_seq'] = protein_seq['wt_seq'].apply(lambda x: ' '.join(x)).apply(
            lambda x: re.sub(r"[UZOB]", "X", x))
    protein_seq['mt_seq'] = protein_seq['mt_seq'].apply(lambda x: ' '.join(x)).apply(
            lambda x: re.sub(r"[UZOB]", "X", x))

    protein_seq['label'].astype(str)
    label_names = set(protein_seq['label'])
    
    protein_test = protein_seq[:2]
    input_emb = utils.get_embedding(protein_test=protein_test, model=model, tokenizer=tokenizer )
    model2 = MLPClassifier2(1024,1024,2)
    
    for i in range(len(input_emb)):
        aa_index = input_emb[i]['aa_index']
        pred = model2(input_emb[i]['wt_seq'], input_emb[i]['mt_seq'],aa_index, input_emb[i]['label'])  

    # read embeddings for downstream tasks


    # X_train, y_train, X_test, y_test = utils.train_test_dataset(situation, gene_constrain, mode)

    # 切分数据集
    # X_train, X_test, y_train, y_test= train_test_split(data_X, data_y,
    #                                                     test_size=0.2,
    #                                                     stratify=data_y,
    #                                                    random_state=SEED)
    # 切分出valid数据集
    # X_valid, X_test, y_valid, y_test = utils.train_test_split(X_test,y_test,
    #                                             test_size=0.3,
    #                                             shuffle=True,
    #                                             stratify=y_test,
    #                                             random_state=config.SEED)

    # train_dataset = utils.TestDataset(X_train, y_train)
    # test_dataset = utils.TestDataset(X_test, y_test)
    # val_dataset = utils.TestDataset(X_valid, y_valid)

    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size)


    # #
    # print('\n\n\n\n')
    # print('=============== MLP model training start ===============')
    # # # ==================== MLP classifier ====================================
    # if gene_constrain == 'ohe_constrain':
    #     input_size = 2088
        
    # elif gene_constrain == 'gene_constrain':
    #     input_size = 2048
    # else:
    #     print('constrain_error')
        
    # model = utils.MLPClassifier(num_input = input_size, num_hidden = 1024, num_output = config.label_num).to(config.device)


    # wandb.init(
    #     project="t5-lstm",
    #     config={
    #         "epochs": config.n_epochs,
    #         "batch_size": config.batch_size,
    #         "lr": config.learning_rate,
    #         "dropout": 0.1
    #         })
        
    # utils.trainer(train_loader, val_loader, model, config.device)
        
        
    # wandb.finish()


    # print('\n\n\n\n')
    # print('=============== Predicting & Evaluating the trained model ===============')

    # model = utils.MLPClassifier(num_input = input_size, num_hidden = 1024, num_output = config.label_num).to(config.device)
    # model.load_state_dict(torch.load('./models/model.ckpt'))
    # preds, y_true = utils.predict(test_loader, model, config.device)

    # utils.predict_results(y_true, preds, gene_constrain = gene_constrain, mode = mode, name = 'MLP', situation=situation)


    # print('\n\n\n\n')
    # print('=============== No Bug No Error, Finished!!! ===============')
    # os.system('tput bel')

    # print('\n\n\n\n')