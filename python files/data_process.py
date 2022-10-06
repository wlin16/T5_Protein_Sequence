from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import pandas as pd
import re
import os

from termcolor import colored
import dataframe_image as dfi
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

# from IPython.display import Audio, display


def allDone():
    display(Audio(url='https://www.mediacollege.com/downloads/sound-effects/beep/beep-10.wav', autoplay=True))


def train_test_split(raw_df, n):
    test_df = raw_df.sample(n=n, random_state=2022)
    train_df = raw_df.drop(test_df.index)
    return train_df, test_df


def merge_with_emb(train_df, test_df, emb_template):
    emb_template.rename(columns={'mutant_index': 'aa_index', 't_aa': 'mt_aa'}, inplace=True)

    train_emb = pd.merge(train_df, emb_template, on='gene_id', how='left').dropna(axis=0)
    test_emb = pd.merge(test_df, emb_template, on='gene_id', how='left').dropna(axis=0)

    train_emb['aa_index'] = train_emb['aa_index'].astype(int)
    test_emb['aa_index'] = test_emb['aa_index'].astype(int)

    del train_emb['sequence']
    del test_emb['sequence']

    train_emb = train_emb.reset_index(drop=True)
    test_emb = test_emb.reset_index(drop=True)

    return train_emb, test_emb


def get_ohe(enc, raw_df):
    enc_wt = enc.transform([[aa] for aa in raw_df['wt_aa'].tolist()]).toarray()
    enc_mt = enc.transform([[aa] for aa in raw_df['mt_aa'].tolist()]).toarray()

    wt_aa_ohe = pd.DataFrame(enc_wt, columns=enc.categories_)
    mt_aa_ohe = pd.DataFrame(enc_mt, columns=enc.categories_)

    df_ohe = pd.concat([raw_df, wt_aa_ohe, mt_aa_ohe], axis=1)

    return df_ohe


def gene_constrain(df):
    result = None
    for key, item in df.groupby('gene_id'):
        # print(grouped_df.get_group(key), "\n\n")
        try:
            a, b = item.label.value_counts()
            # print(item.label.apply(lambda y: y.value_counts()).get())
            if a >= 1 & b >= 1:
                gene = item.gene_id.unique()[0]
                gene_df = df[df['gene_id'] == gene]

                if result is None:
                    result = gene_df
                else:
                    result = pd.concat([result, gene_df])
        except:
            pass

    return result


def gene_balance(df):
    result = None
    for key, item in df.groupby('gene_id'):
        a, b = item.label.value_counts().sort_index(ascending=True)
        class_a = item[item['label'] == 0]
        class_b = item[item['label'] == 1]

        if a > b:
            class_a = class_a.sample(n=b, random_state=2022)
            if result is None:
                result = class_b
                result = pd.concat([class_a, result])
            else:
                result = pd.concat([result, class_b])
                result = pd.concat([class_a, result])

        elif a < b:
            class_b = class_b.sample(n=a, random_state=2022)
            if result is None:
                result = class_a
                result = pd.concat([class_b, result])
            else:
                result = pd.concat([result, class_a])
                result = pd.concat([class_b, result])
        else:
            if result is None:
                result = class_a
                result = pd.concat([class_b, result])
                result = pd.concat([result, class_a])
    return result


def emb2npy(raw_df, mode, data_mode, balance=False, ohe=False):
    a = []
    for j in range(len(raw_df)):
        wt_list = []
        mt_list = []
        wt_list.append(raw_df['wt_emb'][j].strip('[').strip(']').split(', '))
        mt_list.append(raw_df['mt_emb'][j].strip('[').strip(']').split(', '))

        wt_float = [float(i) for i in wt_list[0]]
        mt_float = [float(i) for i in mt_list[0]]
        stack = np.hstack((wt_float, mt_float))
        a.append(stack)
    arr = np.array(a)

    if ohe == True:
        ohe = 'ohe_constrain'
        print(f'{data_mode} before: ', arr.shape)
        arr = np.concatenate((arr, np.array(raw_df.iloc[:, -40:])), axis=1)
        print(f'{data_mode} after: ', arr.shape)
    else:
        ohe = 'gene_constrain'

    arr = np.concatenate((arr, np.array(raw_df['label']).reshape(-1, 1)), axis=1)

    print(f'{data_mode} with label: ', arr.shape)

    if balance == False:
        path = f'../../T5_Protein_Sequence/data/{ohe}/gene_not_balance/For_ML'
    else:
        path = f'../../T5_Protein_Sequence/data/{ohe}/gene_balance/For_ML'

    np.save(f'{path}/{mode}_{data_mode}.npy', arr)

    return arr


def get_result(train_df, test_df, mode, ohe, balance, enc=None):
    train_df = gene_constrain(train_df).reset_index(drop=True)

    if balance is True and ohe is True:

        train_ohe = get_ohe(enc, train_df)
        test_df = get_ohe(enc, test_df)
        train_df = gene_balance(train_ohe).reset_index(drop=True)

    elif balance is True and ohe is False:

        train_df = gene_balance(train_df).reset_index(drop=True)

    elif balance is False and ohe is True:

        train_df = get_ohe(enc, train_df)
        test_df = get_ohe(enc, test_df)
    else:
        pass

    train_embed, test_embed = emb2npy(train_df, mode=mode, data_mode='train', balance=balance, ohe=ohe), \
                              emb2npy(test_df, mode=mode, data_mode='test', balance=balance, ohe=ohe)

    print('train_wt_mt_ratio:\n', f'{mode}: ', Counter(train_embed[:, -1]))
    print('\ntrain_total: ', f'{mode}: ', train_embed.shape[0])
    print('\ntest_wt_mt_ratio:\n', f'{mode}: ', Counter(test_embed[:, -1]))
    print('\ntest_total: ', f'{mode}: ', test1_embed.shape[0])

    return train_embed, test_embed


config = {
    'embed_type': 'single_aa',
    # embed_type = 'mean_aa',
    'ohe': True,
    'balance': True
               }

clean_seq_1 = pd.read_csv('../data/clean_seq_1.csv')
clean_seq_2 = pd.read_csv('../data/clean_seq_2.csv')

embed_type = config['embed_type']
# embed_type = 'mean_aa'

if config['embed_type'] == 'single_aa':
    path1 = 'mode_1_embeds'
    path2 = 'mode_2_embeds'
elif config['embed_type'] == 'mean_aa':
    path1 = 'mode1_mean_embeds'
    path2 = 'mode2_mean_embeds'

mode1_embed = pd.read_csv(f'../data/gene_not_constrain/imbalance_same_seq/Embedding_results_csv/{path1}/sequence_embeddings(27750).csv')
mode2_embed = pd.read_csv(f'../data/gene_not_constrain/imbalance_same_seq/Embedding_results_csv/{path2}/sequence_embeddings(32394).csv')

mode1_embed['wt_seq'] = mode1_embed['wt_seq'].apply(lambda x: x.replace(' ',''))
mode1_embed['mt_seq'] = mode1_embed['mt_seq'].apply(lambda x: x.replace(' ',''))
mode2_embed['wt_seq'] = mode2_embed['wt_seq'].apply(lambda x: x.replace(' ',''))
mode2_embed['mt_seq'] = mode2_embed['mt_seq'].apply(lambda x: x.replace(' ',''))

mode1_for_train, mode1_for_test = train_test_split(clean_seq_1, 555)
mode2_for_train, mode2_for_test = train_test_split(clean_seq_2, 647)
mode1_for_train, mode1_for_test = merge_with_emb(mode1_for_train, mode1_for_test, mode1_embed)
mode2_for_train, mode2_for_test = merge_with_emb(mode2_for_train, mode2_for_test, mode2_embed)
print('mode1_for_train_with_embed: ', mode1_for_train.shape)
print('mode1_for_test_with_embed: ', mode1_for_test.shape)
print('mode2_for_train_with_embed: ', mode2_for_train.shape)
print('mode2_for_test_with_embed: ', mode2_for_test.shape)

s1 = set(mode1_for_train['wt_seq'])
s2 = set(mode1_for_test['wt_seq'])
result = set.intersection(s1, s2)
print(result)

enc = OneHotEncoder(handle_unknown='ignore')
aa_list = list(set(mode1_for_train['mt_aa']))
enc = enc.fit([[aa] for aa in aa_list])

ohe = config['ohe']
balance = config['balance']
enc = enc

train1_embed, test1_embed = get_result(mode1_for_train, mode1_for_test, 'mode1', ohe=ohe, balance=balance, enc = enc)
train2_embed, test2_embed = get_result(mode2_for_train, mode2_for_test, 'mode2', ohe=ohe, balance=balance, enc = enc)

