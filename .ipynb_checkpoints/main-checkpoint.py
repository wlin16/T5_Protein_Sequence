
import time


import utils
import config
import model
import Trainer
import torch
import pandas as pd
import re


if __name__ == '__main__':
    start_time = time.time()

    print("Loading data...")

    # Load data
    protein_seq = pd.read_csv(config.data_path)
    protein_seq = protein_seq[:10]
    label_names = set(protein_seq['pathogenicity'])

    # data processing
    protein_seq['wt_seq'] = protein_seq['wt_seq'].apply(lambda x: ' '.join(x)).apply(
        lambda x: re.sub(r"[UZOB]", "X", x))
    protein_seq['mt_seq'] = protein_seq['mt_seq'].apply(lambda x: ' '.join(x)).apply(
        lambda x: re.sub(r"[UZOB]", "X", x))



    # get embeddings
    utils.get_embedding(protein_seq)

    # read embeddings for downstream tasks
    embeds_path = 'emb_for_test.pkl'
    data_X, data_y = utils.embeds_for_tasks(embeds_path)
    print(data_X.shape)
    #
    #
    #
    # # split data for traditional ML
    # X_train, X_test, y_train, y_test, X_valid, y_valid = utils.train_test_split(data_X,data_y)
    #
    #
    # # feed data to traditional ML
    # from xgboost import XGBClassifier
    # from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    # from sklearn.metrics import classification_report
    #
    # try:
    #     eval_s = [(X_train, y_train), (X_test, y_test)]
    #     xgb = XGBClassifier()
    #     xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_s, verbose=False)
    #     y_xgb = xgb.predict(X_test)
    #
    #     xgb_report = classification_report(y_test, y_xgb, target_names=label_names)
    #     print(xgb_report)
    #
    # except:
    #     rfc = RandomForestClassifier(random_state=0, n_estimators=100)
    #     gbt = GradientBoostingClassifier(random_state=0, n_estimators=50)
    #
    #     rfc.fit(X_train, y_train)
    #     gbt.fit(X_train, y_train)
    #
    #     y_rfc = rfc.predict(X_test)
    #     y_gbt = gbt.predict(X_test)
    #
    #     # RandomForest report
    #     rfc_report = classification_report(y_test, y_rfc, target_names=label_names)
    #     print(rfc_report)
    #     # GradientBoosting report
    #     gbt_report = classification_report(y_test, y_gbt, target_names=label_names)
    #     print(gbt_report)














