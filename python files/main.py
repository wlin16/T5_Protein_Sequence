import utils
import config
import pandas as pd
import re
import os

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if __name__ == '__main__':
    # start_time = time.time()
    print('=============== Loading data... ===============')

    # Load data
    protein_seq_1 = pd.read_csv(config.data_path_1)
    protein_seq_2 = pd.read_csv(config.data_path_2)

    label_names = set(protein_seq_1['label'])

    # data processing
    protein_seq_1['wt_seq'] = protein_seq_1['wt_seq'].apply(lambda x: ' '.join(x)).apply(
        lambda x: re.sub(r"[UZOB]", "X", x))
    protein_seq_1['mt_seq'] = protein_seq_1['mt_seq'].apply(lambda x: ' '.join(x)).apply(
        lambda x: re.sub(r"[UZOB]", "X", x))

    protein_seq_1.drop(protein_seq_1[protein_seq_1['Length']
                                 >= 4000].index, inplace=True)

    # protein_seq_1 = protein_seq_1[:10]
    # protein_seq_1 = protein_seq_1[2159:2275]

    protein_seq_2['wt_seq'] = protein_seq_2['wt_seq'].apply(lambda x: ' '.join(x)).apply(
    lambda x: re.sub(r"[UZOB]", "X", x))
    protein_seq_2['mt_seq'] = protein_seq_2['mt_seq'].apply(lambda x: ' '.join(x)).apply(
    lambda x: re.sub(r"[UZOB]", "X", x))

    protein_seq_2.drop(protein_seq_2[protein_seq_2['Length']
                                 >= 4000].index, inplace=True)


    # protein_seq_2 = protein_seq_2[:10]
    # protein_seq_2 = protein_seq_2[2159:2275]
    # ==================== Embedding ====================================

    ##### For large GPU ######
    # Get embeddings for downstream task（wt+mt no batch)
    utils.get_embedding_large_GPU(protein_seq_1, save_path = config.model_1_embed_csv_path)
    utils.get_embedding_large_GPU(protein_seq_2, save_path = config.model_2_embed_csv_path)

    # Split data into segments if the data size is too large for memory storage (wt+mt+batch)
    # utils.embed_in_batch_for_large(protein_seq, 3700)
    
    ##### For small GPU ######
    # Split data into segments, wt and mt go to model separately in batch form （wt or mt + batch)
    # utils.embed_in_batch_for_small(protein_seq, 5, path = )
    

    # ==================== Data for Downstream Task ====================================
    # read embeddings for downstream tasks
    # data_X, data_y = utils.data_for_downstream_for_large()

    # # split data for traditional ML
    # X_train, X_test, y_train, y_test, X_valid, y_valid = utils.dataset_split(
    #     data_X, data_y)
    #
    #
    print('\n\n\n\n')
    # print('=============== ML model training start ===============')
    # # ==================== Traditional ML ====================================
    # try:
    #     # XGBoost
    #     eval_s = [(X_train, y_train), (X_test, y_test)]
    #     xgb = XGBClassifier()
    #     xgb.fit(X_train, y_train)
    #     y_xgb = xgb.predict(X_test)
    #     utils.report_save(y_test, y_xgb, 'XGBoost')
            

    #     # CatBoost
    #     cbt = CatBoostClassifier(iterations=500, learning_rate=0.09, depth=10)
    #     cbt.fit(X_train, y_train)
    #     y_cbt = cbt.predict(X_test)
    #     utils.report_save(y_test, y_cbt, 'CatBoost')

            
            
    #     # LightGBM
    #     d_train=lgb.Dataset(X_train, label=y_train)
    #     params={}
    #     # params['learning_rate']=0.41282313322582176
    #     params['learning_rate']=0.1
    #     params['boosting_type']='gbdt' #GradientBoostingDecisionTree
    #     params['objective']='cross_entropy' #Binary target feature
    #     params['metric']='binary_error' #metric for binary classification
    #     params['max_depth']= 10
    #     # params['n_estimators'] = 459
    #     # params['num_leaves'] = 34
    #     # params['reg_lambda'] = 0.9557019573592245
    #     # params['colsample_by_tree'] = 0.8506663985944544
    #     lgb = lgb.train(params,d_train,300)
    #     y_lgb = lgb.predict(X_test)
    #     y_lgb=y_lgb.round(0)
    #     y_lgb = y_lgb.astype(int)
    #     utils.report_save(y_test, y_lgb, 'LightGBM')
            

    # except Exception as e:
    #     print('=============== Bug Here!!! ===============')
    #     print(e)
    #     os.system('tput bel')
    #     # print('XGBoost cannot run')

    # finally:
    #     # Random Forest & GradientBoost
    #     rfc = RandomForestClassifier(random_state=config.SEED, n_estimators=20)
    #     gbt = GradientBoostingClassifier(
    #         random_state=config.SEED, n_estimators=12)

    #     utils.traditional_model(rfc, X_train, y_train, X_test, y_test)
        
    #     utils.traditional_model(gbt, X_train, y_train, X_test, y_test)
        
        
    print('\n\n\n\n')
    print('=============== No Bug No Error, Finished!!! ===============')
    os.system('tput bel')
    
    print('\n\n\n\n')
