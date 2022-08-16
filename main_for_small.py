import utils
import config
import pandas as pd
import re
import os

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if __name__ == '__main__':
    # start_time = time.time()
    print('=============== Loading data... ===============')

    # Load data
    protein_seq = pd.read_csv(config.data_path)

    label_names = set(protein_seq['label'])

    # data processing
    protein_seq['wt_seq'] = protein_seq['wt_seq'].apply(lambda x: ' '.join(x)).apply(
        lambda x: re.sub(r"[UZOB]", "X", x))
    protein_seq['mt_seq'] = protein_seq['mt_seq'].apply(lambda x: ' '.join(x)).apply(
        lambda x: re.sub(r"[UZOB]", "X", x))

    protein_seq.drop(protein_seq[protein_seq['Length']
                                 >= 4000].index, inplace=True)

    # protein_seq = protein_seq[:10]
    # protein_seq = protein_seq[2159:2271]

    # ==================== Embedding ====================================

    ###### For large GPU ######
    # Get embeddings for downstream task（wt+mt no batch)
    # utils.get_embedding(protein_seq)

    # Split data into segments if the data size is too large for memory storage (wt+mt+batch)
    # utils.embed_in_batch_for_large(protein_seq, 3700)
    
    ###### For small GPU ######
    # Split data into segments, wt and mt go to model seperately in batch form （wt or mt + batch)
    utils.embed_in_batch_for_small(protein_seq, 1000)
    

    # ==================== Data for Downstream Task ====================================
    # read embeddings for downstream tasks
    data_X, data_y = utils.data_for_downstream_for_small()

    # split data for traditional ML
    X_train, X_test, y_train, y_test, X_valid, y_valid = utils.dataset_split(
        data_X, data_y)
    #
    #
    print('\n\n\n\n')
    print('=============== ML model training start ===============')
    # ==================== Traditional ML ====================================
    try:
        # XGBoost
        eval_s = [(X_train, y_train), (X_test, y_test)]
        xgb = XGBClassifier(n_estimator = 500, max_depth = 6, eta = 0.1)
        xgb.fit(X_train, y_train, eval_set = eval_s, early_stopping_rounds = 10, verbose = False)
        y_xgb = xgb.predict(X_test)
        utils.report_save(y_test, y_xgb, 'XGBoost')

        # CatBoost
        cbt = CatBoostClassifier(
            iterations=500, learning_rate=0.09, depth=10)
        cbt.fit(X_train, y_train)
        y_cbt = cbt.predict(X_test)
        utils.report_save(y_test, y_cbt, 'CatBoost')

    except Exception as e:
        print('=============== Bug Here!!! ===============')
        print(e)
        # print('XGBoost cannot run')

    finally:
        # Random Forest & GradientBoost
        rfc = RandomForestClassifier(random_state=config.SEED, n_estimators=20)
        gbt = GradientBoostingClassifier(
            random_state=config.SEED, n_estimators=12)

        utils.traditional_model(rfc, X_train, y_train, X_test, y_test)
        utils.traditional_model(gbt, X_train, y_train, X_test, y_test)

        print('=============== No Bug No Error, Finished!!! ===============')
        os.system('tput bel')
        
        print('\n\n')
