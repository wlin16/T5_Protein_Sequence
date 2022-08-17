import utils
import config
import pandas as pd
import numpy as np
from termcolor import colored
import re
import os

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from time import time


if __name__ == '__main__':
    # start_time = time.time()
    print('=============== Loading data... ===============')

   
    data_X, data_y = utils.data_for_downstream_for_large()

    # split data for traditional ML
    X_train, X_test, y_train, y_test, X_valid, y_valid = utils.dataset_split(
        data_X, data_y)
    #
    #
    print('\n\n\n\n')
    print('=============== ML model training start ===============')
    
    
# def quadKappa(act,pred,n=4,hist_range=(0,3)):
    
#     O = confusion_matrix(act,pred)
#     O = np.divide(O,np.sum(O))
    
#     W = np.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             W[i][j] = ((i-j)**2)/((n-1)**2)
            
#     act_hist = np.histogram(act,bins=n,range=hist_range)[0]
#     prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
#     E = np.outer(act_hist,prd_hist)
#     E = np.divide(E,np.sum(E))
    
#     num = np.sum(np.multiply(W,O))
#     den = np.sum(np.multiply(W,E))
#     print('QuadKappa',1-np.divide(num,den))
#     return 1-np.divide(num,den)



def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    
    start = time()
    def objective_function(params):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred>0.5)
        print ("Accuracy:", accuracy)
        #change the metric if you like
        # score = quadKappa(y_test,y_pred)
        return {'loss': -accuracy, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials)
    loss = [x['result']['loss'] for x in trials.trials]
    
    best_param_values = [x for x in best_param.values()]
    
    if best_param_values[0] == 0:
        boosting_type = 'gbdt'
    else:
        boosting_type= 'dart'
    
    clf_best = lgb.LGBMClassifier(learning_rate=best_param_values[2],
                                  num_leaves=int(best_param_values[5]),
                                  max_depth=int(best_param_values[3]),
                                  n_estimators=int(best_param_values[4]),
                                  boosting_type=boosting_type,
                                  colsample_bytree=best_param_values[1],
                                  reg_lambda=best_param_values[6],
                                 )
                                  
    clf_best.fit(X_train, y_train)
    
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Test Score: ", clf_best.score(X_test, y_test))
    print("Time elapsed: ", time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    
    return trials,best_param

num_eval =100
param_hyperopt= {
    'objective':'cross_entropy',
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

results_hyperopt,para = hyperopt(param_hyperopt, X_train, y_train, X_test, y_test, num_eval)
print("results_hyperot:", results_hyperopt)
print("best_parameter:", para)

best_parameter = pd.DataFrame.from_dict(para)
best_parameter.to_csv('best_parameter_for_LightGBM.csv', index = False)


# Evaluation

para['boosting_type']='gbdt'
para['num_leaves'] = int(para['num_leaves'] )
para['max_depth'] = int(para['max_depth'])
para['n_estimators'] = int(para['n_estimators'])
clf = lgb.LGBMClassifier(**para)
clf.fit(X_train, y_train)
y_pred =  clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(colored(
    '\n\t\t\t\t *** LightGBM_report ***:\n\n\n', 'blue', attrs=['bold']), report)
MCC = matthews_corrcoef(y_test, y_pred)
print("LightGBM MCC:", MCC)
