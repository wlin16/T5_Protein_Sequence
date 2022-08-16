import utils
import config
import pandas as pd
import numpy as np
import re
import os

from xgboost import XGBClassifier
import catboost
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import sklearn
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import colorama

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
    # ==================== Traditional ML ====================================
    def hyperparameter_tuning(space = config.space, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
        model = XGBClassifier(n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                            reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                            colsample_bytree=int(space['colsample_bytree']), tree_method = 'gpu_hist', gpu_id = 0, objective='binary:hinge')
        evaluation = [( X_train, y_train), ( X_test, y_test)]
        
        model.fit(X_train, y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10,verbose=False)

        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        print ("Accuracy:", accuracy)
        #change the metric if you like
        return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}


    trials = Trials()
    xgb_best = fmin(fn=hyperparameter_tuning,
                space=config.xgb_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    print (xgb_best)
    
    # N_HYPEROPT_PROBES = 60
    # HYPEROPT_ALGO = tpe.suggest
    # colorama.init()
    
    # D_train = catboost.Pool(X_train, y_train)
    # D_test = catboost.Pool(X_test, y_test)
    
    # def get_catboost_params(space):
    #     params = dict()
    #     params['learning_rate'] = space['learning_rate']
    #     params['depth'] = int(space['depth'])
    #     params['l2_leaf_reg'] = space['l2_leaf_reg']
    #     params['border_count'] = space['border_count']
    #     #params['rsm'] = space['rsm']
    #     return params

    # obj_call_count = 0
    # cur_best_loss = np.inf
    # log_writer = open( 'catboost-hyperopt-log.txt', 'w' )
    
    # def objective(space):
    #     global obj_call_count, cur_best_loss

    #     obj_call_count += 1

    #     print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    #     params = get_catboost_params(space)

    #     sorted_params = sorted(space.items(), key=lambda z: z[0])
    #     params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    #     print('Params: {}'.format(params_str) )

    #     model = catboost.CatBoostClassifier(iterations=100000,
    #                                         learning_rate=params['learning_rate'],
    #                                         depth=int(params['depth']),
    #                                         loss_function='Logloss',
    #                                         use_best_model=True,
    #                                         task_type="GPU",
    #                                         eval_metric='AUC',
    #                                         l2_leaf_reg=params['l2_leaf_reg'],
    #                                         early_stopping_rounds=3000,
    #                                         od_type="Iter",
    #                                         border_count=int(params['border_count']),
    #                                         verbose=False
    #                                         )
    #     model.fit(D_train, eval_set=D_test, verbose=False)
    #     nb_trees = model.tree_count_

    #     print('nb_trees={}'.format(nb_trees))

    #     y_pred = model.predict_proba(D_test.get_features())
    #     test_loss = sklearn.metrics.log_loss(D_test.get_label(), y_pred, labels=[0, 1])
    #     acc = sklearn.metrics.accuracy_score(D_test.get_label(), np.argmax(y_pred, axis=1))
    #     auc = sklearn.metrics.roc_auc_score(D_test.get_label(), y_pred[:,1])

    #     log_writer.write('loss={:<7.5f} acc={} auc={} Params:{} nb_trees={}\n'.format(test_loss, acc, auc, params_str, nb_trees ))
    #     log_writer.flush()

    #     if test_loss<cur_best_loss:
    #         cur_best_loss = test_loss
    #         print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)


    #     return {'loss':test_loss, 'status': STATUS_OK }
    
    # space = {
    # 'depth': hp.quniform("depth", 1, 6, 1),
    # 'border_count': hp.uniform ('border_count', 32, 255),
    # 'learning_rate': hp.loguniform('learning_rate', -5.0, -2),
    # 'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),
    # }

    # trials = Trials()
    # best = hyperopt.fmin(fn=objective,
    #                     space=space,
    #                     algo=HYPEROPT_ALGO,
    #                     max_evals=N_HYPEROPT_PROBES,
    #                     trials=trials,
    #                     verbose=True)
    
    # print('\n\n')
    # print('-'*50)
    # print('The best params:')
    # print( best )
    # print('\n\n')
    
    

    print('=============== No Bug No Error, Finished!!! ===============')
    os.system('tput bel')
    
    print('\n\n')
