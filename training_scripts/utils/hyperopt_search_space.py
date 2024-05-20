from hyperopt import hp


flag_params = [0, 1]
classifier_params = ['dtree', 'logistic', 'adaboost', 'rf', 'xgb']
max_depth_params = [1, 2, 3, 4, 5]

criterion_params = ['gini', 'entropy']
solver0_params = ['lbfgs', 'sag', 'newton-cg']
penalty0_params = ['l2', 'none']
solver1_params = ['saga']
penalty1_params = ['l1', 'l2', None]
C_params = [0.01, 0.1, 1, 10, 100]

n_estimator_params = [50, 100, 200, 300, 500]
learning_rate_params = [0.1, 0.01, 0.001, 0.0001, 1]

dtree_search = {'type': classifier_params[0],
                'max_depth': hp.choice('dtree_max_depth', max_depth_params),
                'criterion': hp.choice('dtree_criterion', criterion_params)
                }

adaboost_search = {'type': classifier_params[2],
                   'n_estimators': hp.choice('ada_n_estimators', n_estimator_params),
                   'learning_rate': hp.choice('ada_learning_rate', learning_rate_params)
                   }

logistic_solver_0_search = {
                            "solver": hp.choice('solver0', solver0_params),
                            "penalty": hp.choice('penalty0', penalty0_params)
                            }

logistic_solver_1_search = {
                            "solver": hp.choice('solver1', solver1_params),
                            "penalty": hp.choice('penalty1', penalty1_params)
                            }

logistic_solver_chooser = hp.choice("solver_chooser", [logistic_solver_0_search, logistic_solver_1_search])

logistic_search = {'type': classifier_params[1],
                   'solver_chooser': logistic_solver_chooser,
                   'C': hp.choice('C', C_params)
                   }

rf_search = {'type': classifier_params[3],
             'max_depth': hp.choice('rf_max_depth', max_depth_params),
             'criterion': hp.choice('rf_criterion', criterion_params),
             'n_estimators': hp.choice('rf_n_estimators', n_estimator_params)
             }

xgb_search = {'type': classifier_params[4],
              'max_depth': hp.choice('xgb_max_depth', max_depth_params),
              'n_estimators': hp.choice('xgb_n_estimators', n_estimator_params),
              'learning_rate': hp.choice('xgb_learning_rate', learning_rate_params)
              }

hyperopt_search_space = {'classifier': hp.choice('classifier', [dtree_search, logistic_search, adaboost_search, rf_search, xgb_search])}
