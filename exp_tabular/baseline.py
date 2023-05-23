from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBClassifier

from flaml.ml import get_val_loss
from data_loader import get_dataset
import numpy as np 
from utils import set_seed

set_seed(1)
def run_baseline(dataset, eval_metric, model_func):   
    if eval_metric == 'roc_auc':
        obj = 'binary'
    else:
        obj = 'regression'

    X_train, y_train, X_test, y_test, train_len, test_group_num, test_group_value, task = get_dataset(dataset, 'test')

    print(dataset)
    test_folds = []
    for i in range(1,6):
        model = model_func(random_state=i)
        model.fit(X_train, y_train)
        print(len(X_test))

        val_loss, metric_for_logging, train_time, pred_time= get_val_loss(
            config=None,
            estimator=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            groups_val=test_group_value,
            weight_val=None,
            eval_metric=eval_metric,
            obj=obj,
            require_train=False,
        )
        test_folds.append(metric_for_logging['month'])
        print(metric_for_logging['month'])
    folds = np.array(test_folds)
    print()
    print()
    print('average_per_fold', np.mean(folds, axis=0))
    print('worst_per_fold', np.max(folds, axis=0))
    print('average_mean', np.mean(folds))
    print('median_mean', np.median(np.mean(folds, axis=1)))
    print('average_std', np.mean(np.std(folds, axis=1)))
    print('average_worst' , np.max(np.mean(folds, axis=0)))
    return 


run_baseline('electricity', 'roc_auc', XGBClassifier)
# run_baseline('sales', 'rmse', LGBMRegressor())
run_baseline('vessel', 'rmse', XGBRegressor)
run_baseline('temp', 'rmse', LGBMRegressor)
