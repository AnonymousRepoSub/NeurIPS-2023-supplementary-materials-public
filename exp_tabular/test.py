from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
import argparse
import pickle
import sys
import os
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset
import time
import random
import arff
import warnings
import ast
warnings.filterwarnings("ignore", category=UserWarning)

from train import get_trainable_holdout
from data_loader import get_dataset
from csv_recorder import CSVRecorder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(1)


def save_one_entry(folder_name, algorithm, split, tolerance, folds):
    """ save to [folder_name]/result.csv
    Aggregation results of 10 seeds

    Params:
        split: 'valid' or 'test', depend on the folds you passed in.
        folds: np array: [10 seeds] x [fold length]
    """
    csv_file = f'{folder_name}/result.csv'
    tmp_dict = {
        'algorithm': [algorithm],
        'split': [split],
        'tolerance': [tolerance],
        'average_per_fold': [np.mean(folds, axis=0)],
        'worst_per_fold': [np.max(folds, axis=0)],
        'average_mean' : [np.mean(folds)],
        'median_mean' : [np.median(np.mean(folds, axis=1))], 
        'average_std' : [np.mean(np.std(folds, axis=1))],
        'average_worst' : [np.max(np.mean(folds, axis=0))],
    }
    new_data = pd.DataFrame(tmp_dict)
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        data = pd.concat([data, new_data])
    else:
        data = new_data
    print(f'Save entry to {csv_file}.', flush=True)
    data.to_csv(csv_file, index=False)
    data = pd.read_csv(csv_file)
    data = data.drop_duplicates()
    data.to_csv(csv_file, index=False)



from utils import get_args, get_folder_name
args = get_args()
folder_name = get_folder_name(args)

if not os.path.isdir(f'{folder_name}/test_out'):
    os.makedirs(f'{folder_name}/test_out')
test_out_path = f'{folder_name}/test_out/{args.algorithm}_{args.tolerance}.log'
logpath = open(test_out_path, "w")
sys.stdout = logpath
sys.stderr = logpath


X_train, y_train, X_test, y_test, train_len, group_num, group_value, task = get_dataset(args.dataset, split='test', shuffle=args.shuffle, data_size= args.size) 

trainable = get_trainable_holdout(X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            group_num=group_num,
            group_value=group_value,
            estimator=args.estimator, 
            task=task, 
            metric=args.metric)
            
def test_function(config):
    trainable_return = trainable(config)
    result = {}
    result["val_loss"] = trainable_return["val_loss"]
    result["folds"] = trainable_return["metric_for_logging"]["month"] # change out dict to 'folds'
    result["std"] = np.std(result["folds"])
    result["worst"] = np.max(result["folds"]) # check

    print('result', result)
    return result


config_list = []
seeds = [5]
for seed in seeds:
    seed_str = str(seed)
    if args.algorithm == 'cfo':
        path = f"{folder_name}/{args.algorithm}_out/seed_{seed}/result_info.pckl"
    else:
        path = f"{folder_name}/{args.algorithm}_toler{args.tolerance}_out/seed_{seed}/result_info.pckl"
    f = open(path, "rb")
    resul_info = pickle.load(f)
    f.close()
    config_list.append(resul_info["best_config"])


recorder = CSVRecorder(folder_name=folder_name,
            algorithm=args.algorithm,
            csv_name='test', 
            seed=1, 
            tolerance=args.tolerance,
)
test_folds = []
for i, tem_config in enumerate(config_list):
    print(i, config_list[i], flush=True)
    result = test_function(tem_config) 
    print(result, flush=True)
    test_folds.append(result['folds']) 

    recorder.add_each_seed(result, seeds[i], args.tolerance) 
recorder.save_to_csv()
test_folds = np.array(test_folds)


csv_file = f'{folder_name}/{args.algorithm}/valid.csv' 
valid_df = pd.read_csv(csv_file)
if args.algorithm == 'cfo':
    one_tolerance = valid_df
else:
    one_tolerance = valid_df[valid_df['tolerance'] == args.tolerance]
    print(f'Num of seeds used {len(one_tolerance)}')

valid_folds = np.array([ast.literal_eval(i) for i in one_tolerance['folds']])


save_one_entry(folder_name, args.algorithm, 'valid', args.tolerance, valid_folds)
save_one_entry(folder_name, args.algorithm, 'test', args.tolerance, test_folds)