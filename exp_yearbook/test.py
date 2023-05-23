from csv_recorder import CSVRecorder
from data_loader import get_test_data
from train import get_trainable_holdout, get_trainable_holdout_nn, get_trainable_holdout_nn_mimic
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
import torch
import ast
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(1)


def save_one_entry(folder_name, algorithm, split, tolerance, folds):
    """ save to [folder_name]/result.csv
    Aggregation results of 10 seeds

    Params:
        split: 'valid' or 'test', depend on the folds you passed in.
        folds: np array: [10 seeds] x [fold length]
    """
    print("-----------------")
    print(folds)
    print("-----------------")
    csv_file = f'{folder_name}/result.csv'
    tmp_dict = {
        'algorithm': [algorithm],
        'split': [split],
        'tolerance': [tolerance],
        'average_per_fold': [np.mean(folds, axis=0)],
        'worst_per_fold': [np.max(folds, axis=0)],
        'average_mean': [np.mean(folds)],
        'median_mean': [np.median(np.mean(folds, axis=1))],
        'average_std': [np.mean(np.std(folds, axis=1))],
        'average_worst': [np.mean(np.max(folds, axis=1))],
    }
    new_data = pd.DataFrame(tmp_dict)
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        data = pd.concat([data, new_data])
    else:
        data = new_data

    data.to_csv(csv_file, index=False)


# ------parser------
parser = argparse.ArgumentParser()
# parser.add_argument("--seed", help="seeds", type=int, default=1)
parser.add_argument("--dataset", help="dataset", type=str, default="sales")
parser.add_argument("--estimator", help="estimator", type=str, default='lgbm')
parser.add_argument("--metric", help="metric", type=str, default='rmse')
parser.add_argument("--budget", help="budget", type=int, default=7200)
parser.add_argument("--algorithm", help="algorithm", type=str, default="cfo")
parser.add_argument("--tolerance", help="tolerance", type=float, default=None)
args = parser.parse_args()
if args.algorithm == 'cfo':
    args.tolerance = None


# set the big over folder: drug_xgboost_rmse_b1800
folder_name = f'./out/{args.dataset}_{args.estimator}_{args.metric}_b{str(args.budget)}'


# caution: test have new path
test_out_path = f'{folder_name}/{args.algorithm}_test_out.log'
logpath = open(test_out_path, "w")
sys.stdout = logpath
sys.stderr = logpath

if args.estimator != "nn":
    X_train, y_train, X_test, y_test, train_len, group_num, group_value, task = get_test_data(args.dataset)
    trainable = get_trainable_holdout(X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      group_num=group_num,
                                      group_value=group_value,
                                      estimator=args.estimator,
                                      task=task,
                                      metric=args.metric)
else:
    train_data, test_data, group_num, group_value = get_test_data(args.dataset)
    if args.dataset == "yearbook":
        trainable = get_trainable_holdout_nn(train_data=train_data, test_data=test_data,
                                            group_num=group_num, group_value=group_value)
    elif args.dataset == "mimic":
        trainable = get_trainable_holdout_nn_mimic(train_data=train_data, test_data=test_data,
                                            group_num=group_num, group_value=group_value)

if args.estimator != "nn":
    def test_function(config):
        trainable_return = trainable(config)
        result = {}
        result["val_loss"] = trainable_return["val_loss"]
        result["folds"] = trainable_return["metric_for_logging"]["month"]  # change out dict to 'folds'
        result["std"] = np.std(result["folds"])
        result["worst"] = np.max(result["folds"])  # check
        return result
else:
    def test_function(config):
        trainable_return = trainable(config)
        result = {}
        result["val_loss"] = trainable_return["val_loss"]
        result["folds"] = trainable_return["month_result"]
        result["std"] = np.std(result["folds"])
        result["worst"] = np.max(result["folds"])

        result['folds_erro'] = trainable_return['month_err']
        result['error'] = np.mean(result['folds_erro'])
        result['std_erro'] = np.std(result['folds_erro'])
        result['worst_erro'] = np.max(result["folds_erro"])
        return result

seeds = [1,2]
config_list = []
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

# config_list = []
# arc_config = {'batch_size': 32, 'iteration_limit': 3000, 'n_conv_channels_c1': 32, 
# 'kernel_size_c1': 3, 'has_max_pool_c1': 1, 'n_conv_channels_c2': 32, 'kernel_size_c2': 3, 
# 'n_conv_channels_c3': 32, 'kernel_size_c3': 3, 'n_conv_channels_c4': 32, 'kernel_size_c4': 3, 
# 'n_l_units_l1': 32, 'lr': 0.001, 'has_max_pool_c2': 1, 'has_max_pool_c3': 1, 'has_max_pool_c4': 1}
# config_list.append(arc_config)

# --------record valid.csv---------
recorder = CSVRecorder(folder_name=folder_name,
                       algorithm=args.algorithm,
                       csv_name='test',  # !!!! set to 'test' when testing
                       seed=1,  # this is a zombie arg
                       tolerance=args.tolerance,  # this is a zombie arg
                       )
if args.estimator != "nn":
    test_folds = []
    for i, tem_config in enumerate(config_list):
        result = test_function(tem_config)  # get result
        test_folds.append(result['folds'])  # add to tmp folds
        # different from testing: use add_each_seed
        recorder.add_each_seed(result, seeds[i], args.tolerance)  # add to csv_dict
    recorder.save_to_csv()
    test_folds = np.array(test_folds)
else:
    test_folds = []
    for i, tem_config in enumerate(config_list):
        # print("***********************")
        # print(test_folds)
        # print("***********************")
        result = test_function(tem_config)  # get result
        test_folds.append(result['folds_erro'])  # add to tmp folds
        # different from testing: use add_each_seed
        recorder.add_each_seed(result, seeds[i], args.tolerance)  # add to csv_dict
    recorder.save_to_csv()
    test_folds = np.array(test_folds)

# read corresponding valid.csv and save to results
csv_file = f'{folder_name}/{args.algorithm}/valid.csv'  # !! fixed to cfo/valid
valid_df = pd.read_csv(csv_file)
if args.algorithm in ['cfo', 'CFO']:
    one_tolerance = valid_df
else:
    one_tolerance = valid_df[valid_df['tolerance'] == args.tolerance]
    # assert len(one_tolerance) == 2, 'Should have 10 seeds with the same tolerance'

# folds read from csv is actually str,
# see https://www.geeksforgeeks.org/python-convert-a-string-representation-of-list-into-list/
if args.estimator != "nn":
    valid_folds = np.array([ast.literal_eval(i) for i in one_tolerance['folds']])
else:
    valid_folds = np.array([ast.literal_eval(i) for i in one_tolerance['folds_erro']])
    
# print(valid_folds)
# print(test_folds)

save_one_entry(folder_name, args.algorithm, 'valid', args.tolerance, valid_folds)
save_one_entry(folder_name, args.algorithm, 'test', args.tolerance, test_folds)
# TODO:
# 1. eval all configs
# 2. save to csv
# seed,  val_loss, fold/month_loss

# 3. save to other csv

# parser.add_argument("--dataset", help="dataset", type=str, default="sales")
# parser.add_argument("--estimator", help="estimator", type=str, default='lgbm')
# parser.add_argument("--metric", help="metric", type=str, default='rmse')
# parser.add_argument("--budget", help="budget", type=int, default=7200)
# parser.add_argument("--algorithm", help="algorithm", type=str, default="cfo")
# parser.add_argument("--tolerance", help="tolerance", type=float, default=None)


