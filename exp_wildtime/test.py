from csv_recorder import CSVRecorder
from data_loader import get_test_data
from train import get_trainable_holdout_nn_yearbook, get_trainable_holdout_nn_mimic, get_trainable_holdout_nn_article, get_trainable_holdout_nn_fmow
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset", type=str, default="sales")
parser.add_argument("--estimator", help="estimator", type=str, default='lgbm')
parser.add_argument("--metric", help="metric", type=str, default='rmse')
parser.add_argument("--budget", help="budget", type=int, default=7200)
parser.add_argument("--algorithm", help="algorithm", type=str, default="cfo")
parser.add_argument("--tolerance", help="tolerance", type=float, default=None)
parser.add_argument("--cudaid", help="cudaid", type=str, default=0)
args = parser.parse_args()
if args.algorithm == 'cfo':
    args.tolerance = None

os.environ["CUDA_VISIBLE_DEVICES"] = args.cudaid

folder_name = f'./out/{args.dataset}_{args.estimator}_{args.metric}_b{str(args.budget)}'
test_out_path = f'{folder_name}/{args.algorithm}_test_out.log'
logpath = open(test_out_path, "w")
sys.stdout = logpath
sys.stderr = logpath

train_data, test_data, group_num, group_value = get_test_data(args.dataset)
if args.dataset == "yearbook":
    trainable = get_trainable_holdout_nn_yearbook(train_data=train_data, test_data=test_data,
                                        group_num=group_num, group_value=group_value)
elif args.dataset in ["mimic-mortality", "mimic-readmission"]:
    trainable = get_trainable_holdout_nn_mimic(train_data=train_data, test_data=test_data,
                                        group_num=group_num, group_value=group_value, dataset_name = args.dataset)
elif args.dataset in ["huffpost", "arxiv"]:
    trainable = get_trainable_holdout_nn_article(train_data=train_data, test_data=test_data,
                                        group_num=group_num, group_value=group_value, dataset_name = args.dataset)
elif args.dataset in ["fmow"]:
    trainable = get_trainable_holdout_nn_fmow(train_data=train_data, test_data=test_data,
                                        group_num=group_num, group_value=group_value, dataset_name = args.dataset)

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

seeds = [2,3,4]
config_list = []
for seed in seeds:
    seed_str = str(seed)
    if args.algorithm in ['cfo', 'bo', 'hyperband']:
        path = f"{folder_name}/{args.algorithm}_out/seed_{seed}/result_info.pckl"
    else:
        path = f"{folder_name}/{args.algorithm}_toler{args.tolerance}_out/seed_{seed}/result_info.pckl"
    f = open(path, "rb")
    resul_info = pickle.load(f)
    f.close()
    config_list.append(resul_info["best_config"])

recorder = CSVRecorder(folder_name=folder_name,
                       algorithm=args.algorithm,
                       csv_name='test',  # !!!! set to 'test' when testing
                       seed=1,  # this is a zombie arg
                       tolerance=args.tolerance,  # this is a zombie arg
                       )

test_folds = []
for i, tem_config in enumerate(config_list):
    result = test_function(tem_config)  # get result
    test_folds.append(result['folds_erro'])  # add to tmp folds
    recorder.add_each_seed(result, seeds[i], args.tolerance)  # add to csv_dict
recorder.save_to_csv()
test_folds = np.array(test_folds)
save_one_entry(folder_name, args.algorithm, 'test', args.tolerance, test_folds)
logpath.close()

# python test.py --dataset yearbook --estimator nn --metric accuracy --budget 10800 --algorithm hyperband --cudaid 0

