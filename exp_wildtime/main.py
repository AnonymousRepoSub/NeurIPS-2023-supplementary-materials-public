from train import mytune
import torch
from csv_recorder import CSVRecorder
from data_loader import get_train_data
from train import mytune, get_trainable_cv_nn_yearbook, get_trainable_cv_nn_drug, get_trainable_cv_nn_mimic, get_trainable_cv_nn_article, get_trainable_cv_nn_fmow
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
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", help="cuda", type=str, default=0)
    parser.add_argument("--seed", help="seeds", type=int, default=1)
    parser.add_argument("--budget", help="budget", type=int, default=10)
    parser.add_argument("--algorithm", '-a', help="algorithm", type=str, default="cfo")
    parser.add_argument("--dataset", help="dataset", type=str, default="sales")
    parser.add_argument("--metric", help="metric", type=str, default='accuracy') # set to accuracy for nn
    parser.add_argument("--estimator", help="estimator", type=str, default='nn')
    parser.add_argument("--tolerance", '-t', help="tolerance", type=float, default=None)
    args = parser.parse_args()
    if args.algorithm == 'cfo':
        args.tolerance = None
    return args

def main_nn_function(args):
    import os
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cudaid

    # -----get data info---------
    if args.dataset not in ["mimic-mortality", "mimic-readmission", "huffpost", "arxiv", "fmow"]:
        data, group_num, group_value = get_train_data(args.dataset)
    else:
        Train_train, Train_test, group_num, group_value = get_train_data(args.dataset)
    # -----set result path-------
    args.metric = "accuracy"
    folder_name = f'./out/{args.dataset}_{args.estimator}_{args.metric}_b{str(args.budget)}'
    path = f"{folder_name}/{args.algorithm}_out/seed_{args.seed}/"
    if args.algorithm not in ['cfo', "bo", "hyperband"]:
        path = f"{folder_name}/{args.algorithm}_toler{args.tolerance}_out/seed_{args.seed}/"
    if not os.path.isdir(path):
        os.makedirs(path)
    logpath = open(os.path.join(path, "std.log"), "w")
    sys.stdout = logpath
    sys.stderr = logpath
    # -----set result path-------
    if args.dataset == "yearbook":
        trainable = get_trainable_cv_nn_yearbook(
            seed=args.seed,
            dataset=data,
            group_num=group_num,
            group_value=group_value,
        )
    elif args.dataset == "drug":
        trainable = get_trainable_cv_nn_drug(
            seed=args.seed,
            dataset=data,
            group_num=group_num,
            group_value=group_value,
        )
    elif args.dataset in ["mimic-mortality", "mimic-readmission"]:
        trainable = get_trainable_cv_nn_mimic(
            seed=args.seed,
            dataset=[Train_train, Train_test],
            group_num=group_num,
            group_value=group_value,
            dataset_name = args.dataset,
        )
    elif args.dataset == "huffpost":
        trainable = get_trainable_cv_nn_article(
            seed=args.seed,
            dataset=[Train_train, Train_test],
            group_num=group_num,
            group_value=group_value,
            dataset_name = args.dataset,
        )
    elif args.dataset == "arxiv":
        trainable = get_trainable_cv_nn_article(
            seed=args.seed,
            dataset=[Train_train, Train_test],
            group_num=group_num,
            group_value=group_value,
            dataset_name = args.dataset,
        )
    elif args.dataset == "fmow":
        trainable = get_trainable_cv_nn_fmow(
            seed=args.seed,
            dataset=[Train_train, Train_test],
            group_num=group_num,
            group_value=group_value,
            dataset_name = args.dataset,
        )

    analysis = mytune(
        seed=args.seed,
        budget=args.budget,
        algorithm=args.algorithm,
        tolerance=args.tolerance,
        estimator=args.estimator,
        trainable=trainable,
        local_dir=path,
        data= args.dataset,
        train_len=None)
    
    # ----------save results-------------
    resul_info = {}
    if args.algorithm in ["bo", "hyperband"]:
        trial = analysis.best_trial
        best_config = {}
        for key, value in trial.params.items():
            best_config[key] = value
        resul_info["best_config"] = best_config
        resul_info["best_result"] = trial
    else:
        analysis_best_result =  analysis.best_result
        analysis_best_config = analysis.best_config
        resul_info["best_result"] = analysis_best_result
        resul_info["best_config"] = analysis_best_config

        from csv_recorder import CSVRecorder
        recorder = CSVRecorder(
            folder_name=folder_name,
            algorithm=args.algorithm,
            csv_name='valid',  # set to 'test' when testing
            seed=args.seed,
            tolerance=args.tolerance,
        )
        recorder.add_result(resul_info["best_result"])
        recorder.save_to_csv()
    savepath = os.path.join(path, "result_info.pckl")
    print(resul_info)
    f = open(savepath, "wb")
    pickle.dump(resul_info, f)
    f.close()
    logpath.close()
args = get_args()
set_seed(args.seed)
main_nn_function(args)
# python main.py --cudaid 1 --seed 1 --budget 14400 --algorithm cfo --dataset mimic --metric accuracy --estimator nn
