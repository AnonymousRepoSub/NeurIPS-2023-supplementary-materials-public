from csv_recorder import CSVRecorder
from data_loader import get_train_data
from train import get_trainable_cv, mytune, get_trainable_cv_nn, get_trainable_cv_nn_version2, get_trainable_cv_nn_drug, get_trainable_cv_nn_mimic
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
import torch
from train import get_trainable_cv, mytune, get_tolerances
from data_loader import get_train_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#------parser------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", help="cuda", type=str, default=0)
    parser.add_argument("--seed", help="seeds", type=int, default=1)
    parser.add_argument("--budget", help="budget", type=int, default=10)
    parser.add_argument("--algorithm", '-a', help="algorithm", type=str, default="cfo")
    parser.add_argument("--dataset", help="dataset", type=str, default="sales")
    parser.add_argument("--metric", help="metric", type=str, default='rmse')
    parser.add_argument("--estimator", help="estimator", type=str, default='lgbm')
    parser.add_argument("--tolerance", '-t', help="tolerance", type=float, default=None)
    parser.add_argument("--five", action='store_true')
    args = parser.parse_args()
    if args.algorithm == 'cfo':
        args.tolerance = None
    return args

def main_function(args):
    #-----get data info---------
    X_train, y_train, train_len, group_num, group_value, task = get_train_data(args.dataset)

    #-----set result path-------
    # set the big over folder: drug_xgboost_rmse_b1800
    folder_name = f'./out/{args.dataset}_{args.estimator}_{args.metric}_b{str(args.budget)}'
    # set path for log: /out/drug_xgboost_rmse_b1800/cfo_out/seed_1
    path = f"{folder_name}/{args.algorithm}_out/seed_{args.seed}/"
    if args.algorithm != 'cfo':
        path = f"{folder_name}/{args.algorithm}_toler{args.tolerance}_out/seed_{args.seed}/"

    if not os.path.isdir(path):
        os.makedirs(path)
    # logpath = open(os.path.join(path, "std.log"), "w")
    # sys.stdout = logpath
    # sys.stderr = logpath
    #-------begin HPO----------
    trainable = get_trainable_cv(estimator=args.estimator, 
                                metric=args.metric,
                                seed=args.seed,
                                X_train=X_train,
                                y_train=y_train,
                                group_num=group_num, 
                                group_value = group_value,
                                task=task)

    analysis = mytune(
            seed = args.seed, 
            budget= args.budget, 
            algorithm= args.algorithm, 
            tolerance= args.tolerance, 
            estimator= args.estimator, 
            trainable= trainable, 
            local_dir= path,
            train_len = train_len,
            )


    #----------save results-------------
    resul_info = {}
    resul_info["best_result"] = analysis.best_result
    resul_info["best_config"] = analysis.best_config


    # save to /out/drug_xgboost_rmse_b1800/valid.csv
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

    # save best_result and best_config
    savepath = os.path.join(path, "result_info.pckl")
    f = open(savepath, "wb")
    pickle.dump(resul_info, f)
    f.close()
    print("best_result", analysis.best_result)
    print("best_config", analysis.best_config)
    # logpath.close()

def main_nn_function(args):
    import os
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cudaid
    
    #-----get data info---------
    if args.dataset !="mimic":
        data, group_num, group_value = get_train_data(args.dataset)
    else: 
        Train_train, Train_test, group_num, group_value = get_train_data(args.dataset)

    #-----set result path-------
    args.metric = "accuracy"
    folder_name = f'./out/{args.dataset}_{args.estimator}_{args.metric}_b{str(args.budget)}'
    path = f"{folder_name}/{args.algorithm}_out/seed_{args.seed}/"
    if args.algorithm != 'cfo':
        path = f"{folder_name}/{args.algorithm}_toler{args.tolerance}_out/seed_{args.seed}/"
    if not os.path.isdir(path):
        os.makedirs(path)
    logpath = open(os.path.join(path, "std.log"), "w")
    sys.stdout = logpath
    sys.stderr = logpath
    #-----set result path-------
    if args.dataset == "yearbook":
        trainable = get_trainable_cv_nn_version2(
                                seed=args.seed,
                                dataset = data,
                                group_num = group_num, 
                                group_value = group_value,
                                )
    elif args.dataset == "drug":
        trainable = get_trainable_cv_nn_drug(
                                seed=args.seed,
                                dataset = data,
                                group_num = group_num, 
                                group_value = group_value,
                                )
    elif args.dataset == "mimic":
        trainable = get_trainable_cv_nn_mimic(
                                seed=args.seed,
                                dataset = [Train_train, Train_test],
                                group_num = group_num, 
                                group_value = group_value,
                                )
    analysis = mytune(
                    seed=args.seed, 
                    budget= args.budget, 
                    algorithm= args.algorithm, 
                    tolerance= args.tolerance, 
                    estimator= args.estimator,
                    trainable= trainable, 
                    local_dir= path,
                    data = None if args.dataset == "yearbook" else args.dataset,
                    train_len= None)

    #----------save results-------------
    resul_info = {}
    resul_info["best_result"] = analysis.best_result
    resul_info["best_config"] = analysis.best_config

    # save to /out/drug_xgboost_rmse_b1800/valid.csv
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

    # save best_result and best_config
    savepath = os.path.join(path, "result_info.pckl")
    f = open(savepath, "wb")
    pickle.dump(resul_info, f)
    f.close()
    print("best_result", analysis.best_result)
    print("best_config", analysis.best_config)
    
    logpath.close()


args = get_args()
if args.algorithm != 'cfo' and args.five:
    # CAUTION: make sure cfo is already runned and there is result of 10 seeds
    # in  [folder_name]/cfo/valid.csv 
    tolerances = get_tolerances(args) 

    for i in range(5):
        args.tolerance = tolerances[i] # reset tolerance
        set_seed(args.seed)
        if args.estimator != "nn":
            main_function(args)
        else:
            main_nn_function(args)
else:
    set_seed(args.seed)
    if args.estimator != "nn":
        main_function(args)
    else:
        main_nn_function(args)


# python main.py --cudaid 1 --seed 1 --budget 100 --algorithm cfo --dataset mimic --metric accuracy --estimator nn 