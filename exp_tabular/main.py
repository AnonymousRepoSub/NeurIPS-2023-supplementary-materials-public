from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
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

from train import get_trainable_cv, mytune, get_tolerances
from data_loader import get_dataset
from utils import get_folder_name, get_args, set_seed
from csv_recorder import CSVRecorder



if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    folder_name = get_folder_name(args)
    path = f"{folder_name}/{args.algorithm}_out/seed_{args.seed}/"
    if args.algorithm != 'cfo':
        path = f"{folder_name}/{args.algorithm}_toler{args.tolerance}_out/seed_{args.seed}/"

    if not os.path.isdir(path):
        os.makedirs(path)
    logpath = open(os.path.join(path, "std.log"), "w")
    sys.stdout = logpath
    sys.stderr = logpath

    #-----get data ---------
    X_train, y_train, train_len, group_num, group_value, task = get_dataset(args.dataset, split='train', shuffle=args.shuffle, data_size=args.size) 
    print('X_train', len(X_train), flush=True)

    #-------begin HPO----------
    trainable = get_trainable_cv(estimator=args.estimator, 
                                metric=args.metric,
                                seed=args.seed,
                                X_train=X_train,
                                y_train=y_train,
                                group_num=group_num, 
                                group_value = group_value,
                                task=task)

    analysis = mytune(seed=args.seed, 
                    budget= args.budget, 
                    algorithm= args.algorithm, 
                    tolerance= args.tolerance, 
                    estimator= args.estimator,
                    train_len= train_len, 
                    trainable= trainable, 
                    local_dir= path)

    #----------save results-------------
    resul_info = {}
    if args.algorithm == 'bo' or args.algorithm == 'hyperband':
        fake_dict = {
            'val_loss': analysis.best_value,
            'std': 0, 
            'worst': 0, 
            'folds': [0]*6,
        }
        resul_info["best_result"] = fake_dict
        resul_info["best_config"] = analysis.best_params
    else:
        resul_info["best_result"] = analysis.best_result
        resul_info["best_config"] = analysis.best_config

    print(['Saving results.'], flush=True)
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
    print("best_result", resul_info["best_result"])
    print("best_config", resul_info["best_config"])
    logpath.close()



