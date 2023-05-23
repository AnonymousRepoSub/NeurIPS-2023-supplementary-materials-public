
import argparse
import numpy as np
import random

def get_folder_name(args):
    folder_name = f'./out/{args.dataset}_size{args.size}_{args.estimator}_{args.metric}_b{str(args.budget)}'
    if args.shuffle:
        folder_name += '_shuffle'
    return folder_name

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="size", type=int, default=0)
    parser.add_argument("--dataset", help="dataset", type=str, default="electricity")
    parser.add_argument("--estimator", help="estimator", type=str, default='xgboost')
    parser.add_argument("--metric", help="metric", type=str, default='roc_auc')
    parser.add_argument("--budget", help="budget", type=int, default=7200)
    parser.add_argument("--algorithm",'-a', help="algorithm", type=str, default="hypertime")
    parser.add_argument("--tolerance", help="tolerance", type=float, default=0.01)
    parser.add_argument("--seed", help="seeds", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true')
    args = parser.parse_args()
    if args.algorithm == 'cfo':
        args.tolerance = None
    return args
