from utils import InfiniteDataLoader, set_seed, mixup_data, collate_fn_mimic, FastDataLoader
from network import define_model_mimic, define_model_yearbook, define_model_drug, ArticleNetwork, FMoWNetwork
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from flaml.group import set_test_group
from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
import argparse
import pickle
import itertools
import optuna
import sys
from torch.autograd import Variable
import os
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset
import time
import random
import arff
import torch.nn.functional as F
import warnings
from sklearn import metrics
from functools import partial
warnings.filterwarnings("ignore", category=UserWarning)


def cv_score_agg_func(val_loss_folds, log_metrics_folds):
    ''' 
    Used in get_trainable_cv
    '''
    val_loss_folds.reverse()
    metric_to_minimize = sum(val_loss_folds) / len(val_loss_folds)
    metrics_to_log = None
    for single_fold in log_metrics_folds:
        if metrics_to_log is None:
            metrics_to_log = single_fold
        elif isinstance(metrics_to_log, dict):
            metrics_to_log = {k: metrics_to_log[k] + v for k, v in single_fold.items()}
        else:
            metrics_to_log += single_fold
    if metrics_to_log:
        n = len(val_loss_folds)
        metrics_to_log = (
            {k: v / n for k, v in metrics_to_log.items()}
            if isinstance(metrics_to_log, dict)
            else metrics_to_log / n
        )
        metrics_to_log["std"] = np.std(val_loss_folds)
        metrics_to_log["worst"] = np.max(val_loss_folds)
        metrics_to_log["best"] = np.min(val_loss_folds)
        metrics_to_log["folds"] = val_loss_folds
    return metric_to_minimize, metrics_to_log


def get_lexico_objectives(algorithm, tolerance):
    if algorithm == "lexico_var":
        lexico_objectives = {}
        lexico_objectives["metrics"] = ["val_loss", "std"]
        lexico_objectives["tolerances"] = {"val_loss": 0.01, "std": 0.0}  # check
        lexico_objectives["targets"] = {"val_loss": 0.0, "std": 0.0}
        lexico_objectives["modes"] = ["min", "min"]
    elif algorithm == "hypertime":
        lexico_objectives = {}
        lexico_objectives["metrics"] = ["val_loss", "worst"]
        lexico_objectives["tolerances"] = {"val_loss": 0.01, "worst": 0.0}  # check
        lexico_objectives["targets"] = {"val_loss": 0.0, "worst": 0.0}
        lexico_objectives["modes"] = ["min", "min"]
    else:
        lexico_objectives = None

    return lexico_objectives


def get_search_space(train_len=None, estimator='lgbm', dataset=None):
    if estimator != "nn":
        upper = max(5, min(32768, int(train_len)))
    if estimator == 'lgbm':
        search_space = {
            'n_estimators': raytune.lograndint(4, upper),
            'num_leaves': raytune.lograndint(4, upper),
            "min_child_samples": raytune.lograndint(2, 2**7 + 1),
            "learning_rate": raytune.loguniform(1 / 1024, 1.0),
            "log_max_bin": raytune.lograndint(3, 11),
            "colsample_bytree": raytune.uniform(0.01, 1.0),
            "reg_alpha": raytune.loguniform(1 / 1024, 1024),
            "reg_lambda": raytune.loguniform(1 / 1024, 1024),
            "learner": "lgbm",
        }
        low_cost_partial_config = {
            "n_estimators": 4,
            "num_leaves": 4,
        }
        points_to_evaluate = [{'n_estimators': 4,
                               'num_leaves': 4,
                               "min_child_samples": 20,
                               "learning_rate": 0.1,
                               "log_max_bin": 8,
                               "colsample_bytree": 1.0,
                               "reg_alpha": 1 / 1024,
                               "reg_lambda": 1.0,
                               "learner": "lgbm",
                               }]
    elif estimator == 'xgboost':
        search_space = {
            'max_leaves': raytune.lograndint(4, upper),
            'max_depth': raytune.choice([0, 6, 12]),
            "n_estimators": raytune.lograndint(4, upper),
            "min_child_weight": raytune.loguniform(0.001, 128),
            "learning_rate": raytune.loguniform(1 / 1024, 1.0),
            "subsample": raytune.uniform(0.1, 1.0),
            "colsample_bytree": raytune.uniform(0.01, 1.0),
            "colsample_bylevel": raytune.uniform(0.01, 1.0),
            "reg_alpha": raytune.loguniform(1 / 1024, 1024),
            "reg_lambda": raytune.loguniform(1 / 1024, 1024),
            "learner": "xgboost",
        }
        low_cost_partial_config = {
            "n_estimators": 4,
            "max_leaves": 4,
        }
        points_to_evaluate = [{"n_estimators": 4,
                               "max_leaves": 4,
                               "max_depth": 0,
                               "min_child_weight": 1.0,
                               "learning_rate": 0.1,
                               "subsample": 1.0,
                               "colsample_bylevel": 1.0,
                               "colsample_bytree": 1.0,
                               "reg_alpha": 1 / 1024,
                               "reg_lambda": 1.0,
                               }]
    elif estimator == "nn" and dataset == "yearbook":

        search_space = {
            "lr": raytune.loguniform(1e-4, 1e-1),
            "batch_size": raytune.choice([32, 64, 128, 256]),
            "iteration_limit": raytune.randint(3000, 5000),
            'n_conv_channels_c1': raytune.qlograndint(16, 512, q=2, base=2),
            'kernel_size_c1': raytune.randint(2, 5),
            'has_max_pool_c1': raytune.choice([0, 1]),
            'n_conv_channels_c2': raytune.qlograndint(16, 512, q=2, base=2),
            'kernel_size_c2': raytune.randint(2, 5),
            'has_max_pool_c2': raytune.choice([0, 1]),
            'n_conv_channels_c3': raytune.qlograndint(16, 512, q=2, base=2),
            'kernel_size_c3': raytune.randint(2, 5),
            'has_max_pool_c3': raytune.choice([0, 1]),
            'n_conv_channels_c4': raytune.qlograndint(16, 512, q=2, base=2),
            'kernel_size_c4': raytune.randint(2, 5),
            'has_max_pool_c4': raytune.choice([0, 1]),
        }
        low_cost_partial_config = {
            "lr": 1e-1,
            "batch_size": 32,
            "iteration_limit": 3000,

            'n_conv_channels_c1': 16,
            'kernel_size_c1': 3,
            'n_conv_channels_c2': 16,
            'kernel_size_c2': 3,
            'n_conv_channels_c3': 16,
            'kernel_size_c3': 3,
            'n_conv_channels_c4': 16,
            'kernel_size_c4': 3,

            'has_max_pool_c1': 1,
            'has_max_pool_c2': 1,
            'has_max_pool_c3': 1,
            'has_max_pool_c4': 1,
        }
        points_to_evaluate = None

    elif estimator == "nn" and dataset == "drug":

        search_space = {
            "lr": raytune.loguniform(5e-5, 5e-2),
            "batch_size": raytune.choice([128, 256, 512]),
            "iteration_limit": raytune.randint(5000, 7000),

            "hid_dim_1": raytune.qlograndint(128, 512, q=2, base=2),  # 256
            "hid_dim_2": raytune.qlograndint(128, 512, q=2, base=2),  # 128
        }
        low_cost_partial_config = {
            "lr": 5e-5,
            "batch_size": 256,
            "iteration_limit": 5000,

            "hid_dim_1": 256,  # 256
            "hid_dim_2": 128,  # 128
        }
        points_to_evaluate = None

    elif estimator == "nn" and dataset in ["mimic-mortality", "mimic-readmission"]:

        search_space = {
            "lr": raytune.loguniform(5e-4, 5e-2),
            "train_update_iter": raytune.randint(3000, 5000),

            "head_num": raytune.randint(2, 5),  # 256
            "layer_num": raytune.randint(2, 5),  # 128
            "hidden_size": raytune.choice([64, 128, 256, 512]),  # 128
        }

        low_cost_partial_config = {
            "lr": 5e-4,
            "train_update_iter": 3000,

            "head_num": 2,
            "layer_num": 2,
            "hidden_size": 128,
        }

        points_to_evaluate = None

    elif estimator == "nn" and dataset == "huffpost":

        search_space = {
            "batch_size": raytune.choice([32]),
            "lr": raytune.loguniform(1.5e-5, 1.5e-4),
            "train_update_iter": raytune.randint(6000, 8000),
            "weight_decay": raytune.uniform(0.01, 0.03),
        }

        low_cost_partial_config = {
            "batch_size": 32,
            "lr": 2e-5,
            "train_update_iter": 6000,
            "weight_decay": 0.01,
        }

        points_to_evaluate = None

    elif estimator == "nn" and dataset == "arxiv":

        search_space = {
            "batch_size": raytune.choice([64]),
            "lr": raytune.loguniform(1.5e-5, 1.5e-4),
            "train_update_iter": raytune.randint(6000, 8000),
            "weight_decay": raytune.uniform(0.01, 0.03),
        }

        low_cost_partial_config = {
            "batch_size": 64,
            "lr": 2e-5,
            "train_update_iter": 6000,
            "weight_decay": 0.01,
        }

    elif estimator == "nn" and dataset == "fmow":

        search_space = {
            "train_update_iter": raytune.randint(3000, 6000),
            "lr": raytune.loguniform(1.5e-5, 3e-4),
            "batch_size": raytune.choice([32, 64, 128, 256]),
            "weight_decay": raytune.uniform(0.00, 0.03),
        }

        low_cost_partial_config = {
            "train_update_iter": 3000,
            "lr": 0.0001,
            "batch_size": 64,
            "weight_decay":  0.0,
        }

        points_to_evaluate = None

    return search_space, low_cost_partial_config, points_to_evaluate


def get_trainable_cv_nn_yearbook(seed, dataset, group_num, group_value):
    def train_cifar(configuration):
        loss, accuracy = 0, 0
        lr = configuration.get("lr")
        batch_size = configuration.get("batch_size")
        iteration_limit = configuration.get("iteration_limit")
        net = define_model_yearbook(configuration)
        if net == -1:
            return {"val_loss": -1, "month_result": -1, "month_err": -1}
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
            kfold = GroupKFold(group_num)
            kf = kfold.split(dataset, groups=group_value)
            group_dict = {}
            group_id = 0
            for ids in range(group_value[0], group_value[0] + group_num):
                start_point = group_value.index(ids)
                end_point = start_point + group_value.count(ids) - 1
                group_dict[group_id] = {}
                group_dict[group_id]["start"] = start_point
                group_dict[group_id]["end"] = end_point
                group_id += 1
            loss_dict = {}
            err_dict = {}
            result_group = -1
            true_train_index = []
            all_val_indexes = {}
            for fold, (train_ids, val_index) in enumerate(kf):
                for i in range(group_num):
                    if val_index[0] == group_dict[i]["start"] and val_index[-1] == group_dict[i]["end"]:
                        result_group = i
                        break
                # take random 10%
                test_size = round(0.1 * len(val_index))
                sample_index = random.sample(list(val_index), test_size)
                tmp_train = list(set(val_index).difference(set(sample_index)))
                true_train_index = true_train_index + tmp_train
                all_val_indexes[result_group] = sample_index

            trainloader = InfiniteDataLoader(dataset, batch_size=batch_size, subsetids=true_train_index)
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), torch.squeeze(labels, 1).to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i == iteration_limit:
                    break
    
            for test_fold in range(group_num):
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                test_subsampler = torch.utils.data.SubsetRandomSampler(all_val_indexes[test_fold])
                valloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, sampler=test_subsampler, drop_last=False)
                for i, data in enumerate(valloader, 0):
                    with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), torch.squeeze(labels, 1).to(device)
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        val_loss += loss.cpu().numpy()
                        val_steps += 1
                loss = (val_loss / val_steps)
                accuracy = (correct / total)
                loss_dict[test_fold] = loss
                err_dict[test_fold] = 1 - accuracy
            loss_list = []
            err_list = []
            for i in range(group_num):
                loss_list.append(loss_dict[i])
                err_list.append(err_dict[i])
            return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_cifar


def get_trainable_cv_nn_drug(seed, dataset, group_num, group_value):

    def train_drug(configuration):
        loss, accuracy = 0, 0
        lr = configuration.get("lr")
        batch_size = configuration.get("batch_size")
        iteration_limit = configuration.get("iteration_limit")
        net = define_model_drug(configuration)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        net.to(device)
        criterion = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
        kfold = GroupKFold(group_num)
        kf = kfold.split(dataset, groups=group_value)
        group_dict = {}
        group_id = 0
        for ids in range(group_value[0], group_value[0] + group_num):
            start_point = group_value.index(ids)
            end_point = start_point + group_value.count(ids) - 1
            group_dict[group_id] = {}
            group_dict[group_id]["start"] = start_point
            group_dict[group_id]["end"] = end_point
            group_id += 1
        loss_dict = {}
        err_dict = {}
        result_group = -1
        true_train_index = []
        all_val_indexes = {}
        for fold, (train_ids, val_index) in enumerate(kf):
            for i in range(group_num):
                if val_index[0] == group_dict[i]["start"] and val_index[-1] == group_dict[i]["end"]:
                    result_group = i
                    break
            test_size = round(0.1 * len(val_index))
            sample_index = random.sample(list(val_index), test_size)
            tmp_train = list(set(val_index).difference(set(sample_index)))
            true_train_index = true_train_index + tmp_train
            all_val_indexes[result_group] = sample_index
        trainloader = InfiniteDataLoader(dataset, batch_size=batch_size, subsetids=true_train_index)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.squeeze().double()
            labels = labels.squeeze().double()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break
        for test_fold in range(group_num):
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            test_subsampler = torch.utils.data.SubsetRandomSampler(all_val_indexes[test_fold])
            valloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler, drop_last=False)
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1
            loss = (val_loss / val_steps)
            accuracy = (correct / total)
            loss_dict[test_fold] = loss
            err_dict[test_fold] = 1 - accuracy
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_drug


def get_trainable_cv_nn_mimic(seed, dataset, group_num, group_value, dataset_name):

    def train_mimic(configuration):

        loss, accuracy = 0, 0
        batch_size = 128
        lr = configuration.get("lr")
        iteration_limit = configuration.get("train_update_iter")
        droup_out = configuration.get("droup_out")
        net = define_model_mimic(configuration)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        net.to(device)
        if dataset_name == "mimic-mortality":
            class_weight = torch.FloatTensor(np.array([0.05, 0.95])).cuda()
        else:
            class_weight = torch.FloatTensor(np.array([0.26, 0.74])).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="mean").cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        trainloader = InfiniteDataLoader(dataset[0], batch_size=batch_size, subsetids=None, collate_fn=collate_fn_mimic)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = torch.cat(labels).type(torch.LongTensor).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break
        loss_dict = {}
        err_dict = {}
        net.eval()
        for test_fold in range(group_num):
            sub = torch.utils.data.Subset(dataset[1], group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=collate_fn_mimic, drop_last=True)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                inputs = inputs
                labels = torch.cat(labels).type(torch.LongTensor).cuda()
                with torch.no_grad():
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            if dataset_name == "mimic-mortality":
                metric = metrics.roc_auc_score(labels_all, pred_all)
            elif dataset_name == "mimic-readmission":
                correct = (labels_all == pred_all).sum().item()
                metric = correct / float(labels_all.shape[0])
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        print("val_loss", np.mean(loss_list))
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_mimic


def get_trainable_cv_nn_article(seed, dataset, group_num, group_value, dataset_name):

    def train_article(configuration):
        batch_size = configuration.get("batch_size")
        lr = configuration.get("lr")
        iteration_limit = configuration.get("train_update_iter")
        weight_decay = configuration.get("weight_decay")

        loss, accuracy = 0, 0
        if dataset_name == "huffpost":
            net = ArticleNetwork(num_classes=11).cuda()
        else:
            net = ArticleNetwork(num_classes=172).cuda()
        criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        net.train()
        trainloader = InfiniteDataLoader(dataset[0], batch_size=batch_size, subsetids=None, collate_fn=None)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(dtype=torch.int64).cuda()
            if len(labels.shape) > 1:
                labels = labels.squeeze(1).cuda()
            outputs = net(inputs)
            if len(labels.shape) > 1:
                labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break
        loss_dict = {}
        err_dict = {}
        net.eval()
        for test_fold in range(group_num):
            sub = torch.utils.data.Subset(dataset[1], group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=None, drop_last=True)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                inputs = inputs.to(dtype=torch.int64).cuda()
                if len(labels.shape) > 1:
                    labels = labels.squeeze(1).cuda()
                with torch.no_grad():
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            correct = (labels_all == pred_all).sum().item()
            metric = correct / float(labels_all.shape[0])
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        result = {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
        print(result)
        return result
    return train_article


def get_trainable_cv_nn_fmow(seed, dataset, group_num, group_value, dataset_name):
    # FMoWNetwork
    def train_fmow(configuration):
        # get config
        batch_size = configuration.get("batch_size")
        lr = configuration.get("lr")
        iteration_limit = configuration.get("train_update_iter")
        weight_decay = configuration.get("weight_decay")

        # init training
        loss, accuracy = 0, 0
        if dataset_name == "huffpost":
            net = FMoWNetwork().cuda()
        else:
            net = FMoWNetwork().cuda()
        optimizer = torch.optim.Adam((net.parameters()), lr=lr, weight_decay=weight_decay,
                                     amsgrad=True, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
        criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
        net.train()
        trainloader = InfiniteDataLoader(dataset[0], batch_size=batch_size, subsetids=None, collate_fn=None)
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # prepare_data
            if isinstance(inputs, tuple):
                inputs = (elt.cuda() for elt in inputs)
            else:
                inputs = inputs.cuda()
            if len(labels.shape) > 1:
                labels = labels.squeeze(1).cuda()
            # forward pass
            outputs = net(inputs)
            if len(labels.shape) > 1:
                labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                scheduler.step()
                break

        loss_dict = {}
        err_dict = {}
        net.eval()
        for test_fold in range(group_num):
            sub = torch.utils.data.Subset(dataset[1], group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=None, drop_last=True)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                # prepare data
                if isinstance(inputs, tuple):
                    inputs = (elt.cuda() for elt in inputs)
                else:
                    inputs = inputs.cuda()
                if len(labels.shape) > 1:
                    labels = labels.squeeze(1).cuda()
                # output
                with torch.no_grad():
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            correct = (labels_all == pred_all).sum().item()
            metric = correct / float(labels_all.shape[0])
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        result = {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
        print(result)
        return result
    return train_fmow


def mytune(seed, budget, trainable, local_dir, algorithm, tolerance, estimator, data=None, train_len=None):
    '''
    seed: seed,
    budget: tune time in seconds,
    trainable: a predefined function returned by get_trainable_cv
    local_dir: local_dir for flaml to output log
    algorithm: in ['cfo', 'lexico_var', 'hypertime']
    tolerance: None for cfo, has to be set for other two
    train_len: used to set upper = max(5, min(32768, int(train_len)))
    estimator: choose in ['lgbm', 'xgboost']
    '''
    search_space, low_cost_partial_config, points_to_evaluate = get_search_space(train_len, estimator, data)
    lexico_objectives = get_lexico_objectives(algorithm, tolerance)

    def evaluate_nn(config):

        trainable_return = trainable(config)
        result = {}
        result["std"] = np.std(trainable_return["month_result"])
        result["worst"] = np.max(trainable_return["month_result"])
        result["folds"] = trainable_return["month_result"]
        result["val_loss"] = trainable_return["val_loss"]

        result["std_erro"] = np.std(trainable_return["month_err"])
        result["worst_erro"] = np.max(trainable_return["month_err"])
        result["folds_erro"] = trainable_return["month_err"]
        result['error'] = np.mean(trainable_return["month_err"])
        return result

    if algorithm in ["lexico_var", "hypertime"]:
        if tolerance is None:
            raise ValueError('Have to set tolerance for lexico.')
            exit(1)
        analysis = tune.run(
            evaluate_nn,
            num_samples=-1,
            time_budget_s=budget,
            config=search_space,
            use_ray=False,
            lexico_objectives=lexico_objectives,
            low_cost_partial_config=low_cost_partial_config,
            alg_seed=seed,
            points_to_evaluate=points_to_evaluate,
            local_dir=local_dir,
            verbose=3,
        )
    elif algorithm in ["bo", "hyperband"]:
        def objective(evaluate_function, trial):
            param = {
                "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "iteration_limit": trial.suggest_int("iteration_limit", 3000, 5000),
                'n_conv_channels_c1': trial.suggest_int('n_conv_channels_c1', 16, 512, step=2),
                'kernel_size_c1': trial.suggest_int('kernel_size_c1', 2, 5),
                'has_max_pool_c1': trial.suggest_categorical('has_max_pool_c1', [0, 1]),
                'n_conv_channels_c2': trial.suggest_int('n_conv_channels_c2', 16, 512, step=2),
                'kernel_size_c2': trial.suggest_int('kernel_size_c2', 2, 5),
                'has_max_pool_c2': trial.suggest_categorical('has_max_pool_c2', [0, 1]),
                'n_conv_channels_c3': trial.suggest_int('n_conv_channels_c3', 16, 512, step=2),
                'kernel_size_c3': trial.suggest_int('kernel_size_c3', 2, 5),
                'has_max_pool_c3': trial.suggest_categorical('has_max_pool_c3', [0, 1]),
                'n_conv_channels_c4': trial.suggest_int('n_conv_channels_c4', 16, 512, step=2),
                'kernel_size_c4': trial.suggest_int('kernel_size_c4', 2, 5),
                'has_max_pool_c4': trial.suggest_categorical('has_max_pool_c4', [0, 1]),
            }
            
            result = evaluate_nn(param)
            return result["val_loss"]
        if algorithm == "bo":
            study = optuna.create_study(direction="minimize")
        else:
            study = optuna.create_study(direction="minimize",pruner=optuna.pruners.HyperbandPruner(min_resource="auto", reduction_factor=3),)
        study.optimize(partial(objective,evaluate_nn), timeout=budget)
    else:

        algo = CFO(
            space=search_space,
            metric="val_loss",
            mode="min",
            seed=seed,
            low_cost_partial_config=low_cost_partial_config,
            points_to_evaluate=points_to_evaluate,
        )
        analysis = tune.run(
            evaluate_nn,
            time_budget_s=budget,
            search_alg=algo,
            use_ray=False,
            num_samples=-1,
            metric="val_loss",
            local_dir=local_dir,
            verbose=3,
        )

    if algorithm not in ["bo","hyperband"]:
        return analysis
    else:
        return study
def get_trainable_holdout_nn_yearbook(train_data, test_data, group_num, group_value):
    set_seed(1)

    def train_cifar(configuration):
        loss, accuracy = 0, 0
        lr = configuration.get("lr")
        batch_size = configuration.get("batch_size")
        iteration_limit = configuration.get("iteration_limit")
        net = define_model_yearbook(configuration)
        if net == -1:
            return {"val_loss": -1, "month_result": -1, "month_err": -1}
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
            trainloader = InfiniteDataLoader(train_data, batch_size=batch_size, subsetids=None)

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), torch.squeeze(labels, 1).to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i == iteration_limit:
                    break
    
            loss_list = []
            err_list = []
            for test_fold_num in range(group_num):
                test_ids = group_value[test_fold_num]
                test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
                valloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=batch_size, sampler=test_subsampler, drop_last=True)
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                for i, data in enumerate(valloader, 0):
                    with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), torch.squeeze(labels, 1).to(device)
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        val_loss += loss.cpu().numpy()
                        val_steps += 1
                loss = (val_loss / val_steps)
                accuracy = (correct / total)
                loss_list.append(loss)
                err_list.append(1 - accuracy)
            return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_cifar


def get_trainable_holdout_nn_mimic(train_data, test_data, group_num, group_value, dataset_name):

    def train_mimic(configuration):
        loss, accuracy = 0, 0
        batch_size = 128
        lr = configuration.get("lr")
        iteration_limit = configuration.get("train_update_iter")
        droup_out = configuration.get("droup_out")
        net = define_model_mimic(configuration)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        net.to(device)
        if dataset_name == "mimic-mortality":
            class_weight = torch.FloatTensor(np.array([0.05, 0.95])).cuda()
        else:
            class_weight = torch.FloatTensor(np.array([0.26, 0.74])).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="mean").cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        trainloader = InfiniteDataLoader(train_data, batch_size=batch_size, subsetids=None, collate_fn=collate_fn_mimic)
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = torch.cat(labels).type(torch.LongTensor).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break

        net.eval()
        loss_dict = {}
        err_dict = {}
        for test_fold in range(group_num):
            sub = torch.utils.data.Subset(test_data, group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=collate_fn_mimic, drop_last=True)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                inputs = inputs
                labels = torch.cat(labels).type(torch.LongTensor).cuda()
                with torch.no_grad():
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            if dataset_name == "mimic-mortality":
                metric = metrics.roc_auc_score(labels_all, pred_all)
            elif dataset_name == "mimic-readmission":
                correct = (labels_all == pred_all).sum().item()
                metric = correct / float(labels_all.shape[0])
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_mimic


def get_trainable_holdout_nn_article(train_data, test_data, group_num, group_value, dataset_name):

    def train_article(configuration):
        # get config
        batch_size = configuration.get("batch_size")
        lr = configuration.get("lr")
        iteration_limit = configuration.get("train_update_iter")
        weight_decay = configuration.get("weight_decay")

        # init training
        loss, accuracy = 0, 0
        if dataset_name == "huffpost":
            net = ArticleNetwork(num_classes=11).cuda()
        else:
            net = ArticleNetwork(num_classes=172).cuda()
        criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        net.train()
        trainloader = InfiniteDataLoader(train_data, batch_size=batch_size, subsetids=None)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(dtype=torch.int64).cuda()
            if len(labels.shape) > 1:
                labels = labels.squeeze(1).cuda()
            outputs = net(inputs)
            if len(labels.shape) > 1:
                labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break

        loss_dict = {}
        err_dict = {}
        net.eval()
        for test_fold in range(group_num):
            sub = torch.utils.data.Subset(test_data, group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=None, drop_last=True)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                inputs = inputs.to(dtype=torch.int64).cuda()
                if len(labels.shape) > 1:
                    labels = labels.squeeze(1).cuda()
                with torch.no_grad():
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            correct = (labels_all == pred_all).sum().item()
            metric = correct / float(labels_all.shape[0])
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        result = {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
        print(result)
        return result
    return train_article


def get_trainable_holdout_nn_fmow(train_data, test_data, group_num, group_value, dataset_name):
    # FMoWNetwork
    def train_fmow(configuration):
        # get config
        batch_size = configuration.get("batch_size")
        lr = configuration.get("lr")
        iteration_limit = configuration.get("train_update_iter")
        weight_decay = configuration.get("weight_decay")

        # init training
        loss, accuracy = 0, 0
        if dataset_name == "huffpost":
            net = FMoWNetwork().cuda()
        else:
            net = FMoWNetwork().cuda()
        optimizer = torch.optim.Adam((net.parameters()), lr=lr, weight_decay=weight_decay,
                                     amsgrad=True, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
        criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
        net.train()
        trainloader = InfiniteDataLoader(train_data, batch_size=batch_size, subsetids=None)
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # prepare_data
            if isinstance(inputs, tuple):
                inputs = (elt.cuda() for elt in inputs)
            else:
                inputs = inputs.cuda()
            if len(labels.shape) > 1:
                labels = labels.squeeze(1).cuda()
            # forward pass
            outputs = net(inputs)
            if len(labels.shape) > 1:
                labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                scheduler.step()
                break

        loss_dict = {}
        err_dict = {}
        net.eval()
        for test_fold in range(group_num):
            sub = torch.utils.data.Subset(test_data, group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=None, drop_last=True)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                # prepare data
                if isinstance(inputs, tuple):
                    inputs = (elt.cuda() for elt in inputs)
                else:
                    inputs = inputs.cuda()
                if len(labels.shape) > 1:
                    labels = labels.squeeze(1).cuda()
                # output
                with torch.no_grad():
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            correct = (labels_all == pred_all).sum().item()
            metric = correct / float(labels_all.shape[0])
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        result = {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
        print(result)
        return result
    return train_fmow
