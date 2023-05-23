from utils import InfiniteDataLoader, set_seed, mixup_data, collate_fn_mimic, FastDataLoader
from network import define_model_mimic
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
warnings.filterwarnings("ignore", category=UserWarning)


class YearbookNetwork(nn.Module):

    def __init__(self, layers_dict):
        super(YearbookNetwork, self).__init__()
        self.enc = layers_dict["enc"]
        self.classifier = layers_dict["classifier"]

    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.classifier(x)
        return x


def define_model(config):
    layers_dict = {}
    nn_space = config.copy()
    n_convs = 4
    pre_flat_size = 32
    in_channels = 3
    out_kernel = None

    layers = []
    for i in range(n_convs):
        # if pre_flat_size > 7:
        out_channels = nn_space.get("n_conv_channels_c{}".format(i + 1))
        kernel_size = nn_space.get("kernel_size_c{}".format(i + 1))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        # pre_flat_size = pre_flat_size - kernel_size+1
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        # if nn_space.get("has_max_pool_c{}".format(i+1)) & pre_flat_size > 3:
        layers.append(nn.MaxPool2d(2))
        pre_flat_size = int(pre_flat_size / 2)
        in_channels = out_channels
        out_kernel = kernel_size
    layers_dict["enc"] = nn.Sequential(*layers)

    layers = []
    in_features = out_channels
    # for i in range(n_fulls):
    #     out_features = nn_space.get("n_l_units_l{}".format(i+1))
    #     layers.append(nn.Linear(in_features, out_features))
    #     layers.append(nn.ReLU())
    #     if nn_space.get("has_dropout_l{}".format(i+1)):
    #         p = nn_space.get("dropout_l{}".format(i+1))
    #         layers.append(nn.Dropout(p))
    #     layers.append(nn.LayerNorm(out_features))
    #     in_features = out_features
    layers.append(nn.Linear(in_features, 2))
    # layers.append(nn.LogSoftmax(dim=1))
    layers_dict["classifier"] = nn.Sequential(*layers)
    return YearbookNetwork(layers_dict)


def define_model_drug(config):

    class CNN(nn.Sequential):
        def __init__(self, encoding):
            super(CNN, self).__init__()
            if encoding == 'drug':
                in_ch = [63] + [32, 64, 96]
                kernels = [4, 6, 8]
                layer_size = 3
                self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                     out_channels=in_ch[i + 1],
                                                     kernel_size=kernels[i]) for i in range(layer_size)])
                self.conv = self.conv.double()
                n_size_d = self._get_conv_output((63, 100))
                self.fc1 = nn.Linear(n_size_d, 256)
            elif encoding == 'protein':
                in_ch = [26] + [32, 64, 96]
                kernels = [4, 8, 12]
                layer_size = 3
                self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                     out_channels=in_ch[i + 1],
                                                     kernel_size=kernels[i]) for i in range(layer_size)])
                self.conv = self.conv.double()
                n_size_p = self._get_conv_output((26, 1000))
                self.fc1 = nn.Linear(n_size_p, 256)

        def _get_conv_output(self, shape):
            bs = 1
            input = Variable(torch.rand(bs, *shape))
            output_feat = self._forward_features(input.double())
            n_size = output_feat.data.view(bs, -1).size(1)
            return n_size

        def _forward_features(self, x):
            for l in self.conv:
                x = F.relu(l(x))
            x = F.adaptive_max_pool1d(x, output_size=1)
            return x

        def forward(self, v):
            v = self._forward_features(v.double())
            v = v.view(v.size(0), -1)
            v = self.fc1(v.float())
            return v

    class DTI_Encoder(nn.Sequential):
        def __init__(self, hid_dim_1, hid_dim_2):
            super(DTI_Encoder, self).__init__()
            self.input_dim_drug = 256
            self.input_dim_protein = 256

            self.model_drug = CNN('drug')
            self.model_protein = CNN('protein')

            self.dropout = nn.Dropout(0.1)

            # self.hidden_dims = [256, 128]
            self.hidden_dims = [hid_dim_1, hid_dim_2]

            layer_size = len(self.hidden_dims) + 1
            dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [128]

            self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
            self.n_outputs = 128

        def forward(self, x):
            v_D, v_P = x
            # each encoding
            v_D = self.model_drug(v_D)
            v_P = self.model_protein(v_P)
            # concatenate and classify
            v_f = torch.cat((v_D, v_P), 1)
            for i, l in enumerate(self.predictor):
                v_f = l(v_f)
            return v_f

    def DTI_Classifier(in_features, out_features, is_nonlinear=False):
        if is_nonlinear:
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 2, in_features // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 4, out_features))
        else:
            return torch.nn.Linear(in_features, out_features)

    hid_dim_1 = config["hid_dim_1"]
    hid_dim_2 = config["hid_dim_2"]

    featurizer = DTI_Encoder(hid_dim_1, hid_dim_2)
    classifier = DTI_Classifier(featurizer.n_outputs, 1)
    network = nn.Sequential(featurizer, classifier)
    return network


def get_tolerances(args):
    '''
    Get five tolerances from CFO results.
    '''
    folder_name = f'{args.dataset}_{args.estimator}_{args.metric}_b{str(args.budget)}'
    csv_file = f'./out/{folder_name}/cfo/valid.csv'  # !! fixed to cfo/valid
    valid_df = pd.read_csv(csv_file)
    mean = np.mean(np.array(valid_df['val_loss']))
    tolerances = [round(mean * 0.002 * i, 4) for i in range(1, 6)]
    print(tolerances)
    with open(f'./out/{folder_name}/toler.txt', 'w') as f:
        f.write(str(tolerances))
    return tolerances


def cv_score_agg_func(val_loss_folds, log_metrics_folds):
    ''' Used in get_trainable_cv
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


def get_trainable_cv(X_train,
                     y_train,
                     seed,
                     group_num,
                     group_value,
                     estimator,
                     metric,
                     task):
    ''' Only used in training with holdout.
    '''
    set_test_group()
    length = len(X_train)

    if group_value is None:
        fold_length = int(length / group_num)
        group = []
        for i in range(group_num):
            group += [i] * fold_length
            if i == group_num - 1 and len(group) != length:
                group += [group_num - 1] * (length - len(group))
        group = np.array(group)
    else:
        group = group_value

    automl = AutoML()
    automl._state.eval_only = True
    settings = {
        "time_budget": 0,
        "estimator_list": [
            estimator,
        ],
        "metric": metric,  # check
        "use_ray": False,
        "task": task,
        "max_iter": 0,
        "keep_search_state": True,
        "log_training_metric": True,
        "verbose": 0,
        "eval_method": "cv",
        "mem_thres": 128 * (1024**3),
        "seed": seed,
        "split_type": "group",
        "groups": group,
        "n_splits": group_num,
        "cv_score_agg_func": cv_score_agg_func,
    }
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    trainable = automl.trainable
    return trainable


def get_lexico_objectives(algorithm, tolerance):
    if algorithm == "lexico_var":
        lexico_objectives = {}
        lexico_objectives["metrics"] = ["val_loss", "std"]
        lexico_objectives["tolerances"] = {"val_loss": 0.01, "std": 0.0}  # check
        lexico_objectives["targets"] = {"val_loss": 0.0, "std": 0.0}
        lexico_objectives["modes"] = ["min", "min"]
    elif algorithm == "hyperTime":
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

    elif estimator == "nn" and dataset == "mimic":

        search_space = {
            "lr": raytune.loguniform(5e-4, 5e-2),
            "train_update_iter": raytune.randint(3000, 5000),
            "drop_out": raytune.uniform(0.2, 0.8),

            "head_num": raytune.randint(2, 5),  # 256
            "layer_num": raytune.randint(2, 5),  # 128
            "hidden_size": raytune.choice([64, 128, 256, 512]),  # 128
        }

        low_cost_partial_config = {
            "lr": 5e-4,
            "train_update_iter": 3000,
            "drop_out": 0.5,

            "head_num": 2,
            "layer_num": 2,
            "hidden_size": 128,
        }

        points_to_evaluate = None

    return search_space, low_cost_partial_config, points_to_evaluate


def get_trainable_cv_nn(seed, dataset, group_num, group_value):

    def train_cifar(configuration):
        loss, accuracy = 0, 0
        lr = configuration.get("lr")
        batch_size = configuration.get("batch_size")
        iteration_limit = configuration.get("iteration_limit")
        net = define_model(configuration)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
        kfold = GroupKFold(group_num)
        kf = kfold.split(dataset, groups=group_value)
        ############# get group ids ############
        group_dict = {}
        group_id = 0
        for ids in range(group_value[0], group_value[0] + group_num):
            start_point = group_value.index(ids)
            end_point = start_point + group_value.count(ids) - 1
            group_dict[group_id] = {}
            group_dict[group_id]["start"] = start_point
            group_dict[group_id]["end"] = end_point
            group_id += 1
        ############# get group ids ############
        loss_dict = {}
        err_dict = {}
        result_group = -1
        for fold, (train_ids, test_ids) in enumerate(kf):
            ########################### data split ###########################
            for i in range(group_num):
                if test_ids[0] == group_dict[i]["start"] and test_ids[-1] == group_dict[i]["end"]:
                    result_group = i
                    break
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = InfiniteDataLoader(
                dataset,
                batch_size=batch_size, subsetids=train_ids)
            valloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size, sampler=test_subsampler, drop_last=True)
            ########################### train loop ###########################
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
            ########################### val loop ###########################
            for i, data in enumerate(valloader, 0):
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
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
            loss_dict[result_group] = loss
            err_dict[result_group] = 1 - accuracy
        ######################### change dict to list #########################
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        #######################################################################
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_cifar


def get_trainable_cv_nn_version2(seed, dataset, group_num, group_value):

    def train_cifar(configuration):
        loss, accuracy = 0, 0
        lr = configuration.get("lr")
        batch_size = configuration.get("batch_size")
        iteration_limit = configuration.get("iteration_limit")
        net = define_model(configuration)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
        kfold = GroupKFold(group_num)
        kf = kfold.split(dataset, groups=group_value)
        ############# get group ids ############
        group_dict = {}
        group_id = 0
        for ids in range(group_value[0], group_value[0] + group_num):
            start_point = group_value.index(ids)
            end_point = start_point + group_value.count(ids) - 1
            group_dict[group_id] = {}
            group_dict[group_id]["start"] = start_point
            group_dict[group_id]["end"] = end_point
            group_id += 1
        ############# get group ids ############
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
        ########################### train loop ###########################
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
        ########################### val loop ###########################
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
        ######################### change dict to list #########################
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        #######################################################################
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
        ############# get group ids ############
        group_dict = {}
        group_id = 0
        for ids in range(group_value[0], group_value[0] + group_num):
            start_point = group_value.index(ids)
            end_point = start_point + group_value.count(ids) - 1
            group_dict[group_id] = {}
            group_dict[group_id]["start"] = start_point
            group_dict[group_id]["end"] = end_point
            group_id += 1
        ############# get group ids ############
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
        ########################### train loop ###########################
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
        ########################### val loop ###########################
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
        ######################### change dict to list #########################
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        #######################################################################
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_drug


def get_trainable_cv_nn_mimic(seed, dataset, group_num, group_value):

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
        class_weight = torch.FloatTensor(np.array([0.05, 0.95])).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="mean").cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        ########################### train loop ###########################
        trainloader = InfiniteDataLoader(dataset[0], batch_size=batch_size, subsetids=None, collate_fn=collate_fn_mimic)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs
            labels = torch.cat(labels).type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break

        ########################### val loop ###########################
        loss_dict = {}
        err_dict = {}
        for test_fold in range(group_num):
            # test_subsampler = torch.utils.data.SubsetRandomSampler(group_value[test_fold])
            # sub = dataset[1]
            sub = torch.utils.data.Subset(dataset[1], group_value[test_fold])
            tem_sampler = torch.utils.data.RandomSampler(sub, replacement=False)
            valloader = torch.utils.data.DataLoader(
                sub,
                batch_size=batch_size, sampler=tem_sampler, collate_fn=collate_fn_mimic, drop_last=True)
            # valloader = FastDataLoader(
            #     dataset[1], batch_size=batch_size, subsetids=group_value[test_fold], collate_fn=collate_fn_mimic)
            pred_all = []
            labels_all = []
            for i, data in enumerate(valloader, 0):
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data

                with torch.no_grad():
                    inputs = inputs
                    labels = torch.cat(labels).type(torch.LongTensor).cuda()
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            metric = metrics.roc_auc_score(labels_all, pred_all)
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        ######################### change dict to list #########################
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        #######################################################################
        # print({"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list})
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_mimic

def mytune(seed, budget, trainable, local_dir, algorithm, tolerance, estimator, data=None, train_len=None):
    '''
    seed: seed,
    budget: tune time in seconds,
    trainable: a predefined function returned by get_trainable_cv
    local_dir: local_dir for flaml to output log
    algorithm: in ['cfo', 'lexico_var', 'hyperTime']
    tolerance: None for cfo, has to be set for other two
    train_len: used to set upper = max(5, min(32768, int(train_len)))
    estimator: choose in ['lgbm', 'xgboost']
    '''
    search_space, low_cost_partial_config, points_to_evaluate = get_search_space(train_len, estimator, data)
    lexico_objectives = get_lexico_objectives(algorithm, tolerance)

    def evaluate_function(config):
        trainable_return = trainable(config)
        result = {}
        result["val_loss"] = trainable_return["val_loss"]
        result["std"] = trainable_return["metric_for_logging"]["std"]
        result["worst"] = trainable_return["metric_for_logging"]["worst"]
        result["folds"] = trainable_return["metric_for_logging"]["folds"]
        return result

    def evaluate_nn(config):

        trainable_return = trainable(config)
        result = {}
        result["val_loss"] = trainable_return["val_loss"]

        result["std"] = np.std(trainable_return["month_result"])
        result["worst"] = np.max(trainable_return["month_result"])
        result["folds"] = trainable_return["month_result"]

        result["std_erro"] = np.std(trainable_return["month_err"])
        result["worst_erro"] = np.max(trainable_return["month_err"])
        result["folds_erro"] = trainable_return["month_err"]
        result['error'] = np.mean(trainable_return["month_err"])
        return result

    if algorithm in ["lexico_var", "hyperTime"]:
        if tolerance is None:
            raise ValueError('Have to set tolerance for lexico.')
            exit(1)
        analysis = tune.run(
            evaluate_function if estimator != "nn" else evaluate_nn,
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
            evaluate_function if estimator != "nn" else evaluate_nn,
            time_budget_s=budget,
            search_alg=algo,
            use_ray=False,
            num_samples=-1,
            metric="val_loss",
            local_dir=local_dir,
            verbose=3,
        )
    return analysis

def get_trainable_holdout(X_train,
                          y_train,
                          X_test,
                          y_test,
                          group_num,
                          group_value,
                          estimator,
                          task,
                          metric):
    ''' Only used in testing.
    '''
    length = len(X_train)
    set_test_group(group_value=group_value, group_num=group_num, train_len=length)

    automl = AutoML()
    automl._state.eval_only = True
    settings = {
        "time_budget": 0,
        "estimator_list": [
            estimator,
        ],
        "metric": metric,
        "use_ray": False,
        "task": task,
        "max_iter": 0,
        "keep_search_state": True,
        "log_training_metric": True,
        "verbose": 0,
        "eval_method": "holdout",
        "mem_thres": 128 * (1024**3),
        "seed": 1,
    }
    automl.fit(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, **settings)
    trainable = automl.trainable
    return trainable


def get_trainable_holdout_nn(train_data, test_data, group_num, group_value):
    set_seed(1)

    def train_cifar(configuration):
        loss, accuracy = 0, 0
        lr = configuration.get("lr")
        batch_size = configuration.get("batch_size")
        iteration_limit = configuration.get("iteration_limit")
        net = define_model(configuration)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
        trainloader = InfiniteDataLoader(train_data, batch_size=batch_size, subsetids=None)
        ########################### train loop ###########################
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
        ########################### val loop ###########################
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
        # loss_list.reverse()
        # err_list.reverse()
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_cifar


def get_trainable_holdout_nn_mimic(train_data, test_data, group_num, group_value):

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
        class_weight = torch.FloatTensor(np.array([0.05, 0.95])).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction="mean").cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        ########################### train loop ###########################
        trainloader = InfiniteDataLoader(train_data, batch_size=batch_size, subsetids=None, collate_fn=collate_fn_mimic)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs
            labels = torch.cat(labels).type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == iteration_limit:
                break

        ########################### val loop ###########################
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

                with torch.no_grad():
                    inputs = inputs
                    labels = torch.cat(labels).type(torch.LongTensor).cuda()
                    logits = net(inputs)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    labels_all = list(labels_all) + labels.cpu().numpy().tolist()
            pred_all = np.array(pred_all)
            labels_all = np.array(labels_all)
            metric = metrics.roc_auc_score(labels_all, pred_all)
            loss_dict[test_fold] = 1 - metric
            err_dict[test_fold] = 1 - metric
        ######################### change dict to list #########################
        loss_list = []
        err_list = []
        for i in range(group_num):
            loss_list.append(loss_dict[i])
            err_list.append(err_dict[i])
        #######################################################################
        return {"val_loss": np.mean(loss_list), "month_result": loss_list, "month_err": err_list}
    return train_mimic
