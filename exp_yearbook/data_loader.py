
import pandas as pd
import arff
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import OneHotEncoder

# --------------------- year book ---------------------


class YearbookBase(Dataset):

    def __init__(self):
        super().__init__()
        with open("/Data/yearbook.pkl", 'rb') as myfile:
            self.datasets = pickle.load(myfile)
        self.num_classes = 2
        self.mode = 2
        self.ENV = list(sorted(self.datasets.keys()))
        self.num_years = len(self.ENV)
        self.num_examples = {i: len(self.datasets[i][self.mode]['labels']) for i in self.ENV}
        self.task_idxs = {}
        start_idx = 0
        for i in self.ENV:
            end_idx = start_idx + len(self.datasets[i][self.mode]['labels'])
            self.task_idxs[i] = {}
            self.task_idxs[i] = [start_idx, end_idx]
            start_idx = end_idx

        reform_data_all = {}
        reform_data_all["images"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["images"].append(self.datasets[i][self.mode]['images'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])

        reform_data_train = {}
        reform_data_train["images"] = []
        reform_data_train["labels"] = []
        train_start = self.task_idxs[1930][0]
        train_end = self.task_idxs[1969][1]
        reform_data_train["images"] = reform_data_all['images'][train_start:train_end]
        reform_data_train["labels"] = reform_data_all['labels'][train_start:train_end]

        reform_data_test = {}
        reform_data_test["images"] = []
        reform_data_test["labels"] = []
        test_start = self.task_idxs[1970][0]
        test_end = self.task_idxs[2013][1]
        reform_data_test["images"] = reform_data_all['images'][test_start:test_end]
        reform_data_test["labels"] = reform_data_all['labels'][test_start:test_end]

        self.datasets = reform_data_train
        self.testdata = reform_data_test

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'yearbook'


class YearbookTest(YearbookBase):

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        image = self.testdata["images"][index]
        label = self.testdata["labels"][index]
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        label_tensor = torch.LongTensor([label])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.testdata['labels'])


class YearbookTrain(YearbookBase):

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        image = self.datasets["images"][index]
        label = self.datasets["labels"][index]
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        label_tensor = torch.LongTensor([label])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets['labels'])

# --------------------- drug ---------------------


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
              'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
               '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
               'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
               'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
               'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))


def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T


def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray().T


class TdcDtiDgBase(Dataset):

    def __init__(self):
        super().__init__()

        self.datasets = pickle.load(
            open("/Data/drug_preprocessed.pkl", 'rb'))

        self.ENV = [i for i in list(range(2013, 2021))]
        self.input_shape = [(26, 100), (63, 1000)]
        self.mode = 2

        self.task_idxs = {}
        start_idx = 0
        end_idx = 0
        for i in self.ENV:
            if i != 2019:
                start_idx = end_idx
                end_idx = start_idx + len(self.datasets[i][self.mode])
            elif i == 2019:
                start_idx = 0
                end_idx = len(self.datasets[i][self.mode])
            self.task_idxs[i] = {}
            self.task_idxs[i] = [start_idx, end_idx]

        reform_data_all = {}
        reform_data_all["input"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode])
            for k in range(length):
                reform_data_all["input"].append((self.datasets[i][self.mode].iloc[k].Drug_Enc,
                                                self.datasets[i][self.mode].iloc[k].Target_Enc))
                reform_data_all["labels"].append(self.datasets[i][self.mode].iloc[k].Y)

        reform_data_train = {}
        reform_data_train["input"] = []
        reform_data_train["labels"] = []
        train_start = self.task_idxs[2013][0]
        train_end = self.task_idxs[2018][1]
        reform_data_train["input"] = reform_data_all['input'][train_start:train_end]
        reform_data_train["labels"] = reform_data_all['labels'][train_start:train_end]

        reform_data_test = {}
        reform_data_test["input"] = []
        reform_data_test["labels"] = []
        test_start = self.task_idxs[2019][0]
        test_end = self.task_idxs[2020][1]
        reform_data_test["input"] = reform_data_all['input'][test_start:test_end]
        reform_data_test["labels"] = reform_data_all['labels'][test_start:test_end]

        self.datasets = reform_data_train
        self.testdata = reform_data_test

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class TdcDtiDgTrain(TdcDtiDgBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        d, t = self.datasets["input"][index]

        d = drug_2_embed(d)
        t = protein_2_embed(t)

        y = self.datasets["labels"][index]

        return (d, t), y

    def __len__(self):
        return len(self.datasets["labels"])


class TdcDtiDgTest(TdcDtiDgBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        d, t = self.testdata["input"][index]

        d = drug_2_embed(d)
        t = protein_2_embed(t)

        y = self.testdata["labels"][index]

        return (d, t), y

    def __len__(self):
        return len(self.testdata["labels"])

# --------------------- drug ---------------------


# --------------------- mimc ---------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_anchor_year(anchor_year_group):
    year_min = int(anchor_year_group[:4])
    year_max = int(anchor_year_group[-4:])
    assert year_max - year_min == 2
    return year_min


def assign_readmission_label(row):
    curr_subject_id = row.subject_id
    curr_admittime = row.admittime

    next_row_subject_id = row.next_row_subject_id
    next_row_admittime = row.next_row_admittime

    if curr_subject_id != next_row_subject_id:
        label = 0
    elif (next_row_admittime - curr_admittime).days > 15:
        label = 0
    else:
        label = 1

    return label


def diag_icd9_to_3digit(icd9):
    if icd9.startswith('E'):
        if len(icd9) >= 4:
            return icd9[:4]
        else:
            print(icd9)
            return icd9
    else:
        if len(icd9) >= 3:
            return icd9[:3]
        else:
            print(icd9)
            return icd9


def diag_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def diag_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + diag_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + diag_icd10_to_3digit(icd[6:])
    else:
        raise


def list_join(lst):
    return ' <sep> '.join(lst)


def proc_icd9_to_3digit(icd9):
    if len(icd9) >= 3:
        return icd9[:3]
    else:
        print(icd9)
        return icd9


def proc_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def proc_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + proc_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + proc_icd10_to_3digit(icd[6:])
    else:
        raise


class MIMICStay:

    def __init__(self,
                 icu_id,
                 icu_timestamp,
                 mortality,
                 readmission,
                 age,
                 gender,
                 ethnicity):
        self.icu_id = icu_id    # str
        self.icu_timestamp = icu_timestamp  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity

        self.diagnosis = []     # list of tuples (timestamp in min (int), diagnosis (str))
        self.treatment = []     # list of tuples (timestamp in min (int), treatment (str))

    def __repr__(self):
        return f'MIMIC ID-{self.icu_id}, mortality-{self.mortality}, readmission-{self.readmission}'


def get_stay_dict(save_dir):
    mimic_dict = {}
    input_path = process_mimic_data(save_dir)
    fboj = open(input_path)
    name_list = fboj.readline().strip().split(',')
    for eachline in fboj:
        t = eachline.strip().split(',')
        tempdata = {eachname: t[idx] for idx, eachname in enumerate(name_list)}
        mimic_value = MIMICStay(icu_id=tempdata['hadm_id'],
                                icu_timestamp=tempdata['real_admit_year'],
                                mortality=tempdata['mortality'],
                                readmission=tempdata['readmission'],
                                age=tempdata['age'],
                                gender=tempdata['gender'],
                                ethnicity=tempdata['ethnicity'])
        mimic_value.diagnosis = tempdata['diagnoses'].split(' <sep> ')
        mimic_value.treatment = tempdata['procedure'].split(' <sep> ')
        mimic_dict[tempdata['hadm_id']] = mimic_value

    pickle.dump(mimic_dict, open(os.path.join(save_dir, 'mimic_stay_dict.pkl'), 'wb'))


def preprocess_reduced_train_set(args):
    print(
        f'Preprocessing reduced train proportion dataset and saving to mimic_{args.prediction_type}_wildtime_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_codes = dataset[year][0]['code']
        train_labels = dataset[year][0]['labels']

        num_train_samples = len(train_labels)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_codes = np.array(train_codes)[train_idxs]
        new_train_labels = np.array(train_labels)[train_idxs]
        dataset[year][0]['code'] = np.stack(new_train_codes, axis=0)
        dataset[year][0]['labels'] = np.array(new_train_labels)

    preprocessed_data_file = os.path.join(
        args.data_dir, f'mimic_{args.prediction_type}_wildtime_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))


def preprocess_MIMIC(data, args):
    np.random.seed(0)
    ENV = [i for i in list(range(2008, 2020, 3))]
    datasets = {}
    temp_datasets = {}

    for i in ENV:
        datasets[i] = {}
        temp_datasets[i] = {'code': [], 'labels': []}

    for eachadmit in data:
        year = int(data[eachadmit].icu_timestamp)
        if (year - 2008) % 3 > 0:
            year = 3 * int((year - 2008) / 3) + 2008
        if year in temp_datasets:
            if args.prediction_type not in temp_datasets[year]:
                temp_datasets[year][args.prediction_type] = []
            if args.prediction_type == 'mortality':
                temp_datasets[year]['labels'].append(data[eachadmit].mortality)
            elif args.prediction_type == 'readmission':
                temp_datasets[year]['labels'].append(data[eachadmit].readmission)
            dx_list = ['dx' for _ in data[eachadmit].diagnosis]
            tr_list = ['tr' for _ in data[eachadmit].treatment]
            temp_datasets[year]['code'].append(
                [data[eachadmit].diagnosis + data[eachadmit].treatment, dx_list + tr_list])

    for eachyear in temp_datasets.keys():
        temp_datasets[eachyear]['labels'] = np.array(temp_datasets[eachyear]['labels'])
        temp_datasets[eachyear]['code'] = np.array(temp_datasets[eachyear]['code'])
        num_samples = temp_datasets[eachyear]['labels'].shape[0]
        seed_ = np.random.get_state()
        np.random.seed(0)

        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        datasets[eachyear][0] = {}
        datasets[eachyear][0]['code'] = temp_datasets[eachyear]['code'][idxs[:num_train_samples]]
        datasets[eachyear][0]['labels'] = temp_datasets[eachyear]['labels'][idxs[:num_train_samples]]

        datasets[eachyear][1] = {}
        datasets[eachyear][1]['code'] = temp_datasets[eachyear]['code'][idxs[num_train_samples:]]
        datasets[eachyear][1]['labels'] = temp_datasets[eachyear]['labels'][idxs[num_train_samples:]]

        datasets[eachyear][2] = {}
        datasets[eachyear][2]['code'] = temp_datasets[eachyear]['code']
        datasets[eachyear][2]['labels'] = temp_datasets[eachyear]['labels']

    with open(os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime.pkl'), 'wb') as f:
        pickle.dump(datasets, f)


def preprocess_orig(args):
    if not os.path.exists(os.path.join(args.data_dir, 'mimic_stay_dict.pkl')):
        get_stay_dict(args.data_dir)
    data = pickle.load(open(os.path.join(args.data_dir, 'mimic_stay_dict.pkl'), 'rb'))
    preprocess_MIMIC(data, args)


def preprocess(args):
    if not os.path.isfile(os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'mimic_{args.prediction_type}_wildtime_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)


class MIMICBase(Dataset):
    def __init__(self):
        super().__init__()

        self.data_file = '/Data/mimic_mortality_wildtime.pkl'
        self.datasets = pickle.load(open(self.data_file, 'rb'))

        self.num_classes = 2
        self.current_time = 0
        self.mini_batch_size = 128
        self.mode = 2

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: self.datasets[i][self.mode]['labels'].shape[0] for i in self.ENV}

        # create a datasets object
        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.task_idxs = {}

        start_idx = 0
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs[i] = [start_idx, end_idx]
            start_idx = end_idx

        reform_data_all = {}
        reform_data_all["code"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["code"].append(self.datasets[i][self.mode]['code'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])

        reform_data_train = {}
        reform_data_train["code"] = []
        reform_data_train["labels"] = []
        train_start = self.task_idxs[2008][0]
        train_end = self.task_idxs[2011][1]
        reform_data_train["code"] = reform_data_all['code'][train_start:train_end]
        reform_data_train["labels"] = reform_data_all['labels'][train_start:train_end]

        reform_data_test = {}
        reform_data_test["code"] = []
        reform_data_test["labels"] = []
        test_start = self.task_idxs[2014][0]
        test_end = self.task_idxs[2017][1]
        reform_data_test["code"] = reform_data_all['code'][test_start:test_end]
        reform_data_test["labels"] = reform_data_all['labels'][test_start:test_end]

        ################## train val ##################
        self.mode = 0
        start_idx = 0
        self.task_idxs_traintrain = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs_traintrain[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["code"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["code"].append(self.datasets[i][self.mode]['code'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])
        reform_data_train_val = {}
        reform_data_train_val["code"] = []
        reform_data_train_val["labels"] = []
        train_start = self.task_idxs_traintrain[2008][0]
        train_end = self.task_idxs_traintrain[2011][1]
        reform_data_train_val["code"] = reform_data_all['code'][train_start:train_end]
        reform_data_train_val["labels"] = reform_data_all['labels'][train_start:train_end]

        self.mode = 1
        start_idx = 0
        self.task_idxs_traintest = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs_traintest[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["code"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["code"].append(self.datasets[i][self.mode]['code'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])
        reform_data_test_val = {}
        reform_data_test_val["code"] = []
        reform_data_test_val["labels"] = []
        train_start = self.task_idxs_traintest[2008][0]
        train_end = self.task_idxs_traintest[2011][1]
        reform_data_test_val["code"] = reform_data_all['code'][train_start:train_end]
        reform_data_test_val["labels"] = reform_data_all['labels'][train_start:train_end]

        ################## record result ##################
        self.traindata = reform_data_train
        self.testdata = reform_data_test

        self.train_train = reform_data_train_val
        self.train_test = reform_data_test_val

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'mmic'


class MIMIC_Train(MIMICBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        code = self.traindata['code'][index]
        label = int(self.traindata['labels'][index])
        label_tensor = torch.LongTensor([label])

        return code, label_tensor

    def __len__(self):
        return len(self.traindata['labels'])


class MIMIC_Test(MIMICBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        code = self.testdata['code'][index]
        label = int(self.testdata['labels'][index])
        label_tensor = torch.LongTensor([label])

        return code, label_tensor

    def __len__(self):
        return len(self.testdata['labels'])


class MIMIC_Train_train(MIMICBase):
    
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        code = self.train_train['code'][index]
        label = int(self.train_train['labels'][index])
        label_tensor = torch.LongTensor([label])
        return code, label_tensor

    def __len__(self):
        return len(self.train_train['labels'])


class MIMIC_Train_test(MIMICBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        code = self.train_test['code'][index]
        label = int(self.train_test['labels'][index])
        label_tensor = torch.LongTensor([label])
        return code, label_tensor

    def __len__(self):
        return len(self.train_test['labels'])


# --------------------- mimc ---------------------


def get_train_data(dataset_name):
    if dataset_name == 'drug':
        return init_drug()
    # TODO
    # elif dataset_name == 'drug':
        # return init_drug()
    elif dataset_name == "sales":
        return init_sales()
    elif dataset_name == "temper":
        return init_temper()
    elif dataset_name == "yearbook":
        return init_yearbook()
    elif dataset_name == "mimic":
        return init_mimic()

def init_drug():
    import argparse
    import pandas as pd
    data = TdcDtiDgTrain()
    task_idxs = data.task_idxs
    train_start = task_idxs[2013][0]
    train_end = task_idxs[2018][1]
    split_points = [task_idxs[2013][1], task_idxs[2014][1], task_idxs[2015]
                    [1], task_idxs[2016][1], task_idxs[2017][1], task_idxs[2018][1]]
    group_value = []
    for i in range(train_end - train_start):
        index = i + train_start
        if index >= train_start and index <= split_points[0]:
            group_value.append(1)
        elif index > split_points[0] and index <= split_points[1]:
            group_value.append(2)
        elif index > split_points[1] and index <= split_points[2]:
            group_value.append(3)
        elif index > split_points[2] and index <= split_points[3]:
            group_value.append(4)
        elif index > split_points[3] and index <= split_points[4]:
            group_value.append(5)
        elif index > split_points[4] and index <= split_points[5]:
            group_value.append(6)
    group_num = 6
    return data, group_num, group_value


def init_sales():
    alldata = arff.load(open('/Data/sales.arff', 'r'), encode_nominal=True)
    df = pd.DataFrame(alldata["data"])
    df.columns = ["productId", "machineId", "temp", "weather_condition_id", "isholiday", "daysoff",
                  "year", "month", "day", "week_day", "avail0", "avail1", "avail2", "sales", "stdv"]
    order = ["year", "month", "day", "week_day", "productId", "machineId", "temp",
             "weather_condition_id", "isholiday", "daysoff", "avail0", "avail1", "avail2", "stdv", "sales"]
    df = df[order]  # 6288 9:7
    X_train, y_train = df.iloc[:6288, :-1], df.iloc[:6288, -1]
    train_len = len(X_train)
    group_num = 9
    task = 'regression'
    group_index = list(set(X_train["month"]))
    group_index.sort()
    group_value = []
    for index, month in enumerate(group_index):
        num = len(X_train[X_train["month"] == month])
        group_value += [index] * num
    group_value = np.array(group_value)
    return X_train, y_train, train_len, group_num, group_value, task


def init_temper():
    import xarray as xr
    ds = xr.open_dataset("./data/temper/003_2006_2080.nc")
    train = ds.sel(time=slice("2006-01-02", "2045-12-31")).to_dataframe().reset_index()
    test = ds.sel(time=slice("2046-01-01", "2080-12-31")).to_dataframe().reset_index()
    order = ['time', 'lat', 'lon', 'FLNS', 'FSNS', 'PRECT', 'PRSN', 'QBOT', 'TREFHT', 'UBOT', 'VBOT', 'TREFMXAV_U']
    train = train[order]
    test = test[order]
    X_train, y_train = train.iloc[:, 1:-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, 1:-1], test.iloc[:, -1]
    train_len = X_train.shape[0]
    group_num = 4
    task = 'regression'
    group_value = None
    return X_train, y_train, train_len, group_num, group_value, task

def init_yearbook():
    import argparse
    import pandas as pd
    data = YearbookTrain()
    task_idxs = data.task_idxs
    train_start = task_idxs[1930][0]
    train_end = task_idxs[1969][1]
    split_points = [task_idxs[1934][1], task_idxs[1939][1], task_idxs[1944][1], task_idxs[1949]
                    [1], task_idxs[1954][1], task_idxs[1959][1], task_idxs[1964][1], task_idxs[1969][1]]
    group_value = []
    for i in range(train_end - train_start):
        index = i + train_start
        if index >= train_start and index <= split_points[0]:
            group_value.append(1)
        elif index > split_points[0] and index <= split_points[1]:
            group_value.append(2)
        elif index > split_points[1] and index <= split_points[2]:
            group_value.append(3)
        elif index > split_points[2] and index <= split_points[3]:
            group_value.append(4)
        elif index > split_points[3] and index <= split_points[4]:
            group_value.append(5)
        elif index > split_points[4] and index <= split_points[5]:
            group_value.append(6)
        elif index > split_points[5] and index <= split_points[6]:
            group_value.append(7)
        elif index > split_points[6] and index <= split_points[7]:
            group_value.append(8)
    group_num = 8
    return data, group_num, group_value


def init_mimic():
    import argparse
    import pandas as pd
    
    Train_train = MIMIC_Train_train()
    Train_test = MIMIC_Train_test()

    task_idxs = Train_test.task_idxs_traintest
    test_start = task_idxs[2008][0]
    test_end = task_idxs[2011][1]
    split_points = [task_idxs[2008][1], task_idxs[2011][1]]
    group_value = []
    tem_group1 = []
    tem_group2 = []
    for i in range(test_end - test_start):
        index = i + test_start
        if index >= test_start and index <= split_points[0]:
            tem_group1.append(i)
        elif index > split_points[0] and index <= split_points[1]:
            tem_group2.append(i)
    group_num = 2
    group_value.append(tem_group1)
    group_value.append(tem_group2)
    return Train_train, Train_test, group_num, group_value


def get_test_data(dataset_name):

    if dataset_name == "drug":
        order = ["Drug_ID", "Drug", "Target_ID", "Target", "Year", "Y"]
        train = pd.read_csv("./data/drug/train_val.csv")
        test = pd.read_csv("./data/drug/test.csv")
        train = train[order]
        test = test[order]

        train_length = train.shape[0]
        test_length = test.shape[0]

        X_train, y_train = train.iloc[0:train_length, 0:-1], train.iloc[0:train_length, -1]
        X_test, y_test = test.iloc[0:test_length, 0:-1], test.iloc[0:test_length, -1]
        train_len = len(train)
        group_num = 4
        task = 'regression'
        group_value = None
        return X_train, y_train, X_test, y_test, train_len, group_num, group_value, task

    elif dataset_name == "sales":
        alldata = arff.load(
            open('/Data/sales.arff', 'r'), encode_nominal=True)
        df = pd.DataFrame(alldata["data"])
        df.columns = ["productId", "machineId", "temp", "weather_condition_id", "isholiday", "daysoff",
                      "year", "month", "day", "week_day", "avail0", "avail1", "avail2", "sales", "stdv"]
        order = ["year", "month", "day", "week_day", "productId", "machineId", "temp",
                 "weather_condition_id", "isholiday", "daysoff", "avail0", "avail1", "avail2", "stdv", "sales"]
        df = df[order]  # 6288 9:7
        X_train, y_train = df.iloc[:6288, :-1], df.iloc[:6288, -1]
        X_test, y_test = df.iloc[6288:, :-1], df.iloc[6288:, -1]
        train_len = len(X_train)
        group_num = 4
        task = 'regression'
        group_index = list(set(X_test["month"]))
        group_index.sort()
        group_value = []
        for index, month in enumerate(group_index):
            num = len(X_test[X_test["month"] == month])
            group_value += [index] * num
        return X_train, y_train, X_test, y_test, train_len, group_num, group_value, task

    elif dataset_name == "temper":
        import xarray as xr
        ds = xr.open_dataset("./Data/temper/003_2006_2080.nc")
        train = ds.sel(time=slice("2006-01-02", "2045-12-31")).to_dataframe().reset_index()
        test = ds.sel(time=slice("2046-01-01", "2080-12-31")).to_dataframe().reset_index()
        order = ['time', 'lat', 'lon', 'FLNS', 'FSNS', 'PRECT', 'PRSN', 'QBOT', 'TREFHT', 'UBOT', 'VBOT', 'TREFMXAV_U']
        train = train[order]
        test = test[order]
        X_train, y_train = train.iloc[:, 1:-1], train.iloc[:, -1]
        X_test, y_test = test.iloc[:, 1:-1], test.iloc[:, -1]
        train_len = X_train.shape[0]
        group_num = 4
        task = 'regression'
        group_value = None
        return X_train, y_train, train_len, group_num, group_value, task

    elif dataset_name == "yearbook":
        import argparse
        import pandas as pd
        train_data = YearbookTrain()
        test_data = YearbookTest()
        task_idxs = train_data.task_idxs
        test_start = task_idxs[1970][0]
        test_end = task_idxs[2013][1]
        split_points = [task_idxs[1974][1], task_idxs[1979][1], task_idxs[1984][1], task_idxs[1989][1],
                        task_idxs[1994][1], task_idxs[1999][1], task_idxs[2004][1], task_idxs[2009][1], task_idxs[2013][1]]
        group_value = []
        tem_group1 = []
        tem_group2 = []
        tem_group3 = []
        tem_group4 = []
        tem_group5 = []
        tem_group6 = []
        tem_group7 = []
        tem_group8 = []
        tem_group9 = []
        start_value = task_idxs[1970][0]
        for i in range(test_end - test_start):
            index = i + start_value
            if index >= test_start and index <= split_points[0]:
                tem_group1.append(i)
            elif index > split_points[0] and index <= split_points[1]:
                tem_group2.append(i)
            elif index > split_points[1] and index <= split_points[2]:
                tem_group3.append(i)
            elif index > split_points[2] and index <= split_points[3]:
                tem_group4.append(i)
            elif index > split_points[3] and index <= split_points[4]:
                tem_group5.append(i)
            elif index > split_points[4] and index <= split_points[5]:
                tem_group6.append(i)
            elif index > split_points[5] and index <= split_points[6]:
                tem_group7.append(i)
            elif index > split_points[6] and index <= split_points[7]:
                tem_group8.append(i)
            elif index > split_points[7] and index <= split_points[8]:
                tem_group9.append(i)
        group_value.append(tem_group1)
        group_value.append(tem_group2)
        group_value.append(tem_group3)
        group_value.append(tem_group4)
        group_value.append(tem_group5)
        group_value.append(tem_group6)
        group_value.append(tem_group7)
        group_value.append(tem_group8)
        group_value.append(tem_group9)
        group_num = 9
        return train_data, test_data, group_num, group_value

    elif dataset_name == "mimic":
        import argparse
        import pandas as pd
        train_data = MIMIC_Train()
        test_data = MIMIC_Test()
        task_idxs = train_data.task_idxs
        test_start = task_idxs[2014][0]
        test_end = task_idxs[2017][1]
        split_points = [task_idxs[2014][1], task_idxs[2017][1]]
        group_value = []
        tem_group1 = []
        tem_group2 = []
        start_value = task_idxs[2014][0]
        for i in range(test_end - test_start):
            index = i + start_value
            if index >= test_start and index <= split_points[0]:
                tem_group1.append(i)
            elif index > split_points[0] and index <= split_points[1]:
                tem_group2.append(i)
        group_value.append(tem_group1)
        group_value.append(tem_group2)
        group_num = 2
        return train_data, test_data, group_num, group_value


