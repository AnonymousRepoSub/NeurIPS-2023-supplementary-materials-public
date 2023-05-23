
import pandas as pd
import arff
import numpy as np
import matplotlib.pyplot as plt

# --------- data processing section -------------

def get_dataset(dataset_name, split, shuffle=False, data_size=0):
    if split not in ['train', 'test']:
        raise ValueError(f"Split has to be in [train, test], got {split}")
    print(f'Loading {dataset_name} with split {split}.')
    
    if dataset_name == "sales":
        return init_sales(split=split, shuffle=shuffle)
    elif dataset_name == "temp":
        return init_temp(data_size, split=split, shuffle=shuffle)
    elif dataset_name == 'vessel':
        return init_vessel(data_size, split=split, shuffle=shuffle)
    elif dataset_name == 'electricity':
        return init_electricity(split=split, shuffle=shuffle)
    else:
        raise ValueError(f"Method to load {dataset_name} is not found. Please add it in data_loader.")

def init_electricity(split='train', shuffle=False):
    alldata = arff.load(open('electricity.arff', 'r'), encode_nominal=True)
    df = pd.DataFrame(alldata["data"])
    data_length = int(df.shape[0])
    train,test = df.iloc[11439:28958, :], df.iloc[28958:data_length, :]
    if shuffle:
        print("shuffling electricity training data.", flush=True)
        train = train.sample(frac=1).reset_index().iloc[:, 1:] # caution: ablation study
    X_train, y_train, X_test, y_test = train.iloc[:, 0:-1], train.iloc[:, -1], test.iloc[:, 0:-1], test.iloc[:, -1]
    
    train_len = X_train.shape[0]
    task = 'classification'

    if split == 'train':
        train_group_num = 6
        train_group_value = None
        return X_train, y_train, train_len, train_group_num, train_group_value, task
    else:
        test_group_num = 6
        test_group_value = None
        return X_train, y_train, X_test, y_test, train_len, test_group_num, test_group_value, task

def init_sales(split='train', shuffle=False):
    alldata = arff.load(open('sales.arff', 'r'), encode_nominal=True)
    df = pd.DataFrame(alldata["data"])
    df.columns = ["productId","machineId","temp","weather_condition_id","isholiday","daysoff","year","month","day","week_day","avail0","avail1","avail2","sales","stdv"]
    order = ["year","month","day","week_day", "productId","machineId","temp","weather_condition_id","isholiday","daysoff","avail0","avail1","avail2","stdv","sales"]
    df = df[order] # 6288 9:4 8,5
    train, test = df.iloc[:5032, :], df.iloc[5032:, :]
    if shuffle:
        print("shuffling sales training data.", flush=True)
        train = train.sample(frac=1).reset_index().iloc[:, 1:] # CAUTION: shuffle 
    X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1]
    X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]

    train_len = len(X_train)
    task = 'regression'

    if split == 'train':
        train_group_num = 4
        train_group_value = None

        return X_train, y_train, train_len, train_group_num, train_group_value, task
    else:
        test_group_num = 5 # test group num
        group_index = list(set(X_test["month"]))
        group_index.sort()
        test_group_value = []
        for index, month in enumerate(group_index):
            num = len(X_test[X_test["month"] == month])
            test_group_value += [index]*num 
        return X_train, y_train, X_test, y_test, train_len, test_group_num, test_group_value, task

def init_temp(data_size, split='train', shuffle=False):
    import xarray as xr
    ds = xr.open_dataset("/temperature/003_2006_2080.nc")
    train = ds.sel(time=slice("2006-01-02", "2045-12-31")).to_dataframe().reset_index()
    test = ds.sel(time=slice("2046-01-01", "2080-12-31")).to_dataframe().reset_index()

    if data_size > 0:
        train = train.sample(data_size)
        train = train.sort_values(by=['time']).reset_index()
    order = ['time', 'lat', 'lon', 'FLNS', 'FSNS', 'PRECT', 'PRSN', 'QBOT', 'TREFHT', 'UBOT', 'VBOT','TREFMXAV_U']
    train = train[order]
    test = test[order]
    if shuffle:
        print("shuffling temp training data.", flush=True)
        print(train.columns, flush=True)
        train = train.sample(frac=1).reset_index().iloc[:, 1:] # CAUTION: shuffle 
        print(train.columns, flush=True)
    X_train, y_train = train.iloc[:,1:-1], train.iloc[:,-1]
    X_test, y_test = test.iloc[:,1:-1], test.iloc[:,-1]

    train_len = X_train.shape[0]
    task = 'regression'

    if split == 'train':
        train_group_num = 8
        train_group_value = None
        return X_train, y_train, train_len, train_group_num, train_group_value, task
    else:
        test_group_num = 7
        test_group_value = None
        return X_train, y_train, X_test, y_test, train_len, test_group_num, test_group_value, task

def init_vessel(data_size, split='train', shuffle=False):
    train = pd.read_csv("/power_consumption_upload/synthetic_data/train.csv")
    if data_size == 100000:
        train = train[:100356] #  up to time id 518400, around 12 month
    test = pd.read_csv("/power_consumption_upload/synthetic_data/dev_out.csv")
    if shuffle:
            print("shuffling vessel training data.", flush=True)
            print(train[:2])
            train = train.sample(frac=1).reset_index().iloc[:, 1:] # CAUTION: shuffle 
            print(train[:2])

    X_train, y_train = train.iloc[:,0:10], train.iloc[:,-1]
    X_test, y_test = test.iloc[:,0:10], test.iloc[:,-1]

    train_len = X_train.shape[0]
    task = 'regression'

    if split == 'train':
        train_group_num = 12 
        train_group_value = None
        return X_train, y_train, train_len, train_group_num, train_group_value, task
    else:
        test_group_num = 7
        test_group_value = None
        return X_train, y_train, X_test, y_test, train_len, test_group_num, test_group_value, task














