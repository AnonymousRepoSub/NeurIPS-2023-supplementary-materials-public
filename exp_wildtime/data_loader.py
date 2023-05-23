
import pandas as pd
import arff
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import OneHotEncoder
from transformers import DistilBertTokenizer
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image
from wilds import get_dataset


class YearbookBase(Dataset):

    def __init__(self):
        super().__init__()
        with open("yearbook.pkl", 'rb') as myfile:
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class MIMICBase(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == "mimic-mortality":
            print("mimic-mortality")
            self.data_file = 'mimic_mortality_wildtime.pkl'
        else:
            print("mimic-readmission_data")
            self.data_file = 'mimic_readmission_wildtime.pkl'

        self.datasets = pickle.load(open(self.data_file, 'rb'))
        self.num_classes = 2
        self.current_time = 0
        self.mini_batch_size = 128
        self.mode = 2

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: self.datasets[i][self.mode]['labels'].shape[0] for i in self.ENV}

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
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def __getitem__(self, index):

        code = self.traindata['code'][index]
        label = int(self.traindata['labels'][index])
        label_tensor = torch.LongTensor([label])

        return code, label_tensor

    def __len__(self):
        return len(self.traindata['labels'])


class MIMIC_Test(MIMICBase):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def __getitem__(self, index):

        code = self.testdata['code'][index]
        label = int(self.testdata['labels'][index])
        label_tensor = torch.LongTensor([label])

        return code, label_tensor

    def __len__(self):
        return len(self.testdata['labels'])


class MIMIC_Train_train(MIMICBase):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def __getitem__(self, index):

        code = self.train_train['code'][index]
        label = int(self.train_train['labels'][index])
        label_tensor = torch.LongTensor([label])
        return code, label_tensor

    def __len__(self):
        return len(self.train_train['labels'])


class MIMIC_Train_test(MIMICBase):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def __getitem__(self, index):

        code = self.train_test['code'][index]
        label = int(self.train_test['labels'][index])
        label_tensor = torch.LongTensor([label])
        return code, label_tensor

    def __len__(self):
        return len(self.train_test['labels'])


PREPROCESSED_FILE = 'huffpost.pkl'
MAX_TOKEN_LENGTH = 300
RAW_DATA_FILE = 'News_Category_Dataset_v2.json'
ID_HELD_OUT = 0.1


def initialize_distilbert_transform(max_token_length):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        x = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x
    return transform


class HuffPostBase(Dataset):
    def __init__(self):
        super().__init__()
        self.datasets = pickle.load(open("huffpost.pkl", 'rb'))
        self.ENV = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
        self.num_classes = 11  
        self.num_tasks = len(self.ENV)
        self.current_time = 0
        self.mini_batch_size = 32
        self.task_indices = {}
        self.transform = initialize_distilbert_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = 2

        start_idx = 0
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0
        for i, year in enumerate(self.ENV):
            end_idx = start_idx + len(self.datasets[year][self.mode]['category'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            num_examples = len(self.datasets[year][self.mode]['category'])
            cumulative_batch_size += min(self.mini_batch_size, num_examples)
            self.input_dim.append(cumulative_batch_size)

        reform_data_all = {}
        reform_data_all["headline"] = []
        reform_data_all["category"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['category'])
            for k in range(length):
                reform_data_all["headline"].append(self.datasets[i][self.mode]['headline'][k])
                reform_data_all["category"].append(self.datasets[i][self.mode]['category'][k])

        reform_data_train = {}
        reform_data_train["headline"] = []
        reform_data_train["category"] = []
        train_start = self.task_idxs[2012][0]
        train_end = self.task_idxs[2015][1]
        reform_data_train["headline"] = reform_data_all['headline'][train_start:train_end]
        reform_data_train["category"] = reform_data_all['category'][train_start:train_end]

        reform_data_test = {}
        reform_data_test["headline"] = []
        reform_data_test["category"] = []
        test_start = self.task_idxs[2016][0]
        test_end = self.task_idxs[2018][1]
        reform_data_test["headline"] = reform_data_all['headline'][test_start:test_end]
        reform_data_test["category"] = reform_data_all['category'][test_start:test_end]

        self.mode = 0
        start_idx = 0
        self.task_idxs_traintrain = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['category'].shape[0]
            self.task_idxs_traintrain[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["headline"] = []
        reform_data_all["category"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['category'])
            for k in range(length):
                reform_data_all["headline"].append(self.datasets[i][self.mode]['headline'][k])
                reform_data_all["category"].append(self.datasets[i][self.mode]['category'][k])
        reform_data_train_val = {}
        reform_data_train_val["headline"] = []
        reform_data_train_val["category"] = []
        train_start = self.task_idxs_traintrain[2012][0]
        train_end = self.task_idxs_traintrain[2015][1]
        reform_data_train_val["headline"] = reform_data_all['headline'][train_start:train_end]
        reform_data_train_val["category"] = reform_data_all['category'][train_start:train_end]

        self.mode = 1
        start_idx = 0
        self.task_idxs_traintest = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['category'].shape[0]
            self.task_idxs_traintest[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["headline"] = []
        reform_data_all["category"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['category'])
            for k in range(length):
                reform_data_all["headline"].append(self.datasets[i][self.mode]['headline'][k])
                reform_data_all["category"].append(self.datasets[i][self.mode]['category'][k])
        reform_data_test_val = {}
        reform_data_test_val["headline"] = []
        reform_data_test_val["category"] = []
        train_start = self.task_idxs_traintest[2012][0]
        train_end = self.task_idxs_traintest[2015][1]
        reform_data_test_val["headline"] = reform_data_all['headline'][train_start:train_end]
        reform_data_test_val["category"] = reform_data_all['category'][train_start:train_end]

        self.traindata = reform_data_train
        self.testdata = reform_data_test

        self.train_train = reform_data_train_val
        self.train_test = reform_data_test_val

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'huffpost'


class HuffPost_Train(HuffPostBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        headline = self.traindata['headline'][index]
        category = self.traindata['category'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.traindata['category'])


class HuffPost_Test(HuffPostBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        headline = self.testdata['headline'][index]
        category = self.testdata['category'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.testdata['category'])


class HuffPost_Train_train(HuffPostBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        headline = self.train_train['headline'][index]
        category = self.train_train['category'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.train_train['category'])


class HuffPost_Train_test(HuffPostBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        headline = self.train_test['headline'][index]
        category = self.train_test['category'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.train_test['category'])


MAX_TOKEN_LENGTH = 300
RAW_DATA_FILE = 'arxiv-metadata-oai-snapshot.json'
ID_HELD_OUT = 0.1


class ArXivBase(Dataset):
    def __init__(self):
        super().__init__()
        self.datasets = pickle.load(open("arxiv.pkl", 'rb'))

        self.ENV = [year for year in range(2007, 2023)]
        self.num_tasks = len(self.ENV)
        self.num_classes = 172
        self.mini_batch_size = 64
        self.task_indices = {}
        self.transform = initialize_distilbert_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = 2

        start_idx = 0
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0

        for i, year in enumerate(self.ENV):
            # Store task indices
            end_idx = start_idx + len(self.datasets[year][self.mode]['category'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            # Store input dim
            num_examples = len(self.datasets[year][self.mode]['category'])
            cumulative_batch_size += min(self.mini_batch_size, num_examples)
            self.input_dim.append(cumulative_batch_size)

        reform_data_all = {}
        reform_data_all["title"] = []
        reform_data_all["category"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['category'])
            for k in range(length):
                reform_data_all["title"].append(self.datasets[i][self.mode]['title'][k])
                reform_data_all["category"].append(self.datasets[i][self.mode]['category'][k])

        reform_data_train = {}
        reform_data_train["title"] = []
        reform_data_train["category"] = []
        train_start = self.task_idxs[2007][0]
        train_end = self.task_idxs[2016][1]
        reform_data_train["title"] = reform_data_all['title'][train_start:train_end]
        reform_data_train["category"] = reform_data_all['category'][train_start:train_end]
        reform_data_test = {}
        reform_data_test["title"] = []
        reform_data_test["category"] = []
        test_start = self.task_idxs[2017][0]
        test_end = self.task_idxs[2022][1]
        reform_data_test["title"] = reform_data_all['title'][test_start:test_end]
        reform_data_test["category"] = reform_data_all['category'][test_start:test_end]

        self.mode = 0
        start_idx = 0
        self.task_idxs_traintrain = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['category'].shape[0]
            self.task_idxs_traintrain[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["title"] = []
        reform_data_all["category"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['category'])
            for k in range(length):
                reform_data_all["title"].append(self.datasets[i][self.mode]['title'][k])
                reform_data_all["category"].append(self.datasets[i][self.mode]['category'][k])
        reform_data_train_val = {}
        reform_data_train_val["title"] = []
        reform_data_train_val["category"] = []
        train_start = self.task_idxs_traintrain[2007][0]
        train_end = self.task_idxs_traintrain[2016][1]
        reform_data_train_val["title"] = reform_data_all['title'][train_start:train_end]
        reform_data_train_val["category"] = reform_data_all['category'][train_start:train_end]

        self.mode = 1
        start_idx = 0
        self.task_idxs_traintest = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['category'].shape[0]
            self.task_idxs_traintest[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["title"] = []
        reform_data_all["category"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['category'])
            for k in range(length):
                reform_data_all["title"].append(self.datasets[i][self.mode]['title'][k])
                reform_data_all["category"].append(self.datasets[i][self.mode]['category'][k])
        reform_data_test_val = {}
        reform_data_test_val["title"] = []
        reform_data_test_val["category"] = []
        train_start = self.task_idxs_traintest[2007][0]
        train_end = self.task_idxs_traintest[2016][1]
        reform_data_test_val["title"] = reform_data_all['title'][train_start:train_end]
        reform_data_test_val["category"] = reform_data_all['category'][train_start:train_end]

        self.traindata = reform_data_train
        self.testdata = reform_data_test

        self.train_train = reform_data_train_val
        self.train_test = reform_data_test_val

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'arxiv'


class ArXiv_Train(ArXivBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        title = self.traindata['title'][index]
        category = self.traindata['category'][index]

        x = self.transform(text=title)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.traindata['category'])


class ArXiv_Test(ArXivBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        title = self.testdata['title'][index]
        category = self.testdata['category'][index]

        x = self.transform(text=title)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.testdata['category'])


class ArXiv_Train_train(ArXivBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        title = self.train_train['title'][index]
        category = self.train_train['category'][index]

        x = self.transform(text=title)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.train_train['category'])


class ArXiv_Train_test(ArXivBase):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):

        title = self.train_test['title'][index]
        category = self.train_test['category'][index]

        x = self.transform(text=title)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.train_test['category'])



ID_HELD_OUT = 0.1


class FMoWBase(Dataset):
    def __init__(self):
        super().__init__()

        self.datasets = pickle.load(open("fmow.pkl", 'rb'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(dataset="fmow", download=False)
        self.root = dataset.root

        self.num_classes = 62
        self.num_tasks = 17
        self.ENV = [year for year in range(0, self.num_tasks - 1)]
        self.resolution = 224
        self.mode = 2
        self.task_idxs = {}
        start_idx = 0

        for year in sorted(self.datasets.keys()):
            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = {}
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

        reform_data_all = {}
        reform_data_all["image_idxs"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["image_idxs"].append(self.datasets[i][self.mode]['image_idxs'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])

        reform_data_train = {}
        reform_data_train["image_idxs"] = []
        reform_data_train["labels"] = []
        train_start = self.task_idxs[0][0]
        train_end = self.task_idxs[10][1]
        reform_data_train["image_idxs"] = reform_data_all['image_idxs'][train_start:train_end]
        reform_data_train["labels"] = reform_data_all['labels'][train_start:train_end]
        reform_data_test = {}
        reform_data_test["image_idxs"] = []
        reform_data_test["labels"] = []
        test_start = self.task_idxs[11][0]
        test_end = self.task_idxs[15][1]
        reform_data_test["image_idxs"] = reform_data_all['image_idxs'][test_start:test_end]
        reform_data_test["labels"] = reform_data_all['labels'][test_start:test_end]

        self.mode = 0
        start_idx = 0
        self.task_idxs_traintrain = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs_traintrain[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["image_idxs"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["image_idxs"].append(self.datasets[i][self.mode]['image_idxs'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])
        reform_data_train_val = {}
        reform_data_train_val["image_idxs"] = []
        reform_data_train_val["labels"] = []
        train_start = self.task_idxs_traintrain[0][0]
        train_end = self.task_idxs_traintrain[10][1]
        reform_data_train_val["image_idxs"] = reform_data_all['image_idxs'][train_start:train_end]
        reform_data_train_val["labels"] = reform_data_all['labels'][train_start:train_end]

        self.mode = 1
        start_idx = 0
        self.task_idxs_traintest = {}
        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs_traintest[i] = [start_idx, end_idx]
            start_idx = end_idx
        reform_data_all = {}
        reform_data_all["image_idxs"] = []
        reform_data_all["labels"] = []
        for i in self.ENV:
            length = len(self.datasets[i][self.mode]['labels'])
            for k in range(length):
                reform_data_all["image_idxs"].append(self.datasets[i][self.mode]['image_idxs'][k])
                reform_data_all["labels"].append(self.datasets[i][self.mode]['labels'][k])
        reform_data_test_val = {}
        reform_data_test_val["image_idxs"] = []
        reform_data_test_val["labels"] = []
        train_start = self.task_idxs_traintest[0][0]
        train_end = self.task_idxs_traintest[10][1]
        reform_data_test_val["image_idxs"] = reform_data_all['image_idxs'][train_start:train_end]
        reform_data_test_val["labels"] = reform_data_all['labels'][train_start:train_end]

        self.traindata = reform_data_train
        self.testdata = reform_data_test

        self.train_train = reform_data_train_val
        self.train_test = reform_data_test_val

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'fmow'


class FMoW_Train(FMoWBase):
    def __init__(self):
        super().__init__()

    def get_input(self, idx):
        idx = self.traindata['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        image_tensor = self.transform(self.get_input(idx))
        label_tensor = torch.LongTensor([self.traindata['labels'][idx]])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.traindata['labels'])


class FMoW_Test(FMoWBase):
    def __init__(self):
        super().__init__()

    def get_input(self, idx):
        idx = self.testdata['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        image_tensor = self.transform(self.get_input(idx))
        label_tensor = torch.LongTensor([self.testdata['labels'][idx]])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.testdata['labels'])


class FMoW_Train_train(FMoWBase):
    def __init__(self):
        super().__init__()

    def get_input(self, idx):
        idx = self.train_train['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        image_tensor = self.transform(self.get_input(idx))
        label_tensor = torch.LongTensor([self.train_train['labels'][idx]])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.train_train['labels'])


class FMoW_Train_test(FMoWBase):
    def __init__(self):
        super().__init__()

    def get_input(self, idx):
        idx = self.train_test['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        image_tensor = self.transform(self.get_input(idx))
        label_tensor = torch.LongTensor([self.train_test['labels'][idx]])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.train_test['labels'])



def get_train_data(dataset_name):
    if dataset_name == 'drug':
        return init_drug()
    elif dataset_name == "sales":
        return init_sales()
    elif dataset_name == "zhonghua":
        return init_zhonghua()
    elif dataset_name == "yearbook":
        return init_yearbook()
    elif dataset_name in ["mimic-mortality", "mimic-readmission"]:
        return init_mimic(dataset_name)
    elif dataset_name == "huffpost":
        return init_huffpost()
    elif dataset_name == "arxiv":
        return init_arxiv()
    elif dataset_name == "fmow":
        return init_fmow()


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
    alldata = arff.load(open('sales.arff', 'r'), encode_nominal=True)
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


def init_zhonghua():
    import xarray as xr
    ds = xr.open_dataset("./data/zhonghua/003_2006_2080.nc")
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


def init_mimic(dataset_name):
    import argparse
    import pandas as pd

    Train_train = MIMIC_Train_train(dataset_name)
    Train_test = MIMIC_Train_test(dataset_name)

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


def init_huffpost():
    import argparse
    import pandas as pd

    Train_train = HuffPost_Train_train()
    Train_test = HuffPost_Train_test()

    task_idxs = Train_test.task_idxs_traintest
    test_start = task_idxs[2012][0]
    test_end = task_idxs[2015][1]
    split_points = [task_idxs[2012][1], task_idxs[2013][1], task_idxs[2014][1], task_idxs[2015][1]]
    group_value = []
    tem_group1 = []
    tem_group2 = []
    tem_group3 = []
    tem_group4 = []
    for i in range(test_end - test_start):
        index = i + test_start
        if index >= test_start and index <= split_points[0]:
            tem_group1.append(i)
        elif index > split_points[0] and index <= split_points[1]:
            tem_group2.append(i)
        elif index > split_points[1] and index <= split_points[2]:
            tem_group3.append(i)
        elif index > split_points[2] and index <= split_points[3]:
            tem_group4.append(i)
    group_num = 4
    group_value.append(tem_group1)
    group_value.append(tem_group2)
    group_value.append(tem_group3)
    group_value.append(tem_group4)
    return Train_train, Train_test, group_num, group_value


def init_arxiv():
    import argparse
    import pandas as pd

    Train_train = ArXiv_Train_train()
    Train_test = ArXiv_Train_test()

    task_idxs = Train_test.task_idxs_traintest
    test_start = task_idxs[2007][0]
    test_end = task_idxs[2016][1]
    split_points = [task_idxs[2007][1], task_idxs[2008][1], task_idxs[2009][1], task_idxs[2010][1], task_idxs[2011]
                    [1], task_idxs[2012][1], task_idxs[2013][1], task_idxs[2014][1], task_idxs[2015][1], task_idxs[2016][1]]
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
    tem_group10 = []
    for i in range(test_end - test_start):
        index = i + test_start
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
        elif index > split_points[8] and index <= split_points[9]:
            tem_group10.append(i)
    group_num = 10
    group_value.append(tem_group1)
    group_value.append(tem_group2)
    group_value.append(tem_group3)
    group_value.append(tem_group4)
    group_value.append(tem_group5)
    group_value.append(tem_group6)
    group_value.append(tem_group7)
    group_value.append(tem_group8)
    group_value.append(tem_group9)
    group_value.append(tem_group10)
    return Train_train, Train_test, group_num, group_value


def init_fmow():
    import argparse
    import pandas as pd

    Train_train = FMoW_Train_train()
    Train_test = FMoW_Train_test()

    task_idxs = Train_test.task_idxs_traintest
    test_start = task_idxs[0][0]
    test_end = task_idxs[10][1]
    split_points = [task_idxs[0][1], task_idxs[1][1], task_idxs[2][1], task_idxs[3][1], task_idxs[4]
                    [1], task_idxs[5][1], task_idxs[6][1], task_idxs[7][1], task_idxs[8][1], task_idxs[9][1], task_idxs[10][1]]
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
    tem_group10 = []
    tem_group11 = []
    for i in range(test_end - test_start):
        index = i + test_start
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
        elif index > split_points[8] and index <= split_points[9]:
            tem_group10.append(i)
        elif index > split_points[9] and index <= split_points[10]:
            tem_group11.append(i)
    group_num = 11
    group_value.append(tem_group1)
    group_value.append(tem_group2)
    group_value.append(tem_group3)
    group_value.append(tem_group4)
    group_value.append(tem_group5)
    group_value.append(tem_group6)
    group_value.append(tem_group7)
    group_value.append(tem_group8)
    group_value.append(tem_group9)
    group_value.append(tem_group10)
    group_value.append(tem_group11)
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
            open('sales.arff', 'r'), encode_nominal=True)
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

    elif dataset_name == "zhonghua":
        import xarray as xr
        ds = xr.open_dataset("./data/zhonghua/003_2006_2080.nc")
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

    elif dataset_name in ["mimic-mortality", "mimic-readmission"]:
        import argparse
        import pandas as pd
        train_data = MIMIC_Train(dataset_name)
        test_data = MIMIC_Test(dataset_name)
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

    elif dataset_name == "huffpost":
        import argparse
        import pandas as pd
        train_data = HuffPost_Train()
        test_data = HuffPost_Test()
        task_idxs = train_data.task_idxs
        test_start = task_idxs[2016][0]
        test_end = task_idxs[2018][1]
        split_points = [task_idxs[2016][1], task_idxs[2017][1], task_idxs[2018][1]]
        group_value = []
        tem_group1 = []
        tem_group2 = []
        tem_group3 = []
        start_value = task_idxs[2016][0]
        for i in range(test_end - test_start):
            index = i + start_value
            if index >= test_start and index <= split_points[0]:
                tem_group1.append(i)
            elif index > split_points[0] and index <= split_points[1]:
                tem_group2.append(i)
            elif index > split_points[1] and index <= split_points[2]:
                tem_group3.append(i)
        group_value.append(tem_group1)
        group_value.append(tem_group2)
        group_value.append(tem_group3)
        group_num = 3
        return train_data, test_data, group_num, group_value

    elif dataset_name == "arxiv":
        import argparse
        import pandas as pd
        train_data = ArXiv_Train()
        test_data = ArXiv_Test()
        task_idxs = train_data.task_idxs
        test_start = task_idxs[2017][0]
        test_end = task_idxs[2022][1]
        split_points = [task_idxs[2017][1], task_idxs[2018][1], task_idxs[2019]
                        [1], task_idxs[2020][1], task_idxs[2021][1], task_idxs[2022][1]]
        group_value = []
        tem_group1 = []
        tem_group2 = []
        tem_group3 = []
        tem_group4 = []
        tem_group5 = []
        tem_group6 = []
        start_value = task_idxs[2017][0]
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
        group_value.append(tem_group1)
        group_value.append(tem_group2)
        group_value.append(tem_group3)
        group_value.append(tem_group4)
        group_value.append(tem_group5)
        group_value.append(tem_group6)
        group_num = 6
        return train_data, test_data, group_num, group_value

    elif dataset_name == "fmow":
        import argparse
        import pandas as pd
        train_data = FMoW_Train()
        test_data = FMoW_Test()
        task_idxs = train_data.task_idxs
        test_start = task_idxs[11][0]
        test_end = task_idxs[15][1]
        split_points = [task_idxs[11][1], task_idxs[12][1], task_idxs[13]
                        [1], task_idxs[14][1], task_idxs[15][1]]
        group_value = []
        tem_group1 = []
        tem_group2 = []
        tem_group3 = []
        tem_group4 = []
        tem_group5 = []
        start_value = task_idxs[11][0]
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
        group_value.append(tem_group1)
        group_value.append(tem_group2)
        group_value.append(tem_group3)
        group_value.append(tem_group4)
        group_value.append(tem_group5)
        group_num = 5
        return train_data, test_data, group_num, group_value