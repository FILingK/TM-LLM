import os
import numpy as np
import pandas as pd
import random
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

# Abilene
class Dataset_net_abilene(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='abilene_tm.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, sample_num=1000, seasonal_patterns=None):

        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path

        self.data_scale = 0
        self.tot_len = 0
        self.sample_num = sample_num

        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), header=None)
        # Check if the last column is entirely empty
        if df_raw.iloc[:, -1].isnull().all():
            df_raw = df_raw.iloc[:, :-1]

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.target in cols and 'data' in cols:
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]

        all_data = 3000

        # Divide in an 8:1:1 ratio
        num_train = int(all_data * 0.8)
        num_test = int(all_data * 0.1)
        num_vali = int(all_data - num_train - num_test)
        if self.set_type == 0:
            self.data_scale = 1
            self.tot_len = num_train - self.seq_len
        elif self.set_type == 1:
            self.data_scale = 1/8
            self.tot_len = num_vali - self.seq_len
        else:
            self.data_scale = 1/8
            self.tot_len = num_test - self.seq_len

        # self.data_scale allocates samples to train, test, and validation sets.
        self.sample_num = int(self.sample_num*self.data_scale)

        border1s = [0, num_train - self.seq_len, all_data - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, all_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # Remove non-data columns
        if self.target in cols and 'data' in cols:
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
        else:
            df_data = df_raw
        df_data = df_data.iloc[:3000]
        data = df_data.values
        # standardize
        if self.scale:
            data = data / 1e9

        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        s_begin = random.randint(0, self.tot_len)
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, :]

        return seq_x

    def __len__(self):
        return self.sample_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# geant
class Dataset_net_geant(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='abilene_tm.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, sample_num=1000, seasonal_patterns=None):

        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
        else:
            self.seq_len = size[0]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path

        self.data_scale = 0
        self.tot_len = 0
        self.sample_num = sample_num
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), header=None)
        # Check if the last column is entirely empty.
        if df_raw.iloc[:, -1].isnull().all():
            df_raw = df_raw.iloc[:, :-1]

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)

        # Remove non-data columns
        if self.target in cols and 'data' in cols:
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]
        else:
            df_data = df_raw

        df_data = df_data.iloc[:3000]
        # Standardize
        if self.scale:
            df = df_data / 1e7

        data = df.values

        all_data = 3000
        num_train = int(all_data * 0.8)
        num_test = int(all_data * 0.1)
        num_vali = int(all_data - num_train-num_test)

        if self.set_type == 0:
            self.data_scale = 1
            # tot_len is used to determine the position of the first data point when randomly selecting samples.
            self.tot_len = num_train - self.seq_len
        elif self.set_type == 1:
            self.data_scale = 1/8
            self.tot_len = num_vali - self.seq_len
        else:
            self.data_scale = 1/8
            self.tot_len = num_test - self.seq_len

        # Divide in an 8:1:1 ratio
        self.sample_num = int(self.sample_num*self.data_scale)

        border1s = [0, num_train - self.seq_len, all_data - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, all_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        self.data_x = data[border1:border2]



    def __getitem__(self, index):
        s_begin = random.randint(0, self.tot_len)
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end, :]
        return seq_x

    def __len__(self):
        return self.sample_num

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
