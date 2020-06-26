#!/usr/bin/env python
#coding=utf-8
# @file  : fill_embed
# @time  : 5/17/2020 10:13 PM
# @author: shishishu

import os
import pandas as pd
import numpy as np
from conf import config
from lib.utils.fileOperation import FileOperation
from lib.utils.utils import collect_log_content, collect_log_key_content

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'dataset.log'),
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class DataVecMaker:

    def __init__(self, embedding_size, input_dir, weight_col=None, task_type='train', data_version='002'):
        logging.info('\n')
        logging.info('data_version is: {}'.format(data_version))

        self.embedding_size = embedding_size
        self.input_dir = input_dir
        self.weight_col = weight_col
        self.task_type = task_type
        self.output_dir = os.path.join(input_dir, self.task_type, data_version)
        FileOperation.safe_mkdir(self.output_dir)
        self.log_dict = dict()
        self.log_path = os.path.join(self.output_dir, 'log_dict.json')
        collect_log_key_content(self.log_dict, 'embedding_size', self.embedding_size, self.log_path)
        collect_log_key_content(self.log_dict, 'weight_col', self.weight_col, self.log_path)

    def parse_w2v(self):
        w2v_data = []
        w2v_vector_file = 'w2v_embed_' + str(self.embedding_size) + '.txt'
        w2v_path = os.path.join(config.MODEL_DIR, 'w2v_' + str(self.embedding_size), w2v_vector_file)
        with open(w2v_path, 'r', encoding='utf-8') as fr:
            next(fr)  # skip header line
            for line in fr:
                line = line.split()
                assert len(line) == self.embedding_size + 1, 'A wrong word embedding occurs...'
                data = []
                data.append(line[0])
                data.extend(list(map(float, line[1:])))
                w2v_data.append(data)
        cols = ['creative_str']
        embed_cols = ['embed_' + str(idx) for idx in range(self.embedding_size)]
        cols.extend(embed_cols)
        df_w2v = pd.DataFrame(data=w2v_data, columns=cols)
        logging.info('shape of df w2v is: {}'.format(df_w2v.shape))
        del w2v_data
        w2v_mean = list(df_w2v[embed_cols].mean())
        return df_w2v, embed_cols, w2v_mean

    def parse_log(self):
        file_path = os.path.join(self.input_dir, self.task_type, 'click_log_clean.csv')
        df = FileOperation.load_csv(file_path)
        df['creative_str'] = df['creative_id'].apply(lambda x: 'creative_' + str(x))
        sel_cols = ['user_id', 'creative_str']
        logging.info('shape of df is: {}'.format(df.shape))
        if self.weight_col:
            weight_col_vals = sorted(list(set(df[self.weight_col])))
            weight_col_vals = list(map(int, weight_col_vals))
            weight_col_vals_expand = [[i] * i for i in weight_col_vals]
            weight_col_vals_expand = sum(weight_col_vals_expand, [])  # flatten
            logging.info('weight_col_vals_expand is: {}'.format(weight_col_vals_expand))

            df_weight = pd.DataFrame(data={self.weight_col: weight_col_vals_expand})
            df = pd.merge(df, df_weight, on=self.weight_col, how='left')
            logging.info('shape of df after expand is: {}'.format(df.shape))
        return df[sel_cols]

    def prepare_dataset(self):
        df_log = self.parse_log()
        df_w2v, embed_cols, w2v_mean = self.parse_w2v()
        # df = pd.merge(df_log, df_w2v, on='creative_str', how='outer', indicator=True)
        df = pd.merge(df_log, df_w2v, on='creative_str', how='left')
        logging.info('shape of df is: {}'.format(df.shape))
        del df_log, df_w2v
        # df_fetch = df[df['_merge'] == 'both']
        # df_fetch.drop(columns=['_merge'], inplace=True)
        df_fetch = df[~df[embed_cols[0]].isna()]
        logging.info('shape of df_fetch is: {}'.format(df_fetch.shape))
        collect_log_content(self.log_dict, 'shape of df_fetch is: {}'.format(df_fetch.shape), self.log_path)
        # df_fill = df[df['_merge'] == 'left_only']
        # df_fill.drop(columns=['_merge'], inplace=True)
        df_fill = df[df[embed_cols[0]].isna()]
        logging.info('shape of df_fill is: {}'.format(df_fill.shape))
        collect_log_content(self.log_dict, 'shape of df_fill is: {}'.format(df_fill.shape), self.log_path)
        assert df_fill.shape[1] == len(w2v_mean) + 2, 'wrong dim between df_fill and w2v...'
        del df
        df_fill.loc[:, embed_cols] = w2v_mean
        # df_fill.loc[:, embed_cols] = list(2 * np.random.rand(self.embedding_size) - 1)
        df_new = pd.concat([df_fetch, df_fill], axis=0)
        logging.info('shape of df_new is: {}'.format(df_new.shape))
        del df_fetch, df_fill
        df_user = df_new.groupby('user_id')[embed_cols].mean().reset_index()
        df_user['user_id'] = df_user['user_id'].astype(int)
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        assert df_user.shape[0] == 900000 if self.task_type == 'train' else 1000000,'wrong num in user_id...'
        del df_new

        file_path = os.path.join(self.output_dir, 'dataset.csv')
        FileOperation.save_csv(df_user, file_path)
        return


if __name__ == '__main__':

    dataVecMaker_train = DataVecMaker(
        embedding_size=64,
        input_dir=os.path.join(config.DATA_DIR, 'base_0524'),
        weight_col=None,
        task_type='train',
        data_version='003'
    )
    dataVecMaker_train.prepare_dataset()

    dataVecMaker_test = DataVecMaker(
        embedding_size=64,
        input_dir=os.path.join(config.DATA_DIR, 'base_0524'),
        weight_col=None,
        task_type='test',
        data_version='003'
    )
    dataVecMaker_test.prepare_dataset()


