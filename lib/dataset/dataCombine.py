#!/usr/bin/env python
#coding=utf-8
# @file  : dataCombine
# @time  : 6/18/2020 11:33 PM
# @author: shishishu

import os
import pickle
import sys
import numpy as np
import pandas as pd
import random
from conf import config
from lib.utils.fileOperation import FileOperation
from lib.dataset.createSeq import CreateSeq
from lib.dataset.dataUtils import DataUtils
from itertools import product
from joblib import Parallel, delayed

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'dataset.log'),
                    filemode='a')


class DataCombine:

    def __init__(self, input_dir, src_dir, combine_strategy='cross', model_type='rnn21', data_version='001', pred_gender_model='rnn2_gender_0010'):
        self.input_dir = input_dir
        self.src_dir = src_dir
        self.combine_strategy = combine_strategy
        self.output_dir = os.path.join(config.DATA_DIR, model_type, data_version)
        FileOperation.safe_mkdir(self.output_dir)
        self.pred_gender_path = os.path.join(config.MODEL_DIR, pred_gender_model, 'gender_export.csv')
        self.n_jobs = 10

    def gene_train_data(self):
        product_category_path = os.path.join(self.input_dir, 'train', 'user_features', 'product_category_sparse_onehot.csv')
        df_cate = FileOperation.load_csv(product_category_path)
        logging.info('shape of df_cate is: {}'.format(df_cate.shape))
        user_path = os.path.join(config.DATA_DIR, 'raw', 'train', 'user.csv')
        df_user = FileOperation.load_csv(user_path)
        df_user = df_user[['user_id', 'gender']]
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        return df_cate, df_user

    def gene_test_data(self):
        product_category_path = os.path.join(self.input_dir, 'test', 'user_features', 'product_category_sparse_onehot.csv')
        df_cate = FileOperation.load_csv(product_category_path)
        logging.info('shape of df_cate is: {}'.format(df_cate.shape))
        df_user = FileOperation.load_csv(self.pred_gender_path)
        df_user.columns = ['user_id', 'gender']
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        return df_cate, df_user

    def gene_cross_data(self, df_cate, df_user):
        df_user['gender_onehot_0'], df_user['gender_onehot_1'] = zip(*df_user['gender'].apply(lambda x: (1, 0) if x == 1 else (0, 1)))
        df_user = df_user[['user_id', 'gender_onehot_0', 'gender_onehot_1']]
        cate_cols = list(df_cate.columns)[1:]
        num_cate = len(cate_cols)
        gender_cols = list(df_user.columns)[1:]
        num_gender = len(gender_cols)
        cross_cols = ['cross_idx_' + str(idx) for idx in range(num_cate * num_gender)]
        df = pd.merge(df_user, df_cate, on='user_id', how='left')
        logging.info('shape of df after merge is: {}'.format(df.shape))
        del df_cate, df_user
        df_grouped = np.split(df, self.n_jobs, axis=0)
        def apply_cross(df1):
            df1[cross_cols] = df1.apply(lambda row: list(map(lambda x: x[0] * x[1], product(row[cate_cols], row[gender_cols]))), axis=1, result_type='expand')
            return df1
        df = DataUtils.apply_parallel(df_grouped, apply_cross, self.n_jobs)
        cols = ['user_id']
        cols.extend(cross_cols)
        df_res = df[cols]
        logging.info('shape of df after cross is: {}'.format(df_res.shape))
        del df
        return df_res

    def gene_concat_data(self, df_cate, df_user):
        df_user['gender_onehot'] = df_user['gender'].apply(lambda x: int(x - 1))
        df_user = pd.merge(df_user[['user_id', 'gender_onehot']], df_cate, on='user_id', how='left')
        logging.info('shape of df_user after concat is: {}'.format(df_user.shape))
        del df_cate
        return df_user

    def gene_data_map(self):
        file_names = ['tr_age.txt', 'va_1_age.txt', 'va_2_age.txt', 'te_age.txt']
        old_paths = list(map(lambda x: os.path.join(self.src_dir, x), file_names))
        new_paths = list(map(lambda x: os.path.join(self.output_dir, x), file_names))
        df_list = [FileOperation.load_csv(file_path, sep=' ', has_header=False) for file_path in old_paths]
        old_dim = int(df_list[0].shape[1] - 1)
        old_cols = ['user_id']
        old_dim_cols = ['old_idx_' + str(idx) for idx in range(old_dim)]
        old_cols.extend(old_dim_cols)
        df_tr_cate, df_tr_user = self.gene_train_data()
        df_te_cate, df_te_user = self.gene_test_data()
        df_res = None
        if self.combine_strategy == 'cross':
            df_tr = self.gene_cross_data(df_tr_cate, df_tr_user)
            df_te = self.gene_cross_data(df_te_cate, df_te_user)
        elif self.combine_strategy == 'concat':
            df_tr = self.gene_concat_data(df_tr_cate, df_tr_user)
            df_te = self.gene_concat_data(df_te_cate, df_te_user)
        elif self.combine_strategy == 'category_only':
            df_tr = df_tr_cate.copy()
            df_te = df_te_cate.copy()
        elif self.combine_strategy == 'gender_only':
            df_tr = df_tr_user.copy()
            df_te = df_te_user.copy()
        else:
            raise ValueError('current strategy is: {} and it is not allowed...'.format(self.combine_strategy))
        df_res = pd.concat([df_tr, df_te], axis=0)
        del df_tr, df_te
        logging.info('shape of new df_res is: {}'.format(df_res.shape))

        for (old_df, new_path) in zip(df_list, new_paths):
            old_df.columns = old_cols
            df = pd.merge(old_df, df_res, on='user_id', how='left')
            logging.info('current path is: {}'.format(new_path))
            logging.info('shape of df is: {}'.format(df.shape))
            FileOperation.save_csv(df, new_path, sep=' ', has_header=False)
            del df
        return


if __name__ == '__main__':

    dataCombine = DataCombine(
        input_dir=os.path.join(config.DATA_DIR, 'base_0611'),
        src_dir=os.path.join(config.DATA_DIR, 'rnn2', '001'),
        combine_strategy='cross',
        model_type='rnn2cross',
        data_version='001',
        pred_gender_model='rnn2_gender_0010'
    )
    dataCombine.gene_data_map()





