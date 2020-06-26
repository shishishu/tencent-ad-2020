#!/usr/bin/env python
#coding=utf-8
# @file  : dataMaker
# @time  : 5/25/2020 12:03 AM
# @author: shishishu

import os
import sys
import pandas as pd
from conf import config
from lib.utils.fileOperation import FileOperation

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


class DataMaker:

    def __init__(self, input_dir, feat_dict, task_type='train', data_version='001'):
        self.task_type = task_type
        self.input_dir = input_dir
        self.output_dir = os.path.join(input_dir, self.task_type, data_version)
        FileOperation.safe_mkdir(self.output_dir)
        self.feat_dict = feat_dict
        FileOperation.save_json(self.feat_dict, os.path.join(self.output_dir, 'feat_dict.json'))

    def parse_feature(self, fea_sign, fea_col):
        file_name = fea_col + '_' + fea_sign + '.csv'
        file_path = os.path.join(self.input_dir, self.task_type, 'user_features', file_name)
        if not os.path.exists(file_path):
            logging.error('no file in the dest and exit: {}'.format(file_path))
            sys.exit(1)
        df = FileOperation.load_csv(file_path)
        df['user_id'] = df['user_id'].astype(int)
        logging.info('fea sign is: {}, fea col is: {}, df shape is: {}'.format(fea_sign, fea_col, df.shape))
        if fea_sign in ['age_dist', 'gender_dist', 'sparse_ratio']:
            df = df.iloc[:, :-1]
            logging.info('df shape after drop last col is: {}'.format(df.shape))
        return df

    def merge_feature(self, df_user, df_col):
        logging.info('shape of current df_concat is: {}'.format(df_user.shape))
        df_user = pd.merge(df_user, df_col, on='user_id', how='left')
        logging.info('shape of new df_concat is: {}'.format(df_user.shape))
        del df_col
        return df_user

    def prepare_dataset(self):
        click_path = os.path.join(self.input_dir, self.task_type, 'click_log_clean.csv')
        df_click = FileOperation.load_csv(click_path)
        logging.info('shape of df_click is: {}'.format(df_click.shape))
        df_user = df_click.drop_duplicates(subset=['user_id'])[['user_id']]
        df_user['user_id'] = df_user['user_id'].astype(int)
        logging.info('shape of df_user after dedup is: {}'.format(df_user.shape))
        del df_click

        for fea_sign, fea_cols in self.feat_dict.items():
            for fea_col in fea_cols:
                df_col = self.parse_feature(fea_sign, fea_col)
                df_user = self.merge_feature(df_user, df_col)
        logging.info('final shape of dataset is: {}'.format(df_user.shape))
        fea_cols_expand = list(df_user.columns)[1:]
        logging.info('dim of final dataset is: {}'.format(len(fea_cols_expand)))
        FileOperation.save_json(fea_cols_expand, os.path.join(self.output_dir, 'fea_cols_expand.json'))
        file_path = os.path.join(self.output_dir, 'dataset.csv')
        FileOperation.save_csv(df_user, file_path)
        return


if __name__ == '__main__':

    # feat_dict = {
    #     'cnt': ['creative_id', 'ad_id', 'advertiser_id', 'product_id', 'industry', 'product_category'],
    #     'age_dist': ['advertiser_id', 'product_id', 'industry', 'product_category'],
    #     'gender_dist': ['advertiser_id', 'product_id', 'industry', 'product_category'],
    #     'sparse_onehot': ['product_category'],
    #     'sparse_ratio': ['product_category']
    # }
    # dataMaker_train = DataMaker(
    #     input_dir=os.path.join(config.DATA_DIR, 'base_0524'),
    #     feat_dict=feat_dict,
    #     task_type='train',
    #     data_version='001'
    # )
    # dataMaker_train.prepare_dataset()
    #
    # dataMaker_test = DataMaker(
    #     input_dir=os.path.join(config.DATA_DIR, 'base_0524'),
    #     feat_dict=feat_dict,
    #     task_type='test',
    #     data_version='001'
    # )
    # dataMaker_test.prepare_dataset()
    pass
