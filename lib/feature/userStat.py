#!/usr/bin/env python
#coding=utf-8
# @file  : user_stat
# @time  : 5/22/2020 10:13 PM
# @author: shishishu

import os
import pandas as pd
from conf import config
from lib.feature.feaCountSign import FeaCountSign
from lib.feature.feaDistSign import FeaDistSign
from lib.feature.feaSparseSign import FeaSparseSign
from lib.utils.fileOperation import FileOperation

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'feature.log'),
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s [line:%(lineno)d] %(levelname)-8s %(message)s', datefmt='%Y %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class UserStat:

    def __init__(self, input_dir, task_type='train'):
        self.task_type = task_type
        self.input_dir = input_dir
        self.output_dir = os.path.join(self.input_dir, self.task_type, 'user_features')
        FileOperation.safe_mkdir(self.output_dir)
        self.df_click, self.df_user = self.load_data()

    def load_data(self):
        # ad_path = os.path.join(config.DATA_DIR, 'raw', self.task_type, 'ad.csv')
        click_path = os.path.join(self.input_dir, self.task_type, 'click_log_clean.csv')
        # df_ad = FileOperation.load_csv(ad_path)
        # df_ad = df_ad.applymap(lambda x: int(x) if x != '\\N' else 0)  # convert to int
        # logging.info('shape of df_ad is: {}'.format(df_ad.shape))
        df_click = FileOperation.load_csv(click_path)
        logging.info('shape of df_click is: {}'.format(df_click.shape))

        # df_click = pd.merge(df_click, df_ad, on='creative_id')
        # logging.info('shape of df_click after merge ad info is: {}'.format(df_click.shape))
        # del df_ad

        df_user = df_click.drop_duplicates(subset=['user_id'])[['user_id']]
        df_user['user_id'] = df_user['user_id'].astype(int)
        logging.info('shape of df_user after dedup is: {}'.format(df_user.shape))
        return df_click, df_user

    def gene_count_features(self):
        feaCountSign = FeaCountSign(
            df_click=self.df_click,
            df_user=self.df_user,
            stat_dir=os.path.join(self.input_dir, 'ad_stat')
        )
        for sel_col in config.COUNT_COLS:
            logging.info('current col in progress is: {}'.format(sel_col))
            if sel_col == 'creative_id':
                df_user_stat = feaCountSign.user_item_weighted_cnt(sel_col, weight_col='click_times')
            else:
                df_user_stat = feaCountSign.user_item_unique_cnt(sel_col)
            file_path = os.path.join(self.output_dir, sel_col + '_cnt.csv')
            FileOperation.save_csv(df_user_stat, file_path)
        return

    def gene_dist_features(self):
        feaDistSign = FeaDistSign(
            df_click=self.df_click,
            df_user=self.df_user,
            stat_dir=os.path.join(self.input_dir, 'ad_stat')
        )
        for sel_col in config.DIST_COLS:
            logging.info('current col in progress is: {}'.format(sel_col))
            for user_attr in ['age', 'gender']:
                logging.info('current user_attr in progress is: {}'.format(user_attr))
                df_group = feaDistSign.user_item_dist_weighted_mean(sel_col, weight_col='click_times', user_attr=user_attr)
                file_path = os.path.join(self.output_dir, sel_col + '_' + user_attr + '_dist.csv')
                FileOperation.save_csv(df_group, file_path)
        return

    def gene_sparse_features(self):
        feaSparseSign = FeaSparseSign(
            df_click=self.df_click,
            df_user=self.df_user,
            stat_dir=os.path.join(self.input_dir, 'ad_stat')
        )
        for sel_col in config.SPARSE_COLS:
            logging.info('current col in progress is: {}'.format(sel_col))
            df_onehot = feaSparseSign.user_item_one_hot(sel_col)
            file_path = os.path.join(self.output_dir, sel_col + '_sparse_onehot.csv')
            FileOperation.save_csv(df_onehot, file_path)
            # df_ratio = feaSparseSign.user_item_ratio(sel_col)
            # file_path = os.path.join(self.output_dir, sel_col + '_sparse_ratio.csv')
            # FileOperation.save_csv(df_ratio, file_path)
        return


if __name__ == '__main__':

    logging.info('Prepare train part....')
    userStat_train = UserStat(
        input_dir=os.path.join(config.DATA_DIR, 'base_0611'),
        task_type='train'
    )
    # logging.info('start gene cnt features...')
    # userStat_train.gene_count_features()
    # logging.info('start gene dist features...')
    # userStat_train.gene_dist_features()
    logging.info('start gene sparse features...')
    userStat_train.gene_sparse_features()

    logging.info('Prepare test part....')
    userStat_test = UserStat(
        input_dir=os.path.join(config.DATA_DIR, 'base_0611'),
        task_type='test'
    )
    # logging.info('start gene cnt features...')
    # userStat_test.gene_count_features()
    # logging.info('start gene dist features...')
    # userStat_test.gene_dist_features()
    logging.info('start gene sparse features...')
    userStat_test.gene_sparse_features()