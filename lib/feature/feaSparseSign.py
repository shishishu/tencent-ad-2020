#!/usr/bin/env python
#coding=utf-8
# @file  : feaSparseSign
# @time  : 5/24/2020 10:27 PM
# @author: shishishu

import os
import pandas as pd
import sys
from conf import config
from lib.feature.feaCountSign import FeaCountSign
from collections import Counter

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'feature.log'),
                    filemode='a')


class FeaSparseSign:

    def __init__(self, df_click, df_user, stat_dir):
        self.df_click = df_click
        self.df_user = df_user
        self.stat_dir = stat_dir
        self.stat_cols = config.SPARSE_COLS

    def user_item_one_hot(self, sel_col):
        df_sel, qual_vals = self.clean_record(sel_col)
        df_sel = df_sel[['user_id', sel_col]]

        user_stat_cols = [sel_col + '_onehot_' + str(qual_val) for qual_val in qual_vals]
        list_col = sel_col + '_list'
        df_group = df_sel.groupby('user_id')[sel_col].apply(lambda x: list(x)).to_frame(list_col).reset_index()
        df_group[user_stat_cols] = df_group.apply(lambda row: FeaSparseSign.list2onehot(row[list_col], qual_vals), axis=1, result_type='expand')
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_sel
        return FeaSparseSign.fill_user_stat(self.df_user, df_group, user_stat_cols)

    def user_item_ratio(self, sel_col):
        df_sel, qual_vals = self.clean_record(sel_col)
        df_sel = df_sel[['user_id', sel_col]]

        user_stat_cols = [sel_col + '_ratio_' + str(qual_val) for qual_val in qual_vals]
        list_col = sel_col + '_list'
        df_group = df_sel.groupby('user_id')[sel_col].apply(lambda x: list(x)).to_frame(list_col).reset_index()
        df_group[user_stat_cols] = df_group.apply(lambda row: FeaSparseSign.list2ratio(row[list_col], qual_vals), axis=1, result_type='expand')
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_sel
        return FeaSparseSign.fill_user_stat(self.df_user, df_group, user_stat_cols)

    def clean_record(self, sel_col):
        if sel_col not in self.stat_cols:
            logging.error('sel_col {} is out of stat field and exit...'.format(sel_col))
            sys.exit(1)
        df_qual = FeaCountSign.ad_item_qualified(self.stat_dir, sel_col)
        qual_vals = sorted(list(set(df_qual[sel_col])), reverse=False)
        logging.info('qual_vals are: {}'.format(qual_vals))
        df_sel = pd.merge(self.df_click, df_qual, on=sel_col)
        logging.info('shape of df_sel is: {}'.format(df_sel.shape))
        del df_qual
        return df_sel, qual_vals

    @staticmethod
    def list2onehot(list_val, qual_vals):
        input_dict = Counter(list_val)
        onehot_list = [1 if qual_val in input_dict else 0 for qual_val in qual_vals]
        return onehot_list

    @staticmethod
    def list2ratio(list_val, qual_vals):
        input_dict = Counter(list_val)
        ratio_list = [input_dict.get(qual_val, 0) / sum(list(input_dict.values())) for qual_val in qual_vals]
        return ratio_list

    @staticmethod
    def fill_user_stat(df_user, df_group, user_stat_cols):
        df_user_stat = pd.merge(df_user, df_group, on='user_id', how='outer', indicator=True)
        df_fetch = df_user_stat[df_user_stat['_merge'] == 'both']
        logging.info('shape of df_fetch is: {}'.format(df_fetch.shape))
        df_fill = df_user_stat[df_user_stat['_merge'] == 'left_only']
        logging.info('shape of df_fill is: {}'.format(df_fill.shape))
        del df_user_stat
        df_fill[user_stat_cols] = list([0] * len(user_stat_cols))
        df_new = pd.concat([df_fetch, df_fill], axis=0)
        new_cols = ['user_id']
        new_cols.extend(user_stat_cols)
        df_new = df_new[new_cols]
        logging.info('shape of df_new is: {}'.format(df_new.shape))
        del df_fetch, df_fill
        FeaCountSign.check_user_missing(df_user, df_new)
        return df_new