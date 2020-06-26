#!/usr/bin/env python
#coding=utf-8
# @file  : feaSign
# @time  : 5/24/2020 8:20 PM
# @author: shishishu

import os
import pandas as pd
import sys
from conf import config
from lib.feature.feaCountSign import FeaCountSign

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'feature.log'),
                    filemode='a')


class FeaDistSign:

    def __init__(self, df_click, df_user, stat_dir):
        self.df_click = df_click
        self.df_user = df_user
        self.stat_dir = stat_dir
        self.stat_cols = config.COL_STAT_CLIP_DICT.keys()

    def user_item_dist_mean(self, sel_col, user_attr='age'):
        df_new, user_attr_cols = self.clean_record(sel_col, user_attr)
        new_cols = ['user_id']
        new_cols.extend(user_attr_cols)
        df_new = df_new[new_cols]
        df_group = df_new.groupby('user_id')[user_attr_cols].mean().reset_index()
        df_group['user_id'] = df_group['user_id'].astype(int)
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_new

        FeaCountSign.check_user_missing(self.df_user, df_group)
        return df_group

    def user_item_dist_weighted_mean(self, sel_col, weight_col, user_attr='age'):
        df_new, user_attr_cols = self.clean_record(sel_col, user_attr)
        new_cols = ['user_id', weight_col]
        new_cols.extend(user_attr_cols)
        df_new = df_new[new_cols]

        weight_col_vals = sorted(list(set(df_new[weight_col])))
        weight_col_vals = list(map(int, weight_col_vals))
        weight_col_vals_expand = [[i] * i for i in weight_col_vals]
        weight_col_vals_expand = sum(weight_col_vals_expand, [])  # flatten
        logging.info('weight_col_vals_expand is: {}'.format(weight_col_vals_expand))

        df_weight = pd.DataFrame(data={weight_col: weight_col_vals_expand})
        df_expand = pd.merge(df_new, df_weight, on=weight_col, how='left')
        logging.info('shape of df_expand is: {}'.format(df_expand.shape))
        del df_new

        df_group = df_expand.groupby('user_id')[user_attr_cols].mean().reset_index()
        df_group['user_id'] = df_group['user_id'].astype(int)
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_expand
        return df_group

    def clean_record(self, sel_col, user_attr):
        if sel_col not in self.stat_cols:
            logging.error('sel_col {} is out of stat field and exit...'.format(sel_col))
            sys.exit(1)
        df_qual = FeaDistSign.ad_item_qualified(self.stat_dir, sel_col, user_attr)
        df_qual_cols = list(df_qual.columns)
        header_col = df_qual_cols[0]
        user_attr_cols = df_qual_cols[1:]
        user_attr_cols2 = [sel_col + '_' + user_attr_col for user_attr_col in user_attr_cols]
        df_qual_cols2 = [header_col]
        df_qual_cols2.extend(user_attr_cols2)
        df_qual.columns = df_qual_cols2

        df_sel = pd.merge(self.df_click, df_qual, on=sel_col, how='outer', indicator=True)
        df_fetch = df_sel[df_sel['_merge'] == 'both']
        logging.info('shape of df_fetch is: {}'.format(df_fetch.shape))
        df_fill = df_sel[df_sel['_merge'] == 'left_only']
        logging.info('shape of df_fill is: {}'.format(df_fill.shape))
        del df_sel

        na_fill = list(df_qual[df_qual[sel_col] == -1][user_attr_cols2].values[0])
        logging.info('na fill is: {}'.format(na_fill))
        del df_qual

        df_fill[user_attr_cols2] = df_fill.apply(lambda row: na_fill, axis=1, result_type='expand')
        logging.info('shape of df_fill after na fill is: {}'.format(df_fill.shape))

        df_new = pd.concat([df_fetch, df_fill], axis=0)
        df_new.drop(columns=['_merge'])
        logging.info('shape of df_new is: {}'.format(df_new.shape))
        del df_fetch
        del df_fill
        return df_new, user_attr_cols2

    @staticmethod
    def ad_item_qualified(stat_dir, sel_col, user_attr='age'):
        logging.info('col selected is: {}'.format(sel_col))
        file_name = sel_col + '_' + user_attr + '_stat.csv'
        file_path = os.path.join(stat_dir, file_name)
        df_qual = pd.read_csv(file_path)
        cols = list(df_qual.columns)
        cols[0] = sel_col.strip('_norm')
        df_qual.columns = cols  # diff cols
        logging.info('shape of df_qual is: {}'.format(df_qual.shape))
        return df_qual