#!/usr/bin/env python
#coding=utf-8
# @file  : feaSign
# @time  : 5/24/2020 8:20 PM
# @author: shishishu

import os
import pandas as pd
from conf import config

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(config.LOG_DIR, 'feature.log'),
                    filemode='a')


class FeaCountSign:

    def __init__(self, df_click, df_user, stat_dir):
        self.df_click = df_click
        self.df_user = df_user
        self.stat_dir = stat_dir
        self.stat_cols = config.COL_STAT_CLIP_DICT.keys()

    def user_item_cnt(self, sel_col):
        df_sel = self.clean_record(sel_col)
        df_sel = df_sel[['user_id', sel_col]]

        user_stat_col = sel_col + '_cnt'
        df_group = df_sel.groupby('user_id')[sel_col].count().to_frame(user_stat_col).reset_index()
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_sel

        df_user_stat = FeaCountSign.fill_user_stat(self.df_user, df_group, user_stat_col)
        return df_user_stat

    def user_item_weighted_cnt(self, sel_col, weight_col):
        df_sel = self.clean_record(sel_col)
        df_sel = df_sel[['user_id', sel_col, weight_col]]

        user_stat_col = sel_col + '_weighted_cnt'
        df_group = df_sel.groupby('user_id')[weight_col].sum().to_frame(user_stat_col).reset_index()
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_sel

        df_user_stat = FeaCountSign.fill_user_stat(self.df_user, df_group, user_stat_col)
        return df_user_stat

    def user_item_unique_cnt(self, sel_col):
        df_sel = self.clean_record(sel_col)
        df_sel = df_sel[['user_id', sel_col]]

        user_stat_col = sel_col + '_unique_cnt'
        df_group = df_sel.groupby('user_id')[sel_col].apply(lambda x: len(set(x))).to_frame(user_stat_col).reset_index()
        logging.info('shape of df_group is: {}'.format(df_group.shape))
        del df_sel

        df_user_stat = FeaCountSign.fill_user_stat(self.df_user, df_group, user_stat_col)
        return df_user_stat

    def clean_record(self, sel_col):
        if sel_col in self.stat_cols:
            df_qual = FeaCountSign.ad_item_qualified(self.stat_dir, sel_col)
            df_sel = pd.merge(self.df_click, df_qual, on=sel_col)
            del df_qual
        else:
            df_sel = self.df_click.copy()
        logging.info('shape of df_sel is: {}'.format(df_sel.shape))
        return df_sel

    @staticmethod
    def ad_item_qualified(stat_dir, sel_col):
        logging.info('col selected is: {}'.format(sel_col))
        file_name = sel_col + '_age_stat.csv'
        file_path = os.path.join(stat_dir, file_name)
        norm_col = sel_col + '_norm'
        df_qual = pd.read_csv(file_path)[[norm_col]]
        df_qual.columns = [sel_col.strip('_norm')]  # diff cols
        logging.info('shape of df_qual is: {}'.format(df_qual.shape))
        df_qual = df_qual[df_qual[sel_col] >= 0]  # -1 as low-frequent
        logging.info('shape of df_qual after remove -1 is: {}'.format(df_qual.shape))
        return df_qual

    @staticmethod
    def fill_user_stat(df_user, df_group, user_stat_col):
        df_user_stat = pd.merge(df_user, df_group, on='user_id', how='left')
        df_user_stat[user_stat_col].fillna(0, inplace=True)
        logging.info('shape of df_user_stat is: {}'.format(df_user_stat.shape))
        FeaCountSign.check_user_missing(df_user, df_user_stat)
        return df_user_stat[['user_id', user_stat_col]]

    @staticmethod
    def check_user_missing(df_user, df_user_stat):
        assert df_user.shape[0] == df_user_stat.shape[0], 'missing rows of user...'