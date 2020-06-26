#!/usr/bin/env python
#coding=utf-8
# @file  : ad_stat
# @time  : 5/22/2020 10:13 PM
# @author: shishishu

import os
import pandas as pd
from conf import config
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


class AdStat:

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.output_dir = os.path.join(self.input_dir, 'ad_stat')
        FileOperation.safe_mkdir(self.output_dir)
        self.df = self.load_data()
        self.df_age = pd.DataFrame(data=list(zip(*[list(range(1, 11)), [0] * 10])), columns=['age', 'dummy_key'])
        self.df_gender = pd.DataFrame(data=list(zip(*[list(range(1, 3)), [0] * 2])), columns=['gender', 'dummy_key'])

    def load_data(self):
        user_path = os.path.join(config.DATA_DIR, 'raw', 'train', 'user.csv')
        ad_path = os.path.join(config.DATA_DIR, 'raw', 'train', 'ad.csv')
        click_path = os.path.join(self.input_dir, 'train', 'click_log_clean.csv')
        df_user = FileOperation.load_csv(user_path)
        logging.info('shape of df_user is: {}'.format(df_user.shape))
        df_ad = FileOperation.load_csv(ad_path)
        df_ad = df_ad.applymap(lambda x: int(x) if x != '\\N' else 0)  # convert to int
        logging.info('shape of df_ad is: {}'.format(df_ad.shape))
        df_click = FileOperation.load_csv(click_path)
        logging.info('shape of df_click is: {}'.format(df_click.shape))

        df = pd.merge(df_click, df_user, on='user_id')
        logging.info('shape of df after merge user info is: {}'.format(df.shape))
        del df_user
        df = pd.merge(df, df_ad, on='creative_id')
        logging.info('shape of df after merge ad info is: {}'.format(df.shape))
        del df_ad
        return df

    def pv_count_clip(self, ad_col):
        logging.info('current col is: {}'.format(ad_col))
        thresh_cnt = config.COL_STAT_CLIP_DICT[ad_col]
        logging.info('thresh cnt is: {}'.format(thresh_cnt))

        df_clip = self.df[ad_col].value_counts().to_frame().reset_index()
        record_col = ad_col + '_record'
        df_clip.columns = [ad_col, record_col]
        logging.info('shape of df_clip is: {}'.format(df_clip.shape))

        norm_col = ad_col + '_norm'
        df_clip[norm_col] = df_clip.apply(lambda row: row[ad_col] if row[record_col] >= thresh_cnt else -1, axis=1)
        logging.info('shape of df_clip after normalization is: {}'.format(df_clip.shape))
        return df_clip

    def general_stat(self, ad_col, user_attr='age'):
        logging.info('current user_attr is: {}'.format(user_attr))
        df_user_attr = self.df_age if user_attr == 'age' else self.df_gender
        user_attr_list = range(1, 11) if user_attr == 'age' else range(1, 3)

        norm_col = ad_col + '_norm'
        df_clip = self.pv_count_clip(ad_col)
        df_clip_exp = df_clip.drop_duplicates(subset=[norm_col])
        df_clip_exp = df_clip_exp[[norm_col]]
        df_clip_exp['dummy_key'] = 0
        df_attr = pd.merge(df_clip_exp, df_user_attr, on='dummy_key')
        df_attr.drop(columns=['dummy_key'], inplace=True)
        logging.info('shape of df_attr is: {}'.format(df_attr.shape))

        df_sel = pd.merge(self.df, df_clip, on=ad_col, how='left')
        logging.info('shape of df_sel is: {}'.format(df_sel.shape))

        df_attr_stat = df_sel.groupby([norm_col, user_attr])['user_id'].count().to_frame('user_record').reset_index()
        logging.info('shape of df_attr_stat is: {}'.format(df_attr_stat.shape))
        del df_sel

        df_attr_stat = pd.merge(df_attr, df_attr_stat, on=[norm_col, user_attr], how='left')
        df_attr_stat['user_record'].fillna(0, inplace=True)
        logging.info('shape of df_attr_stat after merge with df_attr is: {}'.format(df_attr_stat.shape))

        df_attr_stat_sum = df_attr_stat.groupby(norm_col)['user_record'].sum().to_frame('user_sum').reset_index()
        df_combine = pd.merge(df_attr_stat, df_attr_stat_sum, on=norm_col, how='left')
        logging.info('shape of df_combine is: {}'.format(df_combine.shape))

        rate_col = 'user_rate_' + user_attr
        df_combine[rate_col] = df_combine.apply(lambda row: round(row['user_record'] / row['user_sum'], 4), axis=1)
        for i in user_attr_list:
            df_combine[user_attr + '_rate_' + str(i)] = df_combine.apply(
                lambda row: row[rate_col] if row[user_attr] == i else 0, axis=1)
        df_stat = df_combine.drop(columns=['user_record', 'user_sum', user_attr, rate_col]).groupby(norm_col).max().reset_index()
        logging.info('shape of df_stat is: {}'.format(df_stat.shape))
        stat_path = os.path.join(self.output_dir, ad_col + '_' + user_attr + '_stat.csv')
        FileOperation.save_csv(df_stat, stat_path)
        return

    def stat_pipeline(self):
        ad_cols = config.COL_STAT_CLIP_DICT.keys()
        for ad_col in ad_cols:
            self.general_stat(ad_col, 'age')
            self.general_stat(ad_col, 'gender')
        return


if __name__ == '__main__':

    adStat = AdStat(
        input_dir=os.path.join(config.DATA_DIR, 'base_0524')
    )
    adStat.stat_pipeline()