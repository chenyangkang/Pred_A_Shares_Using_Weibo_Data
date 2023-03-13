import pickle
import datetime
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt


from weibo_scraper import get_weibo_profile
import weibo_scraper
# weibo_profile = get_weibo_profile(name='天津股侠',)
from tqdm import tqdm
import pickle
from weibo_scraper import  get_formatted_weibo_tweets_by_name

import os




def get_influencer_list():
    name_list = ['腾腾爸','股市狼头哥','股市-白泽','胡墨雨说股市','短线小妖',
             '赢庄光头哥','老股民的老研究','A股长虹A','八月俊杰','股市刀锋','东方股侠5178','闲人说股',
             '钱局长本人','苏州陶小姐','妖哥财经','每天赚一个ETH','峰哥财经','股市辨别','深圳张逸轩',
             'o小怂包','强强记事','庚白星君','光远看经济','广西小wu','墨說財經','何天恩','黑夜之睛滚雪球',
             '武汉潘唯杰','阿K寻龙','喜欢玩基金的小瑜哥','湾区大佬','付鹏的财经世界','涨涨看市','曦阳思慕','老龍博弈',
             '煮酒论财经','有理想的蓝大','索哥_','历史的进城','真当没想到','别梦依稀笑逝川','基大队长','刘彦斌','短线股票池王哥',
             'ViVKin','股东十八掌','麻辣新鲜','泽丰有好股','股道_智慧传奇','大雄作手','广东三阿哥','梦若神机0','喝红茶的三叔','琅琊榜首张大仙',
             '财联社APP','杨易君黄金与金融投资','馨月','马哥在股海','短线小仙女','概率财经','短线掘金老张','邹健论道','Kainin20221027','财经平头哥',
             '三不投资','一哥股市论金','A股任盈盈','锡安财智何','老乐微妙','老鬼哥','股海伏笔','姬永锋','A股独立连','上海猎龙者','股侠骑牛',
             '一休哥没有V','指汇盈',

             '神嘛事儿','洪榕','见识财经','福布斯中文网','环球市场播报','哈利宝舵主','厦门短线侠客','一股红运','杰神Leo','擒牛师叶汶昌',
            '海豚佩佩','天涯尼丹小','月相数据员','彦桢宏观','张学林','星话大白',
             
            '金融人事','杨东辉','特特理财','蓝海经济观察','价值投资日志','张起论市','财经豪哥掘金','風雨顺德人','Degg_GlobalMacroFin',

             '今日灼见','牛犇财经','第一财经广播','雪球','第一财经日报','央视财经','财经网','第一财经','新浪财经','云财经',
             '北京商报']
    name_list = list(set(name_list))
    return name_list




def get_data_by_usr_name(user_name,pages=None):
    result_iterator = get_formatted_weibo_tweets_by_name(name=user_name, pages=pages)
    data=[]
    for user_meta in result_iterator:
        if user_meta is not None:
            for tweetMeta in user_meta.cards_node:
                a = vars(tweetMeta)
                res = {
                    'user_name':user_name,
                    'created_at':a['card_node']['mblog']['created_at'],
                    'text':a['card_node']['mblog']['text'],
                    # 'textLength':a['card_node']['mblog']['textLength'],
                    'reposts_count':a['card_node']['mblog']['reposts_count'],
                    'comments_count':a['card_node']['mblog']['comments_count'],
                    'attitudes_count':a['card_node']['mblog']['attitudes_count']  
                }
                data.append(res)
    return data


