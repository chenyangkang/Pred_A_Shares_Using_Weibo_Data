import pickle
import datetime
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt


from utiles.weibo_scraper import get_weibo_profile
import utiles.weibo_scraper
from tqdm import tqdm
import pickle
from utiles.weibo_scraper import  get_formatted_weibo_tweets_by_name
from sentence_transformers import SentenceTransformer
import os
from  utiles.Ashare.Ashare import *


from utiles.Weibo_scrapper_utiles import get_influencer_list,get_data_by_usr_name
from utiles.Sentence_embedding_utiles import get_date
from utiles.get_stock_codes_utiles import get_stock_codes

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBRFClassifier, XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


stock_codes_dict = get_stock_codes()

from utiles.Agent_utiles import Wei_trade_model

## Predict today (Run every trading day morning ~8:30)
######### pred today
pred_res = []
for count,stock in enumerate(stock_codes_dict.items()):
    print(f'Predicting ---- {count+1}')
    ### building bot
    try:
        bot = pickle.load(open(f'./Trained_bot/{stock[0]}.pkl', 'rb'))

        ### pred today one model (one stock)
        # if count==0:
        #     bot.predict_today(force=True) #False
        # else:
        bot.predict_today(force=False) #False
            
        pred_res.append({
            'stock_id':bot.stock_id, 
            'stock_name':bot.stock_name,
            'total_influencer_used':bot.today_influencer_count,
            'pred_today_cls':bot.pred_today_cls,
            'pred_today_reg':bot.pred_today_reg,
            'f1':round(bot.cls_scores.loc['test_f1_score','mean'], 3),
            'f1_std':round(bot.cls_scores.loc['test_f1_score','std'], 3),
            'precision':round(bot.cls_scores.loc['test_precision','mean'], 3),
            'precision_std':round(bot.cls_scores.loc['test_precision','std'], 3),
            'recall':round(bot.cls_scores.loc['test_recall','mean'], 3),
            'recall_std':round(bot.cls_scores.loc['test_recall','std'], 3),
            'roc_auc':round(bot.cls_scores.loc['test_roc_auc','mean'], 3),
            'roc_auc_std':round(bot.cls_scores.loc['test_roc_auc','std'], 3),
            'r2_score':round(bot.reg_scores.loc['test_r2_score','mean'], 3),
            'r2_score_std':round(bot.reg_scores.loc['test_r2_score','std'], 3),
            'spearmanr':round(bot.reg_scores.loc['test_spearmanr','mean'], 3),
            'spearmanr_std':round(bot.reg_scores.loc['test_spearmanr','std'], 3)
        })

    except Exception as e:
        print(e)
        continue

    # print(bot.stock_id, bot.stock_name, bot.pred_today_cls, bot.pred_today_reg)
pred_res = pd.DataFrame(pred_res).sort_values(by=['f1','pred_today_cls','pred_today_reg'],ascending=[False,False,False])
pred_res['today'] = str(datetime.date.today())
pred_res['today_influencer_count'] = bot.today_influencer_count

    

## Construct your own index
from sklearn.preprocessing import MinMaxScaler,minmax_scale
pred_res['index1'] = 2/(1/minmax_scale(pred_res['pred_today_cls']) + 1/minmax_scale(pred_res['f1']))
pred_res['index2'] = 2/(1/minmax_scale(pred_res['pred_today_reg']) + 1/minmax_scale(pred_res['spearmanr']))
pred_res['index3'] = 2/(1/minmax_scale(pred_res['index1']) + 1/minmax_scale(pred_res['index2']))
pred_res['index4'] = 5/( 1/minmax_scale(pred_res['precision']) + 1/minmax_scale(pred_res['spearmanr'] + \
                            1/(pred_res['pred_today_cls']) + 1/(pred_res['pred_today_reg'])))

### save today result
pred_res.to_csv(f'./Data_{str(datetime.date.today())}/today_pred.csv')

## Send email every morning
#### send

top10 = pred_res[pred_res.pred_today_cls>0.5].sort_values(by='index4',ascending=False).head(10)
top10 = top10[['stock_id','stock_name','total_influencer_used','pred_today_cls','pred_today_reg','f1','precision','index4']].reset_index(drop=False)

from utiles.send_email_utiles import send_email
text = f'''
    Top 10 today:
    {top10}
'''

send_email(text)





