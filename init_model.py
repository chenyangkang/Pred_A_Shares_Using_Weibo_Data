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

### initialize
if not os.path.exists('./Trained_bot'):
    os.mkdir('./Trained_bot')

for index,stock in tqdm(enumerate(stock_codes_dict.items()),total=len(stock_codes_dict)):
    ### building bot
    bot = Wei_trade_model(stock_id=stock[0],
                        stock_name=stock[1])

    ##### we don't need to re-download these historical tweets for a while (like a month)
    ### get historical data
    # if index==0:
    #     bot.get_historical_tweets(force=False)
    # else:
    bot.get_historical_tweets(force=False)

    ### feature engineering historical tweets
    try:
        bot.all_mean_features = pd.read_csv('mean_features.csv')
    except:
        # if index==0:
        #     bot.merging_historical_tweets(force_embedding=True) #False
        # else:
        bot.merging_historical_tweets(force_embedding=False) #False

    ### formatting input
    bot.formatting_input_data()

    ### training
    bot.build_classifier()
    bot.build_regressor()
    print('--------------------------------------------------------')
    print(f'#{index}', bot.stock_id, bot.stock_name,'\n', 
                        bot.cls_scores.to_markdown(),'\n', 
                        bot.reg_scores.to_markdown(),'\n')
    print('--------------------------------------------------------')

    pickle.dump(bot, open(f'./Trained_bot/{stock[0]}.pkl', 'wb'))


