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
from sentence_transformers import SentenceTransformer
import os
from  Ashare.Ashare import *


from Weibo_scrapper_utiles import get_influencer_list,get_data_by_usr_name
from Sentence_embedding_utiles import get_date

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBRFClassifier, XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, r2_score, mean_squared_error

from scipy.stats import spearmanr

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


class Wei_trade_model():
    def __init__(self, stock_id, stock_name, classifier=RandomForestClassifier(random_state=42,oob_score=True), 
                        regressor=RandomForestRegressor(random_state=42,oob_score=True), 
                        influencer_list=get_influencer_list()):
        self.influencer_list = influencer_list
        self.classifier = classifier
        self.regressor = regressor
        self.encoder = SentenceTransformer('uer/sbert-base-chinese-nli')
        self.stock_id = str(stock_id)
        self.stock_name = str(stock_name)

    def get_historical_tweets(self, force=False):
        if not os.path.exists('./Scrapped_weibo'):
            os.mkdir('./Scrapped_weibo')
        for influencer in tqdm(self.influencer_list):
            if os.path.exists(f'./Scrapped_weibo/{influencer}_historical.pkl'):
                if force:
                    # print(f'File exists but still downloading: ./Scrapped_weibo/{influencer}_historical.pkl')
                    data = get_data_by_usr_name(influencer,pages=None)
                    pickle.dump(data, open(f'./Scrapped_weibo/{influencer}_historical.pkl','wb'))
                else:
                    # print(f'File exists, SKIP (you can set force=True): ./Scrapped_weibo/{influencer}_historical.pkl')
                    continue
            
            try:
                data = get_data_by_usr_name(influencer,pages=None)
                pickle.dump(data, open(f'./Scrapped_weibo/{influencer}_historical.pkl','wb'))
            except Exception as e:
                print(influencer,e)
                continue


    def embedding_historical_tweets(self, force=False):
        if not os.path.exists('./Embedded_tweets/'):
            os.mkdir('./Embedded_tweets/')
        data_list = os.listdir('./Scrapped_weibo/')
        #### strategy one: take mean of all influencers
        all_ = []
        for index,data_name in tqdm(enumerate(data_list),total=len(data_list)):
            data_name_base = data_name.split('_historical')[0]
            try:
                fex = os.path.exists(f'./Embedded_tweets/{data_name_base}_embedded.pkl')
                if (fex and force) or (not fex):
                    if (fex and force):
                        # print(f'File exists but still downloading: ./Embedded_tweets/{data_name_base}_embedded.pkl')
                        pass

                    data = pickle.load(open(f'./Scrapped_weibo/{data_name}','rb'))
                    name = data[0]['user_name']
                    dates =[ get_date(i['created_at']) for i in data]
                    text = [i['text'] for i in data]
                    res = self.encoder.encode(text)
                    responses = [i['reposts_count'] for i in data]
                    comments_count = [i['comments_count'] for i in data]
                    attitudes_count = [i['attitudes_count'] for i in data]
                    features = pd.DataFrame({
                        'date':dates,
                        'responses':responses,
                        'comments_count':comments_count,
                        'attitudes_count':attitudes_count,
                        'influencer':[name]*len(dates)
                    })

                    for i in range(res.shape[1]):
                        features[f's{i}']=res[:,i]

                    pickle.dump(features, open(f'./Embedded_tweets/{data_name_base}_embedded.pkl','wb'))
                    all_.append(features)

                else:
                    # print(f'File exists, SKIP (you can set force=True): ./Embedded_tweets/{data_name_base}_embedded.pkl')
                    features = pickle.load(open(f'./Embedded_tweets/{data_name_base}_embedded.pkl','rb'))
                    all_.append(features)

            except Exception as e:
                print(data_name_base,e)
                continue

        all_df = pd.concat(all_)
        all_df.to_csv('features.csv',index=False)
        print('Embedded features saved: features.csv')
        return all_df


    def calc_mean_features(self, all_df):
        
        all_df['time'] = all_df.date.dt.time
        all_df['day'] = all_df.date.dt.date
        all_df['year'] = all_df.date.dt.year
        all_df['DOW'] = all_df.date.dt.day_of_week
        all_df = all_df[all_df.year>=2018]
        unique_date = sorted(all_df.date.dt.date.unique())

        mean_res = {}
        for date in unique_date:
            DOW = pd.DataFrame([pd.to_datetime(date)])[0].dt.day_of_week.values[0]
            if DOW==1:
                sub=all_df[(all_df.date>=datetime.datetime.strptime(
                    str(date-datetime.timedelta(days=2))+'-15:00:00', '%Y-%m-%d-%H:%M:%S'
                    )) & (all_df.date<=datetime.datetime.strptime(
                    str(date)+'-9:30:00', '%Y-%m-%d-%H:%M:%S'
                        ))
                        ]
            else:
                sub=all_df[(all_df.date>=datetime.datetime.strptime(
                str(date-datetime.timedelta(days=1))+'-15:00:00', '%Y-%m-%d-%H:%M:%S'
                )) & (all_df.date<=datetime.datetime.strptime(
                str(date)+'-9:30:00', '%Y-%m-%d-%H:%M:%S'
                    ))
                    ]

            if len(sub)<10: ### must be more than 10 tweets
                continue
            if len(sub.influencer.unique())<10: ### must have at least 5 influencer talk about something
                continue

            mean_features = sub.groupby('influencer').mean()
            ind_count = mean_features.shape[0]
            mean_mean_features = mean_features[[f's{i}' for i in range(0,768)]].mean(axis=0)

            mean_mean_features={
                **{'ind_count':ind_count},
                **dict(mean_mean_features)

            }

            mean_res[date] = mean_mean_features
        all_mean_features = pd.DataFrame(mean_res).T.reset_index(drop=False).rename(columns={'index':'date'})
        return all_mean_features

    def merging_historical_tweets(self, force_embedding=False):
        all_df = self.embedding_historical_tweets(force=force_embedding)
        all_mean_features = self.calc_mean_features(all_df)
        all_mean_features.to_csv('mean_features.csv',index=False)
        self.all_mean_features = all_mean_features
        print('Mean embedded features saved: mean_features.csv, also in model.all_mean_features')


    def formatting_input_data(self):
        try:
            features = self.all_mean_features
        except Exception as e:
            print(e)
            print('Trying to read from mean_features.csv')
        
            try:
                features = pd.read_csv('mean_features.csv')
            except Exception as e:
                print(e)
                raise

        features['date'] = pd.to_datetime(features['date'])
        features = features.set_index('date')

        
        df=get_price(str(self.stock_id),frequency='1d',count=365*5)
        df['change'] = df['close'] - df['open']
        df['change_frac'] = (df['close'] - df['open'])/df['open']
        df['high_frac'] = (df['high'] - df['open'])/df['open']
        df['low_frac'] = (df['low'] - df['open'])/df['open']
        df['change_bi'] = np.where(df['change']>0,1,0)


        data_cla = pd.concat([df['change_bi'], features],axis=1)
        data_reg = pd.concat([df['change_frac'], features],axis=1)
        data_cla = data_cla.dropna()
        data_reg = data_reg.dropna()

        self.data_cla = data_cla
        self.data_reg = data_reg
        print('Input data formatted')

    def build_classifier(self):

        ### classifier
        X = self.data_cla.iloc[:,1:]
        y = self.data_cla.iloc[:,0]

        ### model
        model = self.classifier

        #### store metrics
        scorers = {
            'precision': precision_score,
            'recall':recall_score,
            'roc_auc':roc_auc_score,
            'f1_score':f1_score
            }
        scores = {
            'precision':[],
            'recall':[],
            'roc_auc':[],
            'f1_score':[]
        }

        ### shuffle
        X_new = X.copy().reset_index(drop=True)
        y_new = pd.DataFrame(np.array(y.copy())).reset_index(drop=True)
        data_new = pd.concat([X_new,y_new],axis=1).sample(frac=1,replace=False)
        X_new,y_new = data_new.iloc[:,:-1], pd.DataFrame(data_new.iloc[:,-1])

        ### CV
        each_size = int(X_new.shape[0]/10)
        for i in range(10):
            if not i==9:
                test_X = X_new.iloc[i*each_size:(i+1)*each_size,:]
                test_y = y_new.iloc[i*each_size:(i+1)*each_size,:]
            else:
                test_X = X_new.iloc[i*each_size:,:]
                test_y = y_new.iloc[i*each_size:,:]
            train_X = X_new.loc[[k for k in list(X_new.index) if not k in list(test_X.index)],:]
            train_y = y_new.loc[[k for k in list(y_new.index) if not k in list(test_y.index)],:]
            train_sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
            model.fit(train_X,train_y,sample_weight = train_sample_weights)
            pred = model.predict(test_X)
            for scorer in scorers.keys():
                scores[scorer].append(scorers[scorer](test_y, pred))

        mean_scores = {f'test_{i}':{'mean':np.mean(scores[i]), 'std':np.std(scores[i])} for i in scores.keys()}
        self.cls_scores = pd.DataFrame(mean_scores).T
        print('Classifier stored in model.classifier; scores stored in model.cls_scores')

        #### final fit
        train_sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        model = model.fit(X,y,sample_weight=train_sample_weights)
        self.classifier=model ### put it back


    def build_regressor(self):

        ### regressor
        X = self.data_reg.iloc[:,1:]
        y = self.data_reg.iloc[:,0]

        ### model
        model = self.regressor

        #### store scores
        def my_spearmanr(array1, array2):
            return spearmanr(array1,array2)[0]

        scorer = {
            'r2_score': make_scorer(r2_score),
            'mean_squared_error':make_scorer(mean_squared_error),
            'spearmanr':make_scorer(my_spearmanr)
            }
        
        scores = cross_validate(self.regressor, X, y, cv=10, scoring=scorer) #Cross-validate with a decision tree classifier
        scores = {i:{'mean':np.mean(scores[i]), 'std':np.std(scores[i])} for i in scores.keys() if 'test' in i}
        self.reg_scores = pd.DataFrame(scores).T
        print('Regressor stored in model.regressor; scores stored in model.reg_scores')

        ### final fit
        model = model.fit(X,y)
        self.regressor=model ### put it back



    def get_and_embed_today_data(self,force=False):
        datetime_today = datetime.date.today()
        if not os.path.exists(f'./Data_{str(datetime_today)}'):
            os.mkdir(f'./Data_{str(datetime_today)}')

        all_=[]
        for influencer in tqdm(self.influencer_list):
            try:
                fex = os.path.exists(f'./Data_{str(datetime_today)}/{influencer}_embedded.pkl')
                if (fex and force) or (not fex):
                    if (fex and force):
                        # print(f'Today data {influencer} exist, still downloading')
                        pass
                    
                    data = get_data_by_usr_name(influencer,pages=5) #### 5 pages should contain 1 day volumn tweets
                    name = data[0]['user_name']
                    dates =[ get_date(i['created_at']) for i in data]
                    text = [i['text'] for i in data]
                    res = self.encoder.encode(text)
                    responses = [i['reposts_count'] for i in data]
                    comments_count = [i['comments_count'] for i in data]
                    attitudes_count = [i['attitudes_count'] for i in data]
                    features = pd.DataFrame({
                        'date':dates,
                        'responses':responses,
                        'comments_count':comments_count,
                        'attitudes_count':attitudes_count,
                        'influencer':[name]*len(dates)
                    })

                    for i in range(res.shape[1]):
                        features[f's{i}']=res[:,i]
                    
                    pickle.dump(features, open(f'./Data_{str(datetime_today)}/{influencer}_embedded.pkl','wb'))
                    all_.append(features)
                else:
                    # print(f'Today data {influencer} exist. Reading from files.')
                    features = pickle.load(open(f'./Data_{str(datetime_today)}/{influencer}_embedded.pkl','rb'))
                    all_.append(features)

            except Exception as e:
                print(e)
                continue

        all_df = pd.concat(all_)
        all_df = all_df[all_df['date'].dt.date==datetime_today] ### only include today features
        self.today_influencer_count = len(all_df['influencer'].unique())
        return all_df

    def merging_today_tweets(self, force=False):
        all_df_today = self.get_and_embed_today_data(force=force)
        all_mean_features = self.calc_mean_features(all_df_today)
        all_mean_features.to_csv('mean_features_today.csv',index=False)
        all_mean_features['date'] = pd.to_datetime(all_mean_features['date'])
        all_mean_features = all_mean_features.set_index('date')
        self.all_mean_features_today = all_mean_features
        
        print('Today input data formatted')
        print('Today feature processing done')

    def predict_today(self, force=False):
        self.merging_today_tweets(force=force)
        pred_cls = self.classifier.predict_proba(self.all_mean_features_today)[:,1]
        pred_reg = self.regressor.predict(self.all_mean_features_today)
        print(
            'Pred goes up: ',float(pred_cls.flatten()),'\n',
            'Pred perc change: ',float(pred_reg.flatten()),'\n'
        )

        self.pred_today_cls = float(pred_cls.flatten())
        self.pred_today_reg = float(pred_reg.flatten())
        return self.pred_today_cls, self.pred_today_reg




    

