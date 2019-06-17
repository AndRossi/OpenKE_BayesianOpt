# -*- coding: utf-8 -*-
"""
Created on Mon May 02 21:24:47 2016

@author: tvun
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:25:02 2016

@author: Vu
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import KFold
from dateutil.relativedelta import relativedelta

#import matlab.engine
#import matlab
#eng = matlab.engine.start_matlab()
from sklearn.metrics import f1_score 
        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    

class XGBoost_EEML:

    '''
    XGBoost: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 3)),('max_depth',(4,5)),
                                      ('gamma',(2,9)),
                                       ('alpha',(1,6)),('eta',(0.1,0.28))])
    
    #('colsample_bytree',(0.5, 1)), ('subsample',(0.5,1)),
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 20
        self.ismax=-1
        self.name='XGBoost_EEML'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
            
        #dataset = loadtxt('P:/05.BayesianOptimization/PradaBayesianOptimization/prada_bayes_opt/test_functions/data/pima-indians-diabetes.csv', delimiter=",")
        #dataset = loadtxt('bayes_opt/test_functions/data/Skin_NonSkin.txt', delimiter="\t")
        
        self.df = pd.read_csv("bayes_opt/test_functions/data/train_electricity.csv")
        self.df = self.add_datetime_features(self.df)
        
        
        label_col = "Consumption_MW"  # The target values are in this column
        to_drop = ["Date", "Datetime","Dayofyear"]  # Columns we do not need for training
        to_drop_train=[label_col]+to_drop

        #kf = KFold(self.n_splits,shuffle=True)

        
        month_t1_list=[36,36,24,24,12]
        month_t2_list=[12,24,12,24,12]
        
        self.n_splits=len(month_t2_list)

        self.xg_train=[0]*self.n_splits
        self.xg_test=[0]*self.n_splits
        self.label_train=[0]*self.n_splits
        self.label_test=[0]*self.n_splits

      

        
        #for train_index, test_index in kf.split(self.df):
        for ii in range(self.n_splits):
            month_t1=month_t1_list[ii]
            month_t2=month_t1-month_t2_list[ii]
            
            #train_df, valid_df = train_test_split(self.df, test_size=0.8,random_state=ii)
            threshold_1 = self.df['Datetime'].max() + relativedelta(months=-month_t1)  # Here we set the 6 months threshold
            threshold_2 = self.df['Datetime'].max() + relativedelta(months=-month_t2)  # Here we set the 6 months threshold
            #eval_from = self.df['Datetime'].max() + relativedelta(months=-threshold_month[ii])  # Here we set the 6 months threshold
            #train_df = self.df[self.df['Datetime'] < eval_from]
            #valid_df = self.df[self.df['Datetime'] >= eval_from]
            
            train_df = self.df[ ( self.df['Datetime'] < threshold_1) | (self.df['Datetime'] > threshold_2 ) ]
            valid_df = self.df[(self.df['Datetime'] >= threshold_1) & (self.df['Datetime'] <= threshold_2)]
            
            #train_df=self.df.iloc[train_index]
            #valid_df=self.df.iloc[test_index]
            
            
            self.xg_train[ii] = xgb.DMatrix(train_df.drop(columns=to_drop_train), label=train_df[label_col])
            self.xg_test[ii] = xgb.DMatrix(valid_df.drop(columns=to_drop_train), label=valid_df[label_col])
            
            self.label_train[ii]=train_df[label_col]
            self.label_test[ii]=valid_df[label_col]

        
        
        """
        eval_from = self.df['Datetime'].max() + relativedelta(months=-18)  # Here we set the 6 months threshold
        train_df = self.df[self.df['Datetime'] < eval_from]
        valid_df = self.df[self.df['Datetime'] >= eval_from]
        
        self.xg_train = xgb.DMatrix(train_df.drop(columns=to_drop), label=train_df[label_col])
        self.xg_test = xgb.DMatrix(valid_df.drop(columns=to_drop), label=valid_df[label_col])
        
        self.label_train=train_df[label_col]
        self.label_test=valid_df[label_col]
        """
 
    def add_datetime_features(self,df):
         
             

        #features = ["Year", "Week", "Dayofyear","Dayofweek","Is_year_end", "Is_year_start","Hour"]

        datetime = pd.to_datetime(df.Date * (10 ** 9))
        features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
                "Hour", "Minute",]
        one_hot_features = ["Month", "Dayofweek"] 
        
        df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later
        
        for feature in features:
            new_column = getattr(datetime.dt, feature.lower())
            if feature in one_hot_features:
                df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)
            else:
                df[feature] = new_column
                
                 # add season
        # "day of year" ranges for the northern hemisphere
        spring = range(80, 172)
        summer = range(172, 264)
        fall = range(264, 355)
        # winter = everything else
        
        def doy_to_season(doy): # winter - summer - else
            if doy in spring:
                season = 1
            elif doy in summer:
                season = 2
            elif doy in fall:
                season = 1
            else:
                season = 3
            return season
            
        temp=df['Dayofyear'].apply(doy_to_season)
        df = pd.concat([df, pd.get_dummies(temp, prefix="season")], axis=1)
    
    
    
        return df

        
    def run_XGBoost(self,X):
        #print(X)
        params={}
        params['min_child_weight'] = int(X[0])
        #params['colsample_bytree'] = max(min(X[1], 1), 0)
        params['max_depth'] = int(X[1])
        #params['subsample'] = max(min(X[3], 1), 0)
        params['gamma'] = max(X[2], 0)
        params['alpha'] = max(X[3], 0)
        params['eta'] = max(X[4], 0)
        #params['n_estimators'] = max(X[5], 0)

        #params['silent'] = 1
    

        num_round=1500
        #print(params)
        rms=[0]*self.n_splits
        #for ii in range(self.n_splits):
        
        ## 5. Train (mostly with default parameters; it overfits like hell)
        
        #watchlist = [(xg_trn_data, "train"), (xg_vld_data, "valid")]
        
        for ii in range(self.n_splits):
            bst = xgb.train(params, self.xg_train[ii], num_round,verbose_eval=0)
            
            pred_score=bst.predict(self.xg_test[ii])
            
            rms[ii]=np.sqrt(np.mean((pred_score-self.label_test[ii])**2))
    
        
        return np.mean(rms)
        
        """        
        bst = xgb.train(params, self.xg_train, num_round,verbose_eval=0)
            
        pred_score=bst.predict(self.xg_test)
        
        rms=np.sqrt(np.mean((pred_score-self.label_test)**2))
    
        
        return rms
        """        
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
        
        #import pandas as pd
        #from sklearn.preprocessing import LabelEncoder
        #from tqdm import tqdm
        
        #print(X)
        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_XGBoost(X)
        else:
            Accuracy=np.apply_along_axis( self.run_XGBoost,1,X)

        #print RMSE    
        return Accuracy*self.ismax 
    

class XGBoost_EEML_Heldout_Year:

    '''
    XGBoost: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 5
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 3)),('max_depth',(4,5)),
                                      ('gamma',(2,9)),
                                       ('alpha',(1,6)),('eta',(0.1,0.28))])
    
    #('colsample_bytree',(0.5, 1)), ('subsample',(0.5,1)),
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 20
        self.ismax=-1
        self.name='XGBoost_EEML_YearOut'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
            
        #dataset = loadtxt('P:/05.BayesianOptimization/PradaBayesianOptimization/prada_bayes_opt/test_functions/data/pima-indians-diabetes.csv', delimiter=",")
        #dataset = loadtxt('bayes_opt/test_functions/data/Skin_NonSkin.txt', delimiter="\t")
        
        self.df = pd.read_csv("bayes_opt/test_functions/data/train_electricity.csv")
        self.df = self.add_datetime_features(self.df)
        
        
        label_col = "Consumption_MW"  # The target values are in this column
        to_drop = ["Date", "Datetime","Dayofyear"]  # Columns we do not need for training
        to_drop_train=[label_col]+to_drop

        #kf = KFold(self.n_splits,shuffle=True)

        
        month_t1_list=[36,24,12]
        month_t2_list=[24,12,12]
        
        self.n_splits=len(month_t2_list)

        self.xg_train=[0]*self.n_splits
        self.xg_test=[0]*self.n_splits
        self.label_train=[0]*self.n_splits
        self.label_test=[0]*self.n_splits

      

        
        #for train_index, test_index in kf.split(self.df):
        for ii in range(self.n_splits):
            month_t1=month_t1_list[ii]
            month_t2=month_t1-month_t2_list[ii]
            
            #train_df, valid_df = train_test_split(self.df, test_size=0.8,random_state=ii)
            threshold_1 = self.df['Datetime'].max() + relativedelta(months=-month_t1)  # Here we set the 6 months threshold
            threshold_2 = self.df['Datetime'].max() + relativedelta(months=-month_t2)  # Here we set the 6 months threshold
            #eval_from = self.df['Datetime'].max() + relativedelta(months=-threshold_month[ii])  # Here we set the 6 months threshold
            #train_df = self.df[self.df['Datetime'] < eval_from]
            #valid_df = self.df[self.df['Datetime'] >= eval_from]
            
            train_df = self.df[ ( self.df['Datetime'] < threshold_1) | (self.df['Datetime'] > threshold_2 ) ]
            valid_df = self.df[(self.df['Datetime'] >= threshold_1) & (self.df['Datetime'] <= threshold_2)]
            
            #train_df=self.df.iloc[train_index]
            #valid_df=self.df.iloc[test_index]
            
            
            self.xg_train[ii] = xgb.DMatrix(train_df.drop(columns=to_drop_train), label=train_df[label_col])
            self.xg_test[ii] = xgb.DMatrix(valid_df.drop(columns=to_drop_train), label=valid_df[label_col])
            
            self.label_train[ii]=train_df[label_col]
            self.label_test[ii]=valid_df[label_col]

        
        
        """
        eval_from = self.df['Datetime'].max() + relativedelta(months=-18)  # Here we set the 6 months threshold
        train_df = self.df[self.df['Datetime'] < eval_from]
        valid_df = self.df[self.df['Datetime'] >= eval_from]
        
        self.xg_train = xgb.DMatrix(train_df.drop(columns=to_drop), label=train_df[label_col])
        self.xg_test = xgb.DMatrix(valid_df.drop(columns=to_drop), label=valid_df[label_col])
        
        self.label_train=train_df[label_col]
        self.label_test=valid_df[label_col]
        """
 
    def add_datetime_features(self,df):
         
             

        features = ["Year", "Week", "Dayofyear","Dayofweek","Is_year_end", "Is_year_start","Hour"]

        datetime = pd.to_datetime(df.Date * (10 ** 9))
        #features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
        #        "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
        #        "Hour", "Minute",]
        one_hot_features = ["Month", "Dayofweek"] 
        
        df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later
        
        for feature in features:
            new_column = getattr(datetime.dt, feature.lower())
            if feature in one_hot_features:
                df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)
            else:
                df[feature] = new_column
                
        return df

        
    def run_XGBoost(self,X):
        #print(X)
        params={}
        params['min_child_weight'] = int(X[0])
        #params['colsample_bytree'] = max(min(X[1], 1), 0)
        params['max_depth'] = int(X[1])
        #params['subsample'] = max(min(X[3], 1), 0)
        params['gamma'] = max(X[2], 0)
        params['alpha'] = max(X[3], 0)
        params['eta'] = max(X[4], 0)
        #params['n_estimators'] = max(X[5], 0)

        params['silent'] = 1
    

        num_round=300
        #print(params)
        rms=[0]*self.n_splits
        #for ii in range(self.n_splits):
        
        ## 5. Train (mostly with default parameters; it overfits like hell)
        
        #watchlist = [(xg_trn_data, "train"), (xg_vld_data, "valid")]
        
        for ii in range(self.n_splits):
            bst = xgb.train(params, self.xg_train[ii], num_round,verbose_eval=0)
            
            pred_score=bst.predict(self.xg_test[ii])
            
            rms[ii]=np.sqrt(np.mean((pred_score-self.label_test[ii])**2))
    
        
        return np.mean(rms)
        
        """        
        bst = xgb.train(params, self.xg_train, num_round,verbose_eval=0)
            
        pred_score=bst.predict(self.xg_test)
        
        rms=np.sqrt(np.mean((pred_score-self.label_test)**2))
    
        
        return rms
        """        
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
        
        #import pandas as pd
        #from sklearn.preprocessing import LabelEncoder
        #from tqdm import tqdm
        
        #print(X)
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_XGBoost(X)
        else:
            RMSE=np.apply_along_axis( self.run_XGBoost,1,X)

        #print RMSE    
        return RMSE*self.ismax
    
    
class LightGBM_EEML_Heldout_Year:

    '''
    LightGBM: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 5
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 3)),('max_depth',(4,5)),
                                      ('gamma',(2,9)),
                                       ('alpha',(1,6)),('eta',(0.1,0.28))])
    
    #('colsample_bytree',(0.5, 1)), ('subsample',(0.5,1)),
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 20
        self.ismax=-1
        self.name='LightGBM_EEML_YearOut'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
        
        self.df = pd.read_csv("bayes_opt/test_functions/data/train_electricity.csv")
        self.df = self.add_datetime_features(self.df)
        
        
        label_col = "Consumption_MW"  # The target values are in this column
        to_drop = ["Date", "Datetime","Dayofyear"]  # Columns we do not need for training
        to_drop_train=[label_col]+to_drop

        #kf = KFold(self.n_splits,shuffle=True)

        
        month_t1_list=[36,24,12]
        month_t2_list=[24,12,12]
        
        self.n_splits=len(month_t2_list)

        self.xg_train=[0]*self.n_splits
        self.xg_test=[0]*self.n_splits
        self.label_train=[0]*self.n_splits
        self.label_test=[0]*self.n_splits

      
        
        #for train_index, test_index in kf.split(self.df):
        for ii in range(self.n_splits):
            month_t1=month_t1_list[ii]
            month_t2=month_t1-month_t2_list[ii]
            
            #train_df, valid_df = train_test_split(self.df, test_size=0.8,random_state=ii)
            threshold_1 = self.df['Datetime'].max() + relativedelta(months=-month_t1)  # Here we set the 6 months threshold
            threshold_2 = self.df['Datetime'].max() + relativedelta(months=-month_t2)  # Here we set the 6 months threshold
            #eval_from = self.df['Datetime'].max() + relativedelta(months=-threshold_month[ii])  # Here we set the 6 months threshold
            #train_df = self.df[self.df['Datetime'] < eval_from]
            #valid_df = self.df[self.df['Datetime'] >= eval_from]
            
            train_df = self.df[ ( self.df['Datetime'] < threshold_1) | (self.df['Datetime'] > threshold_2 ) ]
            valid_df = self.df[(self.df['Datetime'] >= threshold_1) & (self.df['Datetime'] <= threshold_2)]
            
            #train_df=self.df.iloc[train_index]
            #valid_df=self.df.iloc[test_index]
            
            self.X_test[ii]=valid_df.drop(columns=to_drop_train)
            self.lgb_train[ii] = lgb.Dataset(train_df.drop(columns=to_drop_train), label=train_df[label_col])
            self.lgb_eval[ii] = lgb.Dataset(valid_df.drop(columns=to_drop_train), label=valid_df[label_col],
                         reference=lgb_train[ii])

            self.label_train[ii]=train_df[label_col]
            self.label_test[ii]=valid_df[label_col]

 
    def add_datetime_features(self,df):
         

        features = ["Year", "Week", "Dayofyear","Dayofweek","Is_year_end", "Is_year_start","Hour"]

        datetime = pd.to_datetime(df.Date * (10 ** 9))
        #features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
        #        "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
        #        "Hour", "Minute",]
        one_hot_features = ["Month", "Dayofweek"] 
        
        df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later
        
        for feature in features:
            new_column = getattr(datetime.dt, feature.lower())
            if feature in one_hot_features:
                df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)
            else:
                df[feature] = new_column
                
        return df

        
    def run_LightGBM(self,X):
        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': np.int(X[0]),
            'learning_rate': X[1],
            'feature_fraction': X[2],
            'bagging_fraction': X[3],
            'bagging_freq': X[4],
            'verbose': 0
        }

    
        #print(params)
        rms=[0]*self.n_splits
        #for ii in range(self.n_splits):
        
        ## 5. Train (mostly with default parameters; it overfits like hell)
        
        #watchlist = [(xg_trn_data, "train"), (xg_vld_data, "valid")]
        
        for ii in range(self.n_splits):
            
            # train
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=20,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=5)
            
            
            
            pred_score = gbm.predict(self.X_test[ii], num_iteration=gbm.best_iteration)

            rms[ii]=np.sqrt(np.mean((pred_score-self.label_test[ii])**2))
    
        
        return np.mean(rms)
   
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
        
        #import pandas as pd
        #from sklearn.preprocessing import LabelEncoder
        #from tqdm import tqdm
        
        #print(X)
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_LightGBM(X)
        else:
            RMSE=np.apply_along_axis( self.run_LightGBM,1,X)

        #print RMSE    
        return RMSE*self.ismax 
    
    
    
class XGBoost_EEML_319:

    '''
    XGBoost: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 7
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 3)),('colsample_bytree',(0.5,1)),('max_depth',(4,5)),
                                     ('subsample',(0.5,1)), ('gamma',(2,9)),
                                       ('alpha',(1,6)),('eta',(0.1,0.28))])
    
    #('colsample_bytree',(0.5, 1)), ('subsample',(0.5,1)),
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 20
        self.ismax=-1
        self.name='XGBoost_EEML_319'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
            
        #dataset = loadtxt('P:/05.BayesianOptimization/PradaBayesianOptimization/prada_bayes_opt/test_functions/data/pima-indians-diabetes.csv', delimiter=",")
        #dataset = loadtxt('bayes_opt/test_functions/data/Skin_NonSkin.txt', delimiter="\t")
        
        self.df = pd.read_csv("bayes_opt/test_functions/data/train_electricity.csv")
        self.df = self.add_datetime_features(self.df)
        df=self.df
        
        test_df = pd.read_csv("bayes_opt/test_functions/data/test_electricity.csv")
        self.test_df = self.add_datetime_features(test_df)
        
        self.label_test=pd.read_csv("bayes_opt/test_functions/data/319_best_submission.csv")
        self.label_test=self.label_test.values[:,1]
        
        label_col = "Consumption_MW"  # The target values are in this column
        to_drop = ["Date", "Datetime","Dayofyear"]  # Columns we do not need for training
        to_drop_train=[label_col]+to_drop

        #kf = KFold(self.n_splits,shuffle=True)

      
        month_t1=0
        month_t2=0
        month_start=12*0
        threshold_1 = self.df['Datetime'].max() + relativedelta(months=-month_t1)  # Here we set the 6 months threshold
        threshold_2 = self.df['Datetime'].max() + relativedelta(months=-month_t2)  # Here we set the 6 months threshold
        threshold_0 = self.df['Datetime'].min() + relativedelta(months=month_start)  # Here we set the 6 months threshold
        
        train_df = self.df[ ( ( df['Datetime'] > threshold_0) & ( df['Datetime'] < threshold_1)) | (df['Datetime'] > threshold_2 ) ]
        

            
        self.xg_train = xgb.DMatrix(train_df.drop(columns=to_drop_train).as_matrix(), label=train_df[label_col])
        self.xg_test = xgb.DMatrix(self.test_df.drop(columns=to_drop).as_matrix())
        
        self.label_train=self.df[label_col]
 
    def add_datetime_features(self,df):
         
        #features = ["Year", "Week", "Dayofyear","Dayofweek","Is_year_end", "Is_year_start","Hour"]

        datetime = pd.to_datetime(df.Date * (10 ** 9))
        features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
                "Hour", "Minute","Quarter"]
        one_hot_features = ["Month", "Dayofweek","Quarter"] 
        
        df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later
        
        for feature in features:
            new_column = getattr(datetime.dt, feature.lower())
            if feature in one_hot_features:
                df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)
            else:
                df[feature] = new_column
    
        return df

        
    def run_XGBoost(self,X):
        #print(X)
        params={}
        params['min_child_weight'] = int(X[0])
        params['colsample_bytree'] = max(min(X[1], 1), 0)
        params['max_depth'] = int(X[2])
        params['subsample'] = max(min(X[3], 1), 0)
        params['gamma'] = max(X[4], 0)
        params['alpha'] = max(X[5], 0)
        params['eta'] = max(X[6], 0)
        #params['n_estimators'] = max(X[5], 0)

        #params['silent'] = 1
    

        num_round=500
        #print(params)
        #for ii in range(self.n_splits):
        
        ## 5. Train (mostly with default parameters; it overfits like hell)
        
        #watchlist = [(xg_trn_data, "train"), (xg_vld_data, "valid")]
        
        bst = xgb.train(params, self.xg_train, num_round,verbose_eval=0)
        
        pred_score=bst.predict(self.xg_test)
        
        pred_score1=pred_score*1.01
        pred_score2=pred_score*1.02
        pred_score3=pred_score*1.03
        rms1=np.sqrt(np.mean((pred_score1-self.label_test)**2))
        rms2=np.sqrt(np.mean((pred_score2-self.label_test)**2))
        rms3=np.sqrt(np.mean((pred_score3-self.label_test)**2))

        
        return min(rms1,rms2,rms3)
    
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
        
        #import pandas as pd
        #from sklearn.preprocessing import LabelEncoder
        #from tqdm import tqdm
        
        #print(X)
        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_XGBoost(X)
        else:
            Accuracy=np.apply_along_axis( self.run_XGBoost,1,X)

        #print RMSE    
        return Accuracy*self.ismax 