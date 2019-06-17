

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
    
class KnowledgeGraphBlackbox:

   
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 3)),('max_depth',(4,5)),
                                      ('gamma',(2,9)),
                                       ('alpha',(1,6)),('eta',(0.1,0.28))])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 20
        self.ismax=-1
        self.name='KnowledgeGraphBlackbox'
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

     
    def run_Blackbox(self,X):


		# given hyper-parameter X
		
		
		
    
        # return the utility
        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 
		
		
		
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
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_XGBoost(X)
        else:
            Accuracy=np.apply_along_axis( self.run_XGBoost,1,X)

        #print RMSE    
        return Accuracy*self.ismax 
    
