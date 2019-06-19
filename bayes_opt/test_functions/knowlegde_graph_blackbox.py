

import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.model_selection import KFold
from dateutil.relativedelta import relativedelta

#import matlab.engine
#import matlab
#eng = matlab.engine.start_matlab()
from sklearn.metrics import f1_score

from main import train_and_evaluate


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
        self.input_dim = 4

        # NOTE FOR ME: Put here all parameters to hyperparameters, with
        #   - name
        #   - range

        # batches_per_epoch: from 50 to 200 (int)
        # learning rate: from 0.1 to 0.0001 (real)
        # embedding_size: from 50 to 300 (int)
        # ent_neg_rate: from 1 to 10 (int)

        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('batches_per_epoch',(50, 200)),
                                       ('learning_rate',(0.1, 0.0001)),
                                      ('embedding_size',(50, 300)),
                                       ('ent_neg_rate',(1,10))])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 20
        self.ismax=-1   # NOTE FOR ME: this means the score must me minimized
        self.name='KnowledgeGraphBlackbox'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None

    # input set of hyperparam values
    def run_Blackbox(self, X):

        batches_per_epoch = np.int(X[0])
        learning_rate = X[1]
        embedding_size = np.int(X[2])
        ent_neg_rate = np.int(X[3])
        utility = train_and_evaluate(batches_per_epoch=batches_per_epoch,
                                     learning_rate=learning_rate,
                                     embedding_size=embedding_size,
                                     ent_neg_rate=ent_neg_rate,
                                     clean_everything_afterwards=True)

        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax