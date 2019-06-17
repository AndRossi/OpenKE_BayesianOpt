import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
sys.path.insert(0,'../../..')


from bayes_opt.sequentialBO.bo_known_optimum_value import BayesOpt_KnownOptimumValue
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt

import numpy as np

from bayes_opt.visualization import vis_ERM



from bayes_opt.test_functions import functions
import warnings


import sys


warnings.filterwarnings("ignore")

#%matplotlib inline.

counter = 0


myfunction=functions.fourier(sd=0)

func=myfunction.func

    
gp_params = {'theta':0.035,'noise_delta':1e-10}

# create an empty object for BO using vanilla GP
acq_func={}
acq_func['name']='ei'
acq_func['surrogate']='gp'
acq_func['dim']=myfunction.input_dim


func_params={}
func_params['bounds']=myfunction.bounds
func_params['f']=func
func_params['function']=myfunction


acq_params={}
acq_params['acq_func']=acq_func

    
gp_params = {'theta':0.04,'noise_delta':1e-8}
bo=BayesOpt_KnownOptimumValue(gp_params,func_params,acq_params)



x0=[3.1,4.4,8,9]


init_X=np.asarray(x0)
init_X=np.reshape(x0,(len(x0),1))
init_Y=func(init_X)
bo.init_with_data(init_X=init_X,init_Y=init_Y)


# number of recommended parameters
NN=1*myfunction.input_dim
for index in range(0,NN):
    vis_ERM.plot_acq_bo_1d(bo,fstar=myfunction.fstar)





    
# create an empty object for BO using transformed GP

acq_func={}
acq_func['name']='kov_erm'
acq_func['surrogate']='tgp'


acq_params={}
acq_params['acq_func']=acq_func

bo_tgp=BayesOpt_KnownOptimumValue(gp_params,func_params,acq_params)
bo_tgp.init_with_data(init_X=init_X,init_Y=init_Y)

NN=1*myfunction.input_dim
for index in range(0,NN):
    vis_ERM.plot_acq_bo_1d_tgp(bo_tgp,fstar=myfunction.fstar)
    