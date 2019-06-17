import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')


from bayes_opt.sequentialBO.bo_known_optimum_value import BayesOpt_KnownOptimumValue
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt

import numpy as np
from bayes_opt import auxiliary_functions


from bayes_opt.test_functions import functions,real_experiment_function
import warnings
#from bayes_opt import acquisition_maximization

import sys

from bayes_opt.utility import export_results
import itertools


np.random.seed(6789)

warnings.filterwarnings("ignore")


counter = 0

  
myfunction_list=[]
myfunction_list.append(real_experiment_function.XGBoost_Skin())


acq_type_list=[]


temp={}
temp['name']='kov_erm' # expected regret minimization
temp['surrogate']='tgp' # recommended to use tgp for ERM
acq_type_list.append(temp)

temp={}
temp['name']='kov_cbm' # confidence bound minimization
temp['surrogate']='tgp' # recommended to use tgp for CBM
acq_type_list.append(temp)


temp={}
temp['name']='kov_mes' # MES+f*
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)


temp={}
temp['name']='kov_ei' # this is EI + f*
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)


temp={}
temp['name']='ucb' # vanilla UCB
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)

temp={}
temp['name']='ei' # vanilla EI
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)



for idx, (myfunction,acq_type,) in enumerate(itertools.product(myfunction_list,acq_type_list)):
    func=myfunction.func
    
    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.05*myfunction.input_dim,'noise_delta':0.0000001} # the lengthscaled parameter will be optimized

    yoptimal=myfunction.fstar*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim
    acq_type['fstar']=myfunction.fstar

    acq_params={}
    acq_params['acq_func']=acq_type
    
    nRepeat=20
    
    ybest=[0]*nRepeat
    MyTime=[0]*nRepeat
    MyOptTime=[0]*nRepeat

    bo=[0]*nRepeat
   
    [0]*nRepeat
    
    
    for ii in range(nRepeat):
        
        if 'kov' in acq_type['name']:
            bo[ii]=BayesOpt_KnownOptimumValue(gp_params,func_params,acq_params)
        else:
            bo[ii]=BayesOpt(gp_params,func_params,acq_params)
  
        ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(bo[ii],gp_params,
             n_init=3*myfunction.input_dim,NN=10*myfunction.input_dim,runid=ii)                                               
        MyOptTime[ii]=bo[ii].time_opt
        print("ii={} BFV={}".format(ii,np.max(ybest[ii])))                                              
        

    Score={}
    Score["ybest"]=ybest
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    
    export_results.print_result_sequential(bo,myfunction,Score,acq_type) 