import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import matplotlib.pyplot as plt

from bayes_opt.sequentialBO.bo_known_optimum_value import BayesOpt_KnownOptimumValue
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from tqdm import tqdm
import numpy as np
from bayes_opt import auxiliary_functions


from bayes_opt.test_functions import functions,knowlegde_graph_blackbox
import warnings
#from bayes_opt import acquisition_maximization

import sys

from bayes_opt.utility import export_results


np.random.seed(6789)

warnings.filterwarnings("ignore")


counter = 0

  
myfunction=knowlegde_graph_blackbox.KnowledgeGraphBlackbox()


acq_type_list=[]


acq_type={}
acq_type['name']='ei' # expected regret minimization
acq_type['surrogate']='gp' # recommended to use tgp for ERM
acq_type['dim']=myfunction.input_dim
acq_type['fstar']=myfunction.fstar


func=myfunction.func

func_params={}
func_params['function']=myfunction

gp_params = {'lengthscale':0.05*myfunction.input_dim,'noise_delta':1e-8} # the lengthscaled parameter will be optimized

yoptimal=myfunction.fstar*myfunction.ismax


acq_params={}
acq_params['acq_func']=acq_type

    
bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
  
# initialize BO using 3*dim number of observations
bo.init(gp_params,n_init_points=2*myfunction.input_dim)

# run for 10*dim iterations
NN=12*myfunction.input_dim
for index in tqdm(range(0,NN)):

    bo.maximize()

    if bo.stop_flag==1:
        break
    
    print("recommemded x={} current y={:.3f}, ymin={:.3f}".format(bo.X_original[-1],myfunction.ismax*bo.Y_original[-1],myfunction.ismax*bo.Y_original.max()))
    
    idxMax=np.argmax(bo.Y_original)
    print("best X ",bo.X_original[idxMax])
    sys.stdout.flush()

fig=plt.figure(figsize=(6, 3))
myYbest=[bo.Y_original[:idx+1].max()*-1 for idx,val in enumerate(bo.Y_original)]
plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Best Found Value',fontsize=14)
plt.title('Performance',fontsize=16)    
