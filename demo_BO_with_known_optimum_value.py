from bayes_opt import BayesOpt
import matplotlib.pyplot as plt
from bayes_opt.test_functions import functions
from bayes_opt.sequentialBO.bo_known_optimum_value import BayesOpt_KnownOptimumValue

import warnings
import sys

warnings.filterwarnings("ignore")




# select the function to be optimized
myfunction=functions.branin(sd=0)
#myfunction=functions.hartman_3d()
#myfunction=functions.hartman_6d()
#myfunction=functions.ackley(input_dim=5)
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1]))
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1]))



func=myfunction.func
    

# specifying the acquisition function
acq_func={}
#acq_func['name']='ei'
#acq_func['name']='ucb'
acq_func['name']='kov_erm'
#acq_func['name']='kov_cbm'
#acq_func['name']='kov_mes'
#acq_func['name']='kov_ei'


# specifying the surrogate model either tgp or gp
acq_func['surrogate']='tgp' # set it to either 'gp' or 'tgp'

acq_func['dim']=myfunction.input_dim
    
# we specify the known optimum value here
acq_func['fstar']=myfunction.fstar



func_params={}
func_params['function']=myfunction


acq_params={}
acq_params['acq_func']=acq_func
gp_params = {'kernel':'SE','lengthscale':0.04*myfunction.input_dim,'noise_delta':1e-8,'flagIncremental':0}


if 'kov' in acq_func['name']:
    bo=BayesOpt_KnownOptimumValue(gp_params,func_params,acq_params)
else:
    bo=BayesOpt(gp_params,func_params,acq_params)
            
if acq_func['surrogate']=='tgp': 
    print("using transform GP with the known optimum value")
else:
    print("using vanilla GP without the known optimum value")

# initialize BO using 3*dim number of observations
bo.init(gp_params,n_init_points=3*myfunction.input_dim)

# run for 10*dim iterations
NN=10*myfunction.input_dim
for index in range(0,NN):

    bo.maximize()

    if bo.stop_flag==1:
        break
    
    if myfunction.ismax==1:
        print("recommemded x={} current y={}, ymax={}".format(bo.X_original[-1],bo.Y_original[-1],bo.Y_original.max()))
    else:
        print("recommemded x={} current y={}, ymin={}".format(bo.X_original[-1],myfunction.ismax*bo.Y_original[-1],myfunction.ismax*bo.Y_original.max()))
    sys.stdout.flush()

fig=plt.figure(figsize=(6, 3))
myYbest=[bo.Y_original[:idx+1].max()*-1 for idx,val in enumerate(bo.Y_original)]
plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Best Found Value',fontsize=14)
plt.title('Performance',fontsize=16)
