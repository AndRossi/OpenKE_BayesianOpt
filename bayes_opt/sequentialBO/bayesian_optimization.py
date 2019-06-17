# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
#from sklearn.gaussian_process import GaussianProcess

from bayes_opt.sequentialBO.bayesian_optimization_base import BO_Sequential_Base
from scipy.optimize import minimize
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt.gaussian_process import GaussianProcess

from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name


import time

#import nlopt

#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt(BO_Sequential_Base):

    def __init__(self, gp_params, func_params, acq_params, verbose=1):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.theta:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        # Find number of parameters
        
        super(BayesOpt, self).__init__(gp_params, func_params, acq_params, verbose)

        """
        bounds=func_params['function'].bounds
        
        self.dim = len(bounds)

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in list(bounds.keys()):
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)

        #print(self.bounds)
 
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        
        # Some function to be optimized
        self.f = func_params['function'].func
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
        # acquisition function type
        
        self.acq=acq_params['acq_func']
        self.acq['scalebounds']=self.scalebounds
        
        if 'debug' not in self.acq:
            self.acq['debug']=0           
        if 'stopping' not in acq_params:
            self.stopping_criteria=0
        else:
            self.stopping_criteria=acq_params['stopping']
        if 'optimize_gp' not in acq_params:
            self.optimize_gp=0
        else:                
            self.optimize_gp=acq_params['optimize_gp']       
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
            
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        
        # performance evaluation at the maximum mean GP (for information theoretic)
        self.Y_original_maxGP = None
        self.X_original_maxGP = None
        
        # value of the acquisition function at the selected point
        self.alpha_Xt=None
        self.Tau_Xt=None
        
        self.time_opt=0

        self.k_Neighbor=2
        
        # Lipschitz constant
        self.L=0
        
        self.gp_params=gp_params       

        # Gaussian Process class
        self.gp=GaussianProcess(gp_params)

        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0
        self.logmarginal=0
        
        # xt_suggestion, caching for Consensus
        self.xstars=[]
        self.xstar_accumulate=[]

        # theta vector for marginalization GP
        self.theta_vector =[]
        
        # PVRS before and after
        self.PVRS_before_after=[]
        self.accummulate_PVRS_before_after=[]
        
        # store ystars
        #self.ystars=np.empty((0,100), float)
        self.ystars=[]
       """
       
       
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, gp_params, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        super(BayesOpt, self).init(gp_params, n_init_points,seed)
        
    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        super(BayesOpt, self).init_with_data(init_X,init_Y)
        
           
    def estimate_L(self,bounds):
        '''
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        '''
        def df(x,model,x0):
            mean_derivative=gp_model.predictive_gradient(self.X,self.Y,x)
            
            temp=mean_derivative*mean_derivative
            if len(temp.shape)<=1:
                res = np.sqrt( temp)
            else:
                res = np.sqrt(np.sum(temp,axis=1)) # simply take the norm of the expectation of the gradient        

            return -res

        gp_model=self.gp
                
        dim = len(bounds)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): samples[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)

        #samples = np.vstack([samples,gp_model.X])
        pred_samples = df(samples,gp_model,0)
        x0 = samples[np.argmin(pred_samples)]

        res = minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (gp_model,x0), options = {'maxiter': 100})        
   
        try:
            minusL = res.fun[0][0]
        except:
            if len(res.fun.shape)==1:
                minusL = res.fun[0]
            else:
                minusL = res.fun
                
        L=-minusL
        if L<1e-6: L=0.0001  ## to avoid problems in cases in which the model is flat.
        
        return L    
        
    
    def maximize_with_lengthscale_derived_by_fstar(self,gp_params):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            return

        # init a new Gaussian Process
        self.gp=GaussianProcess(gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

 
        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        if  len(self.Y)%(3*self.dim)==0:
            fstar_scaled=(self.acq['fstar']-np.mean(self.Y_original))/np.std(self.Y_original)
            newlengthscale = self.gp.optimize_lengthscale_SE_fstar(self.gp_params['lengthscale'],self.gp_params['noise_delta'],fstar_scaled)            
            self.gp_params['lengthscale']=newlengthscale
            print("estimated lengthscale =",newlengthscale)
            
            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=GaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

        if self.acq['name']=='mes':
            self.maximize_mes(gp_params)
            return
        if self.acq['name']=='pvrs':
            self.maximize_pvrs(gp_params)
            return
        if self.acq['name']=='e3i':
            self.maximize_e3i(gp_params)
            return
        if self.acq['name']=='ei_kov' or self.acq['name']=='poi_kov' or self.acq['name']=='ei_fstar':
            self.acq['fstar_scaled']=(self.acq['fstar']-np.mean(self.Y_original))/np.std(self.Y_original)
            
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()
                      
        if 'xstars' not in globals():
            xstars=[]
            
        self.xstars=xstars

        self.acq['xstars']=xstars
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)


        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp)

            self.stop_flag=1
            #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
        if self.gp.flagIncremental==1:
            self.gp.fit_incremental(x_max,self.Y[-1])

            
            
    def maximize(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            
            super(BayesOpt, self).generate_random_point()

            return

        # init a new Gaussian Process
        self.gp=GaussianProcess(self.gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

 
        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        if  len(self.Y)%(2*self.dim)==0:
            self.gp,self.gp_params=super(BayesOpt, self).optimize_gp_hyperparameter()



        if self.acq['name']=='mes':
            self.maximize_mes()
            return
        if self.acq['name']=='pvrs':
            self.maximize_pvrs()
            return
        if self.acq['name']=='e3i':
            self.maximize_e3i()
            return
        if self.acq['name']=='ei_kov' or self.acq['name']=='poi_kov' or self.acq['name']=='ei_fstar':
            self.acq['fstar_scaled']=(self.acq['fstar']-np.mean(self.Y_original))/np.std(self.Y_original)
            
        # Set acquisition function
        start_opt=time.time()

        #y_max = self.Y.max()
                      
        if 'xstars' not in globals():
            xstars=[]
            
        self.xstars=xstars

        self.acq['xstars']=xstars
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)


        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            #val_acq=self.acq_func.acq_kind(x_max,self.gp)

            self.stop_flag=1
            #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        
        super(BayesOpt, self).augment_the_new_data(x_max)

        
  
    def maximize_mes(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
               
 
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()                
            
        # run the acquisition function for the first time to get xstar
        
        self.xstars=[]
        # finding the xt of UCB
        
        y_max=np.max(self.Y)
        #numXtar=10*self.dim
        numXtar=30
        
        y_stars=[]
        temp=[]
        # finding the xt of Thompson Sampling
        for ii in range(numXtar):
            
            xt_TS,y_xt_TS=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,
                                                acq_name="thompson",IsReturnY=True)
            
            #if y_xt_TS>=y_max:
            y_stars.append(y_xt_TS)
            
            temp.append(xt_TS)
            # check if f* > y^max and ignore xt_TS otherwise
            if y_xt_TS>=y_max:
                self.xstars.append(xt_TS)

        if self.xstars==[]:
            #print 'xt_suggestion is empty'
            # again perform TS and take all of them
            self.xstars=temp
            
        self.acq['xstars']=self.xstars   
        self.acq['ystars']=y_stars   


        self.acq_func = AcquisitionFunction(self.acq)
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)

        #xstars_array=np.asarray(self.acq_func.object.xstars)

        val_acq=self.acq_func.acq_kind(x_max,self.gp)
        
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp)
            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        super(BayesOpt, self).augment_the_new_data(x_max)

        
        # convert ystar to original scale
        y_stars=[val*np.std(self.Y_original)+np.mean(self.Y_original) for idx,val in enumerate(y_stars)]
        
        self.ystars.append(np.ravel(y_stars))

    def maximize_e3i(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
               
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()                
            
        # run the acquisition function for the first time to get xstar
        
        self.xstars=[]
        # finding the xt of UCB
        
        y_max=np.max(self.Y)
        numXtar=50*self.dim
        #numXtar=20
        
        y_stars=[]
        temp=[]
        # finding the xt of Thompson Sampling
        for ii in range(numXtar):
            xt_TS,y_xt_TS=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,
                                                acq_name="thompson",IsReturnY=True)
            
            y_stars.append(y_xt_TS)
            
            temp.append(xt_TS)
            # check if f* > y^max and ignore xt_TS otherwise
            #if y_xt_TS>=y_max:
            self.xstars.append(xt_TS)
        
       
        #y_stars=y_stars*np.std(self.Y_original)+np.mean(self.Y_original)
        #print "{:.5f} {:.6f}".format(np.mean(y_stars),np.std(y_stars))
        #print "ymax={:.5f}".format(np.max(self.Y))
        
        if self.acq['debug']==1:
            print('mean y*={:.4f}({:.8f}) y+={:.4f}'.format(np.mean(y_xt_TS),np.std(y_xt_TS),y_max))
            
        if self.xstars==[]:
            #print 'xt_suggestion is empty'
            # again perform TS and take all of them
            self.xstars=temp
            
        self.acq['xstars']=self.xstars   
        self.acq['ystars']=y_stars   

        self.acq_func = AcquisitionFunction(self.acq)
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)

        #xstars_array=np.asarray(self.acq_func.object.xstars)

        val_acq=self.acq_func.acq_kind(x_max,self.gp)
        
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp)
            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        super(BayesOpt, self).augment_the_new_data(x_max)

        
    def maximize_pvrs(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return

                
        if 'n_xstars' in self.acq:
            numXstar=self.acq['n_xstars']
        else:
            numXstar=10*self.dim
   
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()                
        
        # run the acquisition function for the first time to get xstar
        
        self.xstars=[]
        # finding the xt of UCB
        
        numTheta=len(self.theta_vector)
        temp=[]
        # finding the xt of Thompson Sampling
        for ii in range(numXstar):
            if self.theta_vector!=[]:
                # since the numXstar > len(theta_vector)
 
                index=np.random.randint(numTheta)
                #print index
                gp_params['theta']=self.theta_vector[index]    

            # init a new Gaussian Process
            self.gp=GaussianProcess(self.gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])       
        
            xt_TS,y_xt_TS=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,
                                                acq_name="thompson",IsReturnY=True)
            
            
            temp.append(xt_TS)
            # check if f* > y^max and ignore xt_TS otherwise
            #if y_xt_TS>=y_max:
                #self.xstars.append(xt_TS)

        if self.xstars==[]:
            #print 'xt_suggestion is empty'
            # again perform TS and take all of them
            self.xstars=temp       
        
        # check predictive variance before adding a new data points
        var_before=self.gp.compute_var(self.gp.X,self.xstars) 
        var_before=np.mean(var_before)
        
        if self.xstar_accumulate==[]:
            self.xstar_accumulate=np.asarray(self.xstars)
        else:
            self.xstar_accumulate=np.vstack((self.xstar_accumulate,np.asarray(self.xstars)))


        accum_var_before=[self.gp.compute_var(self.gp.X,val)  for idx,val in enumerate(self.xstar_accumulate)]
        accum_var_before=np.mean(accum_var_before)
        
        
        self.gp.lengthscale_vector=self.theta_vector
        self.acq['xstars']=self.xstars    
        self.acq_func = AcquisitionFunction(self.acq)
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)
        #xstars_array=np.asarray(self.acq_func.object.xstars)
    
        val_acq=-self.acq_func.acq_kind(x_max,self.gp)
        
        
        # check predictive variance after
        temp=np.vstack((self.gp.X,x_max))
        var_after=self.gp.compute_var(temp,self.xstars) 
        var_after=np.mean(var_after)
        
        
        accum_var_after=[self.gp.compute_var(temp,val)  for idx,val in enumerate(self.xstar_accumulate)]
        accum_var_after=np.mean(accum_var_after)
        
        if self.PVRS_before_after==[]:
            self.PVRS_before_after=np.asarray([var_before,var_after])
            self.accummulate_PVRS_before_after=np.asarray([accum_var_before,accum_var_after])

        else:
            self.PVRS_before_after=np.vstack((self.PVRS_before_after,np.asarray([var_before,var_after])))  
            self.accummulate_PVRS_before_after=np.vstack((self.accummulate_PVRS_before_after,np.asarray([accum_var_before,accum_var_after])))        

        #print "predictive variance before={:.12f} after={:.12f} val_acq={:.12f}".format(var_before,var_after,np.asscalar(val_acq))

        # check maximum variance
        var_acq={}
        var_acq['name']='pure_exploration'
        var_acq['dim']=self.dim
        var_acq['scalebounds']=self.scalebounds     
        acq_var=AcquisitionFunction(var_acq)
        temp = acq_max(ac=acq_var.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox='scipy')
        
        # get the value f*
        #max_var_after=acq_var.acq_kind(temp,self.gp,y_max=y_max)
        #print "max predictive variance ={:.8f}".format(np.asscalar(max_var_after))

        
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp)
            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
                
        #mean,var=self.gp.predict(x_max, eval_MSE=True)
        #var.flags['WRITEABLE']=True
        #var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        super(BayesOpt, self).augment_the_new_data(x_max)


#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
