# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 12:34:13 2016

@author: V
"""

# define Gaussian Process class


import numpy as np
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from scipy.optimize import minimize

from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
#from eucl_dist.cpu_dist import dist
from sklearn.cluster import KMeans
import scipy.linalg as spla
from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name


from scipy.spatial.distance import squareform

class TransformedGP(object):
    # transform GP given known optimum value: f = f^* - 1/2 g^2
    def __init__ (self,param):
        # init the model
    
        # theta for RBF kernel exp( -theta* ||x-y||)
        if 'kernel' not in param:
            param['kernel']='SE'
            
        kernel_name=param['kernel']
        if kernel_name not in ['SE','ARD']:
            err = "The kernel function " \
                  "{} has not been implemented, " \
                  "please choose one of the kernel SE ARD.".format(kernel_name)
            raise NotImplementedError(err)
        else:
            self.kernel_name = kernel_name
            
        if 'flagIncremental' not in param:
            self.flagIncremental=0
        else:
            self.flagIncremental=param['flagIncremental']
            
        if 'lengthscale' not in param:
            self.lengthscale=param['theta']
        else:
            self.lengthscale=param['lengthscale']
            self.theta=self.lengthscale

        if 'lengthscale_vector' not in param: # for marginalize hyperparameters
            self.lengthscale_vector=[]
        else:
            self.lengthscale_vector=param['lengthscale_vector']
            
        #self.theta=param['theta']
        
        self.gp_params=param
        self.nGP=0
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
        self.KK_x_x_inv=[]
    
        self.fstar=0
        self.X=[]
        self.Y=[]
        self.G=[]
        self.lengthscale_old=self.lengthscale
        self.flagOptimizeHyperFirst=0
        
        self.alpha=[] # for Cholesky update
        self.L=[] # for Cholesky update LL'=A

    def kernel_dist(self, a,b,lengthscale):
        
        if self.kernel_name == 'ARD':
            return self.ARD_dist_func(a,b,lengthscale)
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(a,b)
            return np.exp(-np.square(Euc_dist)/lengthscale)
        
    def ARD_dist_func(self,A,B,length_scale):
        mysum=0
        for idx,val in enumerate(length_scale):
            mysum=mysum+((A[idx]-B[idx])**2)*1.0/val
        dist=np.exp(-mysum)
        return dist

        
    def fit(self,X,Y,fstar):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points 
        y: the outcome y=f(x)
        
        """ 
        ur = unique_rows(X)
        X=X[ur]
        Y=Y[ur]
        
        self.X=X
        self.Y=Y
        self.fstar=fstar
        self.G=np.sqrt(2.0*(fstar-Y))
        #self.G=np.log(1.0*(fstar-Y))
        
        
        
        #KK=pdist(self.X,lambda a,b: self.ARD_dist_func(a,b,self.theta))
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(X,X)
            self.KK_x_x=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(len(X))*self.noise_delta
        else:
            KK=pdist(self.X,lambda a,b: self.kernel_dist(a,b,self.lengthscale)) 
            KK=squareform(KK)
            self.KK_x_x=KK+np.eye(self.X.shape[0])*(1+self.noise_delta)
            
        #Euc_dist=euclidean_distances(X,X)
        #self.KK_x_x=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x")
        
        self.KK_x_x_inv=np.linalg.pinv(self.KK_x_x)
        self.L=np.linalg.cholesky(self.KK_x_x)
        #temp=np.linalg.solve(self.L,self.Y)
        
        tempG=np.linalg.solve(self.L,self.G-np.sqrt(2*self.fstar))
        #self.alpha=np.linalg.solve(self.L.T,temp)
        self.alphaG=np.linalg.solve(self.L.T,tempG)
        

    
    def log_marginal_lengthscale(self,lengthscale,noise_delta):
        """
        Compute Log Marginal likelihood of the GP model w.r.t. the provided lengthscale
        """

        def compute_log_marginal(X,lengthscale,noise_delta):
            # compute K
            ur = unique_rows(self.X)
            myX=self.X[ur]
            #myY=np.sqrt(0.5*(self.fstar-self.Y[ur]))
            myY=self.Y[ur]
            
            if self.flagOptimizeHyperFirst==0:
                if self.kernel_name=='SE':
                    self.Euc_dist_X_X=euclidean_distances(myX,myX)
                    KK=np.exp(-np.square(self.Euc_dist_X_X)/lengthscale)+np.eye(len(myX))*self.noise_delta
                else:
                    KK=pdist(myX,lambda a,b: self.kernel_dist(a,b,lengthscale))
                    KK=squareform(KK)
                    KK=KK+np.eye(myX.shape[0])*(1+noise_delta)
                self.flagOptimizeHyperFirst=1
            else:
                if self.kernel_name=='SE':
                    KK=np.exp(-np.square(self.Euc_dist_X_X)/lengthscale)+np.eye(len(myX))*self.noise_delta
                else:
                    KK=pdist(myX,lambda a,b: self.kernel_dist(a,b,lengthscale))
                    KK=squareform(KK)
                    KK=KK+np.eye(myX.shape[0])*(1+noise_delta)

            try:
                temp_inv=np.linalg.solve(KK,myY)
            except: # singular
                return -np.inf
            
            
            try:
                #logmarginal=-0.5*np.dot(self.Y.T,temp_inv)-0.5*np.log(np.linalg.det(KK+noise_delta))-0.5*len(X)*np.log(2*3.14)
                first_term=-0.5*np.dot(myY.T,temp_inv)
                
                # if the matrix is too large, we randomly select a part of the data for fast computation
                if KK.shape[0]>200:
                    idx=np.random.permutation(KK.shape[0])
                    idx=idx[:200]
                    KK=KK[np.ix_(idx,idx)]
                #Wi, LW, LWi, W_logdet = pdinv(KK)
                #sign,W_logdet2=np.linalg.slogdet(KK)
                chol  = spla.cholesky(KK, lower=True)
                W_logdet=np.sum(np.log(np.diag(chol)))
                # Uses the identity that log det A = log prod diag chol A = sum log diag chol A
    
                #second_term=-0.5*W_logdet2
                second_term=-W_logdet
            except: # singular
                return -np.inf
            
            #print "first term ={:.4f} second term ={:.4f}".format(np.asscalar(first_term),np.asscalar(second_term))

            logmarginal=first_term+second_term-0.5*len(myY)*np.log(2*3.14)
                
            if np.isnan(np.asscalar(logmarginal))==True:
                print("theta={:s} first term ={:.4f} second  term ={:.4f}".format(lengthscale,np.asscalar(first_term),np.asscalar(second_term)))
                #print temp_det

            return np.asscalar(logmarginal)
        
        #print lengthscale
        logmarginal=0
        
        if np.isscalar(lengthscale):
            logmarginal=compute_log_marginal(self.X,lengthscale,noise_delta)
            return logmarginal

        if not isinstance(lengthscale,list) and len(lengthscale.shape)==2:
            logmarginal=[0]*lengthscale.shape[0]
            for idx in range(lengthscale.shape[0]):
                logmarginal[idx]=compute_log_marginal(self.X,lengthscale[idx],noise_delta)
        else:
            logmarginal=compute_log_marginal(self.X,lengthscale,noise_delta)
                
        #print logmarginal

        return logmarginal
    
    
    def leave_one_out_lengthscale(self,lengthscale,noise_delta):
        """
        Compute Log Marginal likelihood of the GP model w.r.t. the provided lengthscale
        """

        def compute_loo_predictive(X,lengthscale,noise_delta):
            # compute K
            ur = unique_rows(self.X)
            myX=self.X[ur]
            myY=self.Y[ur]
            D=np.hstack((myX,myY.reshape(-1,1)))
            LOO_sum=0
            for i in range(0,D.shape[0]):
                D_train=np.delete(D,i,0)
                D_test=D[i,:]
                Xtrain=D_train[:,:-1]
                Ytrain=D_train[:,-1]
                Xtest=D_test[:-1]
                Ytest=D_test[-1]
                gp_params= {'theta':lengthscale,'noise_delta':self.noise_delta}
                gp=TransformedGP(gp_params)
                
                try: # if SVD problem
                    gp.fit(Xtrain, Ytrain)
                    mu, sigma2 = gp.predict(Xtest, eval_MSE=True)
                    logpred=-np.log(np.sqrt(2*3.14))-(2)*np.log(sigma2)-np.square(Ytest-mu)/(2*sigma2)
                except:
                    logpred=-999999
                
                LOO_sum+=logpred
            #return np.asscalar(LOO_sum)
            return LOO_sum
        
        #print lengthscale
        logpred=0
        
        if np.isscalar(lengthscale):
            logpred=compute_loo_predictive(self.X,lengthscale,noise_delta)
            return logpred

        if not isinstance(lengthscale,list) and len(lengthscale.shape)==2:
            logpred=[0]*lengthscale.shape[0]
            for idx in range(lengthscale.shape[0]):
                logpred[idx]=compute_loo_predictive(self.X,lengthscale[idx],noise_delta)
        else:
            logpred=compute_loo_predictive(self.X,lengthscale,noise_delta)
                
        #print logmarginal
        return logpred
    
    def slice_sampling_lengthscale_SE(self,previous_theta,noise_delta,nSamples=10):
        
        print("slice sampling lengthscale")

        nBurnins=1
        # define a bound on the lengthscale
        bounds_lengthscale_min=0.000001*self.dim
        bounds_lengthscale_max=1*self.dim
        mybounds=np.asarray([bounds_lengthscale_min,bounds_lengthscale_max]).T
        
        count=0
        lengthscale_samples=[0]*nSamples
        
        # init x
        x0=np.random.uniform(mybounds[0],mybounds[1],1)
                    
        # marginal_llk at x0
        self.flagOptimizeHyperFirst=0
        y_marginal_llk=self.log_marginal_lengthscale(x0,noise_delta)
        y=np.random.uniform(0,y_marginal_llk,1)

        cut_min=0
        count_reject=0

        # burnins
        while(count<nBurnins and count_reject<=5):

            # sampling x
            x=np.random.uniform(mybounds[0],mybounds[1],1)
                        
            # get f(x)
            new_y_marginal_llk=self.log_marginal_lengthscale(x,noise_delta)
            
            if new_y_marginal_llk>=y: # accept
                #lengthscale_samples[count]=x
                # sampling y
                y=np.random.uniform(cut_min,new_y_marginal_llk,1)
                cut_min=y
                count=count+1
            else:
                count_reject=count_reject+1
        
        count=0
        count_reject=0

        while(count<nSamples):
            # sampling x
            x=np.random.uniform(mybounds[0],mybounds[1],1)
                        
            # get f(x)
            new_y_marginal_llk=self.log_marginal_lengthscale(x,noise_delta)
            
            if new_y_marginal_llk>=y: # accept
                lengthscale_samples[count]=np.asscalar(x)

                # sampling y
                y=np.random.uniform(cut_min,new_y_marginal_llk,1)
                cut_min=y
                count=count+1
            else:
                count_reject=count_reject+1
                
            if count_reject>=3*nSamples:
                lengthscale_samples[count:]=[lengthscale_samples[count-1]]*(nSamples-count)
                break
            
        #print lengthscale_samples 
        if any(lengthscale_samples)==0:
            lengthscale_samples=[previous_theta]*nSamples
        return np.asarray(lengthscale_samples)            
    
    def optimize_lengthscale_SE_loo(self,previous_theta,noise_delta):
        """
        Optimize to select the optimal lengthscale parameter
        """
        
        #print("maximizing lengthscale LOO")
        dim=self.X.shape[1]
        
        # define a bound on the lengthscale
        bounds_lengthscale_min=0.000001*dim
        bounds_lengthscale_max=1*dim
        mybounds=[np.asarray([bounds_lengthscale_min,bounds_lengthscale_max]).T]
       
        
        lengthscale_tries = np.random.uniform(bounds_lengthscale_min, bounds_lengthscale_max,size=(1000*dim, 1))        
        lengthscale_cluster = KMeans(n_clusters=10*dim, random_state=0).fit(lengthscale_tries)

        #print lengthscale_cluster.cluster_centers_
        lengthscale_tries=np.vstack((lengthscale_cluster.cluster_centers_,previous_theta,bounds_lengthscale_min))
        
        #print lengthscale_tries

        # evaluate
        self.flagOptimizeHyperFirst=0 # for efficiency

        logmarginal_tries=self.leave_one_out_lengthscale(lengthscale_tries,noise_delta)
        #print logmarginal_tries

        #find x optimal for init
        idx_max=np.argmax(logmarginal_tries)
        lengthscale_init_max=lengthscale_tries[idx_max]
        #print lengthscale_init_max
        
        myopts ={'maxiter':10,'maxfun':10}

        x_max=[]
        max_log_marginal=None
        
        for i in range(dim):
            res = minimize(lambda x: -self.leave_one_out_lengthscale(x,noise_delta),lengthscale_init_max,
                           bounds=mybounds,method="L-BFGS-B",options=myopts)#L-BFGS-B
            if 'x' not in res:
                val=self.leave_one_out_lengthscale(res,noise_delta)    
            else:
                val=self.leave_one_out_lengthscale(res.x,noise_delta)  
            
            # Store it if better than previous minimum(maximum).
            if max_log_marginal is None or val >= max_log_marginal:
                if 'x' not in res:
                    x_max = res
                else:
                    x_max = res.x
                max_log_marginal = val
            #print res.x
        return x_max
    
    
    
        
    def optimize_lengthscale_SE_maximizing(self,previous_theta,noise_delta):
        """
        Optimize to select the optimal lengthscale parameter
        """
        
        #print("maximizing lengthscale")
        dim=self.X.shape[1]
        
        # define a bound on the lengthscale
        bounds_lengthscale_min=0.0000001
        bounds_lengthscale_max=1*dim
        mybounds=[np.asarray([bounds_lengthscale_min,bounds_lengthscale_max]).T]
       
        
        lengthscale_tries = np.random.uniform(bounds_lengthscale_min, bounds_lengthscale_max,size=(1000*dim, 1))        
        lengthscale_cluster = KMeans(n_clusters=10*dim, random_state=0).fit(lengthscale_tries)

        #print lengthscale_cluster.cluster_centers_
        lengthscale_tries=np.vstack((lengthscale_cluster.cluster_centers_,previous_theta,bounds_lengthscale_min))
        
        #print lengthscale_tries

        # evaluate
        self.flagOptimizeHyperFirst=0 # for efficiency

        logmarginal_tries=self.log_marginal_lengthscale(lengthscale_tries,noise_delta)
        #print logmarginal_tries

        #find x optimal for init
        idx_max=np.argmax(logmarginal_tries)
        lengthscale_init_max=lengthscale_tries[idx_max]
        #print lengthscale_init_max
        
        myopts ={'maxiter':10,'maxfun':10}

        x_max=[]
        max_log_marginal=None
        
        for i in range(1):
            res = minimize(lambda x: -self.log_marginal_lengthscale(x,noise_delta),lengthscale_init_max,
                           bounds=mybounds,method="L-BFGS-B",options=myopts)#L-BFGS-B
            if 'x' not in res:
                val=self.log_marginal_lengthscale(res,noise_delta)    
            else:
                val=self.log_marginal_lengthscale(res.x,noise_delta)  
            
            # Store it if better than previous minimum(maximum).
            if max_log_marginal is None or val >= max_log_marginal:
                if 'x' not in res:
                    x_max = res
                else:
                    x_max = res.x
                max_log_marginal = val
            #print res.x
        return x_max
    
    

    def optimize_lengthscale(self,previous_theta,noise_delta):
        if self.kernel_name == 'ARD':
            return self.optimize_lengthscale_ARD(previous_theta,noise_delta)
        if self.kernel_name=='SE':
            return self.optimize_lengthscale_SE_maximizing(previous_theta,noise_delta)

    

    def compute_var(self,X,xTest):
        """
        compute variance given X and xTest
        
        Input Parameters
        ----------
        X: the observed points
        xTest: the testing points 
        
        Returns
        -------
        diag(var)
        """ 
        
        xTest=np.asarray(xTest)
        xTest=np.atleast_2d(xTest)
        if self.kernel_name=='SE':
            #Euc_dist=euclidean_distances(xTest,xTest)
            #KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
            ur = unique_rows(X)
            X=X[ur]
            if xTest.shape[0]<=800:
                Euc_dist_test_train=euclidean_distances(xTest,X)
                #Euc_dist_test_train=dist(xTest, X, matmul='gemm', method='ext', precision='float32')
                KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
            else:
                KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))

            Euc_dist_train_train=euclidean_distances(X,X)
            self.KK_bucb_train_train=np.exp(-np.square(Euc_dist_train_train)/self.lengthscale)+np.eye(X.shape[0])*self.noise_delta        
        else:
            #KK=pdist(xTest,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            #KK=squareform(KK)
            #KK_xTest_xTest=KK+np.eye(xTest.shape[0])*(1+self.noise_delta)
            ur = unique_rows(X)
            X=X[ur]
            KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            self.KK_bucb_train_train=cdist(X,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))+np.eye(X.shape[0])*self.noise_delta
        try:
            temp=np.linalg.solve(self.KK_bucb_train_train,KK_xTest_xTrain.T)
        except:
            temp=np.linalg.lstsq(self.KK_bucb_train_train,KK_xTest_xTrain.T, rcond=-1)
            temp=temp[0]
            
        #var=KK_xTest_xTest-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.eye(xTest.shape[0])-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.diag(var)
        var.flags['WRITEABLE']=True
        var[var<1e-100]=0
        return var 


    def predict_g2(self,xTest,eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]))
        
        # prevent singular matrix
        ur = unique_rows(self.X)
        X=self.X[ur]
        Y=self.Y[ur]
        G=self.G[ur]
    
        #KK=pdist(xTest,lambda a,b: self.ARD_dist_func(a,b,self.theta))
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(xTest,xTest)
            KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
            
            Euc_dist_test_train=euclidean_distances(xTest,X)
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
        else:
            KK=pdist(xTest,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            KK=squareform(KK)
            KK_xTest_xTest=KK+np.eye(xTest.shape[0])+np.eye(xTest.shape[0])*self.noise_delta
            KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
        
        """
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        mean=np.dot(temp,Y)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)
        """
        
        
        # using Cholesky update
        #mean=np.dot(KK_xTest_xTrain,self.alpha)
        meanG=np.dot(KK_xTest_xTrain,self.alphaG)

        #v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        #var=KK_xTest_xTest-np.dot(v.T,v)
        
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        varG=KK_xTest_xTest-np.dot(v.T,v)
        
        
        # compute mF, varF
        mf=self.fstar-0.5*meanG*meanG
        varf=meanG*varG*meanG
        #varf=varG

        return mf.ravel(),np.diag(varf)     
    
    def predict(self,xTest,eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]))
        
        # prevent singular matrix
        ur = unique_rows(self.X)
        X=self.X[ur]
        Y=self.Y[ur]
        #Gtest=np.log(1.0*(self.fstar-))
    
        #KK=pdist(xTest,lambda a,b: self.ARD_dist_func(a,b,self.theta))
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(xTest,xTest)
            KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
            
            Euc_dist_test_train=euclidean_distances(xTest,X)
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
        else:
            KK=pdist(xTest,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            KK=squareform(KK)
            KK_xTest_xTest=KK+np.eye(xTest.shape[0])+np.eye(xTest.shape[0])*self.noise_delta
            KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
        
        """
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        mean=np.dot(temp,Y)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)
        """
        
        
        # using Cholesky update
        #mean=np.dot(KK_xTest_xTrain,self.alpha)
        meanG=np.dot(KK_xTest_xTrain,self.alphaG)+np.sqrt(2*self.fstar) # non zero prior mean

        #v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        #var=KK_xTest_xTest-np.dot(v.T,v)
        
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        varG=KK_xTest_xTest-np.dot(v.T,v)
        
        
        # compute mF, varF
        mf=self.fstar-0.5*np.square(meanG)
        #mf=self.fstar-np.exp(meanG)
        
        # using linearlisation
        varf=meanG*varG*meanG 

        # using moment matching
        
        """
        temp=np.diag(varG)
        temp=np.atleast_2d(temp)
        temp=np.reshape(temp,(-1,1))
        
        temp2=np.square(meanG)
        temp2=np.atleast_2d(temp2)
        temp2=np.reshape(temp2,(-1,1))

        mf=self.fstar-0.5*(temp2+temp)
        varf=0.5*varG*varG+meanG*varG*meanG 
        """

        return mf.ravel(),np.diag(varf)  

    def predict_G(self,xTest,eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]))
        
        # prevent singular matrix
        ur = unique_rows(self.X)
        X=self.X[ur]
        Y=self.Y[ur]
        G=self.G[ur]
    
        #KK=pdist(xTest,lambda a,b: self.ARD_dist_func(a,b,self.theta))
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(xTest,xTest)
            KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
            
            Euc_dist_test_train=euclidean_distances(xTest,X)
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
        else:
            KK=pdist(xTest,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
            KK=squareform(KK)
            KK_xTest_xTest=KK+np.eye(xTest.shape[0])+np.eye(xTest.shape[0])*self.noise_delta
            KK_xTest_xTrain=cdist(xTest,X,lambda a,b: self.kernel_dist(a,b,self.lengthscale))
        
        """
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        mean=np.dot(temp,Y)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)
        """
        
        
        meanG=np.dot(KK_xTest_xTrain,self.alphaG)+np.sqrt(2*self.fstar) # non zero prior mean

        #v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        #var=KK_xTest_xTest-np.dot(v.T,v)
        
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        varG=KK_xTest_xTest-np.dot(v.T,v)
        


        return meanG.ravel(),np.diag(varG)  

    
    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
        
    
