# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:22:32 2016

@author: Vu
"""
from __future__ import division

import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'..')
import numpy as np
#import mayavi.mlab as mlab
#from scipy.stats import norm
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from prada_bayes_opt import PradaBayOptFn
#from prada_bayes_opt import PradaBayOptBatch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics.pairwise import euclidean_distances
from prada_bayes_opt.acquisition_maximization import acq_max
from scipy.stats import norm as norm_dist

import random
from prada_bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
import os
from pylab import *

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.7),
                  (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0))}

#my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#my_cmap = plt.get_cmap('cubehelix')
my_cmap = plt.get_cmap('Blues')

        
counter = 0

#class Visualization(object):
    
    #def __init__(self,bo):
       #self.plot_gp=0     
       #self.posterior=0
       #self.myBo=bo
       
        
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)
    
def plot_histogram(bo,samples):
    if bo.dim==1:
        plot_histogram_1d(bo,samples)
    if bo.dim==2:
        plot_histogram_2d(bo,samples)

def plot_mixturemodel(g,bo,samples):
    if bo.dim==1:
        plot_mixturemodel_1d(g,bo,samples)
    if bo.dim==2:
        plot_mixturemodel_2d(g,bo,samples)

def plot_mixturemodel_1d(g,bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]

    x_plot = np.linspace(np.min(samples), np.max(samples), len(samples))
    x_plot = np.reshape(x_plot,(len(samples),-1))
    y_plot = g.score_samples(x_plot)[0]
    
    x_plot_ori = np.linspace(np.min(samples_original), np.max(samples_original), len(samples_original))
    x_plot_ori=np.reshape(x_plot_ori,(len(samples_original),-1))
    
    
    fig=plt.figure(figsize=(8, 3))

    plt.plot(x_plot_ori, np.exp(y_plot), color='red')
    plt.xlim(bo.bounds[0,0],bo.bounds[0,1])
    plt.xlabel("X",fontdict={'size':16})
    plt.ylabel("f(X)",fontdict={'size':16})
    plt.title("IGMM Approximation",fontsize=16)
        
def plot_mixturemodel_2d(dpgmm,bo,samples):
    
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    dpgmm_means_original=dpgmm.truncated_means_*bo.max_min_gap+bo.bounds[:,0]

    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myGmm=fig.add_subplot(1,1,1)  

    x1 = np.linspace(bo.scalebounds[0,0],bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0],bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    x_plot=np.c_[x1g.flatten(), x2g.flatten()]
    
    y_plot2 = dpgmm.score_samples(x_plot)[0]
    y_plot2=np.exp(y_plot2)
    #y_label=dpgmm.predict(x_plot)[0]
    
    x1_ori = np.linspace(bo.bounds[0,0],bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0],bo.bounds[1,1], 100)
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)

    CS_acq=myGmm.contourf(x1g_ori,x2g_ori,y_plot2.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    myGmm.scatter(dpgmm_means_original[:,0],dpgmm_means_original[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    myGmm.set_title('IGMM Approximation',fontsize=16)
    myGmm.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    myGmm.set_ylim(bo.bounds[1,0],bo.bounds[1,1])
    myGmm.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        
def plot_acq_bo_1d(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(12, 8))
    #fig.title('Bayesian Optimization with Different Acquisition Functions', fontdict={'size':20})
    
    gs = gridspec.GridSpec(6, 1, height_ratios=[3, 1,1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    acq_TS = plt.subplot(gs[3])
    
    #acq_TS2 = plt.subplot(gs[5])
    acq_ES = plt.subplot(gs[4])
    acq_PES = plt.subplot(gs[5])
    #acq_MRS = plt.subplot(gs[6])
    
    #acq_Consensus = plt.subplot(gs[7])


    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})
    axis.set_title('Bayesian Optimization with Different Acquisition Functions', fontdict={'size':20})
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

       
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    #acq_UCB.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_UCB.set_xlim((np.min(x_original), np.max(x_original)))
    acq_UCB.set_ylabel('UCB', fontdict={'size':16})
    acq_UCB.set_xlabel('x', fontdict={'size':16})
    
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    acq_EI.set_xlabel('x', fontdict={'size':16})
    
    
    
    # TS 
    acq_func={}
    acq_func['name']='thompson'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_TS.plot(x_original, utility, label='Utility Function', color='purple')
    acq_TS.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_POI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_TS.set_xlim((np.min(x_original), np.max(x_original)))
    acq_TS.set_ylabel('TS', fontdict={'size':16})
    acq_TS.set_xlabel('x', fontdict={'size':16})
    
	
	
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_EI.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    """
    # MRS     
    acq_func={}
    acq_func['name']='mrs'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_MRS.plot(x_original, utility, label='Utility Function', color='purple')
    acq_MRS.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_MRS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_MRS.set_xlim((np.min(x_original), np.max(x_original)))
    acq_MRS.set_ylabel('MRS', fontdict={'size':16})
    acq_MRS.set_xlabel('x', fontdict={'size':16})
	"""

    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_PES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_PES.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_PES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_PES.set_xlim((np.min(x_original), np.max(x_original)))
    acq_PES.set_ylabel('PES', fontdict={'size':16})
    acq_PES.set_xlabel('x', fontdict={'size':16})
     
    # TS1   
    """
    acq_func={}
    acq_func['name']='consensus'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_Consensus.plot(x_original, utility, label='Utility Function', color='purple')


    temp=np.asarray(myacq.object.xt_suggestions)
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Consensus.plot(xt_suggestion_original, [np.max(utility)]*xt_suggestion_original.shape[0], 's', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)
   
    max_point=np.max(utility)
    
    acq_Consensus.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        
    #acq_TS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_Consensus.set_xlim((np.min(x_original), np.max(x_original)))
    #acq_TS.set_ylim((np.min(utility)*0.9, np.max(utility)*1.1))
    acq_Consensus.set_ylabel('Consensus', fontdict={'size':16})
    acq_Consensus.set_xlabel('x', fontdict={'size':16})
    """

    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq_ES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ES.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_ES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_ES.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ES.set_ylabel('ES', fontdict={'size':16})
    acq_ES.set_xlabel('x', fontdict={'size':16})
    
    strFileName="{:d}_GP_acquisition_functions.pdf".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

	
def plot_bo_1d(bo):
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(8, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq.plot(x_original, utility, label='Utility Function', color='purple')
    acq.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    acq.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    acq.set_ylabel('Acq', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_bo_1d_variance(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    
    #fig=plt.figure(figsize=(8, 5))
    fig, ax1 = plt.subplots(figsize=(8.5, 4))

    mu, sigma = bo.posterior(x)
    mu_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))


    def distance_function(x,X):            
        Euc_dist=euclidean_distances(x,X)
          
        dist=Euc_dist.min(axis=1)
        return dist
        
    utility_distance=distance_function(x.reshape((-1, 1)),bo.X)
    idxMaxVar=np.argmax(utility)
    #idxMaxVar=[idx for idx,val in enumerate(utility) if val>=0.995]
    ax1.plot(x_original, utility, label='GP $\sigma(x)$', color='purple')  

    
    ax1.scatter(x_original[idxMaxVar], utility[idxMaxVar], marker='s',label='x=argmax $\sigma(x)$', color='blue',linewidth=2)            
          
    #ax1.scatter(x_original[idxMaxVar], utility[idxMaxVar], label='$||x-[x]||$', color='blue',linewidth=2)            

    ax1.plot(bo.X_original.flatten(), [0]*len(bo.X_original.flatten()), 'D', markersize=10, label=u'Observations', color='r')


    idxMaxDE=np.argmax(utility_distance)
    ax2 = ax1.twinx()
    ax2.plot(x_original, utility_distance, label='$d(x)=||x-[x]||^2$', color='black') 
    ax2.plot(x_original[idxMaxDE], utility_distance[idxMaxDE], 'o',label='x=argmax d(x)', color='black',markersize=10)            
           
    ax2.set_ylim((0, 0.45))


         
    ax1.set_xlim((np.min(x_original)-0.01, 0.01+np.max(x_original)))
    ax1.set_ylim((-0.02, np.max(utility) + 0.05))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    ax1.set_ylabel(ur'$\sigma(x)$', fontdict={'size':18})
    ax2.set_ylabel('d(x)', fontdict={'size':18})

    ax1.set_xlabel('x', fontdict={'size':18})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #ax1.legend(loc=2, bbox_to_anchor=(1.1, 1), borderaxespad=0.,fontsize=14)
    #ax2.legend(loc=2, bbox_to_anchor=(1.1, 0.3), borderaxespad=0.,fontsize=14)

    plt.title('Exploration by GP variance vs distance',fontsize=22)
    ax1.legend(loc=3, bbox_to_anchor=(0.05,-0.32,1, -0.32), borderaxespad=0.,fontsize=14,ncol=4)
    ax2.legend(loc=3, bbox_to_anchor=(0.05,-0.46,1, -0.46), borderaxespad=0.,fontsize=14,ncol=2)

    #plt.legend(fontsize=14)
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\demo_geometric"

    strFileName="{:d}_var_DE.eps".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')
    
def plot_acq_bo_2d(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 80)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 80)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 80)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 80)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

    #y_original = func(x_original)
    #y = func(x)
    #y_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(14, 20))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1]) 
    axis_mean2d = fig.add_subplot(4, 2, 1)
    axis_variance2d = fig.add_subplot(4, 2, 2)
    acq_UCB = fig.add_subplot(4, 2, 3)
    acq_EI =fig.add_subplot(4, 2,4)
    #acq_POI = plt.subplot(gs[3])
    

    acq_ES = fig.add_subplot(4, 2, 5)
    acq_PES = fig.add_subplot(4, 2, 6)
    acq_MRS = fig.add_subplot(4, 2, 7)
    acq_Consensus = fig.add_subplot(4, 2, 8)

    
    mu, sigma = bo.posterior(X)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    #mu_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    #sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)**2
    
    
    # mean
    CS=axis_mean2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_mean2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_mean2d.set_title('Gaussian Process Mean',fontsize=16)
    axis_mean2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_mean2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_mean2d, shrink=0.9)
    


    # variance
    CS=axis_variance2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis_variance2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis_variance2d.set_title('Gaussian Process Variance',fontsize=16)
    axis_variance2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis_variance2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis_variance2d, shrink=0.9)

    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp, np.max(bo.Y))
    CS_acq=acq_UCB.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_UCB.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_UCB.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_UCB.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')
    
    xt_UCB=X[idxBest,:]
    
    acq_UCB.set_title('UCB',fontsize=16)
    acq_UCB.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_UCB.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_UCB, shrink=0.9)
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp, np.max(bo.Y))
    CS_acq=acq_EI.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_EI.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_EI.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_EI.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_EI=X[idxBest,:]


    acq_EI.set_title('EI',fontsize=16)
    acq_EI.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_EI.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_EI, shrink=0.9)
    
    # MRS         
    acq_func={}
    acq_func['name']='mrs'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp, np.max(bo.Y))
    CS_acq=acq_MRS.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_MRS.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_MRS.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_MRS.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_MRS.set_title('MRS',fontsize=16)
    acq_MRS.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_MRS.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_MRS, shrink=0.9)
	

    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp, np.max(bo.Y))
    CS_acq=acq_PES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_PES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_PES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_PES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_PES=X[idxBest,:]


    acq_PES.set_title('PES',fontsize=16)
    acq_PES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_PES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_PES, shrink=0.9)
    
    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(X, bo.gp, np.max(bo.Y))
    CS_acq=acq_ES.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_ES.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')
    acq_ES.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    xt_ES=X[idxBest,:]


    acq_ES.set_title('ES',fontsize=16)
    acq_ES.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_ES.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_ES, shrink=0.9)
    
    xt_suggestions=[]
    xt_suggestions.append(xt_UCB)
    xt_suggestions.append(xt_EI)
    xt_suggestions.append(xt_ES)
    xt_suggestions.append(xt_PES)
    
    # Consensus     
    acq_func={}
    acq_func['name']='consensus'
    acq_func['kappa']=2
    acq_func['dim']=2
    acq_func['scalebounds']=bo.scalebounds
    acq_func['xt_suggestions']=xt_suggestions

    myacq=AcquisitionFunction(acq_func)

    utility = myacq.acq_kind(X, bo.gp, np.max(bo.Y))
    CS_acq=acq_Consensus.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq_Consensus.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    temp=np.asarray(myacq.object.xt_suggestions) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Consensus.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],marker='s',color='y',s=100,label='xt_suggestions')
    acq_Consensus.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='*',color='r',s=300,label='Peak')

    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    #acq_ES.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq_Consensus.set_title('Consensus',fontsize=16)
    acq_Consensus.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq_Consensus.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS_acq, ax=acq_Consensus, shrink=0.9)
    
    strFileName="{:d}_GP2d_acquisition_functions.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Selected')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_bo_2d_unbounded(bo,myfunction):

    global counter
    counter=counter+1
    
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\plot_Nov_2016"

    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
        
  
    fig = plt.figure(figsize=(10, 3.5))
    
    #axis2d = fig.add_subplot(1, 2, 1)
    
    # plot invasion set
    acq_expansion = fig.add_subplot(1, 2, 1)

    x1 = np.linspace(bo.b_limit_lower[0], bo.b_limit_upper[0], 100)
    x2 = np.linspace(bo.b_limit_lower[1], bo.b_limit_upper[1], 100)
    x1g_ori_limit,x2g_ori_limit=np.meshgrid(x1,x2)
    X_plot=np.c_[x1g_ori_limit.flatten(), x2g_ori_limit.flatten()]
    Y = myfunction.func(X_plot)
    Y=-np.log(np.abs(Y))
    CS_expansion=acq_expansion.contourf(x1g_ori_limit,x2g_ori_limit,Y.reshape(x1g_ori.shape),cmap=my_cmap,origin='lower')
  
    if len(bo.X_invasion)!=0:
        myinvasion_set=acq_expansion.scatter(bo.X_invasion[:,0],bo.X_invasion[:,1],color='m',s=1,label='Invasion Set')   
    else:
        myinvasion_set=[] 
    
    myrectangle=patches.Rectangle(bo.bounds_bk[:,0], bo.max_min_gap_bk[0],bo.max_min_gap_bk[1],
                                              alpha=0.3, fill=False, facecolor="#00ffff",linewidth=3)
                                              
    acq_expansion.add_patch(myrectangle)
    
    acq_expansion.set_xlim(bo.b_limit_lower[0]-0.2, bo.b_limit_upper[0]+0.2)
    acq_expansion.set_ylim(bo.b_limit_lower[1]-0.2, bo.b_limit_upper[1]+0.2)
    


    if len(bo.X_invasion)!=0:
        acq_expansion.legend([myrectangle,myinvasion_set],[ur'$X_{t-1}$',ur'$I_t$'],loc=4,ncol=1,prop={'size':16},scatterpoints = 5)
        strTitle_Inv="[t={:d}] Invasion Set".format(counter)

        acq_expansion.set_title(strTitle_Inv,fontsize=16)
    else:
        acq_expansion.legend([myrectangle,myinvasion_set],[ur'$X_{t-1}$',ur'Empty $I_t$'],loc=4,ncol=1,prop={'size':16},scatterpoints = 5)
        strTitle_Inv="[t={:d}] Empty Invasion Set".format(counter)
        acq_expansion.set_title(strTitle_Inv,fontsize=16)
   
        
    """
    temp=np.linspace(bo.bounds_bk[0,0], bo.bounds_bk[0,1], num=30)
    acq_expansion.plot(temp,'ro')
    temp=np.linspace(bo.bounds_bk[1,0], bo.bounds_bk[1,1], num=30)
    acq_expansion.plot(temp,'ro')
    temp=np.linspace(bo.bounds_bk[0,1], bo.bounds_bk[1,1], num=30)
    acq_expansion.plot(temp,'ro')
    temp=np.linspace(bo.bounds_bk[0,0], bo.bounds_bk[1,0], num=30)
    acq_expansion.plot(temp,'ro')
    """

    #CS2_acq_expansion = plt.contour(CS_acq_expansion, levels=CS_acq_expansion.levels[::2],colors='r',origin='lower',hold='on')

    # plot acquisition function    
    
    acq2d = fig.add_subplot(1, 2, 2)

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    myrectangle=patches.Rectangle(bo.bounds[:,0], bo.max_min_gap[0],bo.max_min_gap[1],
                                  alpha=0.3, fill=False, facecolor="#00ffff",linewidth=3)
                                              
    acq2d.add_patch(myrectangle)
    
    #acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=30,label='Current Peak')
    myobs=acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',s=6,label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')

    #acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    #acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])

    
    acq2d.set_xlim(bo.b_limit_lower[0]-0.2, bo.b_limit_upper[0]+0.2)
    acq2d.set_ylim(bo.b_limit_lower[1]-0.2, bo.b_limit_upper[1]+0.2)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq2d.legend(loc='center left',bbox_to_anchor=(1.2, 0.5))
    #acq2d.legend(loc=4)
    acq2d.legend([myrectangle,myobs],[ur'$X_{t}$','Data'],loc=4,ncol=1,prop={'size':16}, scatterpoints = 3)

    strTitle_Acq="[t={:d}] Acquisition Func".format(counter)
    acq2d.set_title(strTitle_Acq,fontsize=16)

    fig.colorbar(CS_expansion, ax=acq_expansion, shrink=0.9)
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    strFileName="{:d}_unbounded.eps".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')
    
def plot_bo_2d_withGPmeans(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis2d, shrink=0.9)

    #plt.colorbar(ax=axis2d)

    #axis.plot(x, mu, '--', color='k', label='Prediction')
    
    
    #axis.set_xlim((np.min(x), np.max(x)))
    #axis.set_ylim((None, None))
    #axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

def plot_bo_2d_withGPmeans_Sigma(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 3))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))

    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis2d, shrink=0.9)

    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('Gaussian Process Variance',fontsize=16)
    #acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    #acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)
  
def plot_original_function(myfunction):
    
    origin = 'lower'

    func=myfunction.func


    if myfunction.input_dim==1:    
        x = np.linspace(myfunction.bounds['x'][0], myfunction.bounds['x'][1], 1000)
        y = func(x)
    
        fig=plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        strTitle="{:s}".format(myfunction.name)

        plt.title(strTitle)
    
    if myfunction.input_dim==2:    
        
        # Create an array with parameters bounds
        if isinstance(myfunction.bounds,dict):
            # Get the name of the parameters        
            bounds = []
            for key in myfunction.bounds.keys():
                bounds.append(myfunction.bounds[key])
            bounds = np.asarray(bounds)
        else:
            bounds=np.asarray(myfunction.bounds)
            
        x1 = np.linspace(bounds[0][0], bounds[0][1], 50)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 50)
        x1g,x2g=np.meshgrid(x1,x2)
        X_plot=np.c_[x1g.flatten(), x2g.flatten()]
        Y = func(X_plot)
    
        #fig=plt.figure(figsize=(8, 5))
        
        #fig = plt.figure(figsize=(12, 3.5))
        fig = plt.figure(figsize=(14, 4))
        
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        
        alpha = 0.7
        ax3d.plot_surface(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,alpha=alpha) 
        
        
        idxBest=np.argmax(Y)
        #idxBest=np.argmin(Y)
    
        ax3d.scatter(X_plot[idxBest,0],X_plot[idxBest,1],Y[idxBest],marker='*',color='r',s=200,label='Peak')
    
        
        #mlab.view(azimuth=0, elevation=90, roll=-90+alpha)

        strTitle="{:s}".format(myfunction.name)
        #print strTitle
        ax3d.set_title(strTitle)
        #ax3d.view_init(40, 130)

        
        idxBest=np.argmax(Y)
        CS=ax2d.contourf(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,origin=origin)   
       
        #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin=origin,hold='on')
        ax2d.scatter(X_plot[idxBest,0],X_plot[idxBest,1],marker='*',color='r',s=300,label='Peak')
        plt.colorbar(CS, ax=ax2d, shrink=0.9)

        ax2d.set_title(strTitle)

        
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\plot_2017"
    strFileName="{:s}.eps".format(myfunction.name)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')
        
