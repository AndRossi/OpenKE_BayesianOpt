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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec

import random
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
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
       
        
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)
    

def plot_acq_bo_1d_tgp(bo_tgp,fstar=0):
    
    global counter
    counter=counter+1
    
    func=bo_tgp.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo_tgp.scalebounds[0,0], bo_tgp.scalebounds[0,1], 1000)
    x_original=x*bo_tgp.max_min_gap+bo_tgp.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(7, 13))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(5, 1, height_ratios=[3,1,1,1,1]) 
    axis_tgp = plt.subplot(gs[0])
    #axis_tgp_g = plt.subplot(gs[1])


    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    
    acq_CBM= plt.subplot(gs[3])


    acq_ERM_TGP = plt.subplot(gs[4])
    #acq_ERM = plt.subplot(gs[2])
    
    #acq_TS2 = plt.subplot(gs[5])

    temp=np.abs(y_original-fstar)
    idx=np.argmin(temp)
    

    axis_tgp.hlines(fstar, xmin=bo_tgp.bounds[0,0], xmax=bo_tgp.bounds[0,1], colors='r', linestyles='solid')
    axis_tgp.text(7.3, fstar+1,'Known Value $f^*$',fontsize=14)
    axis_tgp.vlines(x=x_original[idx], ymin=-11, ymax=13, colors='r', linestyles='solid')
    axis_tgp.text(3, 14,'Unknown Location $x^*$',fontsize=14)
    axis_tgp.set_ylim([-19,18])
    #axis.set_yticks([])
    axis_tgp.set_xticks([])
    
    axis_tgp.set_title('f(x)=$f^*$-0.5*$g^2$(x), g$\sim$GP( $\sqrt{2f^*}$ ,K)',fontsize=18)
    

    
    
    # TGP
    mu, sigma = bo_tgp.posterior_tgp(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo_tgp.Y_original)+np.mean(bo_tgp.Y_original)
    sigma_original=sigma*np.std(bo_tgp.Y_original)+np.mean(bo_tgp.Y_original)**2
    
    #axis_tgp.plot(x_original, y_original, linewidth=3, label='f(x)=$f^*$-0.5*$g^2$(x)')
    axis_tgp.plot(x_original, y_original, linewidth=3, label='f(x)')
    axis_tgp.plot(bo_tgp.X_original.flatten(), bo_tgp.Y_original, 'D', markersize=8, label=u'Obs', color='r')
    axis_tgp.plot(x_original, mu_original, '--', color='k', label='$\mu_f(x)$')
    
    #samples*bo_tgp.max_min_gap+bo_tgp.bounds[:,0]
    axis_tgp.set_yticks([])
    axis_tgp.set_xticks([])
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo_tgp.max_min_gap+bo_tgp.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    #temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.0 * sigma, (mu + 1.0 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo_tgp.Y_original)+np.mean(bo_tgp.Y_original)
    axis_tgp.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='$\sigma_f(x)$')
    
    axis_tgp.set_xlim((np.min(x_original), np.max(x_original)))
    #axis_tgp.set_ylim((-23, 18))
    axis_tgp.set_ylabel('f(x)', fontdict={'size':16})
    
    
    
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo_tgp.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo_tgp.gp)
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    acq_UCB.vlines(x=x_original[idx], ymin=-1.3, ymax=2.6, colors='r', linestyles='solid')

    # check batch BO     
    try:
        nSelectedPoints=np.int(bo_tgp.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    #acq_UCB.plot(bo_tgp.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_UCB.set_xlim((np.min(x_original), np.max(x_original)))
    acq_UCB.set_ylabel('UCB', fontdict={'size':16})
    #acq_UCB.set_xlabel('x', fontdict={'size':16})
    acq_UCB.set_xticks([])
    acq_UCB.set_yticks([])


    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo_tgp.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo_tgp.gp)
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    acq_EI.vlines(x=x_original[idx], ymin=0, ymax=0.17, colors='r', linestyles='solid')

    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
 
    acq_EI.set_xticks([])
    acq_EI.set_yticks([])
    
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    #acq_EI.set_xlabel('x', fontdict={'size':16})
    
    
    
    #Confidence Bound Minimization
    acq_func={}
    acq_func['name']='kov_cbm'
    acq_func['dim']=1
    acq_func['scalebounds']=bo_tgp.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo_tgp.Y_original))/np.std(bo_tgp.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo_tgp.gp)
    acq_CBM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_CBM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_CBM.vlines(x=x_original[idx], ymin=-5.5, ymax=0.1, colors='r', linestyles='solid')

    
    acq_CBM.set_xticks([])
    acq_CBM.set_yticks([])
    
    acq_CBM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_CBM.set_ylabel('CBM', fontdict={'size':16})
    acq_CBM.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc="lower center",prop={'size':16},ncol=4)
    
    
    
    
    # ERM TGP
    acq_func={}
    acq_func['name']='kov_tgp'
    acq_func['dim']=1
    acq_func['scalebounds']=bo_tgp.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo_tgp.Y_original))/np.std(bo_tgp.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo_tgp.gp)
    acq_ERM_TGP.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ERM_TGP.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_ERM_TGP.vlines(x=x_original[idx], ymin=-3.5, ymax=0.1, colors='r', linestyles='solid')

    
    acq_ERM_TGP.set_xticks([])
    acq_ERM_TGP.set_yticks([])
    
    acq_ERM_TGP.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ERM_TGP.set_ylabel('ERM', fontdict={'size':16})
    acq_ERM_TGP.set_xlabel('x', fontdict={'size':16})
    
    axis_tgp.legend(loc="lower center",prop={'size':16},ncol=4)
    #axis_tgp_g.legend(loc="lower center",prop={'size':16},ncol=4)

    
    

    strFileName="{:d}_GP_AF_ERM_TGP.pdf".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


    
def plot_acq_bo_1d(bo,fstar=0):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(7, 13))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    acq_CBM= plt.subplot(gs[3])
    acq_ERM = plt.subplot(gs[4])
    
    #acq_TS2 = plt.subplot(gs[5])

    temp=np.abs(y_original-fstar)
    idx=np.argmin(temp)
    axis.hlines(fstar, xmin=bo.bounds[0,0], xmax=bo.bounds[0,1], colors='r', linestyles='solid')
    axis.text(7.3, fstar+1,'Known Value $f^*$',fontsize=14)
    axis.vlines(x=x_original[idx], ymin=-11, ymax=13, colors='r', linestyles='solid')
    axis.text(3, 14,'Unknown Location $x^*$',fontsize=14)
    axis.set_ylim([-19,18])
    axis.set_yticks([])
    axis.set_xticks([])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='f(x)')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Obs', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='$\mu(x)$')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    #temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.3 * sigma, (mu + 1.3 * sigma)[::-1]])

    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='$\sigma(x)$')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    
    axis.set_title('f$\sim$ GP(0,K)',fontsize=18)

    #axis.set_xlabel('x', fontdict={'size':16})
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    acq_UCB.vlines(x=x_original[idx], ymin=-1.5, ymax=3, colors='r', linestyles='solid')

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
    #acq_UCB.set_xlabel('x', fontdict={'size':16})
    acq_UCB.set_xticks([])
    acq_UCB.set_yticks([])


    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    acq_EI.vlines(x=x_original[idx], ymin=0, ymax=0.65, colors='r', linestyles='solid')

    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
 
    acq_EI.set_xticks([])
    acq_EI.set_yticks([])
    
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    #acq_EI.set_xlabel('x', fontdict={'size':16})
  
    
    # Confidence Bound Minimization
    acq_func={}
    acq_func['name']='kov_cbm'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo.Y_original))/np.std(bo.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_CBM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_CBM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_CBM.vlines(x=x_original[idx], ymin=-3.9, ymax=0.1, colors='r', linestyles='solid')

    
    acq_CBM.set_xticks([])
    acq_CBM.set_yticks([])
    
    acq_CBM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_CBM.set_ylabel('CBM', fontdict={'size':16})
    acq_CBM.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc="lower center",prop={'size':16},ncol=4)
    
    
    
    
    # ERM 
    acq_func={}
    acq_func['name']='kov_erm'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo.Y_original))/np.std(bo.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_ERM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ERM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_ERM.vlines(x=x_original[idx], ymin=-3.2, ymax=0.1, colors='r', linestyles='solid')

    
    acq_ERM.set_xticks([])
    acq_ERM.set_yticks([])
    
    acq_ERM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ERM.set_ylabel('ERM', fontdict={'size':16})
    acq_ERM.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc="lower center",prop={'size':16},ncol=4)

    strFileName="{:d}_GP_AF_ERM.pdf".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    
"""
def plot_acq_bo_1d_ucb(bo,fstar=0):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(7, 11))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    acq_CBM = plt.subplot(gs[3])
    acq_ERM = plt.subplot(gs[4])
    
    #acq_TS2 = plt.subplot(gs[5])

    temp=np.abs(y_original-fstar)
    idx=np.argmin(temp)
    axis.hlines(fstar, xmin=bo.bounds[0,0], xmax=bo.bounds[0,1], colors='r', linestyles='solid')
    axis.text(7, fstar+1,'Known Value $f^*$',fontsize=14)
    axis.vlines(x=x_original[idx], ymin=-11, ymax=13, colors='r', linestyles='solid')
    axis.text(3.5, 14,'Unknown Location $x^*$',fontsize=14)
    axis.set_ylim([-19,18])
    axis.set_yticks([])
    axis.set_xticks([])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='f(x)')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Obs', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='$\mu(x)$')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='$\sigma(x)$')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    acq_UCB.vlines(x=x_original[idx], ymin=-1.5, ymax=3, colors='r', linestyles='solid')

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
    #acq_UCB.set_xlabel('x', fontdict={'size':16})
    acq_UCB.set_xticks([])
    acq_UCB.set_yticks([])


    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    acq_EI.vlines(x=x_original[idx], ymin=0, ymax=0.44, colors='r', linestyles='solid')

    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
 
    acq_EI.set_xticks([])
    acq_EI.set_yticks([])
    
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    #acq_EI.set_xlabel('x', fontdict={'size':16})
  
    
    # UCB_Target or Minimize Confidence Bound
    acq_func={}
    acq_func['name']='mcb'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo.Y_original))/np.std(bo.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_CBM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_CBM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_CBM.vlines(x=x_original[idx], ymin=-3.5, ymax=0.1, colors='r', linestyles='solid')

    
    acq_CBM.set_xticks([])
    acq_CBM.set_yticks([])
    
    acq_CBM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_CBM.set_ylabel('CBM', fontdict={'size':16})
    acq_CBM.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc="lower center",prop={'size':16},ncol=4)
    
    
    
    
    # ERM 
    acq_func={}
    acq_func['name']='erm'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo.Y_original))/np.std(bo.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_ERM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ERM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_ERM.vlines(x=x_original[idx], ymin=-3.5, ymax=0.1, colors='r', linestyles='solid')

    
    acq_ERM.set_xticks([])
    acq_ERM.set_yticks([])
    
    acq_ERM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ERM.set_ylabel('ERM', fontdict={'size':16})
    acq_ERM.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc="lower center",prop={'size':16},ncol=4)

    strFileName="{:d}_GP_AF_ERM_UCB.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_acq_bo_1d_2(bo,fstar=0):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(7, 11))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    acq_CBM=plt.subplot(gs[3])
    acq_ERM = plt.subplot(gs[4])
    
    #acq_TS2 = plt.subplot(gs[5])

    temp=np.abs(y_original-fstar)
    idx=np.argmin(temp)
    axis.hlines(fstar, xmin=bo.bounds[0,0], xmax=bo.bounds[0,1], colors='r', linestyles='solid')
    axis.text(7, fstar+1,'Known Value $f^*$',fontsize=14)
    axis.vlines(x=x_original[idx], ymin=-11, ymax=13, colors='r', linestyles='solid')
    axis.text(3.5, 14,'Unknown Location $x^*$',fontsize=14)
    axis.set_ylim([-19,18])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='f(x)')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Obs', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='$\mu(x)$')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='$\sigma(x)$')
    
    axis.set_yticks([])
    axis.set_xticks([])
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    acq_UCB.vlines(x=x_original[idx], ymin=-1.6, ymax=2.5, colors='r', linestyles='solid')

    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    #acq_UCB.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_UCB.set_yticks([])
    acq_UCB.set_xticks([])

    
    acq_UCB.set_xlim((np.min(x_original), np.max(x_original)))
    acq_UCB.set_ylabel('UCB', fontdict={'size':16})
    #acq_UCB.set_xlabel('x', fontdict={'size':16})
    
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    acq_EI.vlines(x=x_original[idx], ymin=0, ymax=0.33, colors='r', linestyles='solid')

    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
 
    acq_EI.set_xticks([])
    acq_EI.set_yticks([])
    
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    #acq_EI.set_xlabel('x', fontdict={'size':16})
  
    
    # Confidence Bound Minimization
    acq_func={}
    acq_func['name']='kov_cbm'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo.Y_original))/np.std(bo.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_CBM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_CBM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_CBM.vlines(x=x_original[idx], ymin=-5.9, ymax=-2.6, colors='r', linestyles='solid')

    
    acq_CBM.set_xticks([])
    acq_CBM.set_yticks([])
    
    acq_CBM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_CBM.set_ylabel('CBM', fontdict={'size':16})
    acq_CBM.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc="lower center",prop={'size':16},ncol=4)
    
    
    
    # ERM 
    acq_func={}
    acq_func['name']='kov_erm'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    acq_func['fstar_scaled']=(fstar-np.mean(bo.Y_original))/np.std(bo.Y_original)

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_ERM.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ERM.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq_ERM.vlines(x=x_original[idx], ymin=-5.6, ymax=-2.5, colors='r', linestyles='solid')

    acq_ERM.set_xticks([])
    acq_ERM.set_yticks([])
    
                 
    acq_ERM.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ERM.set_ylabel('ERM', fontdict={'size':16})
    acq_ERM.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis.legend(loc="lower center",prop={'size':16},ncol=4)

    
    strFileName="{:d}_GP_AF_ERM2_UCB.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
"""  
    
	
def plot_bo_1d(bo):
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(8, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    #temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp)
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
    
    #plt.legend(fontsize=14)
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization"

    strFileName="{:d}_GP_BO_1d.pdf".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')
    
    
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

    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=140,label='Selected')
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')

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

        
    strFolder=""
    strFileName="{:s}.eps".format(myfunction.name)
    strPath=os.path.join(strFolder,strFileName)
    #fig.savefig(strPath, bbox_inches='tight')
  

def plot_gp_sequential_batch(bo,x_seq,x_batch):
    
    global counter
    counter=counter+1
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    fig=plt.figure(figsize=(10, 3))
    
  
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d_seq = fig.add_subplot(1, 2, 1)
    acq2d_batch = fig.add_subplot(1, 2, 2)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d_seq.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')

    acq2d_seq.scatter(x_seq[0],x_seq[1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    acq2d_seq.set_title('Sequential Bayesian Optimization',fontsize=16)
    acq2d_seq.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d_seq.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)

    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq2d.legend(loc='center left',bbox_to_anchor=(1.01, 0.5))
      
    fig.colorbar(CS_acq, ax=acq2d_seq, shrink=0.9)
    
    
    
    CS_acq_batch=acq2d_batch.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq_batch = plt.contour(CS_acq_batch, levels=CS_acq_batch.levels[::2],colors='r',origin='lower',hold='on')

    acq2d_batch.scatter(x_batch[:,0],x_batch[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    acq2d_batch.set_title('Batch Bayesian Optimization',fontsize=16)
    acq2d_batch.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d_batch.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
    
    fig.colorbar(CS_acq_batch, ax=acq2d_batch, shrink=0.9)

        
    strFolder="V:\\plot_2017\\sequential_batch"
    strFileName="{:d}.eps".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')