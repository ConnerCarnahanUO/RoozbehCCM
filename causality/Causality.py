# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:29:24 2020

@author: Amin
"""
from operator import itemgetter
from itertools import groupby
from copy import deepcopy
from scipy import stats
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sea
import pickle

# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:29:24 2020
@author: Amin
"""

# %% helpers
def average_treatment_effect(pre,pst):
    '''mean post stimulation and pre stimulation difference
    '''
    cnn,pvalue = np.nan,np.nan
    for i in range(len(pre)):
        df_f = abs(pst[i].mean()-pre[i].mean())
        cnn = np.nansum((cnn,df_f))
    return cnn/len(pre),pvalue

def aggregated_kolmogorov_smirnov(pre,pst):
    '''ks distribution distance between pre and post stimulation activity
    '''
    if np.array(pre).size > 0 and np.array(pst).size > 0:
        ks,p = stats.mstats.ks_2samp(np.hstack(pre),np.hstack(pst))
        cnn = ks
        pvalue = p
    else:
        cnn,pvalue = np.nan,np.nan
    return cnn,pvalue

def mean_kolmogorov_smirnov(pre,pst):
    '''mean ks distribution distance between pre and post stimulation activity 
    computer on individual instances of stimulation
    '''
    cnn,pvalue = np.nan,np.nan
    for i in range(len(pre)):
        if len(pre[i]) > 0 and len(pst[i]) > 0:
            ks,p = stats.mstats.ks_2samp(pre[i],pst[i])
            cnn = np.nansum((cnn,ks))
            pvalue = np.nansum((pvalue,p))
    return cnn/len(pre),pvalue/len(pre)

def DistTest(pre,post):
    """KSTest(dist1,dist2,ttest0,ttest1)
    Test whether two distributions of time series come from the same distribution (indicating perturbation effect)
    """
    pvalues={}
    ttest = stats.ttest_ind(post,pre)
    ranksum = stats.ranksums(post,pre)
    kstest = stats.kstest(post,pre)
    pvalues['ttest']= ttest.pvalue
    pvalues['ranksum']=ranksum.pvalue
    pvalues['kstest']=kstest.pvalue
    test_stats={}
    test_stats['ttest'] = ttest.statistic
    test_stats['ranksum'] = ranksum.statistic
    test_stats['kstest'] = kstest.statistic
    return test_stats,pvalues


def ExtendedInterventionalConnectivity(preactivity,postactivity,varnum=3,skip_stim_pre=10,skip_post_stim_values = np.arange(1,10),window=10,method='kstest',VarLabels=['X','Y','Z'],plot=False,figsize = (20,20),savepath=None):
    '''
    ExtendedInterventionalConnectivity:
    
    Args:
    preactivity list[list[ndarray]]: a list organized first by the stimulated variable, then the stimulation number, amd then the activity prior to a stimulation
    postactivity list[list[ndarray]]: a list organized first by the stimulated variable, then the stimulation number, amd then the activity after a stimulation
    skip_stim_pre int: how many steps before a stimulation should be ignored in creating the prior distribution
    skip_post_stim_values ndarray(int): how many steps after a stimulation should be tested for the optimum lag
    window int: the window size for the moving average created to test the distributions
    VarLabels list[str]: a list of strings that indicate the variable names, ordered by their order in the pre/postactivity lists
    plot bool: whether plots should be generated
    savepath bool: Where plots and values will be saved, if left None will not save
    figsize tuple float: the size of the plots
    method string: The test used to differentiate the pre and post distributions, currently must be kstest,ttest, or ranksum
    varnum int: the number of variables that will be tested
    '''
    pre_dists = [np.transpose(np.dstack(preactivity[n]),(0,2,1)) for n in range(len(preactivity))]
    post_dists = [np.transpose(np.dstack(postactivity[n]),(0,2,1)) for n in range(len(postactivity))]

    all_stats = {}
    fig = None
    axs = None
    save = savepath is not None

    if plot:
        fig,axs = plt.subplots(nrows=len(preactivity),ncols=varnum,figsize = figsize)
    for n in range(len(preactivity)):
        n_stats = {}
        for m in range(varnum):
            pre = pre_dists[n][-(skip_stim_pre+window):-skip_stim_pre,:,m]
            pre = np.nanmean(pre,axis=0)
            best_test_stats = {}
            best_pvalues = {'kstest':1,'ttest':1,'ranksum':1}
            best_lag = 0
            if plot:
                sea.ecdfplot(pre,ax=axs[n,m],label='Pre CDF')
                axs[n,m].set_title(f'{VarLabels[m]} when {VarLabels[n]} Stimulated')
            for l in skip_post_stim_values:
                post = post_dists[n][l:l+window,:,m]
                post = np.nanmean(post,axis=0)
                test_stat, pvalue = DistTest(pre,post)
                if best_pvalues[method] >= pvalue[method]:
                    best_lag = l
                    best_pvalues[method] = pvalue[method]
                    best_test_stats[method] = test_stat[method]
            
            n_stats[f'{VarLabels[m]}'] = {'lag':best_lag,'method':method,'statistic':best_test_stats[method],'pvalue':best_pvalues[method]}
            
            if plot:
                post = post_dists[n][best_lag:best_lag+window,:,m]
                post = np.nanmean(post,axis=0)
                sea.ecdfplot(post,ax=axs[n,m],label=f'Post CDF, lag = {best_lag}, {method}:{round(best_test_stats[method],3)}, pvalue:{round(best_pvalues[method],3)}')
                axs[n,m].legend()
        all_stats[f'Stim{VarLabels[n]}'] = n_stats
    if plot:
        if save:
            plt.savefig(savepath+'ecdfs.svg')
            plt.savefig(savepath+'ecdfs.png')
        plt.show()
    if save:
        with open(savepath+'ExtendedICStatistics.pkl', 'wb') as f:
            pickle.dump(all_stats, f)
    return all_stats



# %%
def interventional_connectivity(
        activity,stim,mask=None,t=None,
        bin_size=10,skip_pre=10,skip_pst=4,
        method='aggr_ks',
        save=False,load=False,file=None
    ):
    '''Compute interventional connectivity by measure statistical difference between pre 
    and post stimulation activity
    
    Args:
        activity (np.ndarray): (N,T) np array of signals in the perturbed state
        stim (array): Array of tuples of channel, stimulation start, and end time [(chn_i,str_i,end_i)]
        t (np.ndarray): If the activity is rates instead of spiking the timing of the sampled signals is given in t
        bin_size (float): Time window used for binning the activity and computing pre vs. post stimulation firing distribution
        skip_pre (float): Time to skip before the stimulation for pre distribution
        skip_pst (float): Time to skip before the stimulation for pre distribution
        method (string): Metrics for interventional connectivity: aggr_ks, mean_isi, mean_ks
        save_data (bool): If True the computed values will be saved in a mat file
        file (string): Name of the file used to save the mat file
    
    Returns:
        cnn: Dinterventional connectivity matrix evaluated for each given input metric
        pvalue: corresponding significance
    '''
    
    N = len(activity)
    T = activity[0].shape[0]

    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue']

    stim_ = deepcopy(stim)
    
    for i in range(len(stim)):
        if t is None:
            """pre_time = np.zeros(T,dtype=bool)
            pre_time[stim_[i][1]-bin_size-skip_pre:stim_[i][1]-skip_pre] = True
            pst_time = np.zeros(T,dtype=bool)
            pst_time[stim_[i][2]+skip_pst:stim[i][1]+skip_pst+bin_size] = True
            pst_isi = [np.diff(activity[j][pst_time]) for j in range(N)]
            pre_isi = [np.diff(activity[j][pre_time]) for j in range(N)]"""
            pst_isi = [np.diff(activity[j][(activity[j] <  stim_[i][2]+skip_pst+bin_size) & (activity[j] >= stim_[i][2]+skip_pst)]) for j in range(len(activity))]
            pre_isi = [np.diff(activity[j][(activity[j] >= stim_[i][1]-skip_pre-bin_size) & (activity[j] <  stim_[i][1]-skip_pre)]) for j in range(len(activity))]
        else:
            pst_isi = [activity[j,:][(t <  stim_[i][2]+skip_pst+bin_size) & (t >= stim_[i][2]+skip_pst)] for j in range(len(activity))]
            pre_isi = [activity[j,:][(t >= stim_[i][1]-skip_pre-bin_size) & (t <  stim_[i][1]-skip_pre)] for j in range(len(activity))]
        
        stim_[i] += (pre_isi,pst_isi)
        
    stim_g = [(int(k), [(x3,x4) for _,x1,x2,x3,x4 in g]) for k, g in groupby(sorted(stim_,key=itemgetter(0)), key=itemgetter(0))]
    #return stim_g
    cnn = np.zeros((len(activity), len(activity)))*np.nan
    pvalue = np.zeros((len(activity), len(activity)))*np.nan
    
    print(len(stim_g))

    for i in range(len(stim_g)): # stimulation channel
        for n in range(len(activity)): # post-syn channel
            aggr_pre_isi = [stim_g[i][1][j][0][n] for j in range(len(stim_g[i][1]))]
            aggr_pst_isi = [stim_g[i][1][j][1][n] for j in range(len(stim_g[i][1]))]
            
            if method == 'mean_ks':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = mean_kolmogorov_smirnov(aggr_pre_isi,aggr_pst_isi)
            if method == 'mean_isi':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = average_treatment_effect(aggr_pre_isi,aggr_pst_isi)
            if method == 'aggr_ks':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = aggregated_kolmogorov_smirnov(aggr_pre_isi,aggr_pst_isi)
            if method == 'wasserstein':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = Wasserstein_Metric(aggr_pre_isi,aggr_pst_isi)
        
    if mask is None: mask = np.zeros((len(activity),len(activity))).astype(bool)
    
    cnn = cnn.T
    pvalue = pvalue.T
    
    cnn[mask] = np.nan
    pvalue[mask] = np.nan
    
    if save: np.save(file,{'cnn':cnn,'pvalue':pvalue})
    
    return cnn,pvalue


def interventional_connectivity_arb_func(
        activity,stim,mask=None,t=None,
        bin_size=10,skip_pre=10,skip_pst=4,
        method=aggregated_kolmogorov_smirnov,
        save=False,load=False,file=None
    ):
    '''Compute interventional connectivity by measure statistical difference between pre 
    and post stimulation activity
    
    Args:
        activity (np.ndarray): (N,T) np array of signals in the perturbed state
        stim (array): Array of tuples of channel, stimulation start, and end time [(chn_i,str_i,end_i)]
        t (np.ndarray): If the activity is rates instead of spiking the timing of the sampled signals is given in t
        bin_size (float): Time window used for binning the activity and computing pre vs. post stimulation firing distribution
        skip_pre (float): Time to skip before the stimulation for pre distribution
        skip_pst (float): Time to skip before the stimulation for pre distribution
        method (string): Metrics for interventional connectivity: aggr_ks, mean_isi, mean_ks
        save_data (bool): If True the computed values will be saved in a mat file
        file (string): Name of the file used to save the mat file
    
    Returns:
        cnn: Dinterventional connectivity matrix evaluated for each given input metric
        pvalue: corresponding significance
    '''
    
    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue']

    stim_ = deepcopy(stim)
    
    for i in range(len(stim)):
        if t is None:
            pst_isi = [np.diff(activity[j][(activity[j] <  stim_[i][2]+skip_pst+bin_size) & (activity[j] >= stim_[i][2]+skip_pst)]) for j in range(len(activity))]
            pre_isi = [np.diff(activity[j][(activity[j] >= stim_[i][1]-skip_pre-bin_size) & (activity[j] <  stim_[i][1]-skip_pre)]) for j in range(len(activity))]

        else:
            pst_isi = [activity[j][(t <  stim_[i][2]+skip_pst+bin_size) & (t >= stim_[i][2]+skip_pst)] for j in range(len(activity))]
            pre_isi = [activity[j][(t >= stim_[i][1]-skip_pre-bin_size) & (t <  stim_[i][1]-skip_pre)] for j in range(len(activity))]
        
        stim_[i] += (pre_isi,pst_isi)
        
    stim_g = [(int(k), [(x3,x4) for _,x1,x2,x3,x4 in g]) for k, g in groupby(sorted(stim_,key=itemgetter(0)), key=itemgetter(0))]
    #return stim_g
    cnn = np.zeros((len(activity), len(activity)))*np.nan
    pvalue = np.zeros((len(activity), len(activity)))*np.nan
    
    for i in range(len(stim_g)): # stimulation channel
        for n in range(len(activity)): # post-syn channel
            aggr_pre_isi = [stim_g[i][1][j][0][n] for j in range(len(stim_g[i][1]))]
            aggr_pst_isi = [stim_g[i][1][j][1][n] for j in range(len(stim_g[i][1]))]
            cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = method(aggr_pre_isi,aggr_pst_isi)
        
    if mask is None: mask = np.zeros((len(activity),len(activity))).astype(bool)
    
    cnn = cnn.T
    pvalue = pvalue.T
    
    cnn[mask] = np.nan
    pvalue[mask] = np.nan
    
    if save: np.save(file,{'cnn':cnn,'pvalue':pvalue})
    
    return cnn,pvalue

def Wasserstein_Metric(pre,pst,pad=True):
    '''Wasserstein Metric computes the sum of distances between th ith order statistics
    It can also be computed via the integral over the absolute value of the difference of the CDFs
    TODO: Come up with a p-value computation for this (take the distribution of no stim and see where this falls in the actual stim data)
    '''
    cnn,pvalue = np.nan,np.nan
    if np.array(pre).size > 0 and np.array(pst).size > 0:
        for i in range(len(pre)):
            X = pre[i]
            #print(X)
            Y = pst[i]
            if len(X) != len(Y):
                m = np.min([len(X),len(Y)])
                X = (np.random.permutation(X))[:m]
                Y = (np.random.permutation(Y))[:m]
                ws = stats.wasserstein_distance(X,Y)
                cnn = np.nansum([ws,cnn])
                pvalue = np.nansum([np.nan,pvalue])
        return cnn/len(pre),pvalue/len(pre)
    else:
        return np.nan,np.nan
    
def IC2(
        activity,stim,mask=None,t=None,
        bin_size=10,skip_pre=10,skip_pst=4,
        method='aggr_ks',
        save=False,load=False,file=None
    ):
    '''Compute interventional connectivity by measure statistical difference between pre 
    and post stimulation activity
    
    Args:
        activity (np.ndarray): (N,T) np array of signals in the perturbed state
        stim (array): Array of tuples of channel, stimulation start, and end time [(chn_i,str_i,end_i)]
        t (np.ndarray): If the activity is rates instead of spiking the timing of the sampled signals is given in t
        bin_size (float): Time window used for binning the activity and computing pre vs. post stimulation firing distribution
        skip_pre (float): Time to skip before the stimulation for pre distribution
        skip_pst (float): Time to skip before the stimulation for pre distribution
        method (string): Metrics for interventional connectivity: aggr_ks, mean_isi, mean_ks
        save_data (bool): If True the computed values will be saved in a mat file
        file (string): Name of the file used to save the mat file
    
    Returns:
        cnn: Dinterventional connectivity matrix evaluated for each given input metric
        pvalue: corresponding significance
    '''
    
    N,Tlen = activity.shape

    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue']

    stim_ = deepcopy(stim)
    
    pre_acts = [[]]
    pst_acts = [[]]

    pre_stim_times = [[]]
    pst_stim_times = [[]]
    stim_nodes = []

    for i in range(len(stim)):
        pst_stim_times.append(t[(t <  stim_[i][2]+skip_pst+bin_size) & (t >= stim_[i][2]+skip_pst)])
        pre_stim_times.append(t[(t >= stim_[i][1]-skip_pre-bin_size) & (t <  stim_[i][1]-skip_pre)])
        stim_nodes.append(stim[i][0])
    pre_stim_times.pop(0)
    pst_stim_times.pop(0)
    stim_nodes = np.array(stim_nodes)

    pre_act_means = np.zeros((len(stim),N,N))*np.nan
    pst_act_means = np.zeros((len(stim),N,N))*np.nan

    for i in range(len(pre_stim_times)):
        pre_act_means[i,:,stim_[i][0]] = np.mean(activity[:,pre_stim_times[i]],axis=1)
        pst_act_means[i,:,stim_[i][0]] = stim_[i][0]



    IC = np.zeros((N,N))*np.nan
    ICpval = np.zeros((N,N))*np.nan

    

    for i in range(len(stim)):
        if t is None:
            pst_isi = [np.diff(activity[j][(activity[j] <  stim_[i][2]+skip_pst+bin_size) & (activity[j] >= stim_[i][2]+skip_pst)]) for j in range(len(activity))]
            pre_isi = [np.diff(activity[j][(activity[j] >= stim_[i][1]-skip_pre-bin_size) & (activity[j] <  stim_[i][1]-skip_pre)]) for j in range(len(activity))]
        else:
            pst_isi = [activity[j,:][(t <  stim_[i][2]+skip_pst+bin_size) & (t >= stim_[i][2]+skip_pst)] for j in range(len(activity))]
            pre_isi = [activity[j,:][(t >= stim_[i][1]-skip_pre-bin_size) & (t <  stim_[i][1]-skip_pre)] for j in range(len(activity))]
        
        stim_[i] += (pre_isi,pst_isi)
        
    stim_g = [(int(k), [(x3,x4) for _,x1,x2,x3,x4 in g]) for k, g in groupby(sorted(stim_,key=itemgetter(0)), key=itemgetter(0))]
    #return stim_g
    cnn = np.zeros((len(activity), len(activity)))*np.nan
    pvalue = np.zeros((len(activity), len(activity)))*np.nan
    
    print(len(stim_g))

    for i in range(len(stim_g)): # stimulation channel
        for n in range(len(activity)): # post-syn channel
            aggr_pre_isi = [stim_g[i][1][j][0][n] for j in range(len(stim_g[i][1]))]
            aggr_pst_isi = [stim_g[i][1][j][1][n] for j in range(len(stim_g[i][1]))]
            
            if method == 'mean_ks':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = mean_kolmogorov_smirnov(aggr_pre_isi,aggr_pst_isi)
            if method == 'mean_isi':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = average_treatment_effect(aggr_pre_isi,aggr_pst_isi)
            if method == 'aggr_ks':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = aggregated_kolmogorov_smirnov(aggr_pre_isi,aggr_pst_isi)
            if method == 'wasserstein':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = Wasserstein_Metric(aggr_pre_isi,aggr_pst_isi)
        
    if mask is None: mask = np.zeros((len(activity),len(activity))).astype(bool)
    
    cnn = cnn.T
    pvalue = pvalue.T
    
    cnn[mask] = np.nan
    pvalue[mask] = np.nan
    
    if save: np.save(file,{'cnn':cnn,'pvalue':pvalue})
    
    return cnn,pvalue
# %%
