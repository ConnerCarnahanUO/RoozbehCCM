#%%
# Imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from functools import partial
import scipy as sci
from scipy import interpolate
from scipy.io import savemat
from scipy import stats
from scipy import signal
from sklearn.model_selection import KFold
import ipywidgets as wdgs
import matplotlib.pyplot as plt
import time
import tqdm
from tqdm.contrib.concurrent import process_map
import os
import sys
import pandas as pd
from datetime import datetime
import yaml
from sklearn.linear_model import LinearRegression

#%%
# Main computation helper functions
def create_delay_vector(sequence,delay,dim,roll_method = True):
    """Create delay vectors from rate or sequence data
    
    Args:
        sequence (numpy.ndarray): Time series (TxN) corresponding to a single node 
            but can be multidimensional
        delay (integer): Delay used in the embedding (t-delay,t-2*delay,...)
        dim (integer): Embedding dimensionality
        
    Returns:
        numpy.ndarray: Delay coordinates of the embedded sequence
    
    """

        
    T = sequence.shape[0]   #Number of time-points we're working with
    tShift = np.abs(delay)*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Make sure vector is 2D
    sequence = np.squeeze(sequence)
    if len(np.shape(sequence))==1:
        sequence = np.reshape(sequence, (T,1))

    #Number of neurons 
    N = sequence.shape[1]

    #Preallocate delay vectors
    dvs = np.zeros((tDelay,N*dim)) #[length of delay vectors x # of delay vectors]
    #Create fn to flatten matrix if time series is multidimensional
    vec = lambda x: np.ndarray.flatten(x)

    if roll_method:
        for i in range(dim):
            dvs[:,N*i:N*(i+1)] = np.roll(sequence,-delay*i,axis=0)[:tDelay,:]
    #Loop through delay time points
    else:
        for idx in range(tDelay):
            # note: shift+idx+1 has the final +1 because the last index of the slice gets excluded otherwise
            if delay >= 0:
                dvs[idx,:] = vec(sequence[idx:tShift+idx+1:delay,:]) 
    return dvs

def MutualInformation(sig1,sig2,bins=50):
    pXYjoint = np.histogram2d(sig1,sig2,density = True,bins=bins)[0]
    pXYindep = np.outer(np.sum(pXYjoint,axis=1),np.sum(pXYjoint,axis=0))
    return sci.stats.entropy(pXYjoint.flatten(),pXYindep.flatten())

def remote_ApproximatebestTau(iK, sig, taumin=0, taumax=20, bins=50, plot=True, threshold_ratio = 0.001,**args):
    return (iK, ApproximatebestTau(sig, taumin, taumax, bins, plot, threshold_ratio,**args)[0])

def ApproximatebestTau(sig,dt=1, taumin=0, taumax=20, bins=50, plot=False, threshold_ratio = 0.1):
    MIs = np.zeros(taumax-taumin)*np.nan
    total_points = sig.shape[0]
    
    for tau in np.arange(MIs.shape[0]):
        tau_len = total_points - tau
        MIs[tau] = MutualInformation(sig[:tau_len],sig[tau:])
    
    tau_star = 0
    H_min = np.inf
    i = 0
    dMIs = np.diff(MIs)/dt
    while i < dMIs.shape[0]:
        if dMIs[i] > 0:
            tau_star = i
            i = MIs.shape[0]
        i+=1
    if tau_star == 0:
        tau_star = np.argmax(dMIs)

    """    
    for i in range(MIs.shape[0]):
        if (MIs[i] < H_min):
            H_min = MIs[i]
        else:
            tau_star = i-1
            break
    if tau_star == 0:
        connecting_line = MIs[0]+(MIs[-1]-MIs[0])/MIs.shape[0] * np.arange(MIs.shape[0])
        tau_star = np.argmax(np.abs(connecting_line-MIs))
    """
    if plot:
        fig = plt.figure(figsize=(12,10))
        a = plt.axes()
        a.plot(np.arange(taumin,taumax),MIs)
        a.axvline(tau_star,linestyle='--',color = (0,0,0),label=f'Optimum $\\tau$ = {tau_star+taumin}')
        plt.xlabel('Tau')
        plt.ylabel('$I[y(t);y(t-\\tau)]$')
        plt.title('Mutual Information between delays')
        plt.legend()
        plt.show()
    
    return tau_star, MIs

def remote_ApproximatebestTau(iK, sig, taumin=0, taumax=20, bins=50, plot=True, threshold_ratio = 0.001,**args):
    return (iK, ApproximatebestTau(sig, taumin, taumax, bins, plot, threshold_ratio,**args)[0])


def build_nn_single(X,train_indices,test_indices,n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(X[train_indices,:])
    distances, indices = nbrs.kneighbors(X[test_indices,:])
    # This really should scale the distance by 1/min{distances} but it keeps throwing a fit so I am not dealing with that rn
    weights = np.exp(-distances)
    weights = weights/(weights.sum(axis=1)[:,np.newaxis])
    return (weights,indices)

def remote_build_nn_single(index,Xdel,train_indices,test_indices,n_neighbors):
    return index, build_nn_single(Xdel,train_indices,test_indices,n_neighbors)

def indexed_remote_Reconstruction(index,Predicted,Predictor,nns,test_indices):
    return index, CCMReconstruction(Predicted,Predictor,nns,test_indices)

def CCMRegression(Efferent,Afferent,nns,test_indices):
    """ CCMRegression: computes the reconstruction of a lagged time series, its accuracy, and the coefficient for the real regressed on the reconstructed
        
        Args:
        Predicted: txN numpy array, the time series to be preducted
        Predictor: TxN numpy array, the time series used as the predictor
        nns: tuple of indices and weights computed by build_nns, indicates the weights and indices of the nearest neighbors of the predicted time series
        test_indices: t numpy array, the indices of the test set partition
        lag: int, the lag of the time series
        
        returns:
        reconstruction: txN numpy array, the reconstructed lagged time series
        recconstruction_accuracy: the correlation between the lagged time series and the predicted lagged time series
    """
    reconstruction = np.array([np.matmul(nns[0][idx,:],Afferent[nns[1][idx,:],:]) for idx in range(len(test_indices))])
    reconstruction_error = sequential_correlation(reconstruction,Efferent)
    return reconstruction.T, reconstruction_error#, np.corrcoef(reconstruction[:,0],Predicted[:,0])

def CCMReconstruction(Predicted,Predictor,nns,test_indices):
    """ CCMReconstruction: computes the reconstruction of a lagged time series and its accuracy
        
        Args:
        Predicted: txN numpy array, the time series to be preducted
        Predictor: TxN numpy array, the time series used as the predictor
        nns: tuple of indices and weights computed by build_nns, indicates the weights and indices of the nearest neighbors of the predicted time series
        test_indices: t numpy array, the indices of the test set partition
        lag: int, the lag of the time series
        
        returns:
        reconstruction: txN numpy array, the reconstructed lagged time series
        recconstruction_accuracy: the correlation between the lagged time series and the predicted lagged time series
    """
    reconstruction = np.array([np.matmul(nns[0][idx,:],Predictor[nns[1][idx,:],:]) for idx in range(len(test_indices))])
    reconstruction_error = sequential_correlation(reconstruction,Predicted)
    return reconstruction.T, reconstruction_error#, np.corrcoef(reconstruction[:,0],Predicted[:,0])

def SequentialLaggedReconstruction(Predicted,Predictor,nns,test_indices,lags=np.array([0])):
    reconstructions = [[] for l in range(lags.shape[0])]
    reconstruction_error = np.zeros(lags.shape[0])
    reconstruction_corr = np.zeros(lags.shape[0])
    for l in range(lags.shape[0]):
        L = lags[l]
        curr_predictor = np.roll(Predictor,L,axis=0)
        reconstruction = np.array([nns[0][idx,:]@curr_predictor[nns[1][idx,:],:] for idx in range(len(test_indices))])
        reconstructions[l] = reconstruction.T
        reconstruction_error[l] = sequential_correlation(reconstruction,np.roll(Predicted,L,axis=0))[0]
        reconstruction_corr[l] = np.corrcoef(reconstruction[:,0],np.roll(Predicted,L,axis=0)[:,0])[0,1]
    return reconstruction_error, reconstruction_corr #reconstructions, reconstruction_error#, np.corrcoef(reconstruction[:,0],Predicted[:,0])

def indexed_SequentialLaggedReconstruction(index,Predicted,Predictor,nns,test_indices,lags=np.array([0])):
    slr = SequentialLaggedReconstruction(Predicted,Predictor,nns,test_indices,lags=lags)
    return index, slr[0], slr[1]

def reconstruction_column(index,nns,targets,lib_targets,test_indices,lags):
    recon_accuracies = np.zeros((lib_targets.shape[2],lags.shape[0]))
    recon_corrs = np.zeros((lib_targets.shape[2],lags.shape[0]))
    for i in range(lib_targets.shape[2]):
        recon_accuracies[i,:], recon_corrs[i,:] = SequentialLaggedReconstruction(targets[:,:,i],lib_targets[:,:,i],nns,test_indices,lags)
    return index, recon_accuracies, recon_corrs

def FullCCMColumn(index,TimeSeries,efferent,train_indices,test_indices,lags):
    """ FullCCMColum:
        args:
    index:      an argument that is just passed at the end as the first element in the tuple. Can be anything you want to use
                as an index for this computation
    TimeSeries: The Full delay embedded timeseries used
    afferent: Index of the afferent time series
    train_indices: the indices that are used to create the reconstructions
    test_indices: The indices that are used to compute the reconstruction accuracy
    lags: All Lags that are used to make the lagged reconstructions
    
    """
    targets = TimeSeries[test_indices,:,:]
    lib_targets = TimeSeries[train_indices,:,:]
    n_neighbors = targets.shape[1]+1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(TimeSeries[train_indices,:,efferent])
    distances, indices = nbrs.kneighbors(TimeSeries[test_indices,:,efferent])
    # TODO? This really should scale the distance by 1/min{distances} but it keeps throwing a fit 
    #       so I am not dealing with that rn
    weights = np.exp(-distances)
    weights = weights/(weights.sum(axis=1)[:,np.newaxis])
    return reconstruction_column(index,(weights,indices),targets,lib_targets,test_indices,lags)

def CCMColumn(index,X_delay,efferent,train_indices,test_indices,lags):
    N = X_delay.shape[2]
    targets = X_delay[test_indices,:,:]
    lib_targets = X_delay[train_indices,:,:]
    n_neighbors = targets.shape[1]+1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(X_delay[train_indices,:,efferent])
    distances, indices = nbrs.kneighbors(X_delay[test_indices,:,efferent])
    # TODO? This really should scale the distance by 1/min{distances} but it keeps throwing a fit 
    #       so I am not dealing with that rn
    weights = np.exp(-distances)
    weights = weights/(weights.sum(axis=1)[:,np.newaxis])
    
    reconstructions = [[[] for l in range(lags.shape[0])] for n in range(N)]
    reconstruction_error = np.zeros((N,lags.shape[0]))
    reconstruction_corr = np.zeros((N,lags.shape[0]))
    
    for l in range(lags.shape[0]):
        L = lags[l]
        for n in range(N):
            curr_afferent = np.roll(lib_targets[:,:,n],L,axis=0)
            reconstruction = np.array([weights[idx,:]@curr_afferent[indices[idx,:],:] for idx in range(len(test_indices))])
            #print(reconstruction.shape)
            #print(targets[:,:,n].shape)
            reconstructions[n][l] = reconstruction

            reconstruction_error[n,l] = sequential_correlation(reconstruction,np.roll(targets[:,:,n],L,axis=0))[0]
            reconstruction_corr[n,l] = np.corrcoef(reconstruction[:,0],np.roll(targets[:,0,n],L))[0,1]

    return index, reconstruction_error, reconstruction_corr, reconstructions, (weights,indices)


def sequential_correlation(trails1,trails2):
    """Compute the correlation between two signals from their delay representations
    
    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal (shape Txd)
        trails2 (numpy.ndarray): delay vectors of the second signal (shape Txd)
        
    Returns:
        float: Mean correlation between the two delay representations
    
    """
    
    nTrails, trailDim = np.shape(trails1)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trails1[:,idx], trails2[:,idx])[0,1])

    corrcoef=np.nanmean(np.array(corrcoefs))
    
    return corrcoef, corrcoefs[0]

def SurrGen(channel,X,ind,L,eln,seed=0):
    #print(X.shape)
    if seed == 0:
        np.random.seed(time.time_ns()%(2**32-1))
    else:
        np.random.seed(seed)
    kn=np.random.randint(0,L,1)[0]-1
    surr = np.zeros(L)
    for j in range(L):
        kn += 1
        surr[j] = X[kn,0]
        kn = ind[kn][np.random.randint(0,eln[kn],1)[0]]
        if kn==L-1:
            kn=L//2
    return channel, surr

def indexed_SurrGen(index,channel,X,ind,L,eln,seed=0):
    return index, SurrGen(channel,X,ind,L,eln,seed=seed)[1]

def SurrogateMap(X,channel):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    d, indices = nbrs.kneighbors(X)
    threshold = np.percentile(d[:,1],10)
    #print('Twin Surrogate Threshold: ' + str(threshold))
    
    nbrs = NearestNeighbors(radius=threshold, algorithm='ball_tree').fit(X)
    d, indices = nbrs.radius_neighbors(X)
    #indices = [list(i) for i in indices]
    #print(indices)
    #u,a = np.unique(indices,return_inverse=True)
    ind = indices # [u[a[i]] for i in range(len(a))]
    eln = [len(i) for i in ind]
    return channel, ind, eln

def indexed_SurrogateMap(passthrough,X,channel):
    channel, ind, eln = SurrogateMap(X,channel)
    return passthrough, ind, eln

def Curr_channel_Gen(channel,X_channel,ind,eln,T,n_surrogates,return_full_vector=False):
    D = X_channel.shape[1]
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    curr_surrs = [[] for sn in range(n_surrogates)]
    if return_full_vector:
        for sn in range(n_surrogates):
            kn = np.random.randint(0,T,1)[0]-1
            curr_surrs[sn] = np.zeros(T,D)
            for j in range(T):
                kn+=1
                curr_surrs[sn][j] = X_channel[kn,:]
                kn = ind[kn][np.random.randint(0,eln[kn],1)[0]]
                if kn == T-1:
                    kn = T//2
    else: 
        for sn in range(n_surrogates):
            kn = np.random.randint(0,T,1)[0]-1
            curr_surrs[sn] = np.zeros(T)
            for j in range(T):
                kn+=1
                curr_surrs[sn][j] = X_channel[kn,0]
                kn = ind[kn][np.random.randint(0,eln[kn],1)[0]]
                if kn == T-1:
                    kn = T//2
    return channel, curr_surrs

def parallel_twin_surrogates(X,n_surrogates,Tshift=0,max_processes=64,seed=0,return_full_vectors=False):
    '''Create twin surrogates for significance evaluation and hypothesis testing with each surrogate created in parallel
    
    Args:
        X (numpy.ndarray): (Dimension x Time x Channels) multivariate signal for which we want to generate surrogate time series
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    np.random.seed(seed)
    T,D,C = X.shape

    elns = [[] for c in range(C)]
    inds = [[] for c in range(C)]

    surrogate_time = time.time_ns()
    surrs = [[] for sn in range(n_surrogates)]

    #Generate the surrogate swapping indices for each channel in parallel
    with Pool(processes=max_processes) as p:
        processes = tqdm.tqdm([p.apply_async(indexed_SurrogateMap,args = (channel,X[:,:,channel],channel)) for channel in range(C)],desc='Surrogate NNS')

        for proc in processes:
            curr_map = proc.get()
            elns[curr_map[0]] = curr_map[2]
            inds[curr_map[0]] = curr_map[1]
    
        print(f'Surrogate Map generation time: {(time.time_ns()-surrogate_time)*10**-9} s')
        for sn in range(n_surrogates):
            surrs[sn] = np.zeros((T,C))
        # Generate all of the surrogates in parallel for each channel
        
        seeds = np.random.randint(0,2**32-1,n_surrogates)
        
        channel_proccesses = tqdm.tqdm([p.apply_async(Curr_channel_Gen,args=(c,X[:,:,c],inds[c],elns[c],T,n_surrogates,return_full_vectors)) for c in range(C)])

        for proc in channel_proccesses:
            channel, curr_surrs = proc.get()
            for sn in range(n_surrogates):
                surrs[sn][:,channel] = curr_surrs[sn]
    return surrs

def AddNextOptimum(pair_mask, dims, maxFCFs, smoothing=0.5,fullrange = False):
    ys = [[] for p in range(pair_mask[pair_mask].shape[0])]
    yinds = [[] for p in range(pair_mask[pair_mask].shape[0])]
    ind = 0
    for n,m in zip(*pair_mask):
        yinds[ind] = (n,m)
        ys[ind] = maxFCFs[:,n,m][~np.isnan(maxFCFs[:,n,m])]
    
    ys = np.array(ys)
    ys = ys.T
    opts = FindFCFOptimums(dims,ys,smoothing=smoothing, fullrange=fullrange)

    return opts,yinds

def FindFCFOptimums(x,y,smoothing=0.5,fullrange = False,plot=False,to_plot=1):
    """ FindFCFOptimums:
        Args:
        x: L length array
        y: L x N matrix
        smoothing = 0.5: float for the degree of smoothing applied to the y curve

        returns:

        opt1: N length array, opt2: N length array
        
        where: 
        elbow_y: [arg,val] the elbow of each curve defined by the y values for all N smoothed by a value smoothing
        threshold_y: [arg,val] the value closest to and under the 95% cutoff for the max (TODO)
    """

    print(f'shape x: {x.shape}, shape y: {y.shape}')
    max_y = y[:,-1] # should maybe be y[-1]?
    min_y = y[:,0] # should maybe be y[0]?

    yline = np.repeat(x[:,np.newaxis],y.shape[0],axis=1).T
    yline = (yline-np.min(x))/(np.max(x)-np.min(x))
    yline = np.array([(yline[i,:]*(max_y[i]-min_y[i])+min_y[i]) for i in range(y.shape[0])])

    spliney = [ interpolate.splrep(x,y[i,:],s=smoothing) for i in range(y.shape[0])]
    spliney_line = [ interpolate.splrep(x,yline[i,:],s=smoothing) for i in range(y.shape[0]) ]
    x_span = x
    if fullrange:
        x_span = np.arange(x[0],x[-1])
    smooth_y = np.array([interpolate.splev(x_span,spliney[i],der=0) for i in range(y.shape[0])])
    extrap_line = np.array([interpolate.splev(x_span,spliney_line[i],der=0) for i in range(y.shape[0])])
    rotate_y = smooth_y - extrap_line

    if plot:
        fig = plt.figure()
        a = plt.axes()
        a.plot(x_span,yline[to_plot,:],label = "Line")
        a.plot(x_span,y[to_plot,:],'x',label = 'Unsmoothed')
        a.plot(x_span,smooth_y[to_plot,:],label='Smoothed')
        plt.legend()
        plt.show()

    elbow_y_arg = np.argmax(rotate_y,axis=1)
    elbow_val = np.zeros(y.shape[0])*np.nan
    for i in range(y.shape[0]):
        if np.isin(elbow_y_arg[i],x):
            elbow_val[i] = y[i,np.argwhere(elbow_y_arg[i]==x)[0]]

    if not fullrange:
        elbow_y_arg = x[elbow_y_arg]

    return elbow_y_arg, elbow_val

def repeated_value_early_stop(vals,n=3):
    uniques, counts = np.unique(vals[~np.isnan(vals)],return_counts=True)
    if (counts >= n).any():
        return True, uniques[np.argmax(counts)]
    else:
        return False, None

def ParallelParameterSearch(X,dim_min=1, dim_max=30, kfolds=5, delay=0,
                     lags=np.arange(-8,9),lagstep=0, random_projection = False , mask = None, transform='fisher',
                     node_ratio = 1., compute_pvalue = False, n_surrogates = 10, normal_pval=False, pval_threshold=0.05, 
                     min_pairs = 1, dim_search_stopping_num = 3, save=True, save_path = './', retain_test_set = True, 
                     max_processes = 64, early_stop = False, only_hubs = False, find_optimum_dims = False, 
                     seed = 0,pval_correction='None'):
    """ ParallelParameterSearch
    Args --
    X: ndarray (time x neuron) that is the input time series,
    dim_min=1: minimum delay space embedding to use,
    dim_max=30: maximum delay space embedding to use, 
    kfolds=5: number of partitions to use for cross validation, 
    delay=0: delay length to use for delay embeddings if using 0 we approximate the best based on Mutual information,
    lags=np.arange(-8,9): array of integer lags to test over for extended cross mapping,
    lagstep=0: optional set amount for how many time index steps the lags count for (if 0 lags gets set to delay*lags, otherwise lagsteo*lags), 
    random_projection = False: if we want all time series to recieve a random projection after delay embedding, 
    mask = None: boolean array that sets pairs we don't want to compute, 
    transform='fisher': whatever transform you want to do to the ccm value. only fisher is implemented right now, any other input will just return the corrlation coefficient,
    node_ratio = 1: The ratio of nodes to do the parameter search on ranked by their average CCM over columns (strength of causality). , 
    compute_pvalue = False: Whether the Significance should be tested for all nodes at the highest embedding dimension, 
    n_surrogates = 10: The number of surrogate series that should be made per kfold for the significance calculation, 
    normal_pval=False: Whether we should assume the null distributions are approximately normally distributed, 
    pval_threshold=0.05: The threshold we should use to ignore non-significant connections, 
    min_pairs = 1: The minimum number of pairs we want to do the parameter search on (accounts for possibly to small pval_threshold), 
    dim_search_stopping_num = 3: The number of times the optimum value of a pair's downstream complexity needs to be achieved before we eliminate it from further iterations of the search, 
    save=True: Whether to save the outputs, 
    save_path = './': The file path desired to save the outputs to, 
    retain_test_set = True: Whether the test set of the Kfolds should be unchanged in surrogate generation, 
    max_processes = 64: Maximum number of processes that should be computed asynchronously at any time, 
    early_stop = False: Whether the parameter search stops with repeated optimum values, 
    only_hubs = False: Whether the parameter search will compute on only the most hubbed nodes, 
    find_optimum_dims = False: Whether to actually do the parameter search TODO: Will be phased out of this method, 
    seed = 0: If a certain seed should be used (0 for a random seed),
    pval_correction='None': Any correction we want to use for the cross validation pvalues (only none or bonferoni are implemented)

    Outputs -- Tuple of values (honestly depending for most purposes you should just load everything from the output file instead of relying on this)
    averaged_accuracy: FCF for every pair averaged over kfolds indexed by [embedding dimension, efferent, afferent, lag], 
    pvals: Pvalues at maximum embedding dimension indexed by [efferent, afferent, lag], 
    efcf: full fcf , 
    surrogate_fcfs, 
    surrogates, 
    lags, 
    test_indices, 
    train_indices, 
    rolled_delay_vectors
    """
    
    T,N = X.shape
    # If a delay is not given, take the test delay to be the average of the Mutual information method approximations for all time series given
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    hub_nodes_needed = int(np.ceil(node_ratio*N))
    
    if seed == 0:
        np.random.seed(time.time_ns()%(2**32-1))



    important_times = {'Date': datetime.today().strftime('%Y-%m-%d') }

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    time_start = time.time_ns()

    with Pool(processes=max_processes) as p:
        if delay == 0:
            print('Finding Optimum Delay')
            process_outputs = [[] for n in range(N)]
            MIDelays = np.zeros(N)
            process_outputs = tqdm.tqdm([p.apply_async(remote_ApproximatebestTau, args = (i,X[:,i],0,10,50,False,0.1)) for i in range(N)],desc='Mutual Information')
            for out in process_outputs:
                MIDelays[out.get()[0]] = out.get()[1]
            delay = int(np.max([1,np.min(MIDelays)]))
            print(MIDelays)

        print(f'Using Delay: {delay}')
        #TODO: Determine the highest dim based on the Takens' Dimension (Considering Nearest Neighbors method)
        test_dims = np.arange(dim_min,dim_max+1)
        num_dims = test_dims.shape[0]

        # TODO: the argument of the function should not be altered in this function
        if lagstep==0:
            lags = delay*lags
        else:
            lags = lagstep*lags

        #First test at the maximum dim to find the limiting value and the significance interval
        tShift = np.abs(delay)*(dim_max-1)  #Max time shift
        tDelay = T - tShift     #Length of delay vectors

        #Determine the size of the test set and the test / train indices
        t_test = tDelay // kfolds
        iStart = tDelay - t_test; iEnd = tDelay
        test_indices = np.arange(iStart,iEnd)
        train_indices = np.arange(0,iStart-tShift)    

        #Build delay space projection TODO: This whole random projection buisness is kinda fishy so I this really isn't implemented.
        delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], X.T)),2)
        un_proj_delay_vectors = np.array(delay_vectors)

        if random_projection:
            proj_A = np.array([np.random.standard_normal((dim_max,dim_max)) for i in range(N)])
            delay_vectors = np.einsum('imd,tdi->tmi',proj_A,delay_vectors)
            if save:
                np.save(save_path+'projection_operator.npy',proj_A)
        
        if save:
            np.save(save_path+'projected_delay_vectors.npy',delay_vectors)

        # build the rolled set of delay projections for each fold, the FCF is invariant over total shifts 
        # so we can always just compute it in the same interval but with rolled data
        rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]

        # Build the test and training sets
        targets = [rolled_delay_vectors[k][test_indices,:,:] for k in range(kfolds)]
        lib_targets = [rolled_delay_vectors[k][train_indices,:,:] for k in range(kfolds)]

        efcf = None
        eccm = None
        if os.path.isdir(save_path+f'dim{dim_max}/'):
            if os.path.isfile(save_path+f'dim{dim_max}/eFCF.npy'):
                efcf = np.load(save_path+f'dim{dim_max}/eFCF.npy')
        else:
            os.mkdir(save_path+f'dim{dim_max}/')   
        if efcf is None:
            eccm_full_process_start_time = time.time_ns()
            # build the weight matrices for all the variables and folds
            #print('Building CCM Mappings')

            reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
            eccm = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
            
            process_outputs = tqdm.tqdm([p.apply_async(CCMColumn,args=((k,j),rolled_delay_vectors[k],j,train_indices,test_indices,lags)) for k in range(kfolds) for j in range(N)],desc='Computing Columns')
            for proc in process_outputs:
                item = proc.get()
                reconstruction_accuracy[item[0][0],:,item[0][1],:] = item[1]
                eccm[item[0][0],:,item[0][1],:] = item[2]
            
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{dim_max}ReconstructionTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{dim_max}ReconstructionPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            efcf = reconstruction_accuracy


            #eccm = corr_accuracy
            if transform == 'fisher':
                efcf = np.arctanh(efcf)
                eccm = np.arctanh(eccm)
        if save:
            np.save(save_path + f'dim{dim_max}/eFCF.npy',efcf)
            np.save(save_path + f'dim{dim_max}/eCCM.npy',eccm)
            np.save(save_path + f'lags.npy',lags)
            with open(save_path+'benchmarkings.yaml','w') as outfile:
                yaml.dump(important_times,outfile,default_flow_style=False)

        # Average over folds and determine the limiting statistics
        averaged_accuracy = np.nanmean(efcf,axis=0)
        hubness = np.nanmean((np.nanmax(averaged_accuracy[:,:,lags>=0],axis=2)*(1-np.identity(N))),axis=1)
        opt_lag_indices = np.argmax(averaged_accuracy,axis=2)
        opt_lags = lags[opt_lag_indices.flatten()].reshape(opt_lag_indices.shape)
        pos_lags = np.argwhere(lags >= 0)
        zero_lag = np.argmin(np.abs(lags))
        opt_pos_lag_indices = np.argmax(averaged_accuracy[:,:,zero_lag:],axis=2)+zero_lag
        maxed_fcfs = averaged_accuracy[:,:,pos_lags]
        maxed_fcf_lags = np.nanargmax(maxed_fcfs,axis=2)
        max_lag_args = maxed_fcf_lags
        maxed_fcf_lags = np.array([[lags[pos_lags[maxed_fcf_lags[i,j]]] for i in range(N)] for j in range(N)])        
        if save:
            np.save(save_path+'max_dim_opt_lag.npy',maxed_fcf_lags)
        
        maxed_fcfs = np.nanmax(maxed_fcfs,axis=2)
        downward_strength = np.nansum(maxed_fcfs,axis=0)
        
        efcf_no_diag = np.array(efcf)
        for i in range(N):
            efcf_no_diag[:,i,i,:] = 0


        hub_fcf_ordered = np.flip(np.argsort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))
        fcf_ordered = np.flip(np.sort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))

        hub_rank_data = {'Channel': hub_fcf_ordered, 'Hubness': fcf_ordered}

        df = pd.DataFrame(hub_rank_data)

        df.to_csv(save_path+'hub_rankings.csv',sep=',', na_rep='Null')

        print(f'Hub Rankings: \n channels: {hub_fcf_ordered} \n values:   {np.flip(np.sort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))}')
        
        if only_hubs:
            return 
        
        unordered_hub_nodes = np.flip(np.argsort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))[:hub_nodes_needed]
        
        hub_nodes = np.sort(unordered_hub_nodes)
        print(f'Hubs to be computed on: {hub_nodes}')
        hub_mask = np.isin(np.arange(N),hub_nodes)
        hub_mask = np.repeat(hub_mask[:,np.newaxis],N,axis=1).T
        #print(f'FCFs to compute for hub nodes: {hub_mask}')

        #averaged_corr_accuracy = np.nanmean(eccm,axis=0)
        surrogate_fcfs = None
        surrogates = None

        if os.path.isdir(save_path+'/surrogates'):
            surrogates = [[[] for k in range(n_surrogates)] for sn in range(kfolds)]
            for f in os.listdir(save_path+'/surrogates'):
                sn = int(f[-7])
                k = int(f[-5])
                surrogates[k][sn] = np.load(save_path+'/surrogates/'+f)
        
        if os.path.isfile(save_path+'surrogate_fcf.npy'):
            surrogate_fcfs = np.load(save_path+'surrogate_fcf.npy')
        elif compute_pvalue:
            # Compute the significance intervals
            # Create Surrogate time series: n_surrogate list of NxT time series
            print('Creating twin surrogates')
            
            eccm_full_process_start_time = time.time_ns()
            already_made = True
            
            surr_data = rolled_delay_vectors
            if retain_test_set:
                surr_data = lib_targets
            if surrogates is None:
                surrogates = [[] for k in range(kfolds)]
                already_made = False
                for k in range(kfolds):
                    print(f'Surrogates for fold {k}')
                    surrogates[k] = parallel_twin_surrogates(surr_data[k],n_surrogates=n_surrogates,Tshift = tShift,seed=np.random.randint(0,2**32-1,1)[0])

            if save and not already_made:
                if not os.path.isdir(save_path+'/surrogates'):
                    os.mkdir(save_path+'/surrogates')
                for k in range(kfolds):
                    for n in range(n_surrogates):
                        np.save(save_path+'/surrogates/'+f'surrogate{n},{k}.npy',surrogates[k][n]) 

            surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,1))
            surrogate_corrs = np.zeros((n_surrogates,kfolds,N,N,1))

            print(f'Finished Surrogate Generation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{dim_max}SurrogateConstructionTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{dim_max}SurrogateConstructionPerProcess'] = (curr_time_taken/(N*kfolds*n_surrogates))*(10**-9)


            with open(save_path+'benchmarkings.yaml','w') as outfile:
                yaml.dump(important_times,outfile,default_flow_style=False)

            eccm_full_process_start_time = time.time_ns()

            # Generate nearest maps for all surrogates
            completed_surrs = np.zeros(n_surrogates,dtype = bool)

            if os.path.isfile(save_path+'completed_surrs.npy'):
                completed_surrs = np.load(save_path+'completed_surrs.npy')

            for sn in range(n_surrogates):
                sn_start_time = time.time_ns()

                if completed_surrs[sn]:
                    continue

                surrogate_nns = [[[] for i in range(N)] for j in range(kfolds)]

                surrogate_delays = [np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], surrogates[k][sn].T)),2) for k in range(kfolds)]
                if retain_test_set:
                    surrogate_delays = [np.concatenate((surrogate_delays[k],rolled_delay_vectors[k][test_indices,:,:]),axis=0) for k in range(kfolds)]

                surr_shape_diff = abs(surrogate_delays[0].shape[0]-X.shape[0])
                train_indices_surr = train_indices[:-surr_shape_diff]
                test_indices_surr = test_indices-surr_shape_diff
                

                if retain_test_set:
                    process_outputs = tqdm.tqdm([p.apply_async(
                    CCMColumn,args=((k,hub_nodes[j],sn),surrogate_delays[k],hub_nodes[j],train_indices_surr,test_indices_surr,np.array([0])))
                      for k in range(kfolds) for j in range(N)],desc=f'Computing Columns for surrogate set {sn}')
                    
                    """process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                            args = ((k,hub_nodes[j],sn),surrogate_delays[k][:,:,hub_nodes[j]],
                                                                    train_indices_surr,test_indices_surr,dim_max+1)) 
                                                            for j in range(hub_nodes_needed) for k in range(kfolds)],desc=f'NNS {sn}')
                    """
                    """ for proc in process_outputs:
                        item = proc.get()
                        #surrogate_nns += [item[1]]
                        #nns_indices += [[item[0][0],item[0][1]]]
                        surrogate_nns[item[0][0]][item[0][1]] = item[1]
                
                    processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                            args = ([k,hub_nodes[i],sn],surrogate_nns[k][hub_nodes[i]],targets[k], surrogate_delays[k],test_indices_surr,np.array([0])))
                                            for k in range(kfolds) for i in range(hub_nodes_needed)],desc=f'Reconstructions {sn}')"""

                    for proc in process_outputs:
                        item = proc.get()
                        surrogate_fcfs[item[0][2],item[0][0],:,item[0][1],:] = item[1]
                        surrogate_corrs[item[0][2],item[0][0],:,item[0][1],:] = item[2]
                else:
                    #TODO
                    process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                            args = ((0,hub_nodes[j],sn),surrogate_delays[:,:,hub_nodes[j]],
                                                                    train_indices_surr,test_indices_surr,dim_max+1)) 
                                                            for j in range(hub_nodes_needed)],desc=f'NNS {sn}')
                
                    for proc in process_outputs:
                        item = proc.get()
                        for k in range(kfolds):
                            surrogate_nns[k][item[0][1]] = item[1]
                #print(f'Finished Surrogate CCM Mappings in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')


                # Compute the fcf
                    processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                            args = ([0,hub_nodes[i],sn],surrogate_nns[k][hub_nodes[i]],targets[k], surrogate_delays[k],test_indices_surr[:],np.array([0])))
                                            for i in range(hub_nodes_needed)],desc=f'Reconstructions {sn}')

                    for proc in processes:
                        item = proc.get()
                        for k in kfolds:
                            surrogate_fcfs[item[0][2],k,item[0][1],:,0] = item[1]
                            surrogate_corrs[item[0][2],k,item[0][1],:,0] = item[2]

                completed_surrs[sn] = True
                
                curr_time_taken = time.time_ns()-sn_start_time
            
                important_times[f'Dim{dim_max}Surrogate{sn}CCMTotal'] = curr_time_taken*(10**-9)
                important_times[f'Dim{dim_max}Surrogate{sn}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
                
                
                if save:
                    np.save(save_path+'intermediate_surrogate_fcf.npy',surrogate_fcfs)
                    np.save(save_path+'completed_surrs.npy',completed_surrs)
                    with open(save_path+'benchmarkings.yaml','w') as outfile:
                        yaml.dump(important_times,outfile,default_flow_style=False)


            print(f'Finished Surrogate evaluation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            important_times[f'Dim{dim_max}AllSurrogatesCCMTotal'] = curr_time_taken*(10**-9)                

            if transform == 'fisher':
                surrogate_fcfs = np.arctanh(surrogate_fcfs)

            if save:
                np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)

        pvals = np.zeros((N,N,lags.shape[0]))*np.nan

        flat_surr_fcf = surrogate_fcfs.reshape(-1,*surrogate_fcfs.shape[2:])
        flat_surr_fcf = flat_surr_fcf[:,:,:,0]
        if os.path.isfile(save_path+'maxdim_pvalues.npy'):
            pvals = np.load(save_path+'maxdim_pvalues.npy')
            corrected_pval = np.load(save_path+'corrected_pvalue_threshold.npy') # Move this into the output yaml
            significant_pair_lags = np.load(save_path+'maxdim_SignificantPairs.npy')    
        else:
            if normal_pval:
                # TODO: I think it might be better to use a skewed-normal but I am going with this right now
                surrogate_means = np.nanmean(flat_surr_fcf,axis=0)
                surrogate_stds = np.nanstd(flat_surr_fcf,axis=0)
                for l in range(lags.shape[0]):
                    for i in range(hub_nodes.shape[0]):
                        for j in range(N):
                            pvals[hub_nodes[i],j,l] = 1-2*np.abs(stats.norm.cdf(averaged_accuracy[hub_nodes[i],j,l],loc=surrogate_means[i,j],scale=surrogate_stds[i,j])-0.5)
            else:
                #TODO
                pvals = 1-2*np.abs(np.array([[[stats.percentileofscore(flat_surr_fcf[:,i,j],averaged_accuracy[hub_nodes[i],j,k],kind='strict') for k in range(lags.shape[0])] for j in range(N)] for i in range(hub_nodes.shape[0])])/100 - .5)

            if pval_correction == 'Bonferroni':
                pvals *= kfolds

            significant_pair_lags = np.argwhere(pvals <= pval_threshold)
            corrected_pval = pval_threshold

            while significant_pair_lags.shape[0] < min_pairs:
                corrected_pval += pval_threshold
                significant_pair_lags = np.argwhere(pvals <= corrected_pval)
                if corrected_pval >= 1:
                    print("Could not find enough significant nodes to satisfy the minimum node number, using all nodes")
                    corrected_pval = 1
                    break
            if save:
                np.save(save_path+'maxdim_pvalues.npy',pvals)
                np.save(save_path+'corrected_pvalue_threshold.npy', np.array([corrected_pval])) # Move this into the output yaml
                np.save(save_path+'maxdim_SignificantPairs.npy',significant_pair_lags)

        # TODO: Make sure this does not throw away too many things
        mask = (pvals <= corrected_pval)
        print(f'Required pvalue: {corrected_pval} to satisfy {min_pairs} with {np.argwhere(mask.any(axis=2)).shape[0]} pairs')

        #Iterate through dimensions for the elbow
        incomplete_pairs = np.repeat(((mask.any(axis=2)).any(axis=0))[:,np.newaxis],N,axis=1).T
        print(f'Checking mask sizes: hub_mask={hub_mask.shape}, pvalue mask: {incomplete_pairs.shape}')
        incomplete_pairs &= hub_mask

        if save:
            np.save(save_path+'significant_hub_nodes.npy', np.argwhere(np.any(incomplete_pairs,axis=0))[:,0])

    
        #print(f'Significant Pairs: {pandas.DataFrame(np.array(incomplete_pairs,dtype=int))}')
        found_nodes = np.zeros((N,N),dtype=bool)


        reconstruction_accuracies = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
        reconstruction_corrs = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan

        reconstruction_accuracies[:,-1,:,:,:] = efcf
        reconstruction_corrs[:,-1,:,:,:] = eccm
        
        if os.path.isfile(save_path+'almost_all_computed_fcfs.npy'):
            reconstruction_accuracies = np.load(save_path+'almost_all_computed_fcfs.npy')
            
        # Run the parallel like this:
        # Build the array of eFCF values for the significant pairs and compute the eFCF over each dim
        # With it parallelized for computing both dim = d_min+i and d_max - i, and a middle dim
        # This generates a curve filled out from both sides and the center, find the elbow of this curve over each i and when it has not changed for a few iterations (heuristic choice) 
        # declare that the early stoping criteria
        tested_optimium_dims = np.ones((dim_max-dim_min,N,N))*np.nan #np.zeros((dims_to_search.shape[0],N,N))
        tested_optimium_vals = np.ones((dim_max-dim_min,N,N))*np.nan
        actual_optimum_dims = np.ones((N,N))*np.nan
        actual_optimum_vals = np.ones((N,N))*np.nan

        searched_dims = np.zeros(dim_max-dim_min+1,dtype=bool)
        searched_dims[-1] = True
        if os.path.isfile(save_path+'searched_dims.npy'):
            searched_dims[np.load(save_path+'searched_dims.npy')] = True

        dims_to_search = np.arange(dim_min,dim_max+1)[~searched_dims]
        opt_num = 0
        
        if not find_optimum_dims:
            return

        while dims_to_search.shape[0] > 0:
            if not incomplete_pairs.any():
                break            
            dims = np.array([dims_to_search[0],dims_to_search[dims_to_search.shape[0] // 2],dims_to_search[-1]])
            dims = np.unique(dims)
            dims_to_search = dims_to_search[~np.isin(dims_to_search,dims)]

            print(f'Computing CCM for dims: {dims} and {np.argwhere(incomplete_pairs).shape[0]} pairs')
            
            job_num = 0
            to_check = [[] for n in range(N)]
            for i,j in np.argwhere(incomplete_pairs):
                to_check[i] += [j]
                job_num += 1
            columns = []
            efferents = np.argwhere(incomplete_pairs.any(axis=0))[:,0]

            for d in range(dims.shape[0]):
                d_start_time = time.time_ns()

                """process_outputs = tqdm.tqdm([p.apply_async(FullCCMColumn,args=((afferents[i],k,dims[d]),rolled_delay_vectors[k][:,:dims[d],:],afferents[i],train_indices,test_indices,lags)) 
                                             for k in range(kfolds) for i in range(len(afferents))],desc=f'FCF for {dims[d]}')
                """
                process_outputs = tqdm.tqdm([p.apply_async(
                    CCMColumn,args=((efferents[i],k,dims[d]),
                                    rolled_delay_vectors[k][:,:dims[d]+1,:],efferents[i],train_indices,test_indices,lags)) 
                                    for k in range(kfolds) 
                                    for i in range(len(efferents))],desc=f'Computing Columns for dim = {dims[d]}')

                
                for iK,proc in enumerate(process_outputs):
                    item = proc.get()
                    eff,kfold,d_curr = item[0]
                    curr_fcfs = item[1]
                    corr_fcfs = item[2]
                    if transform == 'fisher':
                        curr_fcfs = np.arctanh(curr_fcfs)
                        corr_fcfs = np.arctanh(corr_fcfs)

                    reconstruction_accuracies[kfold,d_curr-dim_min,:,eff,:] = curr_fcfs 
                    reconstruction_corrs[kfold,d_curr-dim_min,:,eff,:] = corr_fcfs 

                searched_dims[dims[d]-dim_min] = True
                if save:
                    np.save(save_path+'all_fcfs.npy',reconstruction_accuracies)
                    np.save(save_path+'searched_dims.npy',np.argwhere(searched_dims))
                
                curr_time_taken = time.time_ns()-d_start_time
            
                important_times[f'Dim{dims[d]}CCMTotal'] = curr_time_taken*(10**-9)
                important_times[f'Dim{dims[d]}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
                
                with open(save_path+'benchmarkings.yaml','w') as outfile:
                    yaml.dump(important_times,outfile,default_flow_style=False)

        
           
            y = np.array([np.nanmean(reconstruction_accuracies[:,searched_dims,i,j,max_lag_args[i,j]],axis=0) for i,j in np.argwhere(incomplete_pairs)])
            #y = np.array([np.nanmax(np.nanmean(reconstruction_accuracies,axis=0)[searched_dims,i,j,:],axis=1) for i,j in np.argwhere(incomplete_pairs)])
            curr_opts = FindFCFOptimums(test_dims[searched_dims],y,fullrange=False)
            curr_pair = 0
            print(curr_opts[0].shape)

            for i,j in np.argwhere(incomplete_pairs):
                tested_optimium_dims[opt_num,i,j] = curr_opts[0][curr_pair]
                tested_optimium_vals[opt_num,i,j] = curr_opts[1][curr_pair]
                curr_pair += 1

            for i,j in np.argwhere(incomplete_pairs):
                check, count_max = repeated_value_early_stop(tested_optimium_dims[:,i,j])
                incomplete_pairs[i,j] = ~check or ~early_stop
                found_nodes[i,j] = check
                if check or (dims_to_search.shape[0] == 0):
                    actual_optimum_dims[i,j] = count_max
                    actual_optimum_vals[i,j] = tested_optimium_vals[opt_num,i,j]
                    tested_optimium_dims[opt_num:,i,j] = count_max
                    tested_optimium_vals[opt_num:,i,j] = tested_optimium_vals[opt_num,i,j]
            opt_num += 1

        if save:
            np.save(save_path+'last_computed_fcfs.npy',reconstruction_accuracies)
            np.save(save_path+'current_optimum_dimensions.npy',tested_optimium_dims)
            np.save(save_path+'current_optimum_dimension_values.npy',tested_optimium_vals)

        if save:
            np.save(save_path+'almost_all_computed_fcfs.npy',reconstruction_accuracies)
            np.save(save_path+'almost_all_computed_eccms.npy',reconstruction_corrs)
            np.save(save_path+'final_optimum_dimensions.npy',actual_optimum_dims)

        stopped_too_early = np.logical_and(np.logical_not(np.isnan(actual_optimum_dims)),np.isnan(actual_optimum_vals))

        last_nns = [[[[] for n in range(N)] for m in range(N)] for k in range(kfolds)]
        
        if (np.argwhere(stopped_too_early).shape[0]) > 0:
            print(f'Still need to compute fcf for pairs: {np.argwhere(stopped_too_early)}')
            process_outputs = tqdm.tqdm([p.apply_async(remote_build_nn_single, 
                                        args = ((i,j,k),rolled_delay_vectors[k][:,:int(actual_optimum_dims[i,j]),i],
                                                train_indices,test_indices,int(actual_optimum_dims[i,j])+1)) 
                                                for k in range(kfolds) for i,j in np.argwhere(stopped_too_early)],desc='Final NNS')

            for ik,out in enumerate(process_outputs):
                vals = out.get()
                i,j,k = vals[0]
                last_nns[k][i][j] = vals[1]
            
            process_outputs = tqdm.tqdm([p.apply_async(indexed_SequentialLaggedReconstruction,args =([k,i,j],targets[k][:,:int(actual_optimum_dims[i,j]),j],lib_targets[k][:,:int(actual_optimum_dims[i,j]),j],last_nns[k][i][j],test_indices,lags))
                                            for k in range(kfolds) for i,j in np.argwhere(stopped_too_early)],desc='Final Reconstructions')
            
            for ik, out in enumerate(process_outputs):
                vals = out.get()
                k,i,j = vals[0]
                curr_fcf = vals[1]
                curr_corr = vals[2]
                d_opt = int(actual_optimum_dims[i,j])
                d_index = d_opt-dim_min
                if transform == 'fisher':
                    curr_fcf = np.arctanh(curr_fcf)
                reconstruction_accuracies[k,d_index,i,j,:] = curr_fcf
                actual_optimum_vals[i,j] = np.nanmean(reconstruction_accuracies[:,d_index,i,j,max_lag_args[i,j]])
            
    all_averaged_accuracies = np.nanmean(reconstruction_accuracies,axis=0)
    
    if save:
        np.save(save_path+'all_computed_fcfs.npy',reconstruction_accuracies)
        np.save(save_path+'all_computed_eccms.npy',reconstruction_corrs)


    total_time = time.time_ns()-time_start

    important_times['TotalTime'] = total_time*(10**-9)

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    print(f'Total Time taken: {total_time*(10**-9)} s')
    np.save(save_path+'time_elapsed.npy', np.array([total_time]))

    return averaged_accuracy, pvals, efcf, surrogate_fcfs, surrogates, lags, test_indices, train_indices, rolled_delay_vectors

#%%
# Main Computation
def ParallelFullECCM(X,d_min=1, dim_max=30, kfolds=5, delay=0,
                     lags=np.arange(-8,9), random_projection = False , mask = None, transform='fisher',
                     node_ratio = 0.1, test_pval = True, compute_pvalue = False, n_surrogates = 10, normal_pval=False, pval_threshold=0.05, 
                     min_pairs = 1, dim_search_stopping_num = 3, save=True, save_path = './', retain_test_set = True, 
                     max_processes = 64, early_stop = False, only_hubs = False, find_optimum_dims = False, 
                     seed = 0,pval_correction='Bonferroni'):
    """
    """

    """
    pseudo code:
        approximate taus via Mutual information with mutual information (this should usually be 1 for very chaotic systems) [parallel]
        approximate dims via false NN [parallel]

        compute eCCM at taus and dims and identify the best lag, with significance:
        
        eCCM subroutine, with surrogates at same time:
            -> take the create that delay embeddings with appropriate dim, tau
            -> compute NNS for each var for their respective dim,tau [parallel]
            -> reconstuct each signal with appropriate dim, tau delay embeddings [parallel]
        
        determine the significant / peaked eCCM lags

        compute eCCM over each tau, dim in an interval around the approximated values with lag bounds over the significant, non flat, non-negative, lag values

        The optimums are now: elbow for dim, highest elbow for tau, peak for lag
        Throw out non-significant efcf, (likely) negative lags
                
    """
    T,N = X.shape
    # If a delay is not given, take the test delay to be the average of the Mutual information method approximations for all time series given
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    hub_nodes_needed = int(np.ceil(node_ratio*N))
    
    if seed == 0:
        np.random.seed(time.time_ns()%(2**32-1))



    important_times = {'Date': datetime.today().strftime('%Y-%m-%d') }

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    time_start = time.time_ns()

    with Pool(processes=max_processes) as p:
        if delay == 0:
            print('Finding Optimum Delay')
            process_outputs = [[] for n in range(N)]
            MIDelays = np.zeros(N)
            process_outputs = tqdm.tqdm([p.apply_async(remote_ApproximatebestTau, args = (i,X[:,i],0,10,50,False,0.1)) for i in range(N)],desc='Mutual Information')
            for out in process_outputs:
                MIDelays[out.get()[0]] = out.get()[1]
            delay = int(np.max([1,np.min(MIDelays)]))
            print(MIDelays)

        print(f'Using Delay: {delay}')
        #TODO: Determine the highest dim based on the Takens' Dimension (Considering Nearest Neighbors method)
        test_dims = np.arange(d_min,dim_max+1)
        num_dims = test_dims.shape[0]

        lags = delay*lags

        #First test at the maximum dim to find the limiting value and the significance interval
        tShift = np.abs(delay)*(dim_max-1)  #Max time shift
        tDelay = T - tShift     #Length of delay vectors

        #Determine the size of the test set and the test / train indices
        t_test = tDelay // kfolds
        iStart = tDelay - t_test; iEnd = tDelay
        test_indices = np.arange(iStart,iEnd)
        train_indices = np.arange(0,iStart-tShift)    

        #Build delay space projection
        delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], X.T)),2)
        un_proj_delay_vectors = np.array(delay_vectors)

        if random_projection:
            proj_A = np.array([np.random.standard_normal((dim_max,dim_max)) for i in range(N)])
            delay_vectors = np.einsum('imd,tdi->tmi',proj_A,delay_vectors)
            if save:
                np.save(save_path+'projection_operator.npy',proj_A)
        
        if save:
            np.save(save_path+'projected_delay_vectors.npy',delay_vectors)

        # build the rolled set of delay projections for each fold, the FCF is invariant over total shifts 
        # so we can always just compute it in the same interval but with rolled data
        rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]

        # Build the test and training sets
        targets = [rolled_delay_vectors[k][test_indices,:,:] for k in range(kfolds)]
        lib_targets = [rolled_delay_vectors[k][train_indices,:,:] for k in range(kfolds)]

        efcf = None
        eccm = None
        if os.path.isfile(save_path + f'eFCFTensorXValidated_dim{dim_max}_delay{delay}.npy'):
            efcf = np.load(save_path + f'eFCFTensorXValidated_dim{dim_max}_delay{delay}.npy')
            #np.save(path + f'eCCMTensorXValidated_dim{d_max}_delay{delay}.npy',eccm)
        else:
            eccm_full_process_start_time = time.time_ns()
            # build the weight matrices for all the variables and folds
            all_nns = [[[] for n in range(N)] for k in range(kfolds)]
            #print('Building CCM Mappings')
            process_outputs =tqdm.tqdm([p.apply_async(remote_build_nn_single,
                                                    args = ((k,j),rolled_delay_vectors[k][:,:,j],
                                                            train_indices,test_indices,
                                                            dim_max+1))
                                                            for j in range(N) for k in range(kfolds)],desc=f'NNS')
            
            for proc in process_outputs:
                item = proc.get()
                all_nns[item[0][0]][item[0][1]] = item[1]
            reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
            eccm = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{dim_max}NNSTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{dim_max}NNSPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            #corr_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

            #print(f'First round NNS computed in {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            eccm_full_process_start_time = time.time_ns()


            process_outputs = tqdm.tqdm([p.apply_async(reconstruction_column,args=((k,n),all_nns[k][n],targets[k],lib_targets[k],test_indices,lags))
                                        for n in range(N) for k in range(kfolds)],desc='Reconstructing')

            for proc in process_outputs:
                item = proc.get()
                reconstruction_accuracy[item[0][0],item[0][1],:,:] = item[1]
                eccm[item[0][0],item[0][1],:,:] = item[2]
            
            #print(f'Processed reconstructions in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')

            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{dim_max}ReconstructionTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{dim_max}ReconstructionPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            efcf = reconstruction_accuracy


            #eccm = corr_accuracy
            if transform == 'fisher':
                efcf = np.arctanh(efcf)
                eccm = np.arctanh(eccm)
        if save:
            np.save(save_path + f'eFCFTensorXValidated_dim{dim_max}_delay{delay}.npy',efcf)
            np.save(save_path + f'eCCMTensorXValidated_dim{dim_max}_delay{delay}.npy',eccm)
            np.save(save_path + f'lags.npy',lags)
            with open(save_path+'benchmarkings.yaml','w') as outfile:
                yaml.dump(important_times,outfile,default_flow_style=False)

        # Average over folds and determine the limiting statistics
        averaged_accuracy = np.nanmean(efcf,axis=0)
        pos_lags = np.argwhere(lags >= 0)
        maxed_fcfs = averaged_accuracy[:,:,pos_lags]
        maxed_fcf_lags = np.nanargmax(maxed_fcfs,axis=2)
        max_lag_args = maxed_fcf_lags
        maxed_fcf_lags = np.array([[lags[pos_lags[maxed_fcf_lags[i,j]]] for i in range(N)] for j in range(N)])
        
        if save:
            np.save(save_path+'max_dim_opt_lag.npy',maxed_fcf_lags)
        
        maxed_fcfs = np.nanmax(maxed_fcfs,axis=2)
        downward_strength = np.nansum(maxed_fcfs,axis=0)
        
        efcf_no_diag = np.array(efcf)
        for i in range(N):
            efcf_no_diag[:,i,i,:] = 0

        hub_fcf_ordered = np.flip(np.argsort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))
        fcf_ordered = np.flip(np.sort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))

        hub_rank_data = {'Channel': hub_fcf_ordered, 'Hubness': fcf_ordered}

        df = pd.DataFrame(hub_rank_data)

        df.to_csv(save_path+'hub_rankings.csv',sep=',', na_rep='Null')

        print(f'Hub Rankings: \n channels: {hub_fcf_ordered} \n values:   {np.flip(np.sort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))}')
        
        if only_hubs:
            return
        
        unordered_hub_nodes = np.flip(np.argsort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))[:hub_nodes_needed]
        
        hub_nodes = np.sort(unordered_hub_nodes)
        print(f'Hubs to be computed on: {hub_nodes}')
        hub_mask = np.isin(np.arange(N),hub_nodes)
        hub_mask = np.repeat(hub_mask[:,np.newaxis],N,axis=1).T
        #print(f'FCFs to compute for hub nodes: {hub_mask}')

        #averaged_corr_accuracy = np.nanmean(eccm,axis=0)
        surrogate_fcfs = None
        surrogates = None

        if os.path.isdir(save_path+'/surrogates'):
            surrogates = [[[] for k in range(n_surrogates)] for sn in range(kfolds)]
            for f in os.listdir(save_path+'/surrogates'):
                sn = int(f[-7])
                k = int(f[-5])
                surrogates[k][sn] = np.load(save_path+'/surrogates/'+f)
        
        if os.path.isfile(save_path+'surrogate_fcf.npy'):
            surrogate_fcfs = np.load(save_path+'surrogate_fcf.npy')
        elif compute_pvalue:
            # Compute the significance intervals
            # Create Surrogate time series: n_surrogate list of NxT time series
            print('Creating twin surrogates')
            
            eccm_full_process_start_time = time.time_ns()
            already_made = True
            
            surr_data = rolled_delay_vectors
            if retain_test_set:
                surr_data = lib_targets
            if surrogates is None:
                surrogates = [[] for k in range(kfolds)]
                already_made = False
                for k in range(kfolds):
                    print(f'Surrogates for fold {k}')
                    surrogates[k] = parallel_twin_surrogates(surr_data[k],n_surrogates=n_surrogates,Tshift = tShift,seed=np.random.randint(0,2**32-1,1)[0])

            if save and not already_made:
                if not os.path.isdir(save_path+'/surrogates'):
                    os.mkdir(save_path+'/surrogates')
                for k in range(kfolds):
                    for n in range(n_surrogates):
                        np.save(save_path+'/surrogates/'+f'surrogate{n},{k}.npy',surrogates[k][n]) 

            surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,1))
            surrogate_corrs = np.zeros((n_surrogates,kfolds,N,N,1))

            print(f'Finished Surrogate Generation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{dim_max}SurrogateConstructionTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{dim_max}SurrogateConstructionPerProcess'] = (curr_time_taken/(N*kfolds*n_surrogates))*(10**-9)


            with open(save_path+'benchmarkings.yaml','w') as outfile:
                yaml.dump(important_times,outfile,default_flow_style=False)

            eccm_full_process_start_time = time.time_ns()

            # Generate nearest maps for all surrogates
            completed_surrs = np.zeros(n_surrogates,dtype = bool)

            if os.path.isfile(save_path+'completed_surrs.npy'):
                completed_surrs = np.load(save_path+'completed_surrs.npy')

            for sn in range(n_surrogates):
                sn_start_time = time.time_ns()

                if completed_surrs[sn]:
                    continue

                surrogate_nns = [[[] for i in range(N)] for j in range(kfolds)]

                surrogate_delays = [np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], surrogates[k][sn].T)),2) for k in range(kfolds)]
                if retain_test_set:
                    surrogate_delays = [np.concatenate((surrogate_delays[k],rolled_delay_vectors[k][test_indices,:,:]),axis=0) for k in range(kfolds)]

                surr_shape_diff = abs(surrogate_delays[0].shape[0]-X.shape[0])
                train_indices_surr = train_indices[:-surr_shape_diff]
                test_indices_surr = test_indices-surr_shape_diff

                if retain_test_set:
                    process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                            args = ((k,hub_nodes[j],sn),surrogate_delays[k][:,:,hub_nodes[j]],
                                                                    train_indices_surr,test_indices_surr,dim_max+1)) 
                                                            for j in range(hub_nodes_needed) for k in range(kfolds)],desc=f'NNS {sn}')
                
                    for proc in process_outputs:
                        item = proc.get()
                        #surrogate_nns += [item[1]]
                        #nns_indices += [[item[0][0],item[0][1]]]
                        surrogate_nns[item[0][0]][item[0][1]] = item[1]
                
                    processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                            args = ([k,hub_nodes[i],sn],surrogate_nns[k][hub_nodes[i]],targets[k], surrogate_delays[k],test_indices_surr,np.array([0])))
                                            for k in range(kfolds) for i in range(hub_nodes_needed)],desc=f'Reconstructions {sn}')

                    for proc in processes:
                        item = proc.get()
                        surrogate_fcfs[item[0][2],item[0][0],item[0][1],:,:] = item[1]
                        surrogate_corrs[item[0][2],item[0][0],item[0][1],:,:] = item[2]
                else:
                    #TODO
                    process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                            args = ((0,hub_nodes[j],sn),surrogate_delays[:,:,hub_nodes[j]],
                                                                    train_indices_surr,test_indices_surr,dim_max+1)) 
                                                            for j in range(hub_nodes_needed)],desc=f'NNS {sn}')
                
                    for proc in process_outputs:
                        item = proc.get()
                        for k in range(kfolds):
                            surrogate_nns[k][item[0][1]] = item[1]
                #print(f'Finished Surrogate CCM Mappings in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')


                # Compute the fcf
                    processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                            args = ([0,hub_nodes[i],sn],surrogate_nns[k][hub_nodes[i]],targets[k], surrogate_delays[k],test_indices_surr[:],np.array([0])))
                                            for i in range(hub_nodes_needed)],desc=f'Reconstructions {sn}')

                    for proc in processes:
                        item = proc.get()
                        for k in kfolds:
                            surrogate_fcfs[item[0][2],k,item[0][1],:,0] = item[1]
                            surrogate_corrs[item[0][2],k,item[0][1],:,0] = item[2]

                completed_surrs[sn] = True
                
                curr_time_taken = time.time_ns()-sn_start_time
            
                important_times[f'Dim{dim_max}Surrogate{sn}CCMTotal'] = curr_time_taken*(10**-9)
                important_times[f'Dim{dim_max}Surrogate{sn}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
                
                
                if save:
                    np.save(save_path+'intermediate_surrogate_fcf.npy',surrogate_fcfs)
                    np.save(save_path+'completed_surrs.npy',completed_surrs)
                    with open(save_path+'benchmarkings.yaml','w') as outfile:
                        yaml.dump(important_times,outfile,default_flow_style=False)


            print(f'Finished Surrogate evaluation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            important_times[f'Dim{dim_max}AllSurrogatesCCMTotal'] = curr_time_taken*(10**-9)                

            if transform == 'fisher':
                surrogate_fcfs = np.arctanh(surrogate_fcfs)

            if save:
                np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)

        pvals = np.zeros((N,N,lags.shape[0]))*np.nan

        flat_surr_fcf = surrogate_fcfs.reshape(-1,*surrogate_fcfs.shape[2:])
        flat_surr_fcf = flat_surr_fcf[:,:,:,0]
        if os.path.isfile(save_path+'maxdim_pvalues.npy'):
            pvals = np.load(save_path+'maxdim_pvalues.npy')
            corrected_pval = np.load(save_path+'corrected_pvalue_threshold.npy') # Move this into the output yaml
            significant_pair_lags = np.load(save_path+'maxdim_SignificantPairs.npy')    
        else:
            if normal_pval:
                # TODO: I think it might be better to use a skewed-normal but I am going with this right now
                surrogate_means = np.nanmean(flat_surr_fcf,axis=0)
                surrogate_stds = np.nanstd(flat_surr_fcf,axis=0)
                for l in range(lags.shape[0]):
                    for i in range(hub_nodes.shape[0]):
                        for j in range(N):
                            pvals[hub_nodes[i],j,l] = 1-2*np.abs(stats.norm.cdf(averaged_accuracy[hub_nodes[i],j,l],loc=surrogate_means[i,j],scale=surrogate_stds[i,j])-0.5)
            else:
                #TODO
                pvals = 1-2*np.abs(np.array([[[stats.percentileofscore(flat_surr_fcf[:,i,j],averaged_accuracy[hub_nodes[i],j,k],kind='strict') for k in range(lags.shape[0])] for j in range(N)] for i in range(hub_nodes.shape[0])])/100 - .5)

            if pval_correction == 'Bonferroni':
                pvals *= kfolds

            significant_pair_lags = np.argwhere(pvals <= pval_threshold)
            corrected_pval = pval_threshold

            while significant_pair_lags.shape[0] < min_pairs:
                corrected_pval += pval_threshold
                significant_pair_lags = np.argwhere(pvals <= corrected_pval)
                if corrected_pval >= 1:
                    print("Could not find enough significant nodes to satisfy the minimum node number, using all nodes")
                    corrected_pval = 1
                    break
            if save:
                np.save(save_path+'maxdim_pvalues.npy',pvals)
                np.save(save_path+'corrected_pvalue_threshold.npy', np.array([corrected_pval])) # Move this into the output yaml
                np.save(save_path+'maxdim_SignificantPairs.npy',significant_pair_lags)

        # TODO: Make sure this does not throw away too many things
        mask = (pvals <= corrected_pval)
        print(f'Required pvalue: {corrected_pval} to satisfy {min_pairs} with {np.argwhere(mask.any(axis=2)).shape[0]} pairs')

        #Iterate through dimensions for the elbow
        incomplete_pairs = np.repeat(((mask.any(axis=2)).any(axis=0))[:,np.newaxis],N,axis=1).T
        print(f'Checking mask sizes: hub_mask={hub_mask.shape}, pvalue mask: {incomplete_pairs.shape}')
        incomplete_pairs &= hub_mask

        if save:
            np.save(save_path+'significant_hub_nodes.npy', np.argwhere(np.any(incomplete_pairs,axis=0))[:,0])

    
        #print(f'Significant Pairs: {pandas.DataFrame(np.array(incomplete_pairs,dtype=int))}')
        found_nodes = np.zeros((N,N),dtype=bool)


        reconstruction_accuracies = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
        reconstruction_corrs = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan

        #corr_accuracies = np.zeros((num_dims,N,N,lags.shape[0]))*np.nan
        #corr_accuracies[:,-1,:,:,:] = corr_accuracy
        reconstruction_accuracies[:,-1,:,:,:] = efcf
        reconstruction_corrs[:,-1,:,:,:] = eccm
        
        if os.path.isfile(save_path+'almost_all_computed_fcfs.npy'):
            reconstruction_accuracies = np.load(save_path+'almost_all_computed_fcfs.npy')
            
        # Run the parallel like this:
        # Build the array of eFCF values for the significant pairs and compute the eFCF over each dim
        # With it parallelized for computing both dim = d_min+i and d_max - i, and a middle dim
        # This generates a curve filled out from both sides and the center, find the elbow of this curve over each i and when it has not changed for a few iterations (heuristic choice) 
        # declare that the early stoping criteria
        tested_optimium_dims = np.ones((dim_max-d_min,N,N))*np.nan #np.zeros((dims_to_search.shape[0],N,N))
        tested_optimium_vals = np.ones((dim_max-d_min,N,N))*np.nan
        actual_optimum_dims = np.ones((N,N))*np.nan
        actual_optimum_vals = np.ones((N,N))*np.nan

        searched_dims = np.zeros(dim_max-d_min+1,dtype=bool)
        searched_dims[-1] = True
        if os.path.isfile(save_path+'searched_dims.npy'):
            searched_dims[np.load(save_path+'searched_dims.npy')] = True
        #searched_dims = np.any((~np.isnan(reconstruction_accuracies)),axis=(0,2,3,4))

        dims_to_search = np.arange(d_min,dim_max+1)[~searched_dims]
        opt_num = 0
        
        if not find_optimum_dims:
            return

        while dims_to_search.shape[0] > 0:
            if not incomplete_pairs.any():
                break            
            dims = np.array([dims_to_search[0],dims_to_search[dims_to_search.shape[0] // 2],dims_to_search[-1]])
            dims = np.unique(dims)
            dims_to_search = dims_to_search[~np.isin(dims_to_search,dims)]

            print(f'Computing CCM for dims: {dims} and {np.argwhere(incomplete_pairs).shape[0]} pairs')
            
            job_num = 0
            to_check = [[] for n in range(N)]
            for i,j in np.argwhere(incomplete_pairs):
                to_check[i] += [j]
                job_num += 1
            columns = []
            afferents = np.argwhere(incomplete_pairs.any(axis=1))[:,0]

            for d in range(dims.shape[0]):
                d_start_time = time.time_ns()

                process_outputs = tqdm.tqdm([p.apply_async(FullCCMColumn,args=((afferents[i],k,dims[d]),rolled_delay_vectors[k][:,:dims[d],:],afferents[i],train_indices,test_indices,lags)) 
                                             for k in range(kfolds) for i in range(len(afferents))],desc=f'FCF for {dims[d]}')
                for iK,proc in enumerate(process_outputs):
                    item = proc.get()
                    aff,kfold,d_curr = item[0]
                    curr_fcfs = item[1]
                    corr_fcfs = item[2]
                    if transform == 'fisher':
                        curr_fcfs = np.arctanh(curr_fcfs)
                        corr_fcfs = np.arctanh(corr_fcfs)

                    reconstruction_accuracies[kfold,d_curr-d_min,aff,:,:] = curr_fcfs 
                    reconstruction_corrs[kfold,d_curr-d_min,aff,:,:] = corr_fcfs 

                searched_dims[dims[d]-d_min] = True
                if save:
                    np.save(save_path+'all_fcfs.npy',reconstruction_accuracies)
                    np.save(save_path+'searched_dims.npy',np.argwhere(searched_dims))
                
                curr_time_taken = time.time_ns()-d_start_time
            
                important_times[f'Dim{dims[d]}CCMTotal'] = curr_time_taken*(10**-9)
                important_times[f'Dim{dims[d]}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
                
                with open(save_path+'benchmarkings.yaml','w') as outfile:
                    yaml.dump(important_times,outfile,default_flow_style=False)

        
           
            y = np.array([np.nanmean(reconstruction_accuracies[:,searched_dims,i,j,max_lag_args[i,j]],axis=0) for i,j in np.argwhere(incomplete_pairs)])
            #y = np.array([np.nanmax(np.nanmean(reconstruction_accuracies,axis=0)[searched_dims,i,j,:],axis=1) for i,j in np.argwhere(incomplete_pairs)])
            curr_opts = FindFCFOptimums(test_dims[searched_dims],y,fullrange=False)
            curr_pair = 0
            print(curr_opts[0].shape)

            for i,j in np.argwhere(incomplete_pairs):
                tested_optimium_dims[opt_num,i,j] = curr_opts[0][curr_pair]
                tested_optimium_vals[opt_num,i,j] = curr_opts[1][curr_pair]
                curr_pair += 1

            for i,j in np.argwhere(incomplete_pairs):
                check, count_max = repeated_value_early_stop(tested_optimium_dims[:,i,j])
                incomplete_pairs[i,j] = ~check or ~early_stop
                found_nodes[i,j] = check
                if check or (dims_to_search.shape[0] == 0):
                    actual_optimum_dims[i,j] = count_max
                    actual_optimum_vals[i,j] = tested_optimium_vals[opt_num,i,j]
                    tested_optimium_dims[opt_num:,i,j] = count_max
                    tested_optimium_vals[opt_num:,i,j] = tested_optimium_vals[opt_num,i,j]
            opt_num += 1

        if save:
            np.save(save_path+'last_computed_fcfs.npy',reconstruction_accuracies)
            np.save(save_path+'current_optimum_dimensions.npy',tested_optimium_dims)
            np.save(save_path+'current_optimum_dimension_values.npy',tested_optimium_vals)

        if save:
            np.save(save_path+'almost_all_computed_fcfs.npy',reconstruction_accuracies)
            np.save(save_path+'almost_all_computed_eccms.npy',reconstruction_corrs)
            np.save(save_path+'final_optimum_dimensions.npy',actual_optimum_dims)

        stopped_too_early = np.logical_and(~np.isnan(actual_optimum_dims),np.isnan(actual_optimum_vals))

        last_nns = [[[[] for n in range(N)] for m in range(N)] for k in range(kfolds)]
        
        if (np.argwhere(stopped_too_early).shape[0]) > 0:
            print(f'Still need to compute fcf for pairs: {np.argwhere(stopped_too_early)}')
            process_outputs = tqdm.tqdm([p.apply_async(remote_build_nn_single, 
                                        args = ((i,j,k),rolled_delay_vectors[k][:,:int(actual_optimum_dims[i,j]),i],
                                                train_indices,test_indices,int(actual_optimum_dims[i,j])+1)) 
                                                for k in range(kfolds) for i,j in np.argwhere(stopped_too_early)],desc='Final NNS')

            for ik,out in enumerate(process_outputs):
                vals = out.get()
                i,j,k = vals[0]
                last_nns[k][i][j] = vals[1]
            
            process_outputs = tqdm.tqdm([p.apply_async(indexed_SequentialLaggedReconstruction,args =([k,i,j],targets[k][:,:int(actual_optimum_dims[i,j]),j],lib_targets[k][:,:int(actual_optimum_dims[i,j]),j],last_nns[k][i][j],test_indices,lags))
                                            for k in range(kfolds) for i,j in np.argwhere(stopped_too_early)],desc='Final Reconstructions')
            
            for ik, out in enumerate(process_outputs):
                vals = out.get()
                k,i,j = vals[0]
                curr_fcf = vals[1]
                curr_corr = vals[2]
                d_opt = int(actual_optimum_dims[i,j])
                d_index = d_opt-d_min
                if transform == 'fisher':
                    curr_fcf = np.arctanh(curr_fcf)
                reconstruction_accuracies[k,d_index,i,j,:] = curr_fcf
                actual_optimum_vals[i,j] = np.nanmean(reconstruction_accuracies[:,d_index,i,j,max_lag_args[i,j]])
            
    all_averaged_accuracies = np.nanmean(reconstruction_accuracies,axis=0)
    
    if save:
        np.save(save_path+'all_computed_fcfs.npy',reconstruction_accuracies)
        np.save(save_path+'all_computed_eccms.npy',reconstruction_corrs)


    total_time = time.time_ns()-time_start

    important_times['TotalTime'] = total_time*(10**-9)

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    print(f'Total Time taken: {total_time*(10**-9)} s')
    np.save(save_path+'time_elapsed.npy', np.array([total_time]))

    """noded_pairs = np.array([[i,j,final_optimums[i,j],max_averaged_accuracies[final_optimums[i,j],i,j],np.argmax(all_averaged_accuracies,axis=3)] for i,j in np.argwhere(found_nodes)])

    if save:
        np.save(save_path+'noded_pairs.npy',noded_pairs)"""

    return averaged_accuracy, pvals, efcf, surrogate_fcfs, surrogates, lags, test_indices, train_indices, rolled_delay_vectors

# %%

def DirectConnectionCCM(X,delay=0,connection_dim = 2,surrogate_dim = 3,n_surrogates=10,lags=np.array([0]),retain_test_set = True,normal_pval=True,save = True,save_path='./',seed=0,max_processes = 3,kfolds=5,random_projection=False,transform='fisher'):
    """ DirectConnectionCCM
    Args --
    X: (time x neuron ndarray)
    delay = 0: int corresponding to number of time steps used to delay embed X
    connection_dim = 2: int dimension to do CCM computation in,
    surrogate_dim = 3: dimension to do Surrogate Generation in,
    n_surrogates = 10: Number or surrogates to generate in the significance computation,
    lags = np.array([0]): lags to test for eCCM,
    retain_test_set = True: whether the test set is untouched in surrogate generation,
    normal_pval = True: Assume that the surrogate distribution is nearly normal,
    save = True: Whether you want the data saved,
    save_path = './': where the output should be saved,
    seed = 0: choice of random seed,
    max_processes = 3: Maximum number of parallel processes to run on your computer,
    kfolds = 5: Number of partitions for cross validation,
    random_projection = False: Whether the delay space embeddings should also be subjected to a random projection,
    transform = 'Fisher': the transformation to be applied to the final correlation coefficients (only fisher is currently implemented)
    Output --
    (efcf: matrix of FCF values indexed by [kfolds, efferent, afferent, lag]
    ,surrogate_fcfs : matrix of surrogate fcfs indexed by [kfolds,n_surrogate,efferent,afferent]
    ,fcf_pvals: matrix of pvalues indexed by [efferent, afferent, lag]
    )
    ,
    (eccm: matrix of unlagged correlation values indexed by [kfolds, efferent, afferent, lag]
    ,surrogate_corrs: matrix of surrogate correlations indexed by [kfolds,n_surrogate,efferent,afferent]
    ,eccm_pvals: matrix of pvalues of corr values indexed by [efferent, afferent, lag]
    )
    
    """
    T,N = X.shape
    # If a delay is not given, take the test delay to be the average of the Mutual information method approximations for all time series given
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    if seed == 0:
        np.random.seed(time.time_ns()%(2**32-1))

    important_times = {'Date': datetime.today().strftime('%Y-%m-%d') }

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    time_start = time.time_ns()

    with Pool(processes=max_processes) as p:
        if delay == 0:
            print('Finding Optimum Delay')
            process_outputs = [[] for n in range(N)]
            MIDelays = np.zeros(N)
            process_outputs = tqdm.tqdm([p.apply_async(remote_ApproximatebestTau, args = (i,X[:,i],0,10,50,False,0.1)) for i in range(N)],desc='Mutual Information')
            for out in process_outputs:
                MIDelays[out.get()[0]] = out.get()[1]
            delay = int(np.max([1,np.min(MIDelays)]))
            print(f'Optimum delays using mutual information method: {MIDelays}')

        print(f'Using Delay: {delay}')
        #TODO: Determine the highest dim based on the Takens' Dimension (Considering Nearest Neighbors method)

        lags = delay*lags
        np.save(save_path+'lags.npy',lags)

        #First test at the maximum dim to find the limiting value and the significance interval
        tShift = np.abs(delay)*(surrogate_dim-1)  #Max time shift
        tDelay = T - tShift     #Length of delay vectors

        #Determine the size of the test set and the test / train indices
        t_test = tDelay // kfolds
        iStart = tDelay - t_test; iEnd = tDelay
        test_indices = np.arange(iStart,iEnd)
        train_indices = np.arange(0,iStart-tShift)    

        #Build delay space projection
        delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,surrogate_dim)[:,:,np.newaxis], X.T)),2)
        un_proj_delay_vectors = np.array(delay_vectors)

        rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]

        # Build the test and training sets
        targets = [rolled_delay_vectors[k][test_indices,:,:] for k in range(kfolds)]
        lib_targets = [rolled_delay_vectors[k][train_indices,:,:] for k in range(kfolds)]

        efcf = None
        eccm = None

        if os.path.isfile(save_path + f'eFCF.npy'):
            efcf = np.load(save_path + f'eFCF.npy')
            #np.save(path + f'eCCMTensorXValidated_dim{d_max}_delay{delay}.npy',eccm)
        else:
            eccm_full_process_start_time = time.time_ns()
            
            reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
            eccm = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

            process_outputs = tqdm.tqdm([p.apply_async(CCMColumn,args=((k,n),rolled_delay_vectors[k][:,:connection_dim,:],n,train_indices,test_indices,lags)) for n in range(N) for k in range(kfolds)],
                                        desc='Computing Directionality')

            for proc in process_outputs:
                item = proc.get()
                reconstruction_accuracy[item[0][0],:,item[0][1],:] = item[1]
                eccm[item[0][0],:,item[0][1],:] = item[2]
            
            #print(f'Processed reconstructions in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')

            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{connection_dim}ReconstructionTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{connection_dim}ReconstructionPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            efcf = reconstruction_accuracy
            if transform == 'fisher':
                efcf = np.arctanh(efcf)
                eccm = np.arctanh(eccm)
        if save:
            np.save(save_path+'/efcf.np',efcf)
            np.save(save_path+'/eccm.np',eccm)


        surrogate_fcfs = None
        surrogates = None

        if os.path.isdir(save_path+'/surrogates'):
            surrogates = [[[] for k in range(n_surrogates)] for sn in range(kfolds)]
            for f in os.listdir(save_path+'/surrogates'):
                sn = int(f[-7])
                k = int(f[-5])
                surrogates[k][sn] = np.load(save_path+'/surrogates/'+f)
        
        if os.path.isfile(save_path+'surrogate_fcf.npy'):
            surrogate_fcfs = np.load(save_path+'surrogate_fcf.npy')
        # Compute the significance intervals
        # Create Surrogate time series: n_surrogate list of NxT time series
        print('Creating twin surrogates')
        
        eccm_full_process_start_time = time.time_ns()
        already_made = True
        
        surr_data = rolled_delay_vectors
        if retain_test_set:
            surr_data = lib_targets
        if surrogates is None:
            surrogates = [[] for k in range(kfolds)]
            already_made = False
            for k in range(kfolds):
                print(f'Surrogates for fold {k}')
                surrogates[k] = parallel_twin_surrogates(surr_data[k],n_surrogates=n_surrogates,Tshift = tShift,seed=np.random.randint(0,2**32-1,1)[0])

        if save and not already_made:
            if not os.path.isdir(save_path+'/surrogates'):
                os.mkdir(save_path+'/surrogates')
            for k in range(kfolds):
                for n in range(n_surrogates):
                    np.save(save_path+'/surrogates/'+f'surrogate{n},{k}.npy',surrogates[k][n]) 

        surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,1))
        surrogate_corrs = np.zeros((n_surrogates,kfolds,N,N,1))

        print(f'Finished Surrogate Generation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
        
        curr_time_taken = time.time_ns()-eccm_full_process_start_time

        with open(save_path+'benchmarkings.yaml','w') as outfile:
            yaml.dump(important_times,outfile,default_flow_style=False)

        eccm_full_process_start_time = time.time_ns()

        completed_surrs = np.zeros(n_surrogates,dtype = bool)

        if os.path.isfile(save_path+'completed_surrs.npy'):
            completed_surrs = np.load(save_path+'completed_surrs.npy')

        for sn in range(n_surrogates):
            sn_start_time = time.time_ns()

            if completed_surrs[sn]:
                continue

            surrogate_nns = [[[] for i in range(N)] for j in range(kfolds)]

            surrogate_delays = [np.concatenate(list(map(lambda x: create_delay_vector(x,delay,surrogate_dim)[:,:,np.newaxis], surrogates[k][sn].T)),2) for k in range(kfolds)]
            if retain_test_set:
                surrogate_delays = [np.concatenate((surrogate_delays[k],rolled_delay_vectors[k][test_indices,:,:]),axis=0) for k in range(kfolds)]

            surr_shape_diff = abs(surrogate_delays[0].shape[0]-X.shape[0])
            train_indices_surr = train_indices[:-surr_shape_diff]
            test_indices_surr = test_indices-surr_shape_diff

            if retain_test_set:
                processes = tqdm.tqdm([p.apply_async(CCMColumn,
                                                     args=([k,n,sn],surrogate_delays[k][:,:connection_dim,:]
                                                           ,n,train_indices_surr,test_indices_surr,np.array([0])))
                                                            for k in range(kfolds) for n in range(N)])

                for proc in processes:
                    item = proc.get()
                    surrogate_fcfs[item[0][2],item[0][0],:,item[0][1],:] = item[1]
                    surrogate_corrs[item[0][2],item[0][0],:,item[0][1],:] = item[2]
            else:
                #TODO
                process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                        args = ((0,j,sn),surrogate_delays[:,:,j],
                                                                train_indices_surr,test_indices_surr,connection_dim+1)) 
                                                        for j in range(N)],desc=f'NNS {sn}')
            
                for proc in process_outputs:
                    item = proc.get()
                    for k in range(kfolds):
                        surrogate_nns[k][item[0][1]] = item[1]

            # Compute the fcf
                processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                        args = ([0,i,sn],surrogate_nns[k][i],targets[k], surrogate_delays[k][:,:connection_dim,:],test_indices_surr[:],np.array([0])))
                                        for i in range(N)],desc=f'Reconstructions {sn}')

                for proc in processes:
                    item = proc.get()
                    fcfval = item[1]
                    corrval = item[2]
                    if transform == 'fisher':
                        fcfval = np.atanh(fcfval)
                        corrval = np.atanh(corrval)
                    for k in kfolds:
                        surrogate_fcfs[item[0][2],k,item[0][1],:,0] = fcfval
                        surrogate_corrs[item[0][2],k,item[0][1],:,0] = corrval
                    

            completed_surrs[sn] = True
            
            curr_time_taken = time.time_ns()-sn_start_time
        
            important_times[f'Dim{connection_dim}Surrogate{sn}CCMTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{connection_dim}Surrogate{sn}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            
            if save:
                np.save(save_path+'intermediate_surrogate_fcf.npy',surrogate_fcfs)
                np.save(save_path+'completed_surrs.npy',completed_surrs)
                with open(save_path+'benchmarkings.yaml','w') as outfile:
                    yaml.dump(important_times,outfile,default_flow_style=False)


            print(f'Finished Surrogate evaluation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            important_times[f'Dim{connection_dim}AllSurrogatesCCMTotal'] = curr_time_taken*(10**-9)                
        
        average_FCF = np.nanmean(efcf,axis=0)
        average_eCCM = np.nanmean(eccm,axis=0)
        flat_surr_fcf = (surrogate_fcfs.reshape(-1,*surrogate_fcfs.shape[2:]))[:,:,:,0]
        flat_surr_eccm = (surrogate_corrs.reshape(-1,*surrogate_fcfs.shape[2:]))[:,:,:,0]

        fcf_pvals = np.ones((N,N,lags.shape[0]))
        eccm_pvals = np.ones((N,N,lags.shape[0]))

        if os.path.isfile(save_path+'maxdim_pvalues.npy'):
            fcf_pvals = np.load(save_path+'maxdim_pvalues.npy')
        else:
            if normal_pval:
                # TODO: I think it might be better to use a skewed-normal but I am going with this right now
                surrogate_means = np.nanmean(flat_surr_fcf,axis=0)
                surrogate_stds = np.nanstd(flat_surr_fcf,axis=0)
                for l in range(lags.shape[0]):
                    for i in range(N):
                        for j in range(N):
                            fcf_pvals[i,j,l] = 1-2*np.abs(stats.norm.cdf(average_FCF[i,j,l],loc=surrogate_means[i,j],scale=surrogate_stds[i,j])-0.5)
            else:
                #TODO
                fcf_pvals = 1-2*np.abs(np.array([[[stats.percentileofscore(flat_surr_fcf[:,i,j],average_FCF[i,j,k],kind='strict') for k in range(lags.shape[0])] for j in range(N)] for i in range(N)])/100 - .5)

            if save:
                np.save(save_path+'fcf_pvalues.npy',fcf_pvals)

            if normal_pval:
                # TODO: I think it might be better to use a skewed-normal but I am going with this right now
                surrogate_means = np.nanmean(flat_surr_eccm,axis=0)
                surrogate_stds = np.nanstd(flat_surr_eccm,axis=0)
                for l in range(lags.shape[0]):
                    for i in range(N):
                        for j in range(N):
                            eccm_pvals[i,j,l] = 1-2*np.abs(stats.norm.cdf(average_eCCM[i,j,l],loc=surrogate_means[i,j],scale=surrogate_stds[i,j])-0.5)
            else:
                #TODO
                eccm_pvals = 1-2*np.abs(np.array([[[stats.percentileofscore(flat_surr_eccm[:,i,j],average_eCCM[i,j,k],kind='strict') for k in range(lags.shape[0])] for j in range(N)] for i in range(N)])/100 - .5)

            if save:
                np.save(save_path+'eccm_pvalues.npy',eccm_pvals)

    return (efcf,surrogate_fcfs,fcf_pvals),(eccm,surrogate_corrs,eccm_pvals)

class TimeSeriesData:
    def __init__(self, X, d = 1, kfolds = 1):
        self.X = X
        
def ModularSurrogates(X,n_surrogates,delay,surrogate_dim,connection_dim,lags=np.array([0]),kfolds=5,retain_test_set = True,return_full_vector = False,max_processes = 3,save_path='./',save=True,transform='fisher'):
    important_times = {'Date': datetime.today().strftime('%Y-%m-%d') }

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)
    eccm_full_process_start_time = time.time_ns()

    with Pool(processes=max_processes) as p: 
        T,N = X.shape
        lags = delay*lags

        #First test at the maximum dim to find the limiting value and the significance interval
        tShift = np.abs(delay)*(surrogate_dim-1)  #Max time shift
        tDelay = T - tShift     #Length of delay vectors

        #Determine the size of the test set and the test / train indices
        t_test = tDelay // kfolds
        iStart = tDelay - t_test; iEnd = tDelay
        test_indices = np.arange(iStart,iEnd)
        train_indices = np.arange(0,iStart-tShift)    

        #Build delay space projection
        delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,surrogate_dim)[:,:,np.newaxis], X.T)),2)
        un_proj_delay_vectors = np.array(delay_vectors)

        rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]

        # Build the test and training sets
        targets = [rolled_delay_vectors[k][test_indices,:,:] for k in range(kfolds)]
        lib_targets = [rolled_delay_vectors[k][train_indices,:,:] for k in range(kfolds)]

        surr_data = rolled_delay_vectors
        if retain_test_set:
            surr_data = lib_targets
        if surrogates is None:
            surrogates = [[] for k in range(kfolds)]
            already_made = False
            for k in range(kfolds):
                print(f'Surrogates for fold {k}')
                surrogates[k] = parallel_twin_surrogates(surr_data[k],n_surrogates=n_surrogates,Tshift = tShift,seed=np.random.randint(0,2**32-1,1)[0])

        if save and not already_made:
            if not os.path.isdir(save_path+'/surrogates'):
                os.mkdir(save_path+'/surrogates')
            for k in range(kfolds):
                for n in range(n_surrogates):
                    np.save(save_path+'/surrogates/'+f'surrogate{n},{k}.npy',surrogates[k][n]) 

        surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,1))
        surrogate_corrs = np.zeros((n_surrogates,kfolds,N,N,1))

        # Generate nearest maps for all surrogates
        completed_surrs = np.zeros(n_surrogates,dtype = bool)

        if os.path.isfile(save_path+'completed_surrs.npy'):
            completed_surrs = np.load(save_path+'completed_surrs.npy')

        for sn in range(n_surrogates):
            sn_start_time = time.time_ns()

            if completed_surrs[sn]:
                continue

            surrogate_nns = [[[] for i in range(N)] for j in range(kfolds)]

            surrogate_delays = [np.concatenate(list(map(lambda x: create_delay_vector(x,delay,surrogate_dim)[:,:,np.newaxis], surrogates[k][sn].T)),2) for k in range(kfolds)]
            if retain_test_set:
                surrogate_delays = [np.concatenate((surrogate_delays[k],rolled_delay_vectors[k][test_indices,:,:]),axis=0) for k in range(kfolds)]

            surr_shape_diff = abs(surrogate_delays[0].shape[0]-X.shape[0])
            train_indices_surr = train_indices[:-surr_shape_diff]
            test_indices_surr = test_indices-surr_shape_diff

            if retain_test_set:
                processes = tqdm.tqdm([p.apply_async(CCMColumn,
                                                     args=([k,n,sn],surrogate_delays[k][:,:connection_dim,:]
                                                           ,n,train_indices_surr,test_indices_surr,np.array([0])))
                                                            for k in range(kfolds) for n in range(N)])
                """
                process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                        args = ((k,j,sn),surrogate_delays[k][:,:connection_dim,j],
                                                                train_indices_surr,test_indices_surr,connection_dim+1)) 
                                                        for j in range(N) for k in range(kfolds)],desc=f'NNS {sn}')
            
                for proc in process_outputs:
                    item = proc.get()
                    #surrogate_nns += [item[1]]
                    #nns_indices += [[item[0][0],item[0][1]]]
                    surrogate_nns[item[0][0]][item[0][1]] = item[1]
            
                processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                        args = ([k,i,sn],surrogate_nns[k][i],targets[k], surrogate_delays[k],test_indices_surr,np.array([0])))
                                        for k in range(kfolds) for i in range(N)],desc=f'Reconstructions {sn}')
                """

                for proc in processes:
                    item = proc.get()
                    surrfcf = item[1]
                    surrcorr = item[2]
                    if transform=='fisher':
                        surrfcf = np.atanh(surrfcf)
                        surrcorr = np.atanh(surrcorr)
                    surrogate_fcfs[item[0][2],item[0][0],:,item[0][1],:] = surrfcf
                    surrogate_corrs[item[0][2],item[0][0],:,item[0][1],:] = surrcorr
            else:
                #TODO
                process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                        args = ((0,j,sn),surrogate_delays[:,:,j],
                                                                train_indices_surr,test_indices_surr,connection_dim+1)) 
                                                        for j in range(N)],desc=f'NNS {sn}')
            
                for proc in process_outputs:
                    item = proc.get()
                    for k in range(kfolds):
                        surrogate_nns[k][item[0][1]] = item[1]
            #print(f'Finished Surrogate CCM Mappings in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')


            # Compute the fcf
                processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                        args = ([0,i,sn],surrogate_nns[k][i],targets[k], surrogate_delays[k][:,:connection_dim,:],test_indices_surr[:],np.array([0])))
                                        for i in range(N)],desc=f'Reconstructions {sn}')

                for proc in processes:
                    item = proc.get()
                    surrfcf = item[1]
                    surrcorr = item[2]
                    if transform=='fisher':
                        surrfcf = np.atanh(surrfcf)
                        surrcorr = np.atanh(surrcorr)
                    surrogate_fcfs[item[0][2],item[0][0],:,item[0][1],:] = surrfcf
                    surrogate_corrs[item[0][2],item[0][0],:,item[0][1],:] = surrcorr

            completed_surrs[sn] = True
            
            curr_time_taken = time.time_ns()-sn_start_time
        
            important_times[f'Dim{connection_dim}Surrogate{sn}CCMTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{connection_dim}Surrogate{sn}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            
            if save:
                np.save(save_path+'intermediate_surrogate_fcf.npy',surrogate_fcfs)
                np.save(save_path+'completed_surrs.npy',completed_surrs)
                with open(save_path+'benchmarkings.yaml','w') as outfile:
                    yaml.dump(important_times,outfile,default_flow_style=False)


            print(f'Finished Surrogate evaluation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            important_times[f'Dim{connection_dim}AllSurrogatesCCMTotal'] = curr_time_taken*(10**-9)      

        flat_surr_fcf = (surrogate_fcfs.reshape(-1,*surrogate_fcfs.shape[2:]))[:,:,:,0]
        flat_surr_eccm = (surrogate_corrs.reshape(-1,*surrogate_fcfs.shape[2:]))[:,:,:,0]
    return surrogates,flat_surr_fcf,flat_surr_eccm

# %%
def MuteMultiprocessing():
    sys.stdout = open(os.devnull,'w')


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar (code copied from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters)
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()