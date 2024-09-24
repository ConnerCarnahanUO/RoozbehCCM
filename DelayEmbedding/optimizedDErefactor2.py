###########################
# OptimizedDE
# Creator: Conner Carnahan https://github.com/ConnerCarnahanUO
# Based on Code By Amin Nejatbakhsh https://github.com/amin-nejat
# Last Updated: 9/19/2024
###########################

#%%
# Imports
import tqdm.auto
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
import multiprocessing
from joblib import Parallel, delayed
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

def DelayConvolutionEmbedding(X,C,tau,dim):
    """ DelayConvolutionEmbedding(X,C,tau,dim)
    Args:
    X -- TxN numpy array: Time series to embbed
    C -- T numpy array: Convolution function basis to use (usual delay embedding would use C[t==0] = 1/dt, C[t!=0] = 0)
    tau -- int: time shift to apply to each component of the convolution kernel (C[t]->C[t-n*tau])
    dim -- int: dimension of the embedding 
    """
    T,N = X.shape
    CX = np.zeros((T-tau*dim,dim,N))
    for i in range(dim):
        CX[:,-i,:] = np.apply_along_axis(lambda m: signal.convolve(m,np.roll(C,tau*i),mode='full')[tau*dim:T],axis=0,arr=X)
    return CX

def MutualInformation(sig1,sig2,bins=50):
    pXYjoint = np.histogram2d(sig1,sig2,density = True,bins=bins)[0]
    pXYindep = np.outer(np.sum(pXYjoint,axis=1),np.sum(pXYjoint,axis=0))
    return sci.stats.entropy(pXYjoint.flatten(),pXYindep.flatten())

def remote_ApproximatebestTau(iK, sig, taumin=0, taumax=20, bins=50, plot=True, threshold_ratio = 0.001,**args):
    return (iK, ApproximatebestTau(sig, taumin, taumax, bins, plot, threshold_ratio,**args)[0])

def ApproximatebestTau(source_signal,dt=1, taumin=0, taumax=20, bins=50, plot=False, threshold_ratio = 0.1):
    MIs = np.zeros(taumax-taumin)*np.nan
    total_points = source_signal.shape[0]
    
    for tau in np.arange(MIs.shape[0]):
        tau_len = total_points - tau
        MIs[tau] = MutualInformation(source_signal[:tau_len],source_signal[tau:])
    
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

def roll_matrix_with_lags_vectorized(M, L):
    """
    Rolls matrix M along the first axis by the set of lags L (vectorized).
    
    Parameters:
    M (numpy.ndarray): A TxN matrix.
    L (list or numpy.ndarray): A list or array of lags.
    
    Returns:
    numpy.ndarray: A matrix of shape TxNxlen(L) where A[:,:,i] is M rolled by L[i].
    """
    T, N = M.shape
    L = np.asarray(L)  # Ensure L is an array
    len_l = len(L)
    
    # Create an index array for rolling, accounting for periodic boundary conditions
    indices = (np.arange(T)[:, None] - L[None, :]) % T
    
    # Vectorized extraction of the rolled matrices
    A = M[indices, :]
    A = np.transpose(A,(0,2,1))
    return A

def remote_ApproximatebestTau(iK, sig, taumin=0, taumax=20, bins=50, plot=False, threshold_ratio = 0.001,**args):
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

def compute_permuted_correlation(X_bin, n_bins, N):
    # Generate random permutations
    bin_perms1 = np.array([np.random.permutation(n_bins) for _ in range(N)])
    bin_perms2 = np.array([np.random.permutation(n_bins) for _ in range(N)])
    
    # Apply the permutations
    Xpermed1 = np.take_along_axis(X_bin, bin_perms1[:, :, np.newaxis], axis=1)
    Xpermed2 = np.take_along_axis(X_bin, bin_perms2[:, :, np.newaxis], axis=1)
    
    # Compute permuted cross-correlation
    return np.einsum('ibt,jbt->ijb', Xpermed1, np.roll(Xpermed2,-l))

def NoiseCorrNoLag(X, n_bins=100, n_perms=100, n_jobs=64):
    T_bin = X.shape[1] // n_bins
    N = X.shape[0]

    # Vectorized creation of X_bin
    X_bin = np.reshape(X[:, :n_bins * T_bin], (N, n_bins, T_bin))
    X_bin = X_bin - np.nanmean(X_bin, axis=2, keepdims=True)
    X_bin = X_bin / np.sqrt(np.sum(np.square(X_bin), axis=2, keepdims=True))

    # Compute the original correlation
    orig_corr = np.einsum('ibt,jbt->ijb', X_bin, X_bin)

    # Parallelize permutation correlations using joblib
    p_crosses = Parallel(n_jobs=n_jobs)(
        delayed(compute_permuted_correlation)(X_bin, n_bins, N) for _ in tqdm.tqdm(range(n_perms),desc='Permutation Computations')
    )

    # Stack the results from parallel executions
    p_crosses = np.stack(p_crosses, axis=0)

    return orig_corr, p_crosses

def apply_lag(X, lag):
    """Shifts time series by the given lag, padding with NaNs along the third axis."""
    if lag > 0:
        # Positive lag: pad at the beginning of the third axis (time) and truncate the end
        return np.pad(X, ((0, 0), (0, 0), (lag, 0)), mode='constant', constant_values=0)[:, :, :-lag]
    elif lag < 0:
        # Negative lag: pad at the end of the third axis (time) and truncate the beginning
        return np.pad(X, ((0, 0), (0, 0), (0, -lag)), mode='constant', constant_values=0)[:, :, -lag:]
    else:
        # No lag: return as is
        return X

def compute_permuted_correlation_with_lag(X_bin, n_bins, N, lag):
    # Apply lag to X_bin
    X_lagged = apply_lag(X_bin, lag)

    # Generate random permutations
    bin_perms1 = np.array([np.random.permutation(n_bins) for _ in range(N)])
    bin_perms2 = np.array([np.random.permutation(n_bins) for _ in range(N)])
    
    # Apply the permutations
    Xpermed1 = np.take_along_axis(X_lagged, bin_perms1[:, :, np.newaxis], axis=1)
    Xpermed2 = np.take_along_axis(X_lagged, bin_perms2[:, :, np.newaxis], axis=1)

    # Compute permuted cross-correlation
    return np.einsum('ibt,jbt->ijb', Xpermed1, Xpermed2)

def NoiseCorrWithLag(X, n_bins, n_perms, lags, n_jobs=-1):
    T_bin = X.shape[1] // n_bins
    N = X.shape[0]

    # Vectorized creation of X_bin
    X_bin = np.reshape(X[:, :n_bins * T_bin], (N, n_bins, T_bin))
    X_bin = X_bin - np.nanmean(X_bin, axis=2, keepdims=True)
    X_bin = X_bin / np.sqrt(np.sum(np.square(X_bin), axis=2, keepdims=True))

    # Compute original cross-correlation for all lags
    orig_corrs = []
    for lag in lags:
        X_lagged = apply_lag(X_bin, lag)
        orig_corr = np.einsum('ibt,jbt->ijb', X_lagged, X_bin)
        orig_corrs.append(orig_corr)
    orig_corrs = np.stack(orig_corrs, axis=-1)  # Shape: (N, N, n_bins, n_lags)

    # Parallelize permutation correlations using joblib for each lag
    p_crosses = []
    for lag in tqdm.tqdm(lags,desc='Computing Shuffled Correlations'):
        permuted_crosses = Parallel(n_jobs=n_jobs)(
            delayed(compute_permuted_correlation_with_lag)(X_bin, n_bins, N, lag) for _ in range(n_perms)
        )
        p_crosses.append(np.stack(permuted_crosses, axis=0))  # Shape: (n_perms, N, N, n_bins)
    p_crosses = np.stack(p_crosses, axis=-1)  # Shape: (n_perms, N, N, n_bins, n_lags)

    return orig_corrs, p_crosses

def CCMRowVectorized(index, X_delay, efferent, train_indices, test_indices, lags, kfold,return_only_corr=False, bad_channels=None, save=False, save_path='./', save_transform='fisher',compute_noise_correlations = False):
    """ CCMRow:
    args -- 
    index: passthrough argument for indexing when using multiprocessing
    X_delay: TxDxN array of floats that is the D dimensional delay space embedding of a timeseries of length T with N variables
    efferent: int index of the test efferent (what the shadow manifold operator will be computed with)
    train_indices: array of indices in the training set
    test_indices: array of indices in the test set
    lags: array of integers that will be used for extended causality testing
    
    returns -- index, reconstruction_error, reconstruction_corr, reconstructions, (weights, indices)
    index: passthrough argument
    reconstruction_error: CCM as computed with Amin's Method (average correlation of each vector index)
    reconstruction_corr: CCM as computed with Sugihara's Method (correlation of first index only)
    reconstructions: list of each reconstruction indexed by [var][lag index]
    (weights,indices): Shadow Manifold operator parameters
    """
    N = X_delay.shape[2]
    nrange = np.arange(N)
    if bad_channels is not None:
        nrange = np.delete(nrange, bad_channels)
    
    targets = X_delay[test_indices, :, :]
    lib_targets = X_delay[train_indices, :, :]
    n_neighbors = targets.shape[1] + 1
    D = X_delay.shape[1]
    # Nearest neighbors search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_delay[train_indices, :, efferent])
    distances, indices = nbrs.kneighbors(X_delay[test_indices, :, efferent])
    
    # Compute weights
    weights = np.exp(-distances)
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Initialize results
    if not return_only_corr:
        reconstructions = np.empty((N, lags.shape[0]), dtype=object)
    
    reconstruction_corr = np.full((N, lags.shape[0]), np.nan)
    reconstruction_error = np.full((N, lags.shape[0]), np.nan)
    
    eff_reconstruction_corr = np.full((N, lags.shape[0]), np.nan)
    eff_reconstruction_error = np.full((N, lags.shape[0]), np.nan)
                                   
    # Vectorized calculation of reconstruction and correlation for each lag and variable
    # Vectorized computation for all lags in one batch
    lagged_reconstructions = np.zeros((len(test_indices), D, N, len(lags)))

    for n in nrange:
        curr_afferent = lib_targets[:, :, n]

        # Vectorized rolling of the lib_targets array for all lags at once
        for i, L in enumerate(lags):
            rolled_afferent = np.roll(curr_afferent, L, axis=0)
            lagged_reconstructions[:, :, n, i] = np.einsum('ij,ijk->ik', weights, rolled_afferent[indices])

        for i, L in enumerate(lags):
            rolled_targets = np.roll(targets[:, :, n], L, axis=0)

            # Fix the shape before calling sequential_correlation_batch
            recon_for_corr = np.expand_dims(lagged_reconstructions[:, :, n, i], axis=2)
            rolled_for_corr = np.expand_dims(rolled_targets, axis=2)

            # Error using the vectorized sequential correlation for this lag
            corrcoefs, first_index_corrcoef = sequential_correlation_batch(recon_for_corr, rolled_for_corr)
            
            reconstruction_error[n, i] = corrcoefs 
            # Correlation using Sugihara's method
            reconstruction_corr[n, i] = first_index_corrcoef
        
        lagged_efferents = roll_matrix_with_lags_vectorized(lib_targets[:,:,efferent],lags)
        eff_corrcoef,eff_first_index_corrcoef = sequential_correlation_batch(curr_afferent[:,:,np.newaxis],lagged_efferents)
        
        #Base Correlations for variation on Differential CCM defined in Gerald's Paper
        eff_reconstruction_error[n,:] = eff_corrcoef
        eff_reconstruction_corr[n,:] = eff_first_index_corrcoef

    rho_diff = reconstruction_error-np.abs(eff_reconstruction_error)
    rho0_diff = reconstruction_corr-np.abs(eff_reconstruction_corr)

    # Save results if necessary
    if save:
        np.save(save_path + f'/ccms/row.kfold_{kfold}.var_{efferent}.npy', np.arctanh(reconstruction_error) if save_transform == 'fisher' else reconstruction_error)
        np.save(save_path + f'/ccms_corrs/row.kfold_{kfold}.var_{efferent}.npy', np.arctanh(reconstruction_corr) if save_transform == 'fisher' else reconstruction_corr)
        if compute_noise_correlations:
            np.save(save_path + f'/sequential_cross_correlations/row.kfold_{kfold}.var_{efferent}.npy', eff_reconstruction_error)
            np.save(save_path + f'/cross_correlations/row.kfold_{kfold}.var_{efferent}.npy', eff_reconstruction_corr)
            np.save(save_path + f'/rho_diff/row.kfold_{kfold}.var_{efferent}.npy', np.arctanh(rho_diff) if save_transform == 'fisher' else rho_diff)
            np.save(save_path + f'/rho0_diff/row.kfold_{kfold}.var_{efferent}.npy', np.arctanh(rho0_diff) if save_transform == 'fisher' else rho0_diff)
        
    # Return results
    if return_only_corr:
        if compute_noise_correlations:
            return index, reconstruction_corr, reconstruction_error, rho_diff, rho0_diff,eff_reconstruction_error,eff_reconstruction_corr
        else:
            return index, reconstruction_corr, reconstruction_error
    else:
        return index, reconstruction_error, reconstruction_corr, reconstructions, (weights, indices)
    
def sequential_correlation_batch(trails1, trails2):
    """
    Compute the correlation between two signals from their delay representations in a vectorized way.

    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal, shape (T, d, lags)
        trails2 (numpy.ndarray): delay vectors of the second signal, shape (T, d, lags)

    Returns:
        corrcoef (numpy.ndarray): Mean correlation between the two delay representations across all lags (shape lags)
        first_index_corrcoef (numpy.ndarray): Correlation of the first index of the vectors across all lags (shape lags)
    """
    # Input shapes: T (time), d (dimensions), lags
    #T, d, lags = trails1.shape

    # Subtract mean from each time series (along time axis)
    trails1_mean_centered = trails1 - np.mean(trails1, axis=0, keepdims=True)
    trails2_mean_centered = trails2 - np.mean(trails2, axis=0, keepdims=True)

    # Compute covariance (dot product between centered arrays along the time axis)
    covariance = np.sum(trails1_mean_centered * trails2_mean_centered, axis=0)

    # Compute standard deviations
    std1 = np.sqrt(np.sum(trails1_mean_centered**2, axis=0))
    std2 = np.sqrt(np.sum(trails2_mean_centered**2, axis=0))

    # Compute Pearson correlation for each dimension and lag
    corrcoefs_per_dim = covariance / (std1 * std2)

    # Average correlation across dimensions for each lag
    corrcoefs = np.nanmean(corrcoefs_per_dim, axis=0)

    # Correlation for the first index (first dimension) across all lags
    first_index_corrcoefs = corrcoefs_per_dim[0, :]

    return corrcoefs, first_index_corrcoefs


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

def parallel_twin_surrogates_ld(X,n_surrogates,Tshift=0,max_processes=64,seed=0,return_full_vectors=False,bad_channels = None):
    '''Create twin surrogates for significance evaluation and hypothesis testing with each surrogate created in parallel
    
    Args:
        X (numpy.ndarray): (Dimension x Time x Channels) multivariate signal for which we want to generate surrogate time series
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    np.random.seed(seed)
    T,D,C = X.shape
    crange = np.arange(C)
    if bad_channels is not None:
        crange = np.delete(crange,bad_channels)

    elns = [[] for c in range(C)]
    inds = [[] for c in range(C)]

    surrogate_time = time.time_ns()
    surrs = [[] for sn in range(n_surrogates)]

    #Generate the surrogate swapping indices for each channel in parallel

    #process_outputs = Parallel(n_jobs=max_processes)(delayed(CCMRow)((k,j),rolled_delay_vectors[k],j,train_indices,test_indices,lags,
    # True,bad_channels,True,save_path,'fisher') for k,j in tqdm.tqdm(inds_to_do,desc='Computing Columns'))
    processes = Parallel(n_jobs=max_processes)(delayed(indexed_SurrogateMap)(channel,X[:,:,channel],channel) for channel in tqdm.tqdm(crange,desc='Surrogate NNS'))
    #processes = tqdm.tqdm([p.apply_async(indexed_SurrogateMap,args = (channel,X[:,:,channel],channel)) for channel in crange],desc='Surrogate NNS')

    for proc in list(processes):
        curr_map = proc
        elns[curr_map[0]] = curr_map[2]
        inds[curr_map[0]] = curr_map[1]

    print(f'Surrogate Map generation time: {(time.time_ns()-surrogate_time)*10**-9} s')
    for sn in range(n_surrogates):
        surrs[sn] = np.zeros((T,C))
    # Generate all of the surrogates in parallel for each channel
    
    seeds = np.random.randint(0,2**32-1,n_surrogates)
    
    channel_proccesses = Parallel(n_jobs=max_processes)(delayed(Curr_channel_Gen)(c,X[:,:,c],inds[c],elns[c],T,n_surrogates,return_full_vectors) for c in tqdm.tqdm(crange, desc='Surrogate Time series generation'))#tqdm.tqdm([p.apply_async(Curr_channel_Gen,args=(c,X[:,:,c],inds[c],elns[c],T,n_surrogates,return_full_vectors)) for c in crange])

    for proc in list(channel_proccesses):
        channel, curr_surrs = proc
        for sn in range(n_surrogates):
            surrs[sn][:,channel] = curr_surrs[sn]
    return surrs

def parallel_twin_surrogates(X,n_surrogates,Tshift=0,max_processes=64,seed=0,return_full_vectors=False,bad_channels = None):
    '''Create twin surrogates for significance evaluation and hypothesis testing with each surrogate created in parallel
    
    Args:
        X (numpy.ndarray): (Dimension x Time x Channels) multivariate signal for which we want to generate surrogate time series
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    np.random.seed(seed)
    T,D,C = X.shape
    crange = np.arange(C)
    if bad_channels is not None:
        crange = np.delete(crange,bad_channels)

    elns = [[] for c in range(C)]
    inds = [[] for c in range(C)]

    surrogate_time = time.time_ns()
    surrs = [[] for sn in range(n_surrogates)]

    #Generate the surrogate swapping indices for each channel in parallel
    with Pool(processes=max_processes) as p:

        #process_outputs = Parallel(n_jobs=max_processes)(delayed(CCMRow)((k,j),rolled_delay_vectors[k],j,train_indices,test_indices,lags,
        # True,bad_channels,True,save_path,'fisher') for k,j in tqdm.tqdm(inds_to_do,desc='Computing Columns'))
        processes = tqdm.tqdm([p.apply_async(indexed_SurrogateMap,args = (channel,X[:,:,channel],channel)) for channel in crange],desc='Surrogate NNS')

        for proc in processes:
            curr_map = proc.get()
            elns[curr_map[0]] = curr_map[2]
            inds[curr_map[0]] = curr_map[1]
    
        print(f'Surrogate Map generation time: {(time.time_ns()-surrogate_time)*10**-9} s')
        for sn in range(n_surrogates):
            surrs[sn] = np.zeros((T,C))
        # Generate all of the surrogates in parallel for each channel
        
        seeds = np.random.randint(0,2**32-1,n_surrogates)
        
        channel_proccesses = tqdm.tqdm([p.apply_async(Curr_channel_Gen,args=(c,X[:,:,c],inds[c],elns[c],T,n_surrogates,return_full_vectors)) for c in crange])

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

def SplineCausalPeakFinder(C,lags,smoothing=0):
    all_peaks = []
    tck = interpolate.splrep(lags,C,s=smoothing)
    dCnew = interpolate.splev(lags,tck,der=1)
    for i in range(lags.shape[0]-1):
        if dCnew[i] > 0:
            if dCnew[i+1] <= 0:
                all_peaks += [i]
    all_peaks = np.array(all_peaks)
    if all_peaks.shape[0] == 0:
        all_peaks = np.array([0,np.argmax(C)])
    return all_peaks

def SplineCausalPeaksOverMatrix(CCM,lags,smoothing=0):
    N = CCM.shape[0]
    M = CCM.shape[1]
    return [[SplineCausalPeakFinder(CCM[n,m,:],lags,smoothing=smoothing) for m in range(M)] for n in range(N)]

#Modularizing the whole computation so the Dimension Search isn't the only thing with this stuff
def DirectionalECCM(XDs,delay,kfolds,nrange,lags,save_path,bad_channels,max_processes,transform,compute_noise_correlations = False):
    """ DirectionalECCM ---
    Computes the whole Crossvalidation of the CCM matrix for a specific set of parameters
    
    Args --
    XDs: [TxDxN numpy array for floats] Delay Vector embeddings for the system
    delay: [int] delay length in measurement time steps
    kfolds: [int] number of cross validation partitions
    nrange: [numpy array of non-negative integers] the variables in the data set that we should compute on
    lags: [numpy array of integers] the lags used for the cross mappings
    save_path: [string] file location where all this will be saved
    max_processes = 64: [non negative int] maximum number of processes available to use for the computation. Use -1 for all possible threads
    transform='fisher': [string] possible transform for the data that should be used. 
                                 Currently only Fisher-Z transform ['fisher'] is implemented, 
                                 other strings will just return the untransformed values 
    
    """
    dim = XDs.shape[1]
    N = XDs.shape[2]
    tShift = np.abs(delay)*(dim-1)  #Max time shift
    tDelay = XDs.shape[0]     #Length of delay vectors

    #Determine the size of the test set and the test / train indices
    t_test = tDelay // kfolds
    iStart = tDelay - t_test; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)    
    rolled_delay_vectors = [np.roll(XDs,t_test*k,axis=0) for k in range(kfolds)]
    reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
    extended_correlation = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
    differential_CCM = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
    differential_CCM0 = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

    cross_correlations = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
    sequential_cross_correlations = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan


    inds_to_do = np.array([[k,j] for k in range(kfolds) for j in nrange])
    if not os.path.isdir(save_path+'/ccms/'):
        os.makedirs(save_path+'/ccms/')
        os.makedirs(save_path+'/ccms_corrs/')
        if compute_noise_correlations:
            os.makedirs(save_path+'rho_diff/')
            os.makedirs(save_path+'rho0_diff/')
            os.makedirs(save_path+'sequential_cross_correlations/')
            os.makedirs(save_path+'cross_correlations/')
    else:
        for f in os.listdir(save_path+'/ccms/'):
            k = int(f.split('.')[1].split('_')[1])
            j = int(f.split('.')[2].split('_')[1])
            right_kfolds = (inds_to_do[:,0] == k)
            right_j = (inds_to_do[:,1] == j)
            row_to_remove = np.argwhere(np.logical_and(right_kfolds,right_j)).flatten()[0]
            inds_to_do = np.delete(inds_to_do,(row_to_remove),axis=0)
            reconstruction_accuracy[k,j,:,:] = np.load(save_path+'/ccms/'+f)
            extended_correlation[k,j,:,:] = np.load(save_path+'/ccms_corrs/'+f)
            if compute_noise_correlations:
                sequential_cross_correlations[k,j,:,:] = np.load(save_path+'/sequential_cross_correlations/'+f)
                cross_correlations[k,j,:,:] = np.load(save_path+'/cross_correlations/'+f)
                differential_CCM[k,j,:,:] = np.load(save_path+'/rho_diff/'+f)
                differential_CCM0[k,j,:,:] = np.load(save_path+'/rho0_diff/'+f)

    process_outputs = Parallel(n_jobs=max_processes)(
        delayed(CCMRowVectorized)((k,j),rolled_delay_vectors[k],j,train_indices,test_indices,lags,k,True,bad_channels,
                                  True,save_path,transform,compute_noise_correlations) for k,j in tqdm.tqdm(inds_to_do,desc='Computing Columns'))
    for proc in list(process_outputs):
        item = proc
        reconstruction_accuracy[item[0][0],item[0][1],:,:] = np.arctanh(item[2]) if transform=='fisher' else item[2]
        extended_correlation[item[0][0],item[0][1],:,:] = np.arctanh(item[1]) if transform=='fisher' else item[1]
        if compute_noise_correlations:
            differential_CCM[item[0][0],item[0][1],:,:] = np.arctanh(item[3]) if transform=='fisher' else item[3]
            differential_CCM0[item[0][0],item[0][1],:,:] = np.arctanh(item[4]) if transform=='fisher' else item[4]
            sequential_cross_correlations[item[0][0],item[0][1],:,:] = item[5]
            cross_correlations[item[0][0],item[0][1],:,:] = item[6]

    np.save(save_path + f'eFCF.npy',reconstruction_accuracy)
    np.save(save_path + f'eCCM.npy',extended_correlation)
    if compute_noise_correlations:
        np.save(save_path + f'rho_diff.npy',differential_CCM)
        np.save(save_path + f'rho0_diff.npy',differential_CCM0)
        np.save(save_path + f'sequential_cross_correlations.npy',sequential_cross_correlations)
        np.save(save_path + f'cross_correlations.npy',cross_correlations)
    np.save(save_path + f'lags.npy',lags)
    if compute_noise_correlations:
        return reconstruction_accuracy,extended_correlation,differential_CCM,differential_CCM0,sequential_cross_correlations,cross_correlations
    else:
        return reconstruction_accuracy,extended_correlation

def DimensionParameterSearchLargeDatasetOpt2(X,dim_min=1, dim_max=30, kfolds=5, delay=0,
                             lags=np.arange(-8,9),lagstep=0, random_projection = False , mask = None, transform='fisher',
                             node_ratio = 1., compute_pvalue = False, n_surrogates = 10, normal_pval=False, pval_threshold=0.05, 
                             min_pairs = 1, dim_search_stopping_num = 3, save=True, save_path = './', retain_test_set = True, 
                             max_processes = 64, early_stop = False, only_hubs = False, find_optimum_dims = False, save_surrogates=False,
                             seed = 0,pval_correction='None',causal_peak_smoothing=1,bad_channels = None,noise_correlation_bins = 100,noise_correlation_shuffles = 100,compute_noise_correlations=False):
    """ DimensionParameterSearch, uses CCM quantification
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
    bad_channels = None: array of indices that indicate channels which should be ignored
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
    channel_mask = np.ones(N,dtype=bool)
    nrange = np.arange(N)
    if bad_channels is not None:
        print(bad_channels)
        channel_mask[bad_channels] = False
        nrange = np.delete(nrange,bad_channels)
        hub_nodes_needed = min([int(np.ceil(node_ratio*N)),N-len(bad_channels)])
        print(f'bad channels (python indexing): {bad_channels}')
    else:
        hub_nodes_needed = int(np.ceil(node_ratio*N))
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    if seed == 0:
        np.random.seed(time.time_ns()%(2**32-1))

    important_times = {'Date': datetime.today().strftime('%Y-%m-%d') }

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    time_start = time.time_ns()
    print(f'{multiprocessing.cpu_count()} cpus available, using {max_processes}')
    #with Pool(processes=max_processes) as p:
    #TODO: Fix this for the new joblib garbage
    if delay == 0:
        print('Finding Optimum Delay')
        process_outputs = [[] for n in nrange]
        MIDelays = np.zeros(N)
        process_outputs = tqdm.tqdm([p.apply_async(remote_ApproximatebestTau, args = (i,X[:,i],0,10,50,False,0.1)) for i in nrange],desc='Mutual Information')
        for out in process_outputs:
            MIDelays[out.get()[0]] = out.get()[1]
        delay = int(np.nanmean(np.min(MIDelays)))
        print(MIDelays)

    print(f'Using Delay: {delay}')
    #TODO: Determine the highest dim based on the Takens' Dimension (Considering Nearest Neighbors method)
    test_dims = np.arange(dim_min,dim_max+1)
    num_dims = test_dims.shape[0]

    # TODO: the argument of the function should not be altered in this function
    if lagstep==0:
        lags = delay*lags
        #noise_corr_lags = np.arange(lags[0]-delay*dim_max,lags[-1])
    else:
        lags = lagstep*lags
        #noise_corr_lags = np.arange(lags[0]-delay*dim_max,lags[-1],lagstep)
    noise_corr_lags = lags
    
    #Compute the Noise Correlations of all the pairs
    if not os.path.isfile(f'{save_path}base_noise_correlations.npy') and compute_noise_correlations:
        nc = NoiseCorrWithLag(X.T,noise_correlation_bins,noise_correlation_shuffles,noise_corr_lags,max_processes)
        np.save(f'{save_path}base_noise_correlations.npy',nc[0])
        np.save(f'{save_path}shuffled_noise_correlations.npy',nc[1])
        corrected_noise_correlations = np.nanmean(nc[0],axis=2)-np.nanmean(nc[1],axis=(0,3))
        noise_correlation_baselines = np.nanmean(nc[1],axis=(0,3))
        np.save(f'{save_path}noise_correlation_baselines.npy',noise_correlation_baselines)
        np.save(f'{save_path}corrected_noise_correlations.npy',corrected_noise_correlations)
    
    #this extra check really is for backwards compatibility
    if os.path.isfile(f'{save_path}corrected_noise_correlations.npy') and compute_noise_correlations:
        corrected_noise_correlations = np.load(f'{save_path}corrected_noise_correlations.npy')
        noise_correlation_baselines = np.load(f'{save_path}noise_correlation_baselines.npy')
    elif compute_noise_correlations:
        noise_correlation_baselines = np.nanmean(np.load(f'{save_path}shuffled_noise_correlations.npy'),axis=(0,3))
        np.save(f'{save_path}noise_correlation_baselines.npy',noise_correlation_baselines)
        corrected_noise_correlations = np.nanmean(np.load(f'{save_path}base_noise_correlations.npy'),axis=2)-np.nanmean(np.load(f'{save_path}shuffled_noise_correlations.npy'),axis=(0,3))
        np.save(f'{save_path}corrected_noise_correlations.npy',corrected_noise_correlations)

    #First test at the maximum dim to find the limiting value and the significance interval
    tShift = np.abs(delay)*(dim_max-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Determine the size of the test set and the test / train indices
    t_test = tDelay // kfolds
    iStart = tDelay - t_test; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)    

    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], X.T)),2)
    un_proj_delay_vectors = np.array(delay_vectors)

    if random_projection:
        if os.path.isfile(save_path+'projection_operator.npy'):
            proj_A = np.load(save_path+'projection_operator.npy')
        else:
            proj_A = np.array([np.random.standard_normal((dim_max,dim_max)) for i in range(N)])
        delay_vectors = np.einsum('imd,tdi->tmi',proj_A,delay_vectors)
        if save:
            np.save(save_path+'projection_operator.npy',proj_A)
    
        if save:
            np.save(save_path+'projected_delay_vectors.npy',delay_vectors)

    # build the rolled set of delay projections for each fold, the FCF is invariant over total shifts 
    # so we can always just compute it in the same interval but with rolled data
    rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]
    
    #TODO: this should not be all on its own. Should just be done like the computations for the param search later
    if os.path.isfile(save_path+f'/dims/{dim_max}/eCCM.npy'):
        extended_correlation = np.load(save_path+f'/dims/{dim_max}/eCCM.npy')
        efcf = np.load(save_path+f'/dims/{dim_max}/eFCF.npy')
        if compute_noise_correlations:
            differential_CCM = np.load(save_path+f'/dims/{dim_max}/rho_diff.npy')
            differential_CCM0 = np.load(save_path+f'/dims/{dim_max}/rho0_diff.npy')
            sequential_cross_correlations = np.load(save_path+f'/dims/{dim_max}/sequential_cross_correlations.npy')
            cross_correlations = np.load(save_path+f'/dims/{dim_max}/cross_correlations.npy')
    else:
        eccm_full_process_start_time = time.time_ns()
        # build the weight matrices for all the variables and folds
        #print('Building CCM Mappings')

        if compute_noise_correlations:
            efcf,extended_correlation,differential_CCM,differential_CCM0,sequential_cross_correlations,cross_correlations = DirectionalECCM(
                XDs=delay_vectors,delay=delay,kfolds=kfolds,nrange=nrange,lags=lags,save_path=save_path+f'/dims/{dim_max}/',
                bad_channels=bad_channels,max_processes=max_processes,transform=transform,compute_noise_correlations=compute_noise_correlations)
        else:
            efcf,extended_correlation = DirectionalECCM(
                XDs=delay_vectors,delay=delay,kfolds=kfolds,nrange=nrange,lags=lags,save_path=save_path+f'/dims/{dim_max}/',
                bad_channels=bad_channels,max_processes=max_processes,transform=transform,compute_noise_correlations=compute_noise_correlations)
    # Average over folds and determine the limiting statistics
    # TODO: This should be more sophisticated to pick the causal peaks
    quick_efcf = np.nanmax(np.nanmean(efcf,axis=0)[:,:,lags>=0],axis=2)
    quick_efcf[np.identity(quick_efcf.shape[0],dtype=bool)] = np.nan
    hubness = np.nansum(quick_efcf,axis=0)


    hubness_ranks = np.argsort(-hubness)
    hubness_rank_values = -np.sort(-hubness)
    np.save(save_path+f'hubranks.npy',hubness_ranks)
    np.save(save_path+f'hubrank_values.npy',hubness_rank_values)

    hub_rank_data = {'Channel': hubness_ranks, 'Hubness': hubness_rank_values}
    df = pd.DataFrame(hub_rank_data)
    df.to_csv(save_path+'hub_rankings.csv',sep=',', na_rep='Null')

    print(f'Hub Rankings: \n channels: {hubness_ranks} \n values:   { hubness[hubness_ranks]}')
    
    hub_nodes = nrange
    
    for i in range(len(hub_nodes)):
        if bad_channels is None:
            break
        if hub_nodes[i] in bad_channels:
            hub_nodes = np.delete(hub_nodes,i)
    
    print(f'Hubs to be computed on: {hub_nodes}')
    hub_mask = np.zeros((N,N),dtype=bool)
    for i in hub_nodes:
        hub_mask[i,:] = True
    #hub_mask = np.repeat(hub_mask[:,np.newaxis],N,axis=1).T
    #print(f'FCFs to compute for hub nodes: {hub_mask}')

    best_lag_values = np.full((N,N),np.nan)
    submat = np.ix_(nrange,nrange,np.arange(lags.shape[0]))
    nrange_submat = np.ix_(nrange,nrange)
    valid_lag_indices = np.nanargmax(np.nanmean(efcf,axis=0)[submat],axis=2)
    valid_lags = lags[valid_lag_indices.flatten()].reshape(valid_lag_indices.shape)
    best_lag_values[nrange_submat] = valid_lags
    np.save(save_path+f'optimum_lags',best_lag_values)

    #averaged_corr_accuracy = np.nanmean(eccm,axis=0)
    surrogate_fcfs = None
    surrogates = None

    if os.path.isdir(save_path+'/surrogates'):
        surrogates = [[[] for k in range(n_surrogates)] for sn in range(kfolds)]
        for f in os.listdir(save_path+'/surrogates'):
            sn,k = f.split('.')[1:3]
            sn = int(sn)
            k = int(k)
            #sn = int(f[-7])
            #k = int(f[-5])
            surrogates[k][sn] = np.load(save_path+'/surrogates/'+f)
    
    # TODO: Need to change the surrogate naming conventions to "surrogate.sn.k.npy" so that it splits correctly cause right now the sn isn't loading right
    surrogate_corrs = np.zeros((n_surrogates,kfolds,N,N,1))*np.nan
    surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,1))*np.nan
    completed_surrs = np.zeros(n_surrogates,dtype = bool)
    if os.path.isfile(save_path+'completed_surrs.npy'):
        complete_surr_indices = np.load(save_path+'completed_surrs.npy')
        completed_surrs[complete_surr_indices] = True
    if os.path.isfile(save_path+'surrogate_fcf.npy'):
        isfcf = np.load(save_path+'surrogate_fcf.npy')
        #iscorrs = np.load(save_path+'surrogate_corrs.npy')
        for sn in np.argwhere(completed_surrs).flatten():
            surrogate_fcfs[sn,:,:,:,:] = isfcf[sn,:,:,:,:]
        #    surrogate_corrs[sn,:,:,:,:] = iscorrs[sn,:,:,:,:]

    if os.path.isfile(save_path+'maxdim_pvalues.npy'):
        pvals = np.load(save_path+'maxdim_pvalues.npy')
    elif compute_pvalue and not os.path.isfile(save_path+'maxdim_pvalues.npy'):
        # Compute the significance intervals
        # Create Surrogate time series: n_surrogate list of NxT time series
        print('Creating twin surrogates')
        
        eccm_full_process_start_time = time.time_ns()
        already_made = True
        
        surr_data = rolled_delay_vectors
        if retain_test_set:
            surr_data = [rolled_delay_vectors[k][train_indices,:,:] for k in range(kfolds)]
        if surrogates is None:
            surrogates = [[] for k in range(kfolds)]
            already_made = False
            for k in range(kfolds):
                print(f'Surrogates for fold {k}')
                surrogates[k] = parallel_twin_surrogates_ld(surr_data[k],n_surrogates=n_surrogates,Tshift = tShift,seed=np.random.randint(0,2**32-1,1)[0],bad_channels=bad_channels)

        if (save and not already_made) and save_surrogates:
            if not os.path.isdir(save_path+'/surrogates'):
                os.mkdir(save_path+'/surrogates')
            for k in range(kfolds):
                for n in range(n_surrogates):
                    np.save(save_path+'/surrogates/'+f'surrogate.{n}.{k}.npy',surrogates[k][n]) 

        print(f'Finished Surrogate Generation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
        
        curr_time_taken = time.time_ns()-eccm_full_process_start_time
        
        important_times[f'Dim{dim_max}SurrogateConstructionTotal'] = curr_time_taken*(10**-9)
        important_times[f'Dim{dim_max}SurrogateConstructionPerProcess'] = (curr_time_taken/(N*kfolds*n_surrogates))*(10**-9)

        with open(save_path+'benchmarkings.yaml','w') as outfile:
            yaml.dump(important_times,outfile,default_flow_style=False)

        eccm_full_process_start_time = time.time_ns()

        # Generate nearest maps for all surrogates
        for sn in range(n_surrogates):
            sn_start_time = time.time_ns()

            if completed_surrs[sn]:
                continue

            surrogate_delays = [np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], surrogates[k][sn].T)),2) for k in range(kfolds)]
            if retain_test_set:
                surrogate_delays = [np.concatenate((surrogate_delays[k],rolled_delay_vectors[k][test_indices,:,:]),axis=0) for k in range(kfolds)]

            surr_shape_diff = abs(surrogate_delays[0].shape[0]-X.shape[0])
            train_indices_surr = train_indices[:-surr_shape_diff]
            test_indices_surr = test_indices-surr_shape_diff

            surr_save_path = save_path+f'/surrccms/surr.{sn}/'
            if retain_test_set:
                inds_to_do = np.array([[k,j] for k in range(kfolds) for j in range(hub_nodes.shape[0])])
                if not os.path.isdir(surr_save_path+'/ccms/'):
                    os.makedirs(surr_save_path)
                    os.makedirs(surr_save_path+'/ccms/')
                    os.makedirs(surr_save_path+'/ccms_corrs/')
                    if compute_noise_correlations:
                        os.makedirs(surr_save_path+'rho_diff/')
                        os.makedirs(surr_save_path+'rho0_diff/')
                        os.makedirs(surr_save_path+'sequential_cross_correlations/')
                        os.makedirs(surr_save_path+'cross_correlations/')
                else:
                    for f in os.listdir(save_path+f'/surrccms/surr.{sn}/ccms/'):
                        k = int(f.split('.')[1].split('_')[1])
                        j = int(f.split('.')[2].split('_')[1])
                        right_kfolds = (inds_to_do[:,0] == k)
                        right_j = (inds_to_do[:,1] == j)
                        row_to_remove = np.argwhere(np.logical_and(right_kfolds,right_j)).flatten()[0]
                        inds_to_do = np.delete(inds_to_do,(row_to_remove),axis=0)
                        surrogate_fcfs[k,j,:,:] = np.load(surr_save_path+'/ccms/'+f)
                        surrogate_corrs[k,j,:,:] = np.load(surr_save_path+'/ccms_corrs/'+f) #TODO this isn't right but I just want this to work fast

                process_outputs = Parallel(n_jobs=max_processes)(delayed(CCMRowVectorized)((k,hub_nodes[j],sn),surrogate_delays[k],hub_nodes[j],train_indices_surr,test_indices_surr,
                                                                                           np.array([0]),k,True,bad_channels,True,surr_save_path,transform) 
                                                                 for k,j in tqdm.tqdm(inds_to_do,desc=f'Computing Columns for surrogate set {sn}')  )#tqdm.tqdm([p.apply_async(
                #CCMRow,args=((k,hub_nodes[j],sn),surrogate_delays[k],hub_nodes[j],train_indices_surr,test_indices_surr,np.array([0]),True,bad_channels))
                #                for k in range(kfolds) for j in range(hub_nodes.shape[0])],desc=f'Computing Columns for surrogate set {sn}')

                for proc in list(process_outputs):
                    item = proc
                    surrf = item[2]
                    surrcorr = item[1]
                    if transform == 'fisher':
                        surrf = np.arctanh(surrf)
                        surrcorr = np.arctanh(surrcorr)
                    surrogate_fcfs[item[0][2],item[0][0],item[0][1],:,:] = surrf
                    surrogate_corrs[item[0][2],item[0][0],item[0][1],:,:] = surrcorr

            completed_surrs[sn] = True
            curr_time_taken = time.time_ns()-sn_start_time
        
            important_times[f'Dim{dim_max}Surrogate{sn}CCMTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{dim_max}Surrogate{sn}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            if save:
                np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)
                np.save(save_path+'surrogate_corrs.npy',surrogate_corrs)
                np.save(save_path+'completed_surrs.npy',np.argwhere(completed_surrs).flatten())
                with open(save_path+'benchmarkings.yaml','w') as outfile:
                    yaml.dump(important_times,outfile,default_flow_style=False)

        print(f'Finished Surrogate evaluation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
        curr_time_taken = time.time_ns()-eccm_full_process_start_time
        important_times[f'Dim{dim_max}AllSurrogatesCCMTotal'] = curr_time_taken*(10**-9)                

        if save:
            np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)
            np.save(save_path+'surrogate_corrs.npy',surrogate_corrs)
            np.save(save_path+'completed_surrs.npy',np.argwhere(completed_surrs).flatten())

        pvals = np.zeros((N,N,lags.shape[0]))*np.nan

        flat_surr_fcf = surrogate_fcfs.reshape(-1,*surrogate_fcfs.shape[2:])
        flat_surr_fcf = flat_surr_fcf[:,:,:,0]
        if os.path.isfile(save_path+'maxdim_pvalues.npy'):
            pvals = np.load(save_path+'maxdim_pvalues.npy')
            corrected_pval = np.load(save_path+'corrected_pvalue_threshold.npy') #TODO Move this into the output yaml
            significant_pair_lags = np.load(save_path+'maxdim_SignificantPairs.npy')    
        else:
            #Assumes that the null is normally distributed
            if normal_pval:
                surrogate_means = np.nanmean(flat_surr_fcf,axis=0)
                surrogate_stds = np.nanstd(flat_surr_fcf,axis=0)
                for l in range(lags.shape[0]):
                    for i in range(hub_nodes.shape[0]):
                        for j in nrange:
                            pvals[hub_nodes[i],j,l] = 1-2*np.abs(stats.norm.cdf(np.nanmean(efcf[:,hub_nodes[i],j,l]),loc=surrogate_means[i,j],scale=surrogate_stds[i,j])-0.5)
            else:
                for i in hub_nodes:
                    for j in nrange:
                        for k in range(lags.shape[0]):
                            pvals[i,j,k] = 1-2*np.abs(stats.percentileofscore(flat_surr_fcf[:,i,j],np.nanmean(efcf[:,i,j,k]),kind='strict')/100 - .5)

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
                max_pvals = np.full((N,N),np.nan)
                #TODO: Vectorize this
                for iI,I in enumerate(nrange):
                    for jJ,J in enumerate(nrange):
                        max_pvals[I,J] = pvals[I,J,valid_lag_indices[iI,jJ]]
                network_graph = np.logical_and(max_pvals<= pval_threshold,best_lag_values>=0)
                np.save(save_path+'network_adjacency_matrix.npy',network_graph)
                
    
    # TODO: Make sure this does not throw away too many things
    #Iterate through dimensions for the elbow
    if only_hubs:
        return 
    
    if pvals is not None:
        mask = (pvals <= pval_threshold)
        #print(f'Required pvalue: {corrected_pval} to satisfy {min_pairs} with {np.argwhere(mask.any(axis=2)).shape[0]} pairs')
    else:
        mask = np.ones((N,N,lags.shape[0]),dtype = bool)
    #TODO: Fix this it's being dumb with the sizes I am pretty sure because of the bad channels
    incomplete_pairs = np.ones((N,N),dtype=bool) #np.repeat(((mask.any(axis=2)).any(axis=0))[:,np.newaxis],N,axis=1).T
    print(f'Checking mask sizes: hub_mask={hub_mask.shape}, pvalue mask: {incomplete_pairs.shape}')
    incomplete_pairs &= hub_mask

    if save:
        np.save(save_path+'significant_hub_nodes.npy', np.argwhere(np.any(incomplete_pairs,axis=0))[:,0])

    found_nodes = np.zeros((N,N),dtype=bool)

    searched_dims_mask = np.zeros(dim_max-dim_min+1,dtype=bool)
    searched_dims_mask[-1] = True
    if os.path.isfile(save_path+'searched_dims.npy'):
        last_low_dim = np.load(save_path+'last_saved_low_dimension.npy')
        pre_shifted_dims =np.load(save_path+'searched_dims.npy')
        searched_dim_shifted = np.load(save_path+'searched_dims.npy')+max(0,last_low_dim-dim_min)
        searched_dims_mask[searched_dim_shifted] = True
    dims_to_search = np.arange(dim_min,dim_max+1)[~searched_dims_mask]
    search_iteration = 0

    reconstruction_accuracies = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
    reconstruction_corrs = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
    differential_CCMs = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
    differential_CCM0s = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
    sequential_cross_correlations_all = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
    cross_correlations_all = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
    
    reconstruction_accuracies[:,-1,:,:,:] = efcf
    reconstruction_corrs[:,-1,:,:,:] = extended_correlation
    differential_CCMs[:,-1,:,:,:] = differential_CCM
    differential_CCM0s[:,-1,:,:,:] = differential_CCM0
    sequential_cross_correlations_all[:,-1,:,:,:] = sequential_cross_correlations
    cross_correlations_all[:,-1,:,:,:] = cross_correlations

    if os.path.isfile(save_path+'all_fcfs.npy'):
        reconstruction_accuracies[:,searched_dims_mask,:,:,:] = np.load(save_path+'all_fcfs.npy')[:,np.load(save_path+'searched_dims.npy').flatten(),:,:,:]
        reconstruction_corrs[:,searched_dims_mask,:,:,:] = np.load(save_path+'all_ccms.npy')[:,np.load(save_path+'searched_dims.npy').flatten(),:,:,:]
        
    # Run the parallel like this:
    # Build the array of eFCF values for the significant pairs and compute the eFCF over each dim
    # With it parallelized for computing both dim = d_min+i and d_max - i, and a middle dim
    # This generates a curve filled out from both sides and the center, find the elbow of this curve over each i and when it has not changed for a few iterations (heuristic choice) 
    # declare that the early stoping criteria
    tested_optimium_dims = np.ones((dim_max-dim_min,N,N))*np.nan #np.zeros((dims_to_search.shape[0],N,N))
    tested_optimium_vals = np.ones((dim_max-dim_min,N,N))*np.nan
    actual_optimum_dims = np.ones((N,N))*np.nan
    actual_optimum_vals = np.ones((N,N))*np.nan

    np.save(save_path+'last_saved_low_dimension.npy',np.array([dim_min]))


    while dims_to_search.shape[0] > 0:
        if not incomplete_pairs.any():
            break            
        current_dims = np.array([dims_to_search[0],dims_to_search[dims_to_search.shape[0] // 2],dims_to_search[-1]])
        current_dims = np.unique(current_dims)
        dims_to_search = dims_to_search[~np.isin(dims_to_search,current_dims)]

        print(f'Computing CCM for dims: {current_dims} and {np.argwhere(incomplete_pairs).shape[0]} pairs')
        
        efferents = nrange #TODO, optimize and account for bad channels np.argwhere(incomplete_pairs.any(axis=0))[:,0]

        for d in range(current_dims.shape[0]):
            d_start_time = time.time_ns()
            # We need to rebuild the projected delay vectors so that they don't include more information 
            # from delays that would not be there if we evaluated this in the lower dimension  
            if random_projection:
                proj_A_d = proj_A[:,:current_dims[d]+1,:current_dims[d]+1]
                delay_vectors = np.einsum('imd,tdi->tmi',proj_A_d,un_proj_delay_vectors[:,:current_dims[d]+1,:])
                #rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]
            
            outs = DirectionalECCM(
                        XDs=delay_vectors,delay=delay,kfolds=kfolds,nrange=nrange,lags=lags,save_path=save_path+f'/dims/{current_dims[d]}/',
                        bad_channels=bad_channels,max_processes=max_processes,transform=transform,compute_noise_correlations=compute_noise_correlations)
            
            reconstruction_accuracies[:,current_dims[d]-dim_min,:,:,:]=outs[0]
            reconstruction_corrs[:,current_dims[d]-dim_min,:,:,:]=outs[1]
            if compute_noise_correlations:
                differential_CCMs[:,current_dims[d]-dim_min,:,:,:]=outs[2]
                differential_CCM0s[:,current_dims[d]-dim_min,:,:,:] = outs[3]
                sequential_cross_correlations_all[:,current_dims[d]-dim_min,:,:,:] = outs[4]
                cross_correlations_all[:,current_dims[d]-dim_min,:,:,:] = outs[5]

            searched_dims_mask[current_dims[d]-dim_min] = True
            if save:
                np.save(save_path+'all_fcfs.npy',reconstruction_accuracies)
                np.save(save_path+'all_ccms.npy',reconstruction_corrs)
                np.save(save_path+'searched_dims.npy',np.argwhere(searched_dims_mask))
                np.save(save_path+'all_rhodiff.npy',differential_CCMs)
                np.save(save_path+'all_rho0diff.npy',differential_CCM0s)
                np.save(save_path+'all_cross_correlations.npy',cross_correlations_all)
                np.save(save_path+'all_sequential_cross_correlations.npy',sequential_cross_correlations_all)

            
            curr_time_taken = time.time_ns()-d_start_time
        
            important_times[f'Dim{current_dims[d]}CCMTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{current_dims[d]}CCMPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            with open(save_path+'benchmarkings.yaml','w') as outfile:
                yaml.dump(important_times,outfile,default_flow_style=False)


        # TODO: This currently doesn't work as nicely as I want
        if False: #find_optimum_dims:           
            y = np.array([np.nanmean(reconstruction_corrs[:,searched_dims_mask,i,j,optimum_causal_lags[i,j]],axis=0) for i,j in np.argwhere(incomplete_pairs)])
            #y = np.array([np.nanmax(np.nanmean(reconstruction_accuracies,axis=0)[searched_dims,i,j,:],axis=1) for i,j in np.argwhere(incomplete_pairs)])
            curr_opts = FindFCFOptimums(test_dims[searched_dims_mask],y,fullrange=False)
            curr_pair = 0
            print(curr_opts[0].shape)

            for i,j in np.argwhere(incomplete_pairs):
                tested_optimium_dims[search_iteration,i,j] = curr_opts[0][curr_pair]
                tested_optimium_vals[search_iteration,i,j] = curr_opts[1][curr_pair]
                curr_pair += 1

            for i,j in np.argwhere(incomplete_pairs):
                check, count_max = repeated_value_early_stop(tested_optimium_dims[:,i,j])
                incomplete_pairs[i,j] = ~check or ~early_stop
                found_nodes[i,j] = check
                if check or (dims_to_search.shape[0] == 0):
                    actual_optimum_dims[i,j] = count_max
                    actual_optimum_vals[i,j] = tested_optimium_vals[search_iteration,i,j]
                    tested_optimium_dims[search_iteration:,i,j] = count_max
                    tested_optimium_vals[search_iteration:,i,j] = tested_optimium_vals[search_iteration,i,j]
        search_iteration += 1   
    if save:
        np.save(save_path+'all_fcfs.npy',reconstruction_accuracies)
        np.save(save_path+'all_ccms.npy',reconstruction_corrs)


    total_time = time.time_ns()-time_start

    important_times['TotalTime'] = total_time*(10**-9)

    with open(save_path+'benchmarkings.yaml','w') as outfile:
        yaml.dump(important_times,outfile,default_flow_style=False)

    print(f'Total Time taken: {total_time*(10**-9)} s')
    np.save(save_path+'time_elapsed.npy', np.array([total_time]))
    return

# %%
def DirectConnectionCCM(X,delay=0,connection_dim = 2,surrogate_dim = 3,lags=np.array([0]),lag_step=0,
                        compute_significance=True,n_surrogates=10,retain_test_set = True,normal_pval=True,lag_surrogates=False,save_surrogates=True,
                        save = True,save_path='./',seed=0,max_processes = 3,kfolds=5,random_projection=False,transform='fisher'):
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
    (efcf: matrix of FCF values indexed by [kfolds, efferent, afferent, lag] {The statistic used in the Causal Flow paper}
    ,surrogate_fcfs : matrix of surrogate fcfs indexed by [kfolds,n_surrogate,efferent,afferent]
    ,fcf_pvals: matrix of pvalues indexed by [efferent, afferent, lag]
    )
    ,
    (eccm: matrix of unlagged correlation values indexed by [kfolds, efferent, afferent, lag] {recommended by George Sugihara}
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

        lags = delay*lags if lag_step == 0 else lags
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

        if os.path.isfile(save_path + f'efcf.npy'):
            efcf = np.load(save_path + f'efcf.npy')
            eccm = np.load(save_path + f'eccm.npy')
        else:
            eccm_full_process_start_time = time.time_ns()
            
            reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
            eccm = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

            process_outputs = tqdm.tqdm([p.apply_async(CCMRow,
                                                       args=((k,n),rolled_delay_vectors[k][:,:connection_dim,:],n,train_indices,test_indices,lags)) 
                                                       for n in range(N) for k in range(kfolds)],
                                        desc='Computing Directionality')

            for proc in process_outputs:
                item = proc.get()
                reconstruction_accuracy[item[0][0],item[0][1],:,:] = item[1]
                eccm[item[0][0],item[0][1],:,:] = item[2]
            
            #print(f'Processed reconstructions in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')

            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            
            important_times[f'Dim{connection_dim}ReconstructionTotal'] = curr_time_taken*(10**-9)
            important_times[f'Dim{connection_dim}ReconstructionPerProcess'] = (curr_time_taken/(N*kfolds))*(10**-9)
            
            efcf = reconstruction_accuracy
            if transform == 'fisher':
                efcf = np.arctanh(efcf)
                eccm = np.arctanh(eccm)
        if save:
            np.save(save_path+'/efcf.npy',efcf)
            np.save(save_path+'/eccm.npy',eccm)

        if not compute_significance:
            return [efcf,None,None],[eccm,None,None]
        surrogate_fcfs = None
        surrogates = None

        if os.path.isdir(save_path+'/surrogates'):
            surrogates = [[[] for sn in range(n_surrogates)] for k in range(kfolds)]
            for f in os.listdir(save_path+'/surrogates'):
                sn = int(f[-7])
                k = int(f[-5])
                surrogates[k][sn] = np.load(save_path+'/surrogates/'+f)
        
        surr_lags = lags if lag_surrogates else np.array([0])

        surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,surr_lags.shape[0]))
        surrogate_corrs = np.zeros((n_surrogates,kfolds,N,N,surr_lags.shape[0]))

        if os.path.isfile(save_path+'surrogate_fcf.npy'):
            surrogate_fcfs = np.load(save_path+'surrogate_fcf.npy')
        if os.path.isfile(save_path+'surrogate_corrs.npy'):
            surrogate_corrs = np.load(save_path+'surrogate_corrs.npy')
       
        completed_surrs = np.zeros(n_surrogates,dtype = bool)
        surr_lags = np.array([0])
        if os.path.isfile(save_path+'completed_surrs.npy'):
            completed_surrs = np.load(save_path+'completed_surrs.npy')
        # Compute the significance intervals
        # Create Surrogate time series: n_surrogate list of NxT time series
        print('Creating twin surrogates')
        
        eccm_full_process_start_time = time.time_ns()
        already_made = True
        
        surr_data = rolled_delay_vectors
        if retain_test_set:
            surr_data = lib_targets
        if surrogates is None and (np.count_nonzero(completed_surrs) < n_surrogates):
            surrogates = [[] for k in range(kfolds)]
            already_made = False
            for k in range(kfolds):
                print(f'Surrogates for fold {k}')
                surrogates[k] = parallel_twin_surrogates(surr_data[k],n_surrogates=n_surrogates,Tshift = tShift,seed=np.random.randint(0,2**32-1,1)[0])

        if save_surrogates and not already_made:
            if not os.path.isdir(save_path+'/surrogates'):
                os.mkdir(save_path+'/surrogates')
            for k in range(kfolds):
                for n in range(n_surrogates):
                    np.save(save_path+'/surrogates/'+f'surrogate{n},{k}.npy',surrogates[k][n]) 

        print(f'Finished Surrogate Generation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
        
        curr_time_taken = time.time_ns()-eccm_full_process_start_time

        with open(save_path+'benchmarkings.yaml','w') as outfile:
            yaml.dump(important_times,outfile,default_flow_style=False)

        eccm_full_process_start_time = time.time_ns()

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
                processes = tqdm.tqdm([p.apply_async(CCMRow,
                                                     args=([k,n,sn],surrogate_delays[k][:,:connection_dim,:]
                                                           ,n,train_indices_surr,test_indices_surr,surr_lags))
                                                            for k in range(kfolds) for n in range(N)])

                for proc in processes:
                    item = proc.get()
                    surrogate_fcfs[item[0][2],item[0][0],item[0][1],:,:] = item[1]
                    surrogate_corrs[item[0][2],item[0][0],item[0][1],:,:] = item[2]
            else:
                #TODO really don't ever do this but Idk maybe if I have nothing to do at some point
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
                np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)
                np.save(save_path+'surrogate_corr.npy',surrogate_corrs)
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

        if os.path.isfile(save_path+'pvalues.npy'):
            fcf_pvals = np.load(save_path+'pvalues.npy')
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
            print(surr_shape_diff)
            train_indices_surr = train_indices[:-surr_shape_diff]
            test_indices_surr = test_indices-surr_shape_diff

            if retain_test_set:
                processes = tqdm.tqdm([p.apply_async(CCMRow,
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
                np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)
                np.save(save_path+'completed_surrs.npy',completed_surrs)
                with open(save_path+'benchmarkings.yaml','w') as outfile:
                    yaml.dump(important_times,outfile,default_flow_style=False)


            print(f'Finished Surrogate evaluation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            curr_time_taken = time.time_ns()-eccm_full_process_start_time
            important_times[f'Dim{connection_dim}AllSurrogatesCCMTotal'] = curr_time_taken*(10**-9)      

        flat_surr_fcf = (surrogate_fcfs.reshape(-1,*surrogate_fcfs.shape[2:]))[:,:,:,0]
        flat_surr_eccm = (surrogate_corrs.reshape(-1,*surrogate_fcfs.shape[2:]))[:,:,:,0]
    return surrogates,flat_surr_fcf,flat_surr_eccm

#Helpers
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.auto.tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()