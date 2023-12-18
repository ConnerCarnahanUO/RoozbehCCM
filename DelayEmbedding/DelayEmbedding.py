# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020
 
@author: ff215, Amin

"""
from audioop import cross
from dis import dis
#from this import d
#from webbrowser import MacOSX
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from functools import partial
#%%
import scipy as sci
from scipy import interpolate
from scipy.io import savemat
from scipy import stats
from scipy import signal
import numpy as np

#%%
import re
from sklearn.model_selection import KFold
#from tqdm import tqdm
import ipywidgets as wdgs
import matplotlib.pyplot as plt
import ray
import os

import time

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# %%
@ray.remote
def remote_connectivity(X,**args):
    return connectivity(X,**args)[0]

@ray.remote
def remote_build_nn_single(X,train_indices,test_indices,n_neighbors,**args):
    return build_nn_single(X,train_indices,test_indices,n_neighbors,**args)

@ray.remote
def remote_ApproximatebestTau(sig, taumin=0, taumax=20, bins=50, plot=True, threshold_ratio = 0.001,**args):
    return ApproximatebestTau(sig, taumin, taumax, bins, plot, threshold_ratio,**args)[0]

@ray.remote
def remote_eFCF_mat(X,dim=4,n_neighbors=0,test_ratio = 0.1,delay = 1,lags = np.arange(-10,11),mask = None,transform = 'fisher',return_pvalue = False, n_surrogates = 10,**args):
    return extendedFCF(X,dim,n_neighbors,test_ratio,delay,lags,mask,transform,return_pvalue,n_surrogates,**args)[0]

@ray.remote
def remote_eFCF_mat_1(X,dim=4,n_neighbors=0,test_ratio = 0.1,delay = 1,mask = None,transform = 'fisher',return_pvalue = False, n_surrogates = 10,**args):
    return extendedFCF(X,dim,n_neighbors,test_ratio,delay,np.array([0]),mask,transform,return_pvalue,n_surrogates,**args)[0][:,:,0]

@ray.remote
def remote_TestReconstruction(ActualTS,ReconstructingTS,nns,test_indices,**args):
    return TestReconstruction(ActualTS,ReconstructingTS,nns,test_indices,**args)

@ray.remote
def indexed_remote_TestReconstruction(index,ActualTS,ReconstructingTS,nns,test_indices,**args):
    return index, TestReconstruction(ActualTS,ReconstructingTS,nns,test_indices,**args)

@ray.remote
def remote_LagReconstruction(l,test_indices,lib_targets,targets,mask_idx,dim,nns,**args):
    return LagReconstruction(l,test_indices,lib_targets,targets,mask_idx,dim,nns,**args)

@ray.remote
def remote_build_nn(X,train_indices,test_indices,test_ratio=.02,n_neighbors=4,scaling=1,**args):
    return build_nn(X,train_indices,test_indices,test_ratio,n_neighbors,scaling,**args)

@ray.remote
def remote_twin_surrogates(X,N,**args):
    return twin_surrogates(X,N,**args)

@ray.remote
def remote_parallel_twin_surrogates(X,N,**args):
    return parallel_twin_surrogates(X,N,**args)

@ray.remote
def remote_SurrGen(X,ind,L,eln,seed=0,**args):
    return SurrGen(X,ind,L,eln,seed=seed,**args)

@ray.remote
def remote_extendedFCF(X,dim=4,n_neighbors=0,test_ratio = 0.1,delay = 1,lags = np.arange(-10,11),mask = None,transform = 'fisher',return_pvalue = False,n_surrogates = 10,parallel_nns = False,**args):
    val = extendedFCF(X,dim,n_neighbors,test_ratio,delay,lags,mask,transform,return_pvalue,n_surrogates,parallel_nns,**args)
    return val[0],val[2], val[3]

#%%
def create_delay_vector_spikes(spktimes,dim):
    """Create ISI delay vectors from spike times
    
    Args:
        spktimes (numpy.array): Array of spike times for a single channel
        dim (integer): Embedding dimensionality
        
    Returns:
        numpy.ndarray: Delay coordinates of the embedded spike train
    
    """
    
    return np.array([np.append(spktimes[i-dim:i]-spktimes[i-dim-1:i-1],spktimes[i-dim]) for i in range(dim+1,len(spktimes))])

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

def create_multidelay_vector(X,delays,dim):
    T,N = X.shape
    if len(delays) != N:
        print(f'Declared Delays must be the same length as number of time series. \n DelayLen: {len(delays)}, TS#: {N} ')
    max_delay = np.max(np.abs(delays))
    total_T = T-dim*max_delay+1
    delayedX = np.zeros((T,dim*N))

    for n in range(N):
        for i in range(dim):
            delayedX[0,:] = 0


def random_projection(x,dim):
    """Random projection of delay vectors for a more isotropic representation
    
    Args:
        x (numpy.ndarray): Delay coordinates of a sequence (n,time,delay)
        dim (integer): Projection dimensionality
        
    Returns:
        numpy.ndarray: Random projected signals
    
    """
    P =  np.random.rand(np.shape(x)[1],dim)
    projected = np.array([x[:,:,i]*P for i in range(x.shape[2])]).transpose(1,2,0)
    return projected

def cov2corr(cov):
    """Transform covariance matrix to correlation matrix
    
    Args:
        cov (numpy.ndarray): Covariance matrix (NxN)
        
    Returns:
        numpy.ndarray: Correlation matrix (NxN)
    
    """
    
    diag = np.sqrt(np.diag(cov))[:,np.newaxis]
    corr = np.divide(cov,diag@diag.T)
    return corr

        
def reconstruct(cues,lib_cues,lib_targets,n_neighbors=3,n_tests="all"):
    """Reconstruct the shadow manifold of one time series from another one
        using Convergent Cross Mapping principle and based on k-nearest-neighbours
        method
    
    Args:
        lib_cues (numpy.ndarray): Library of the cue manifold use for 
            reconstruction, the dimensions are L x d1 (where L is a large integer),
            the library is used for the inference of the CCM map
        lib_targets (numpy.ndarray): Library of the target manifold to be used
            for the reconstruction of the missing part, the dimensions are
            L x d2, the library is used for the inference of the CCM map
        cue (numpy.ndarray): The corresponding part in the cue manifold to the
            missing part of the target manifold, cue has dimension N x d1 
            (where N could be one or larger)
        
    Returns:
        numpy.ndarray: Reconstruction of the missing parts of the target manifold
    
    """
    
    nCues,dimCues=np.shape(cues)
    dimTargets=np.shape(lib_targets)[1]


    if n_tests == None:
        n_tests = nCues
        
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(lib_cues)
    
    # distances is a matrix of dimensions N x k , where k = nNeighbors
    # indices is also a matrix of dimensions N x k , where k = nNeighbors
    distances, indices = nbrs.kneighbors(cues)

    # the matrix weight has dimensions N x k 
    # we want to divide each row of the matrix by the sum of that row.
    weights = np.exp(-distances)
    
    # still, weight has dimensions N x k 
    weights = weights/weights.sum(axis=1)[:,None] #using broadcasting 
    
    # We want to comput each row of reconstruction by through a weighted sum of vectors from lib_targets
    reconstruction=np.zeros((nCues,dimTargets))
    for idx in range(nCues):
        reconstruction[idx,:] = weights[idx,:]@lib_targets[indices[idx,:],:] # The product of a Nxk matrix with a kxd2 matrix

    return reconstruction
     

def interpolate_delay_vectors(delay_vectors,times,kind='nearest'):
    """Interpolte delay vectors used for making the spiking ISI delay 
        coordinates look more continuous
    
    Args:
        delay_vectors (numpy.ndarray): 3D (N,time,delay) numpy array of the 
            delay coordinates
        times (numpy.ndarray): The time points in which delay vectors are sampled
        kind (string): Interpolation type (look at interp1d documentation)
        
    Returns:
        numpy.ndarray: Interpolated delay vectors
    
    """
    
    interpolated = np.zeros((len(times), delay_vectors.shape[1]))
    interpolated[:,-1] = times
    
    
    interp = interpolate.interp1d(delay_vectors[:,-1],delay_vectors[:,:-1].T,kind=kind,bounds_error=False)
    interpolated[:,:-1] = interp(times).T
    
    return interpolated
    
def mean_covariance(trials):
    """Compute mean covariance matrix from delay vectors
    
    Args:
        trials (numpy.ndarray): delay vectors
        
    Returns:
        numpy.ndarray: Mean covariance computed from different dimensions
            in the delay coordinates
    
    """
    
    _, nTrails, trailDim = np.shape(trials)

    covariances = []
    for idx in range(trailDim):
        covariances.append(np.cov(trials[:,:,idx]))

    covariances = np.nanmean(np.array(covariances),0)
    
    return covariances

def mean_correlations(trials):
    """Compute mean correlation matrix from delay vectors
    
    Args:
        trials (numpy.ndarray): delay vectors
        
    Returns:
        numpy.ndarray: Mean correlation computed from different dimensions
            in the delay coordinates
    
    """
    
    _, nTrails, trailDim = np.shape(trials)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trials[:,:,idx]))

    corrcoef = np.nanmean(np.array(corrcoefs),0)
    
    return corrcoef
    
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
    
    return corrcoef

def sequential_mse(trails1,trails2):
    """Compute the mean squared error between two signals from their delay 
        representations
    
    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal (shape Txd)
        trails2 (numpy.ndarray): delay vectors of the second signal (shape Txd)
        
    Returns:
        float: Mean squared error between the two delay representations
    
    """
    
    nTrails, trailDim=np.shape(trails1)

    mses = []
    for idx in range(trailDim):
        mses.append(np.nanmean((trails1[:,idx]-trails2[:,idx])**2))

    mses=np.nanmean(np.array(mses))
    
    return mses

def correlation_FC(X,transform='fisher'):
    
    T, N = X.shape
            
    #Loop over each pair and calculate the correlation between signals i and j
    correlation_mat  = np.zeros((N,N))*np.nan
    for i in range(N):
        for j in range(i,N):
            cc = np.corrcoef(X[:,i],X[:,j])[0,1]
            correlation_mat[i,j] = cc
            correlation_mat[j,i] = cc
            
#     #Apply transformation   
#     if transform == 'fisher':
#         correlation_mat = np.arctanh(correlation_mat)
        
    return correlation_mat

def connectivity_with_xval(X,train_indices,test_indices,delay=1,dim=10,n_neighbors=0,scaling = 1,method='corr',mask=None,transform='fisher',return_p_value = False, n_surrogates=20, compute_surrogates_parallel = True, MAX_PROCESSES = 92,print_check = False,return_surrogates=False,overwrite=True):
    T, N = X.shape
    #Reconstruct the attractor manifold for each node in the delay coordinate state space
    #and split train and test delay vectors
    #Get index where training set is split in 2
    #Scale is outdated do not use
    if n_neighbors == 0:
        n_neighbors = dim+1
    timeShiftBufferLength = delay*(dim-1)
    bb = np.where(np.diff(train_indices) != 1)[0]
    if len(bb) == 1:
        train1 = train_indices[0:bb[0]+1]
        train2  = train_indices[bb[0]+1:]

        dv1 = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X[train1].T)),2)
        dv2 = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X[train2].T)),2)
        lib_targets = np.concatenate((dv1,dv2),0)
        targets = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X[test_indices].T)),2)
    
    else:
        #For first and last test set, the training set is 1 continuous block
        lib_targets = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X[train_indices].T)),2)
        targets = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X[test_indices].T)),2)
    
    X_delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    #print(delay_vectors.shape)
    #Calculate reconstruction error only for elements we want to compute
    if mask is None:
        mask = np.zeros((N,N)).astype(bool)
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    #Build nearest neighbour data structures for multiple delay vectors
    #nns is a list of tuples: the first being the weights used for the forecasting technique, 
    #and the second element the elements corresponding to the training delay vector
    nns_ = []  
    for i in range(N):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(lib_targets[:,:,i])
        distances, indices = nbrs.kneighbors(targets[:,:,i])
        weights = np.exp(-distances)
        weights = weights/(weights.sum(axis=1)[:,np.newaxis])
        nns_.append((weights,indices))
    
    nns = [[]]*N
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]
        
    #Loop over each pair and calculate the topological embeddedness for signal i by another signal j
    #This will be quantified based on the correlation coefficient (or MSE) between the true and forecast signals
    reconstruction_error = None
    if return_p_value:
        reconstruction_error = np.zeros((N,N))*np.nan
    else:
        reconstruction_error = np.zeros((N,N))*np.nan
    for i, j in zip(*mask_idx):
            
        #Use k-nearest neigbors forecasting technique to estimate the forecast signal
        reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(targets.shape[0])])

        if method == 'corr':
            reconstruction_error[i,j] = sequential_correlation(reconstruction, targets[:,:,j])
        elif method == 'mse':
            reconstruction_error[i,j] = sequential_mse(reconstruction, targets[:,:,j])
    
    #Apply transformation   
    if transform == 'fisher':
        reconstruction_error = np.arctanh(reconstruction_error)

    #Get Directionality measure as well
    directionality = reconstruction_error - reconstruction_error.T
    
    if return_p_value:
        surrogate_size= len(train_indices)+len(test_indices)-timeShiftBufferLength
        surrogates = np.zeros((N,n_surrogates,surrogate_size))*np.nan
        results = None
        if compute_surrogates_parallel:
            with Pool(MAX_PROCESSES) as p:
                #TODO currently can't do it because there are problems having parallel code that is called when process is already being parallel processed
                print("at the actual computation")
                surrogates[mask_u_idx,:,:] = np.array(list(p.map(partial(twin_surrogates,N=n_surrogates), targets[:,:,mask_u_idx].transpose([2,0,1]))))
                results = p.map(partial(connectivity_with_xval,X,train_indices=train_indices,test_indices=test_indices,delay=delay,dim=dim,n_neighbors=n_neighbors,method=method,mask=mask,transform=transform),surrogates.transpose([1,2,0]))
        else:
            #Create the surrogate time series
            #twin_surrogates(targets[:,:,mask_u_idx].transpose([2,0,1]),n_surrogates)    
            surrogates[mask_u_idx,:,:] = np.array(list(map(lambda x: twin_surrogates(x,n_surrogates,test_indices=test_indices,end_method='overwrite'), X_delay_vectors[:,:,mask_u_idx].transpose([2,0,1]))))
            results = list(map(lambda x: connectivity_with_xval(x,test_indices=test_indices[test_indices<surrogate_size],train_indices=train_indices[train_indices<surrogate_size],delay=delay,dim=dim,n_neighbors=n_neighbors,method=method,mask=mask, transform=transform), surrogates.transpose([1,2,0])))

        #Get surrogate results
        connectivity_surr = np.array([r[0] for r in results]).transpose([1,2,0])
        directionality_surr = np.array([r[1] for r in results]).transpose([1,2,0])


        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval_FCF = 1-2*np.abs(np.array([[stats.percentileofscore(connectivity_surr[i,j,:],reconstruction_error[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
        #print(pval_FCF)
        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval_dir = 1-2*np.abs(np.array([[stats.percentileofscore(directionality_surr[i,j,:],directionality[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

        # One way to get the significance is to calculate the z-score of the FCF relative to the surrogate distribution and pass it through the surival function; this gets similar signifant elements
        # pval_FCF = stats.norm.sf((reconstruction_error - np.mean(connectivity_surr,axis=-1))/np.std(connectivity_surr,axis=-1))
        if return_surrogates:
            return reconstruction_error, pval_FCF, directionality, pval_dir, connectivity_surr
        return reconstruction_error, pval_FCF, directionality, pval_dir
    else:
        return reconstruction_error, directionality
    # %%

def MutualInformation(sig1,sig2,bins=50):
    pXYjoint = np.histogram2d(sig1,sig2,density = True,bins=bins)[0]
    pXYindep = np.outer(np.sum(pXYjoint,axis=1),np.sum(pXYjoint,axis=0))
    return sci.stats.entropy(pXYjoint.flatten(),pXYindep.flatten())

def ApproxRelativeTau(sig1,sig2,taumin=0,taumax=40,bins=50,plot=True,smooth=False,smoothing=0,tau_extrap = 20,threshold = 0.05):
    taus = np.arange(taumin,taumax+1)
    MIs = np.zeros(len(taus))*np.nan
    sig1_len = sig1.shape[0]
    sig2_len = sig2.shape[0]
    total_points = sig1_len
    if sig1_len < sig2_len:
        sig2 = sig2[:total_points]
    else:
        total_points = sig2_len
        sig1 = sig1[:total_points]

    x = np.arange(taumin,taumax)
    for i in np.arange(MIs.shape[0]):
        tau_len = total_points - np.abs(i+taumin)
        MIs[i] = MutualInformation(sig1[:tau_len],np.roll(sig2,i+taumin)[:tau_len])
    
    y = MIs

    if smooth:
        trep = sci.interpolate.splrep(x,MIs,s=smoothing)
        x = np.linspace(taumin,taumax,len(x)*tau_extrap)
        y = sci.interpolate.BSpline(*trep)(x)
    
    """    
    tauhigh = np.argmax(MIs)
    H1min = MIs[tauhigh]
    """
    tau_star = taumin
    H_min = np.inf
    
    for i in range(y.shape[0]):
        if  threshold <= H_min-y[i]:
            H_min = MIs[i]
        else:
            tau_star = int(i)-1-taumin
            break

    if plot:
        fig = plt.figure(figsize=(12,10))
        a = plt.axes()
        a.plot(taus,y,label='Delay Mutual Information')
        a.axvline(tau_star,linestyle='--',color = (0,0,0),label=f'Optimum tau = {tau_star+taumin}')
        plt.xlabel('Tau')
        plt.ylabel('I[y(t);y(t-tau)]')
        plt.legend()
        plt.show()
    
    return tau_star, MIs


def ApproximatebestTau(sig, taumin=0, taumax=20, bins=50, plot=True, threshold_ratio = 0.1):
    MIs = np.zeros(taumax-taumin)*np.nan
    total_points = sig.shape[0]
    
    for tau in np.arange(MIs.shape[0]):
        tau_len = total_points - tau
        MIs[tau] = MutualInformation(sig[:tau_len],sig[tau:])
    
    tau_star = 0
    H_min = np.inf
    for i in range(MIs.shape[0]):
        if (MIs[i] <= H_min):
            H_min = MIs[i]
        else:
            tau_star = i-1
            break
    if tau_star == 0:
        connecting_line = MIs[0]+(MIs[-1]-MIs[0])/MIs.shape[0] * np.arange(MIs.shape[0])
        tau_star = np.argmax(np.abs(connecting_line-MIs))

    if plot:
        fig = plt.figure(figsize=(12,10))
        a = plt.axes()
        a.plot(np.arange(taumin,taumax),MIs,label='Delay Mutual Information')
        a.axvline(tau_star,linestyle='--',color = (0,0,0),label=f'Optimum tau = {tau_star+taumin}')
        plt.xlabel('Tau')
        plt.ylabel('I[y(t);y(t-tau)]')
        plt.legend()
        plt.show()
    
    return tau_star, MIs

def connectivity(
        X,test_ratio=.02,delay=10,dim=3,n_neighbors=0,mask=None,
        transform='fisher',return_pval=False,n_surrogates=20,
        save=False,load=False,file=None
    ):
    '''Pairwise effective connectivity based on convergent cross mapping
    
    Args:
        X (numpy.ndarray): Multivariate signal to compute functional connectivity from (TxN), columns are the time series for different chanenls/neurons/pixels
        test_ratio (float): Fraction of the test/train split (between 0,1)
        delay (integer): Delay embedding time delay 
        dim (integer): Delay embedding dimensionality
        mask (numpy.ndarray): 2D boolean array represeting which elements of the functional connectivity matrix we want to compute
        transform (string): Transofrmation applied to the inferred functional connectivity, choose from ('fisher','identity')
        return_pval (bool): If true the pvales will be computed based on twin surrogates method
        n_surrogates (integer): Number of twin surrogates datasets created for computing the pvalues
        save (bool): If True the results of the computations will be saved in a mat file
        file (string): File address in which the results mat file will be saved
        
    Returns:
        numpy.ndarray: the output is a matrix whose i-j entry (i.e. reconstruction_error[i,j]) is the error level observed when reconstructing channel i from channel, which used as the surrogate for the functional connectivity
        numpy.ndarray: If return_pval is True this function also returns the matrix of pvalues
    '''
#
    if n_neighbors == 0:
        n_neighbors = dim+1


    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue'],result['surrogates']

    T, N = X.shape
    tShift = np.abs(delay)*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    
    # How much data are we going to try and reconstruct?
    tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)

    # Get indices for training and test datasets
    iStart = tDelay - tTest; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    
    # Calculate reconstruction error only for elements we want to compute
    if mask is None: mask = np.zeros((N,N)).astype(bool)
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    # Build nearest neighbour data structures for multiple delay vectors
    # nns is a list of tuples: the first being the weights used for the forecasting technique, 
    # and the second element the elements corresponding to the training delay vector
    nns_ = build_nn(delay_vectors[:,:,mask_u_idx],train_indices,test_indices,test_ratio,n_neighbors)
    nns = [[]]*N
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]

    # Loop over each pair and calculate the topological embeddedness for signal i by another signal j
    # This will be quantified based on the correlation coefficient between the true and forecast signals
    reconstruction_error = np.zeros((N,N))*np.nan
    reconstruction_error_1 = np.zeros((N,N))*np.nan
    reconstruction_error_b = np.zeros((N,N))*np.nan

    for i, j in zip(*mask_idx):
        # Use k-nearest neigbors forecasting technique to estimate the forecast signal
        reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])
        reconstruction_error[i,j] = sequential_correlation(reconstruction, targets[:,:,j])
        """if other_correlations:
            corrcoefs = []
            for idx in range(dim):
                corrcoefs.append(np.corrcoef(reconstruction[:,idx], targets[:,:,j][:,idx])[0,1])
            reconstruction_error_1[i,j] = corrcoefs[0]
            reconstruction_error_b[i,j] = np.max(corrcoefs)"""
    
    # Apply transformation   
    if transform == 'fisher': 
        fcf = np.arctanh(reconstruction_error)
        reconstruction_error_1 = np.arctanh(reconstruction_error_1)
        reconstruction_error_b = np.arctanh(reconstruction_error_b)
    if transform == 'identity': fcf = reconstruction_error
    

    pval_hists = [[]]


    """
    if return_dual_pvals:
        #TODO: maybe adjust the number of surrogates to be n_surrogates//2 for the total set for each test
        refs = [twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
        dual_surrogates = np.array(ray.get(refs))
        for j in range(n_surrogates):
                dual_surrogates[:,j,test_indices] = X[test_indices,:].T
        
        surrogate_delays = [[]] 
        for i in range(n_surrogates):
            surrogate_delays.append(np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], dual_surrogates[:,i,:])),2))
        surrogate_delays.pop(0)
        #surrogate_delays = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], surrogates.T)),2)

        # compute false positive distribution for p(y|\chi_i)
        false_efferents = np.zeros((N,N,n_surrogates))*np.nan
        # 1. take the already computed weights for the y in question
        #    and compute the fcf with each surrogate in place of the efferent
        for k in range(len(surrogate_delays)):
            surr = surrogate_delays[k]
            for i, j in zip(*mask_idx):
                surr_reconstruction = np.array([nns[i][0][idx,:]@surr[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])
                false_efferents[i,j,k] = sequential_correlation(surr_reconstruction, targets[:,:,j])

        dual_surrogates_FCF = false_efferents
        if transform == 'fisher': dual_surrogates_FCF = np.arctanh(dual_surrogates_FCF)
        
        print(dual_surrogates.shape)

        refs = [twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
        dual_surrogates_aff = np.array(ray.get(refs))
        for j in range(n_surrogates):
                dual_surrogates_aff[:,j,test_indices] = X[test_indices,:].T
                
        # compute false positive distribution for p(\eta_i | x)
        # 1. compute the weights for each surrogate
        false_afferents = np.zeros((N,N,n_surrogates))*np.nan
        for i,j in zip(*mask_idx):
            Xij = np.zeros((dual_surrogates_aff.shape[2],dual_surrogates_aff.shape[1]+1))
            Xij[:,0] = X[:Xij.shape[0],j]
            Xij[:,1:] = dual_surrogates[i,:,:].T
            idxmask = np.ones((n_surrogates+1,n_surrogates+1)).astype(bool)
            idxmask[0,:] = np.zeros(n_surrogates+1).astype(bool)
            idxmask[0,0] = True
            fcfXij = connectivity2(Xij,
                                    delay=delay,
                                    dim=dim,
                                    test_ratio=test_ratio,
                                    n_neighbors=n_neighbors,
                                    mask=idxmask)
            false_afferents[i,j] = fcfXij[0][0,1:]

        aff_fcf = false_afferents
        if transform == 'fisher': aff_fcf = np.arctanh(aff_fcf)

        if len(hist_plot_neurons) > 0:
            for ij in hist_plot_neurons:
                fig = plt.figure()
                a = plt.axes()
                i,j = ij
                a.hist(dual_surrogates_FCF[i,j,:], label=f'efferent dist {i},{j}',density=True)
                a.hist(aff_fcf[i,j,:], label=f'afferent dist {i},{j}',density=True)
                a.hist(fcf[i,j],label=f'actual {i},{j}')
                plt.legend()
                plt.title('New Pvals')
                plt.show()

        pval_hists.append(dual_surrogates_FCF)
        pval_hists.append(aff_fcf)
                    

        dual2 = 1-2*np.abs(np.array([[stats.percentileofscore(aff_fcf[i,j,:],fcf[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
        dual_pval = 1-2*np.abs(np.array([[stats.percentileofscore(dual_surrogates_FCF[i,j,:],fcf[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
        aggregate_dist = np.zeros((N,N,2*n_surrogates))
        aggregate_dist[:,:,:n_surrogates] = aff_fcf
        aggregate_dist[:,:,n_surrogates:] = dual_surrogates_FCF
        aggregate_pval = 1-2*np.abs(np.array([[stats.percentileofscore(aggregate_dist[i,j,:],fcf[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
    else:
        dual_pval,dual_surrogates,dual_surrogates_FCF,dual2 = None,None,None,None
        aggregate_pval = None
        """
    if return_pval:
        refs = [remote_twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
        surrogates = np.array(ray.get(refs))
        refs = [remote_connectivity.remote(
                surrogates[:,i].T,
                test_ratio=test_ratio,
                delay=delay,
                dim=dim,
                n_neighbors=n_neighbors,
                mask=mask,
                transform=transform
            ) for i in range(surrogates.shape[1])]
        fcf_surrogates = np.stack(ray.get(refs)).reshape((n_surrogates,N,N))
        print(fcf_surrogates.shape)
        
        # Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval = 1-2*np.abs(np.array([[stats.percentileofscore(fcf_surrogates[:,i,j],fcf[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)
        """
        if len(hist_plot_neurons) > 0:
            for ij in hist_plot_neurons:
                fig = plt.figure()
                a = plt.axes()
                i,j = ij
                xplot = np.linspace(-1,1)
                gplot = Gaussian(xplot,g_stats[0,i,j],g_stats[1,i,j],normalize=False)
                a.plot(xplot,gplot,label='Distribution')
                a.hist(fcf_surrogates[i,j,:], label=f'{i},{j}',density=True)
                a.hist(fcf[i,j],label=f'actual {i},{j}')
                plt.title('Old Pval')
                plt.legend()
                plt.show()
        pval_hists.append(fcf_surrogates)
        """
    else:
        pval,surrogates,fcf_surrogates = None,None,None
        
        
    if save: np.save(file,{'cnn':fcf,'pvalue':pval,'surrogates':surrogates,'transform':transform})

    pval_hists.pop(0)
    return fcf, pval, surrogates, fcf_surrogates


def ExtendedFCFCrossvalidation(X,dim=4,n_neighbors=0,kfolds = 5,delay = 1,lags = np.arange(-10,11),mask = None,transform = 'fisher',return_pvalue = False,n_surrogates = 10, num_cpus = 32, num_gpus = 0, parallel_nns = False, parallel_folds = True,retain_test_set=False):
    start = time.time()
    ray.shutdown()
    ray.init(num_cpus = num_cpus, num_gpus = num_gpus)
    print(time.time()-start)
    start = time.time()

    fold_size = X.shape[0] // kfolds
    fold_ratio = fold_size/X.shape[0]
    foldvals = [[]]*kfolds
    if parallel_folds:
        refs = [remote_extendedFCF.remote(np.roll(X,fold_size*i,axis=0),dim,n_neighbors,fold_ratio,delay,lags,mask,transform,return_pvalue,n_surrogates,parallel_nns) for i in range(kfolds)]
        foldvals = ray.get(refs)
        print(time.time()-start)
    else:
        for i in range(kfolds):
            fold = extendedFCF(np.roll(X,fold_size*i,axis=0),dim,n_neighbors,fold_ratio,delay,lags,mask,transform,return_pvalue,n_surrogates,parallel_nns)
            foldvals[i] = (fold[0], fold[2], fold[3])
        print(time.time()-start)
    output = [[]]*len(foldvals[0])
    for i in range(len(foldvals[0])):
        if foldvals[0][i] is not None:
            output[i] = np.vstack([foldvals[k][i][np.newaxis] for k in range(kfolds)])
        else:
            output[i] = np.array([])
    ray.shutdown()
    return output

def extendedFCF(X,dim=4,n_neighbors=0,test_ratio = 0.1,delay = 1,lags = np.arange(-10,11),mask = None,transform = 'fisher',return_pvalue = False,n_surrogates = 10,parallel_nns = True, retain_test_set = True):
    if n_neighbors == 0:
        n_neighbors = dim+1

    T, N = X.shape
    tShift = np.abs(delay)*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    
    # How much data are we going to try and reconstruct?
    tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)

    # Get indices for training and test datasets
    iStart = tDelay - tTest; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    
    # Calculate reconstruction error only for elements we want to compute
    if mask is None: mask = np.zeros((N,N)).astype(bool)
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    # Build nearest neighbour data structures for multiple delay vectors
    # nns is a list of tuples: the first being the weights used for the forecasting technique, 
    # and the second element the elements corresponding to the training delay vector
    nns_ = None
    if parallel_nns:
        refs = [remote_build_nn_single.remote(delay_vectors[:,:,i],train_indices,test_indices,n_neighbors) for i in range(N)]
        nns_ = ray.get(refs)
    else:
        nns_ = build_nn(delay_vectors[:,:,mask_u_idx],train_indices,test_indices,test_ratio,n_neighbors)
    nns = [[]]*N
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]

    test_lags = lags
    num_delays = test_lags.shape[0]
    # Loop over each pair and calculate the topological embeddedness for signal i by another signal j
    # This will be quantified based on the correlation coefficient between the true and forecast signals
    reconstruction_error = np.zeros((N,N,num_delays))*np.nan
    
    """    lag_refs = [remote_LagReconstruction.remote(test_lags[L],test_indices,lib_targets,targets,mask_idx,dim,nns) for L in range(test_lags.shape[0])]
    all_reconsts = ray.get(lag_refs)

    for L in range(test_lags.shape[0]):
        reconstructions[L], reconstruction_error[:,:,L] = all_reconsts[L]"""

    max_cutoff = np.max([1,np.max(np.abs(test_lags))])
    reconstructions = np.zeros((num_delays,N,N,dim,len(test_indices)-max_cutoff))



    for L in range(test_lags.shape[0]):
        #Generate the shifted time series
        l = test_lags[L]
        cutoff = -max(1,abs(l))
        reconL = np.zeros((N,N,dim,len(test_indices)+cutoff))
        lagged_lib_targets = np.roll(lib_targets,l,axis=0)
        targets_roll = np.roll(targets,l,axis=0)[:cutoff,:,:]
        test_is = test_indices[:cutoff]
        # Use k-nearest neigbors forecasting technique to estimate the forecast signal
        refs = [remote_TestReconstruction.remote(targets_roll[:,:,j],lagged_lib_targets[:,:,j],nns[i],test_is) for i,j in zip(*mask_idx)]
        reconsts_and_errors = ray.get(refs)
        for i, j in zip(*mask_idx):
            reconL[i,j,:,:], reconstruction_error[i,j,L] = reconsts_and_errors[N*i+j]
        reconstructions[L,:,:,:,:] = reconL[:,:,:,:len(test_indices)-max_cutoff]
    
    if transform == 'fisher': 
        fcf = np.arctanh(reconstruction_error)
    elif transform == 'identity':
        fcf = reconstruction_error
    if return_pvalue:
        surrogates = None
        if retain_test_set:        
            # We want to only jumble the training set so we only generate the surrogate with that
            refs = [remote_parallel_twin_surrogates.remote(delay_vectors[:iStart,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
            surrogates = np.array(ray.get(refs))
            surrogates = np.dstack([surrogates, np.repeat(targets[:,0,:].T[:,np.newaxis,:],n_surrogates,axis=1)])
        else:
            refs = [remote_parallel_twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
            surrogates = np.array(ray.get(refs))
        
        # With enough samples it actually does not matter if the surrogate is calculated with lags, so we only compute it for the 0 lag case
        refs = [remote_eFCF_mat_1.remote(
                surrogates[:,i].T,
                test_ratio=test_ratio,
                delay=delay,
                dim=dim,
                n_neighbors=n_neighbors,
                mask=mask,
                transform=transform,
                parallel_nns=True,
                return_pvalue = False
            ) for i in range(surrogates.shape[1])]
        efcf_surrogates = np.stack(ray.get(refs))

        pval = 1-2*np.abs(np.array([[[stats.percentileofscore(efcf_surrogates[:,i,j],fcf[i,j,k],kind='strict') for k in range(test_lags.shape[0])] for j in range(N)] for i in range(N)])/100 - .5)
    else:
        pval = None
        efcf_surrogates = None

    return fcf, reconstructions, pval, efcf_surrogates, nns, delay_vectors, test_indices, train_indices, lags


def TestReconstruction(ActualTS,ReconstructingTS,nns,test_indices):
    """ Build the CCM reconstruction and correlate it with the actual time series using the nearest neighbors and indices for the test
        TODO: Vectorize the reconstruction step
    """
    reconstruction = np.array([np.matmul(nns[0][idx,:],ReconstructingTS[nns[1][idx,:],:])
                                        for idx in range(len(test_indices))])
    reconstruction_error = sequential_correlation(reconstruction,ActualTS)
    return reconstruction.T, reconstruction_error

@ray.remote
def indexed_remote_LaggedReconstruction(index,Predicted,Predictor,nns,test_indices):
    return index, LaggedReconstruction(Predicted,Predictor,nns,test_indices)

def LaggedReconstruction(Predicted,Predictor,nns,test_indices):
    """ LaggedReconstruction: computes the reconstruction of a lagged time series and its accuracy
        
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
    return reconstruction.T, reconstruction_error, np.corrcoef(reconstruction[:,0],Predicted[:,0])

def LagReconstruction(l,test_indices,lib_targets,targets,mask_idx,dim,nns):
    #Generate the shifted time series
    N = lib_targets.shape[2]
    cutoff = -max(1,abs(l))
    reconL = np.zeros((N,N,dim,len(test_indices)+cutoff))
    lagged_lib_targets = np.roll(lib_targets,l,axis=0)
    targets_roll = np.roll(targets,l,axis=0)[:cutoff,:,:]
    test_is = test_indices[:cutoff]
    # Use k-nearest neigbors forecasting technique to estimate the forecast signal
    refs = [remote_TestReconstruction.remote(targets_roll[:,:,j],lagged_lib_targets[:,:,j],nns[i],test_is) for i,j in zip(*mask_idx)]
    reconsts_and_errors = ray.get(refs)
    reconstruction_error = np.zeros((N,N))*np.nan
    #load results into the temporary arrays
    for i, j in zip(*mask_idx):
            reconL[i,j,:,:], reconstruction_error[i,j] = reconsts_and_errors[N*i+j]
    return reconL, reconstruction_error

def ParallelFullECCM(X,d_min=1, d_max=0, kfolds=5, delay=0, lags=np.arange(-8,9), mask = None, transform='fisher', test_pval = True, n_surrogates = 100, save=True, path = './', retain_test_set = False, num_cpus = 64,num_gpus = 0,max_mem = 64*10**9,max_processes = 0):
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
    if delay == 0:
        refs = [remote_ApproximatebestTau.remote(X[:,i],0,10,50,False,0.1) for i in range(N)]
        MIDelays = np.array(ray.get(refs))
        delay = np.max([1,np.min(MIDelays)])
        print(MIDelays)

    #TODO: Determine the highest dim based on the Takens' Dimension (Considering Nearest Neighbors method)
    test_dims = np.arange(d_min,d_max+1)
    num_dims = test_dims.shape[0]

    lags = delay*lags
    #max_dim_fcf = ExtendedFCFCrossvalidation(X,dim=d_max,delay=delay,kfolds=kfolds,lags=lags,num_cpus=num_cpus,num_gpus=num_gpus,return_pvalue=True,n_surrogates=n_surrogates,retain_test_set=retain_test_set)

    #max_dim_peaks, max_dim_synchs = LagPeaks(np.nanmean(max_dim_fcf[0],axis=3),lags)

    #Possibly Need to figure out what to do with the cross-validated pvalue thing (leave it for a sec)

    #First test at the maximum dim to find the limiting value and the significance interval
    tShift = np.abs(delay)*(d_max-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Determine the size of the test set and the test / train indices
    t_test = tDelay // kfolds
    iStart = tDelay - t_test; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)    

    #Build delay space projection
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,d_max)[:,:,np.newaxis], X.T)),2)
    
    # build the rolled set of delay projections for each fold, the FCF is invariant over total shifts 
    # so we can always just compute it in the same interval but with rolled data
    rolled_delay_vectors = [np.roll(delay_vectors,t_test*i,axis=0) for i in range(kfolds)]

    # Build the test and training sets
    targets = [rolled_delay_vectors[i][test_indices,:,:] for i in range(kfolds)]
    lib_targets = [rolled_delay_vectors[i][train_indices,:,:] for i in range(kfolds)]
    
    print(targets[0].shape)
    print(lib_targets[0].shape)


    # build the weight matrices for all the variables and folds
    all_nns = [[]]*kfolds
    for i in range(kfolds):
        refs = [remote_build_nn_single.remote(rolled_delay_vectors[i][:,:,j],train_indices,test_indices,d_max+1) for j in range(N)]
        all_nns[i] = ray.get(refs)
    
    #refs = [[remote_build_nn_single.remote(rolled_delay_vectors[i][:,j,:],train_indices,test_indices,d_max+1,1)
    #         for j in range(N)] for i in range(kfolds)]
    #nns = ray.get(refs)

    # Compute the reconstructions for all the pairs and folds
    # remote_TestReconstruction(ActualTS,ReconstructingTS,nns,test_indices)
    
    max_lag = np.max(np.abs(lags))
    
    #setting max_processes = 0 means just run the maximum available
    if max_processes == 0:
        max_processes = (num_cpus + num_gpus)
    # Queue the reconstructions
    refs = [[]]*max_processes
    ind = 0
    batch = 0
    # We may want to Split this because stuff seems to get unsafe when instantiating this many workers

    # Unload locations
    #reconstructions = np.zeros((kfolds,N,N,lags.shape[0],d_max,len(test_indices)-max_lag))*np.nan
    reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
    corr_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

    eccm_full_process_start_time = time.time_ns()
    for m in range(N):
        for n in range(N):
            for l in range(lags.shape[0]):
                for k in range(kfolds):
                    cutoff = len(test_indices)-abs(lags[l])
                    refs[ind] = indexed_remote_LaggedReconstruction.remote([k,n,m,l],np.roll(targets[k][:,:,n],lags[l],axis=0)[:cutoff,:],
                                                np.roll(lib_targets[k][:,:,m],lags[l],axis=0),all_nns[k][n],test_indices[:cutoff]) #returns reconstruction, reconstruction error
                    ind+=1
                    # Evaluate if we have reached the max number of processes 
                    # (there may be a better way to do this or a way for ray to handle this but I have suffered enough at the hands of the machine and hopefully this will tame it)
                    if ind >= max_processes:
                        print(f'Processing batch: {batch}')
                        start_time = time.time_ns()
                        remote_fcfs = ray.get(refs)
                        refs = [[]]*max_processes
                        ind = 0
                        #for item in remote_fcfs:
                            #reconstructions[item[0][0],item[0][1],item[0][2],item[0][3],:,:] = item[1][0][:,:len(test_indices)-max_lag]
                        for item in remote_fcfs:
                            reconstruction_accuracy[item[0][0],item[0][1],item[0][2],item[0][3]] = item[1][1]
                        for item in remote_fcfs:
                            corr_accuracy[item[0][0],item[0][1],item[0][2],item[0][3]] = item[1][2][0,1]
                        batch += 1
                        print(f'Processed in: {time.time_ns()-start_time} ns')

    if ind < max_processes:
        refs = refs[:ind]    
    
    # Evaluate last batch
    print(f'Processing last batch: {batch}')
    remote_fcfs = ray.get(refs)

    #for item in remote_fcfs:
        #reconstructions[item[0][0],item[0][1],item[0][2],item[0][3],:,:] = item[1][0][:,:len(test_indices)-max_lag]
    for item in remote_fcfs:
        reconstruction_accuracy[item[0][0],item[0][1],item[0][2],item[0][3]] = item[1][1]
    for item in remote_fcfs:
        corr_accuracy[item[0][0],item[0][1],item[0][2],item[0][3]] = item[1][2][0,1]
    
    print(f'Processed all batches in: {time.time_ns()-eccm_full_process_start_time} ns')
    efcf = reconstruction_accuracy
    eccm = corr_accuracy
    if transform == 'fisher':
        efcf = np.arctanh(efcf)
        eccm = np.arctanh(eccm)

    if save:
        np.save(path + f'eFCFTensorXValidated_dim{d_max}_delay{delay}.npy',efcf)
        np.save(path + f'eCCMTensorXValidated_dim{d_max}_delay{delay}.npy',eccm)
        np.save(path + f'lags.npy',lags)

    # Average over folds and determine the limiting statistics
    averaged_accuracy = np.nanmean(efcf,axis=0)
    
    # Compute the significance intervals
    refs = [remote_twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
    surrogates = np.array(ray.get(refs))
    # This is only semi-parallel so
    refs = [remote_connectivity.remote(
            surrogates[:,i].T,
            test_ratio=1/kfolds,
            delay=delay,
            dim=d_max,
            n_neighbors=0,
            mask=mask,
            transform=transform
        ) for i in range(surrogates.shape[1])]
    efcf_surrogates = np.stack(ray.get(refs))

    if save:
        np.save('surrogate_fcf.npy',efcf_surrogates)
        np.save('surrogates.npy',surrogates)

    pval = 1-2*np.abs(np.array([[[stats.percentileofscore(efcf_surrogates[:,i,j],averaged_accuracy[i,j,k],kind='strict') for k in range(lags.shape[0])] for j in range(N)] for i in range(N)])/100 - .5)

        
    #reconstructions
    return averaged_accuracy, pval, efcf, efcf_surrogates, surrogates, lags, test_indices, train_indices, rolled_delay_vectors

    #Test for whether the maximums are significant on the positive side
    #TODO



    #Set up the boolean matrices that determine whether we should continue computing CCM for all pairs
    IsComplete = np.zeros((N,N), dtype = bool)

def LagPeaks(FCFTensor,lags,prominence=0.2,width=1):
    #TODO: Need to decide whether the prominence threshold is the best way or to use the wavelet transform version (cwt)
    N = FCFTensor.shape[0]
    peaks = [[[]]*N]*N
    synchs = [[[]]*N]*N
    for i in range(N):
        for j in range(N):
            #peaksij = signal.find_peaks(FCFTensor[i,j,:],prominence=prominence*np.max(np.abs(FCFTensor[i,j,:])))[0]-low_lag
            peaksij = lags[signal.find_peaks_cwt(FCFTensor[i,j,:],widths = [width])]
            peaks[i][j] = peaksij[peaksij >= 0]
            synchs[i][j] = peaksij[peaksij < 0]
    return peaks,synchs
            

"""def cross_validate(X,nKfold=40,delay=10,dim=3,n_neighbors=4,scaling =1,method='corr',mask=None,transform='fisher',return_pval=False,n_surrogates=20,save_data=False,file=None,parallel=False,MAX_PROCESSES=96):
    
    ##===== Do kfolds in parallel =====##
    T, N = X.shape
    k_fold = KFold(n_splits=nKfold)
    FCF_kfold = np.zeros((nKfold,N,N))*np.nan
    DIR_kfold = np.zeros((nKfold,N,N))*np.nan
    
    process_outputs = []
    with Pool(MAX_PROCESSES) as p:
        for iK, (train_indices, test_indices) in tqdm(enumerate(k_fold.split(X))):
            process_outputs.append(p.apply_async(connectivity_with_xval, args = (X, train_indices,test_indices,delay,dim,n_neighbors,scaling,method,mask,transform)))
            
        for iK,out in enumerate(process_outputs):
            FCF_kfold[iK] = out.get()[0]
            DIR_kfold[iK] = out.get()[1]
                  
    return FCF_kfold, DIR_kfold"""

def connectivity_parameter_search(X,nKfold=10,params={},print_progress = True, MAX_PROC = 52):
    dims = np.arange(1)
    if not "dims" in params.keys():
        print("No dimension set")
        return
    else:
        dims = params["dims"]
    nns = np.array([0])
    if "nns" in params.keys():
        nns = params["nns"]
    scales = np.array([1])
    if "scales" in params.keys():
        scales = params["scales"]
    delays = np.array([1])
    if "delays" in params.keys():
        delays = params["delays"]
    
    N = X.shape[1]
    allFCF_kfold = np.zeros((dims.size,nns.size,scales.size,delays.size,N,N,nKfold))*np.nan
    dimsize = dims.size
    nnssize = nns.size
    scalesize = scales.size
    delaysize = delays.size
    
    #This 100% could probably be multithreaded but I am scared because of the multithreading already in the cross validation
    for i in np.arange(dimsize):
        for j in np.arange(nnssize):
            ns = nns[j]
            if ns == 0:
                ns = i+1
            for k in np.arange(scalesize):
                for l in np.arange(delaysize):
                    allFCF_kfold[i,j,k,l,:,:,:] , _ = cross_validate(X,nKfold=nKfold,delay=delays[l],dim=dims[i],n_neighbors=ns,scaling=scales[k], MAX_PROCESSES=MAX_PROC)
                    #if print_progress:
                        #print(f"Finished delay={dims[i]},scale={scales[k]},nearest neighbors = {ns},delay size = {delays[l]}. Have {np.count_nonzero(np.isnan(allFCF_kfold))/(N*N*nKfold)} left to compute.")

    return allFCF_kfold

def connectivity3(X,test_ratio=.02,delay=10,dim=3,n_neighbors=4,scaling = 1,method='corr',
                 mask=None,transform='fisher',return_pval=False,n_surrogates=20,
                 save_data=False,file=None,parallel=False,MAX_PROCESSES=96,return_surrogates = False):
    """Create point clouds from a video using Matching Pursuit or Local Max algorithms
    
    Args:
        X (numpy.ndarray): Multivariate signal to compute functional 
            connectivity from (TxN), columns are the time series for 
            different chanenls/neurons/pixels
        test_ratio (float): Fraction of the test/train split (between 0,1)
        delay (integer): Delay embedding time delay 
        dim (integer): Delay embedding dimensionality
        method (string): Method used for computing the reconstructability,,
            choose from ('mse','corr')
        mask (numpy.ndarray): 2D boolean array represeting which elements of the 
            functional connectivity matrix we want to compute
        transform (string): Transofrmation applied to the inferred functional 
            connectivity, choose from ('fisher','identity')
        return_pval (bool): If true the pvales will be computed based on twin
            surrogates method
        n_surrogates (integer): Number of twin surrogates datasets created for 
            computing the pvalues
        save_data (bool): If True the results of the computations will be saved 
            in a mat file
        file (string): File address in which the results mat file will be saved
        parallel (bool): If True the computations are done in parallel
        MAX_PROCESSES (integer): Max number of processes instantiated for parallel 
            processing
        
    Returns:
        numpy.ndarray: the output is a matrix whose i-j entry (i.e. reconstruction_error[i,j]) 
            is the error level observed when reconstructing channel i from channel, 
            which used as the surrogate for the functional connectivity
        numpy.ndarray: If return_pval is True this function also returns the 
            matrix of pvalues
    """
    T, N = X.shape
    tShift = delay*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    
    #How much data are we going to try and reconstruct?
    tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)

    #Get indices for training and test datasets
    iStart = tDelay - tTest; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    #print("Train indices shape: " + str(train_indices.shape))
    #print("Test Indices Shape: " + str(test_indices.shape))
    #print("Targets shape: " + str(targets.shape))
    
#     print(f'{lib_targets.shape[0]}, {targets.shape[0]}, f{tDelay-lib_targets.shape[0]-targets.shape[0]}')
#     import pdb; pdb.set_trace()
    #Calculate reconstruction error only for elements we want to compute
    if mask is None:
        mask = np.zeros((N,N)).astype(bool)
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    #Build nearest neighbour data structures for multiple delay vectors
    #nns is a list of tuples: the first being the weights used for the forecasting technique, 
    #and the second element the elements corresponding to the training delay vector
    nns_ = build_nn(delay_vectors[:,:,mask_u_idx],train_indices,test_indices,test_ratio,n_neighbors=n_neighbors,scaling=scaling)
    nns = [[]]*N
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]
    
    #Loop over each pair and calculate the topological embeddedness for signal i by another signal j
    #This will be quantified based on the correlation coefficient (or MSE) between the true and forecast signals
    reconstruction_error = np.zeros((N,N))*np.nan
    #print(delay_vectors.shape)
    reconstructions = np.zeros(shape = (N,N,test_indices.size,targets.shape[1]))

    for i, j in zip(*mask_idx):

        #Use k-nearest neigbors forecasting technique to estimate the forecast signal
        reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])
        #if i==0 and j == 0:
            #print(reconstruction.shape)
        reconstructions[i,j,:,:] = reconstruction

        if method == 'corr':
            reconstruction_error[i,j] = sequential_correlation(reconstruction, targets[:,:,j])
        elif method == 'mse':
            reconstruction_error[i,j] = sequential_mse(reconstruction, targets[:,:,j])
    
    #Apply transformation   
    if transform == 'fisher':
        reconstruction_error = np.arctanh(reconstruction_error)

    #Get Directionality measure as well
    directionality = reconstruction_error - reconstruction_error.T

    if return_pval:
        surrogates = np.zeros((N,n_surrogates,tDelay))*np.nan
        if parallel:
            with Pool(MAX_PROCESSES) as p:
                surrogates[mask_u_idx,:,:] = np.array(list(p.map(partial(twin_surrogates,N=n_surrogates), delay_vectors[:,:,mask_u_idx].transpose([2,0,1]))))
                results = p.map(partial(connectivity3,test_ratio=test_ratio,delay=delay,dim=dim,n_neighbors=n_neighbors,method=method,mask=mask,transform=transform),surrogates.transpose([1,2,0]))

                # connectivity_surr = np.array(list(p.map(partial(connectivity,test_ratio=test_ratio,delay=delay,
                #              dim=dim,n_neighbors=n_neighbors,method=method,mask=mask,transform=transform), surrogates.transpose([1,2,0])))).transpose([1,2,0])
        else:    
            surrogates[mask_u_idx,:,:] = np.array(list(map(lambda x: twin_surrogates(x,n_surrogates), delay_vectors[:,:,mask_u_idx].transpose([2,0,1]))))

            results = list(map(lambda x: connectivity3(x,test_ratio=test_ratio,delay=delay,dim=dim,n_neighbors=n_neighbors,method=method,mask=mask, transform=transform), surrogates.transpose([1,2,0])))
            # connectivity_surr = np.array(list(map(lambda x: connectivity(x,test_ratio=test_ratio,delay=delay,
            #              dim=dim,n_neighbors=n_neighbors,method=method,mask=mask, transform=transform), surrogates.transpose([1,2,0])))).transpose([1,2,0])
            
        #Get surrogate results
        connectivity_surr = np.array([r[0] for r in results]).transpose([1,2,0])
        directionality_surr = np.array([r[1] for r in results]).transpose([1,2,0])

        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval_FCF = 1-2*np.abs(np.array([[stats.percentileofscore(connectivity_surr[i,j,:],reconstruction_error[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval_dir = 1-2*np.abs(np.array([[stats.percentileofscore(directionality_surr[i,j,:],directionality[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

        # #One way to get the significance is to calculate the z-score of the FCF relative to the surrogate distribution and pass it through the surival function; this gets similar signifant elements
        # pval_FCF = stats.norm.sf((reconstruction_error - np.mean(connectivity_surr,axis=-1))/np.std(connectivity_surr,axis=-1))
                    
        if save_data:
            savemat(file+'.mat',{'fcf':reconstruction_error,'pval_FCF':pval_FCF,'directionality':directionality,'pval_dir':pval_dir,'surrogates':surrogates,'connectivity_surr':connectivity_surr,'n_surrogates':n_surrogates,
                                 'test_ratio':test_ratio,'delay':delay,'dim':dim,'n_neighbors':n_neighbors,
                                 'method':method,'mask':mask,'transform':transform})
        if return_surrogates:
            return reconstruction_error, pval_FCF, connectivity_surr
        else:
            return reconstruction_error, pval_FCF, directionality, pval_dir
    else:
        
        if save_data:
            savemat(file+'.mat',{'fcf':reconstruction_error,'directionality':directionality,'test_ratio':test_ratio,'delay':delay,'dim':dim,'n_neighbors':n_neighbors,
                                 'method':method,'mask':mask,'transform':transform})
            
        return reconstruction_error, directionality, reconstructions, test_indices, nns



def build_nn(X,train_indices,test_indices,test_ratio=.02,n_neighbors=4,scaling=1):
    """Build nearest neighbour data structures for multiple delay vectors
    
    Args:
        X (numpy.ndarray): 3D (TxDxN) numpy array of delay vectors for
            multiple signals
        train_indices (array): indices used for the inference of the CCM
            mapping
        test_indices (array): indices used for applying the inferred CCM
            mapping and further reconstruction
    Returns:
        array: Nearest neighbor data structures (see the documentation of 
                 NearestNeighbors.kneighbors)
    
    """
    nns = []
    for i in range(X.shape[2]):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(X[train_indices,:,i])
        distances, indices = nbrs.kneighbors(X[test_indices,:,i])
        weights = np.exp(-distances)  
        #print(distances)
        weights = weights/(weights.sum(axis=1)[:,np.newaxis])
        nns.append((weights,indices))
    return nns

def build_nn_single(X,train_indices,test_indices,n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(X[train_indices,:])
    distances, indices = nbrs.kneighbors(X[test_indices,:])
    distances = distances/np.min(distances)
    weights = np.exp(-distances)
    weights = weights/(weights.sum(axis=1)[:,np.newaxis])
    return (weights,indices)

def buld_nn_parallel(X,train_indices,test_indices,n_neighbors=4,scaling = 1):
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X[train_indices,:])
    distances, indices = nbrs.kneighbors(X[test_indices,:])
    if scaling == 0:
        scaling = np.min(distances)
    weights = np.exp(-distances/scaling)
    weights = weights/(weights.sum(axis=1)[:,np.newaxis])
    return (weights,indices)
        
def estimate_dimension(X, tau, method='fnn'):
    """Estimate the embedding dimension from the data
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate
            the embedding dimension
        tau (integer): Takens time delay 
        method (string): Method for estimating the embedding dimension, choose
            from ('fnn', hilbert)
        
    Returns:
        integer: Estimated embedding dimension
    
    """
    
    # TODO: Implement correlation dimension and box counting dimension methods
    
    L = X.shape[1]
    
    if method == 'hilbert':
        asig=np.fft.fft(X)
        asig[np.ceil(0.5+L/2):]=0
        asig=2*np.fft.ifft(asig)
        esig=[asig.real(), asig.imag()]

    elif 'fnn' in method:
        RT=20
        mm=0.01
        
        spos = [m.start() for m in re.finditer('-',method)]
        
        if len(spos)==1:
            RT=float(method[spos:])
            
        if len(spos)==2:
            RT=float(method[spos[0]+1:spos[1]])
            mm=float(method[spos[1]+1:])
        
        
        pfnn = [1]
        d = 1
        esig = X.copy()
        while pfnn[-1] > mm:
            nbrs = NearestNeighbors(2, algorithm='ball_tree').fit(esig[:,:-tau].T)
            NNdist, NNid = nbrs.kneighbors(esig[:,:-tau].T)
            
            NNdist = NNdist[:,1:]
            NNid = NNid[:,1:]
            
            d=d+1
            EL=L-(d-1)*tau
            esig=np.zeros((d*X.shape[0],EL))
            for dn in range(d):
                esig[dn*X.shape[0]:(dn+1)*X.shape[0],:]=X[:,(dn)*tau:L-(d-dn-1)*tau].copy()
            
            # Checking false nearest neighbors
            FNdist = np.zeros((EL,1))
            for tn in range(esig.shape[1]):
                FNdist[tn]=np.sqrt(((esig[:,tn]-esig[:,NNid[tn,0]])**2).sum())
            
            pfnn.append(len(np.where((FNdist**2-NNdist**2)>((RT**2)*(NNdist**2)))[0])/EL)
            
        D = d-1 
        esig=np.zeros((D*X.shape[0],L-(D-1)*tau))
        
        for dn in range(D):
            esig[dn*X.shape[0]:(dn+1)*X.shape[0],:]=X[:,dn*tau:L-(D-dn-1)*tau].copy()
            
        return D,esig,pfnn


def estimate_timelag(X,method='autocorr'):
    """Estimate the embedding time lag from the data
    
    Args:
        X (numpy.ndarray): (TxN) multivariate signal for which we want to estimate
            the embedding time lag
        method (string): Method for estimating the embedding time lag tau, choose
            from ('autocorr', 'mutinf')
        
    Returns:
        integer: Estimated embedding time lag
    
    TODO: mutinf section does not work

    """
    
    L = len(X)
    if method == 'autocorr':
        x = np.arange(len(X)).T
        FM = np.ones((len(x),4))
        for pn in range(1,4):
            CX = x**pn
            FM[:,pn] = (CX-CX.mean())/CX.std()
        
        csig = X-FM@(np.linalg.pinv(FM)@X)
        acorr = np.real(np.fft.ifft(np.abs(np.fft.fft(csig))**2).min(1))
        tau = np.where(np.logical_and(acorr[:-1]>=0, acorr[1:]<0))[0][0]
        
    elif method == 'mutinf':
        NB=np.round(np.exp(0.636)*(L-1)**(2/5)).astype(int)
        ss=(X.max()-X.min())/NB/10 # optimal number of bins and small shift
        bb=np.linspace(X.min()-ss,X.max()+ss,NB+1)
        bc=(bb[:-1]+bb[1:])/2
        bw=np.mean(np.diff(bb)) # bins boundaries, centers and width
        mi=np.zeros((L))*np.nan; # mutual information
        for kn in range(L-1):
            sig1=X[:L-kn]
            sig2=X[kn:L]
            # Calculate probabilities
            prob1=np.zeros((NB,1))
            bid1=np.zeros((L-kn)).astype(int)
            prob2=np.zeros((NB,1))
            bid2=np.zeros((L-kn)).astype(int)
            jprob=np.zeros((NB,NB))
            
            for tn in range(L-kn):
                cid1=np.floor(0.5+(sig1[tn]-bc[0])/bw).astype(int)
                bid1[tn]=cid1
                prob1[cid1]=prob1[cid1]+1
                cid2=np.floor(0.5+(sig2[tn]-bc[0])/bw).astype(int)
                bid2[tn]=cid2
                prob2[cid2]=prob2[cid2]+1
                jprob[cid1,cid2]=jprob[cid1,cid2]+1
                jid=(cid1,cid2)
                
            prob1=prob1/(L-kn)
            prob2=prob2/(L-kn)
            jprob=jprob/(L-kn)
            prob1=prob1[bid1]
            prob2=prob2[bid2]
            jprob=jprob[jid[0],jid[1]]
            
            # Estimate mutual information
            mi[kn]=np.nansum(jprob*np.log2(jprob/(prob1*prob2)))
            # Stop if minimum occured
            if kn>0 and mi[kn]>mi[kn-1]:
                tau=kn
                break
    else:
        raise Exception('Method {} not implemented'.format(method))
    
    return tau


def twin_surrogates(X,N):
    '''Create twin surrogates for significance evaluation and hypothesis testing
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate the embedding dimension
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    d, indices = nbrs.kneighbors(X)
    threshold = np.percentile(d[:,1],10)
    #print('Twin Surrogate Threshold: ' + str(threshold))
    
    nbrs = NearestNeighbors(radius=threshold, algorithm='ball_tree').fit(X)
    d, indices = nbrs.radius_neighbors(X)
    indices = [list(i) for i in indices]
    #print(indices)
    u,a = np.unique(indices,return_inverse=True)
    ind = [u[a[i]] for i in range(len(a))]
    eln = [len(i) for i in ind]
    surr = np.zeros((N,X.shape[0]))
    L = X.shape[0]
    for sn in range(N):
        kn=np.random.randint(0,L,1)[0]-1
        for j in range(L):
            kn += 1
            surr[sn,j] = X[kn,0]
            kn = ind[kn][np.random.randint(0,eln[kn],1)[0]]
            if kn==L-1:
                kn=L//2
    
    return surr

def parallel_twin_surrogates(X,N):
    '''Create twin surrogates for significance evaluation and hypothesis testing with each surrogate created in parallel
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate the embedding dimension
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    d, indices = nbrs.kneighbors(X)
    threshold = np.percentile(d[:,1],10)
    #print('Twin Surrogate Threshold: ' + str(threshold))
    
    nbrs = NearestNeighbors(radius=threshold, algorithm='ball_tree').fit(X)
    d, indices = nbrs.radius_neighbors(X)
    indices = [list(i) for i in indices]
    #print(indices)
    u,a = np.unique(indices,return_inverse=True)
    ind = [u[a[i]] for i in range(len(a))]
    eln = [len(i) for i in ind]
    refs = [remote_SurrGen.remote(X,ind,X.shape[0],eln) for sn in range(N)]
    surr = np.array(ray.get(refs))
    return surr

def SurrGen(X,ind,L,eln,seed=0):
    kn=np.random.randint(0,L,1)[0]-1
    surr = np.zeros(X.shape[0])
    for j in range(L):
        kn += 1
        surr[j] = X[kn,0]
        kn = ind[kn][np.random.randint(0,eln[kn],1)[0]]
        if kn==L-1:
            kn=L//2
    return surr

def autocorrelation_func(x):
    """Autocorrelation function
        http://stackoverflow.com/q/14297012/190597
        
        Args:
            x (np.ndarray): signal
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def find_correlation_time(x,dt,nlags=100):
    """Autocorrelation time of a multivariate signal
    
    Args:
        x (np.ndarray): is a time series on a regular time grid
        dt (float): is the time interval between consecutive time points
    
    Returns:
        float: Autocorrelation time
    """
    C = autocorrelation_func(x)
    if  len(np.where(C<0)[0])>0:
        C = C[:(np.where(C<0)[0][0])]
    return (dt*np.sum(C)/C[0])

def FindElbow(y, x = np.array([]), smoothing = 1, printerror = True,method = "Kneedle"):
    #Elbow finding algorithm, defined here: https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
    #Modified to be less complicated
    if method == "Kneedle":
        if x.size == 0:
            x = np.arange(y.size)
        
        #Generate the spline interpolation of the data
        tknot = sci.interpolate.splrep(x,y,s=smoothing)
        xnew = np.linspace(x[0],x[-1])
        spl = sci.interpolate.splev(xnew,tknot)
        
        #normalize the domain and spline
        xdiff = x[-1]-x[0]
        ymin = np.min(y)
        ydiff = np.max(y)-ymin
        xnew2 = (xnew - x[0])/xdiff
        splnew = (spl-ymin)/ydiff
        
        #Generate the difference curve
        diffcurve = splnew - xnew2
        
        #Find the local maxima of the difference curve
        dist = np.zeros((x.size))
        for i in np.arange(x.size):
            test_ind = np.abs(xnew-x[i]).argmin()
            dist[i] = diffcurve[test_ind]
            
        return dist.argmax(), y[dist.argmax()]

def NewCrossVal(X, dimbounds = [2,20], taubounds = [1,10], nFolds = 5, return_p_value = False, n_surrogates = 100, return_surrogates = False, MAXCORES = 96):
    #TODO: The loading bar is screwed up
    T, N = X.shape
    ndims = dimbounds[1]-dimbounds[0]
    dmin = dimbounds[0]
    ntaus = taubounds[1]-taubounds[0]
    taumin = taubounds[0]
    fullout = np.zeros((nFolds,ndims,ntaus,N,N))
    crossval_ratio = 1/nFolds
    shift_amount = int(T*crossval_ratio)
    shifted_X = np.zeros((T,N,nFolds))
    loading_bar = wdgs.IntProgress(
        value=0,
        min=0,
        max=ntaus*ndims*nFolds,
        description='Loading:',
        bar_style= 'info',
        style={'bar_color': 'maroon'},
        orientation='horizontal'
    )
    text = wdgs.Text(value = '0')
    textval = 0
    display(loading_bar)
    display(text)
    for d in range(nFolds):
        shifted_X[:,:,d] = np.roll(X,-d*shift_amount,axis = 0)
    for iK in range(nFolds):
        process_outputs = []
        kX = shifted_X[:,:,iK]
        with Pool(MAXCORES) as p:
            for d in range(ndims):
                for tau in range(ntaus):
                    process_outputs.append([iK,d,tau,p.apply_async(connectivity,args=(kX,crossval_ratio,tau+taumin,d+dmin,0,None,'fisher',False,n_surrogates,False,False,None))])
            for iP in process_outputs:
                fullout[iP[0],iP[1],iP[2],:,:] = iP[3].get()[0]
                loading_bar.value += 1
                textval += 1
                text.value = str(textval)
    return fullout

def FCFOptimumSearch(FCF,all_neurons = True,zero_diagonals = False):
    optimums = np.zeros((FCF.shape[2],FCF.shape[3],4)) #Shape is: Predictor, Predicted, [opt_tau,opt_dim,opt_FCF] 
    for n1 in range(FCF.shape[2]):
        for n2 in range(FCF.shape[3]):
            possibles = np.zeros(FCF.shape[1])
            possiblevals = np.zeros(FCF.shape[1])
            for tau in range(FCF.shape[1]):
                possibles[tau], possiblevals[tau] = FindElbow(FCF[:,tau,n1,n2])
            opttau = np.argmax(possiblevals)
            #print(opttau)
            optdim = possibles[opttau]
            optimums[n1,n2,:3] = np.array([opttau+1,optdim+1,possiblevals[opttau]])
    if zero_diagonals:
        for i in range(optimums.shape[0]):
            optimums[i,i,:] = [0,0,0,0]
    optimums[:,:,3] = optimums[:,:,2]/np.max(optimums[:,:,2])
    return optimums

#TODO: Move this to helpers

def Gaussian(x,mu,sigma,normalize = True):
    if normalize:
        return np.exp(-((x-mu)**2)/(2*(sigma**2)))/np.sqrt(np.pi*sigma**2)
    else:
        return np.exp(-((x-mu)**2)/(2*(sigma**2)))

def eCCMCrossval(y,dim=2,n_neighbors=0,delay=1,kfolds = 5,lags=np.arange(-10,11),return_pval = False, n_surrogates = 10,return_all = True):
    T,N = y.shape
    eCCMvals = np.zeros((kfolds,N,N,lags[1]-lags[0]+1))
    eCCMpvals = np.zeros((kfolds,N,N,lags[1]-lags[0]+1))
    eCCMsurrvals = np.zeros((kfolds,n_surrogates,N,N))
    kfoldshift = T // kfolds
    eccms = [[]]*kfolds
    for i in range(kfolds):
        ycurr = np.roll(y,kfoldshift*i,axis=0)
        eccms[i] = extendedFCF(ycurr,delay=delay,dim=dim,n_neighbors=n_neighbors,lags=lags,return_pvalue=return_pval,n_surrogates=n_surrogates)
    if return_all:
        return eccms
    for i in range(kfolds):
        eCCMvals[i,:,:,:] = eccms[i][0]
    if return_pval:
        for i in range(kfolds):
            eCCMpvals[i,:,:,:] = eccms[i][2]
            eCCMsurrvals[i,:,:,:] = eccms[i][3]
    return eCCMvals, eCCMpvals, eCCMsurrvals

# %%
