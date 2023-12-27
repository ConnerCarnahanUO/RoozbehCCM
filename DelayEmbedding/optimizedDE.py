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
import pandas

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
    reconstructions = [[]]*lags.shape[0]
    reconstruction_error = np.zeros(lags.shape[0])
    for l in range(lags.shape[0]):
        L = lags[l]
        curr_predictor = np.roll(Predictor,L,axis=0)
        reconstruction = np.array([nns[0][idx,:]@curr_predictor[nns[1][idx,:],:] for idx in range(len(test_indices))])
        reconstructions[l] = reconstruction.T
        reconstruction_error[l] = sequential_correlation(reconstruction,np.roll(Predicted,L,axis=0))[0]
    return reconstruction_error #reconstructions, reconstruction_error#, np.corrcoef(reconstruction[:,0],Predicted[:,0])

def indexed_SequentialLaggedReconstruction(index,Predicted,Predictor,nns,test_indices,lags=np.array([0])):
    return index, SequentialLaggedReconstruction(Predicted,Predictor,nns,test_indices,lags=lags)

def reconstruction_column(index,nns,targets,lib_targets,test_indices,lags):
    recon_accuracies = np.zeros((lib_targets.shape[2],lags.shape[0]))
    for i in range(lib_targets.shape[2]):
        recon_accuracies[i,:] = SequentialLaggedReconstruction(targets[:,:,i],lib_targets[:,:,i],nns,test_indices,lags)
    return index, recon_accuracies

def FullCCMColumn(index,TimeSeries,afferent,train_indices,test_indices,lags):
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
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree').fit(TimeSeries[train_indices,:,afferent])
    distances, indices = nbrs.kneighbors(TimeSeries[test_indices,:,afferent])
    # TODO? This really should scale the distance by 1/min{distances} but it keeps throwing a fit 
    #       so I am not dealing with that rn
    weights = np.exp(-distances)
    weights = weights/(weights.sum(axis=1)[:,np.newaxis])
    return reconstruction_column(index,(weights,indices),targets,lib_targets,test_indices,lags)

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
    return index, SurrGen(channel,X,ind,L,eln,seed=0)[1]

def SurrogateMap(X,channel):
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
    return channel, ind, eln

def indexed_SurrogateMap(X,channel,kfold):
    channel, ind, eln = SurrogateMap(X,channel)
    return (kfold,channel), ind, eln

def parallel_twin_surrogates(X,N,Tshift=0,max_processes=64):
    '''Create twin surrogates for significance evaluation and hypothesis testing with each surrogate created in parallel
    
    Args:
        X (numpy.ndarray): (Channel x Dimension x Time) multivariate signal for which we want to generate surrogate time series
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    kfolds = len(X)
    T,D,C = X[0].shape

    elns = [[[]]*C]*kfolds
    inds = [[[]]*C]*kfolds

    surrogate_time = time.time_ns()
    #Generate the surrogate swapping indices for each channel in parallel
    with Pool(max_processes,initializer=MuteMultiprocessing) as p:
        processes = tqdm.tqdm([p.apply_async(indexed_SurrogateMap,args = (X[k][:,:,channel],channel,k)) for channel in range(C) for k in range(kfolds)])

        for proc in processes:
            curr_map = proc.get()
            elns[curr_map[0][0]][curr_map[0][1]] = curr_map[2]
            inds[curr_map[0][0]][curr_map[0][1]] = curr_map[1]
    
    print(f'Surrogate Map generation time: {(time.time_ns()-surrogate_time)*10**-9} s')
    """    for channel in range(C):

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
        d, indices = nbrs.kneighbors(X[:,:,C])
        threshold = np.percentile(d[:,1],10)
        #print('Twin Surrogate Threshold: ' + str(threshold))
        
        nbrs = NearestNeighbors(radius=threshold, algorithm='ball_tree').fit(X)
        d, indices = nbrs.radius_neighbors(X)
        indices = [list(i) for i in indices]
        #print(indices)
        u,a = np.unique(indices,return_inverse=True)
        ind = [u[a[i]] for i in range(len(a))]
        inds[channel] = ind
        elns[channel] = [len(i) for i in ind]"""

    surrs = [[[[]]*C]*kfolds]*N

    # Generate all of the surrogates in parallel for each channel
    with Pool(max_processes,initializer=MuteMultiprocessing) as p:    
        processes = tqdm.tqdm([p.apply_async(indexed_SurrGen, args = ((k,channel,sn),channel,X[k][:,:,channel],inds[k][channel],T,elns[k][channel])) for sn in range(N) for channel in range(C) for k in range(kfolds)])
    
    #Load all surrogates into the surrogate array
        for channel_process in processes:
            surr = channel_process.get()
            surrs[surr[0][2]][surr[0][0]][surr[0][1]] = surr[1]

        """for channel_proccesses in processes:
            i = 0
            for sn in channel_proccesses:
                surr = sn.get()
                surrs[i][surr[0]] = surr[1]
                i+=1"""

    #reshape surrogate lists into np arrays:
    surrs = [[np.vstack(surrs[i][k]).T for k in range(kfolds)] for i in range(N)]
    
    print(surrs[0][0].shape)

    return surrs

def AddNextOptimum(pair_mask, dims, maxFCFs, smoothing=0.5,fullrange = False):
    ys = [[]]*pair_mask[pair_mask].shape[0]
    yinds = [[]]*pair_mask[pair_mask].shape[0]
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



#%%
# Main Computation
def ParallelFullECCM(X,d_min=1, dim_max=30, kfolds=5, delay=0,
                     lags=np.arange(-8,9), mask = None, transform='fisher',
                     node_ratio = 0.1, test_pval = True, n_surrogates = 10, normal_pval=False, pval_threshold=0.05, 
                     min_pairs = 1, dim_search_stopping_num = 3, save=True, save_path = './', retain_test_set = False, 
                     max_processes = 64):
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
    
    hub_nodes_needed = int(np.ceil(node_ratio*N))
    """def X_ref():
        global X
        return X"""
    
    time_start = time.time_ns()

    print('Finding Optimum Delay')
    with Pool(processes=max_processes) as p:
        if delay == 0:
            process_outputs = [[]]*N
            MIDelays = np.zeros(N)
            process_outputs = tqdm.tqdm([p.apply_async(remote_ApproximatebestTau, args = (i,X[:,i],0,10,50,False,0.1)) for i in range(N)])
            for out in process_outputs:
                MIDelays[out.get()[0]] = out.get()[1]
            delay = int(np.max([1,np.min(MIDelays)]))
            print(MIDelays)

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
        
        # build the rolled set of delay projections for each fold, the FCF is invariant over total shifts 
        # so we can always just compute it in the same interval but with rolled data
        rolled_delay_vectors = [np.roll(delay_vectors,t_test*k,axis=0) for k in range(kfolds)]

        # Build the test and training sets
        targets = [rolled_delay_vectors[k][test_indices,:,:] for k in range(kfolds)]
        lib_targets = [rolled_delay_vectors[k][train_indices,:,:] for k in range(kfolds)]

        efcf = None
        if os.path.isfile(save_path + f'eFCFTensorXValidated_dim{dim_max}_delay{delay}.npy'):
            efcf = np.load(save_path + f'eFCFTensorXValidated_dim{dim_max}_delay{delay}.npy')
            #np.save(path + f'eCCMTensorXValidated_dim{d_max}_delay{delay}.npy',eccm)
        else:
            eccm_full_process_start_time = time.time_ns()
            # build the weight matrices for all the variables and folds
            all_nns = [[[]]*N]*kfolds
            print('Building CCM Mappings')
            process_outputs =tqdm.tqdm([p.apply_async(remote_build_nn_single,
                                                    args = ((k,j),rolled_delay_vectors[k][:,:,j],
                                                            train_indices,test_indices,
                                                            dim_max+1))
                                                            for j in range(N) for k in range(kfolds)])
            
            for proc in process_outputs:
                item = proc.get()
                all_nns[item[0][0]][item[0][1]] = item[1]
            reconstruction_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan
            #corr_accuracy = np.zeros((kfolds,N,N,lags.shape[0]))*np.nan

            print(f'First round NNS computed in {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            eccm_full_process_start_time = time.time_ns()
            
            """process_outputs = [p.apply_async(indexed_remote_LaggedReconstruction,
                                    args = (([k,n,m,l],np.roll(targets[k][:,:,n],lags[l],axis=0)[:,:], np.roll(lib_targets[k][:,:,m],lags[l],axis=0),all_nns[k][n],test_indices)))
                                    for k in range(kfolds) for l in range(lags.shape[0])
                                    for n in range(N) for m in range(N)]"""
            """process_outputs = tqdm.tqdm([p.apply_async(indexed_SequentialLaggedReconstruction,
                                    args = (([k,i,j],targets[k][:,:,j], lib_targets[k][:,:,j], all_nns[k][i], test_indices, lags)))
                                    for k in range(kfolds)
                                    for i in range(N) for j in range(N)])
            
            for proc in process_outputs:
                item = proc.get()
                reconstruction_accuracy[item[0][0],item[0][1],item[0][2],:] = item[1][1]
                #corr_accuracy[item[0][0],item[0][1],item[0][2],item[0][3]] = item[1][1][1]"""

            process_outputs = tqdm.tqdm([p.apply_async(reconstruction_column,args=((k,n),all_nns[k][n],targets[k],lib_targets[k],test_indices,lags))
                                        for n in range(N) for k in range(kfolds)])

            for proc in process_outputs:
                item = proc.get()
                reconstruction_accuracy[item[0][0],item[0][1],:,:] = item[1]
            
            print(f'Processed reconstructions in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            
            efcf = reconstruction_accuracy
            #eccm = corr_accuracy
            if transform == 'fisher':
                efcf = np.arctanh(efcf)
                #eccm = np.arctanh(eccm)
        if save:
            np.save(save_path + f'eFCFTensorXValidated_dim{dim_max}_delay{delay}.npy',efcf)
            #np.save(path + f'eCCMTensorXValidated_dim{d_max}_delay{delay}.npy',eccm)
            np.save(save_path + f'lags.npy',lags)

        # Average over folds and determine the limiting statistics
        averaged_accuracy = np.nanmean(efcf,axis=0)
        pos_lags = np.argwhere(lags >= 0)
        maxed_fcfs = averaged_accuracy[:,:,pos_lags]
        maxed_fcf_lags = np.nanargmax(maxed_fcfs,axis=2)
        maxed_fcf_lags = np.array([[lags[pos_lags[maxed_fcf_lags[i,j]]] for i in range(N)] for j in range(N)])
        
        if save:
            np.save(save_path+'max_dim_opt_lag.npy',maxed_fcf_lags)
        
        maxed_fcfs = np.nanmax(maxed_fcfs,axis=2)
        downward_strength = np.nansum(maxed_fcfs,axis=0)
        
        efcf_no_diag = efcf
        for i in range(N):
            efcf_no_diag[:,i,i,:] = 0

        fcf_ordered = np.flip(np.argsort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))
        print(f'Hub Rankings: \n channels: {fcf_ordered} \n values:   {np.flip(np.sort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))}')
        unordered_hub_nodes = np.flip(np.argsort(np.nansum(np.nanmax(np.nanmean(efcf_no_diag[:,:,:,(lags >= 0)],axis=0),axis=2),axis=0)))[:hub_nodes_needed]
        
        hub_nodes = np.sort(unordered_hub_nodes)
        print(f'Hubs to be computed on: {hub_nodes}')
        hub_mask = np.isin(np.arange(N),hub_nodes)
        hub_mask = np.repeat(hub_mask[:,np.newaxis],N,axis=1).T
        #print(f'FCFs to compute for hub nodes: {hub_mask}')

        #averaged_corr_accuracy = np.nanmean(eccm,axis=0)
        surrogate_fcfs = None
        surrogates = None

        if os.path.isfile(save_path+'surrogate_fcf.npy'):
            surrogate_fcfs = np.load(save_path+'surrogate_fcf.npy')
            surrogates = np.load(save_path+'surrogates.npy')

        else:
            # Compute the significance intervals
            # Create Surrogate time series: n_surrogate list of NxT time series
            print('Creating twin surrogates')
            
            eccm_full_process_start_time = time.time_ns()
            
            surrogates = parallel_twin_surrogates(lib_targets,N=n_surrogates,Tshift = tShift)
            
            # Create delay vectors for all time series
            surrogate_delays = [[np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim_max)[:,:,np.newaxis], surrogates[n][k].T)),2) for k in range(kfolds)] for n in range(n_surrogates)]
            surrogate_delays = [[np.concatenate((surrogate_delays[n][k],rolled_delay_vectors[k][test_indices,:,:]),0) for k in range(kfolds)] for n in range(n_surrogates)]
            
            print(surrogate_delays[0][0].shape[0]-X.shape[0])
            surr_shape_diff = abs(surrogate_delays[0][0].shape[0]-X.shape[0])
            train_indices_surr = train_indices[:-surr_shape_diff]
            test_indices_surr = test_indices-surr_shape_diff
            print(f'Finished Surrogate Generation in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')
            
            # Current fix is to just not change the test set: (Luca says this is what we should always be doing but we werent for a long while)
            eccm_full_process_start_time = time.time_ns()

            # Generate nearest maps for all surrogates
            surrogate_nns = [[[[]]*N]*kfolds]*n_surrogates
            process_outputs = tqdm.tqdm([ p.apply_async(remote_build_nn_single, 
                                                        args = ((k,hub_nodes[j],n),surrogate_delays[n][k][:,:,hub_nodes[j]],train_indices_surr,test_indices_surr,dim_max+1)) 
                                                        for j in range(hub_nodes_needed) for k in range(kfolds) for n in range(n_surrogates)])
            
            for proc in process_outputs:
                item = proc.get()
                surrogate_nns[item[0][2]][item[0][0]][item[0][1]] = item[1]
            
            print(f'Finished Surrogate CCM Mappings in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')

            eccm_full_process_start_time = time.time_ns()

            # Compute the fcfs
            surrogate_fcfs = np.zeros((n_surrogates,kfolds,N,N,1))
            processes = tqdm.tqdm([p.apply_async(reconstruction_column,
                                    args = ([k,hub_nodes[i],sn],surrogate_nns[sn][k][hub_nodes[i]],targets[k], surrogate_delays[sn][k],test_indices_surr[:],np.array([0])))
                                    for k in range(kfolds) for sn in range(n_surrogates)
                                    for i in range(hub_nodes_needed)])

            for proc in processes:
                item = proc.get()
                surrogate_fcfs[item[0][2],item[0][0],item[0][1],:,0] = item[1][1]

            print(f'Finished Surrogate Reconstructions in: {(time.time_ns()-eccm_full_process_start_time)*10**-9} s')

            if transform == 'fisher':
                surrogate_fcfs = np.arctanh(surrogate_fcfs)

            if save:
                np.save(save_path+'surrogate_fcf.npy',surrogate_fcfs)
                np.save(save_path+'surrogates.npy',surrogates)

        pvals = np.zeros((N,N,lags.shape[0]))*np.nan
        avgd_surrogate_fcfs = np.nanmean(surrogate_fcfs,axis=0)

        if os.path.isfile(save_path+'maxdim_pvalues.npy'):
            pvals = np.load(save_path+'maxdim_pvalues.npy')
            corrected_pval = np.load(save_path+'corrected_pvalue_threshold.npy') # Move this into the output yaml
            significant_pair_lags = np.load(save_path+'maxdim_SignificantPairs.npy')    
        else:

            if normal_pval:
                # TODO: I think it might be better to use a skewed-normal but I am going with this right now
                surrogate_means = np.nanmean(avgd_surrogate_fcfs,axis=0)
                surrogate_stds = np.nanstd(avgd_surrogate_fcfs,axis=0)
                for l in range(lags.shape[0]):
                    for i in range(hub_nodes.shape[0]):
                        for j in range(N):
                            pvals[hub_nodes[i],j,l] = 1-2*np.abs(stats.norm.cdf(averaged_accuracy[hub_nodes[i],j,l],loc=surrogate_means[i,j],scale=surrogate_stds[i,j])-0.5)
            else:
                #TODO
                pvals = 1-2*np.abs(np.array([[[stats.percentileofscore(avgd_surrogate_fcfs[:,i,j],averaged_accuracy[hub_nodes[i],j,k],kind='strict') for k in range(lags.shape[0])] for j in range(N)] for i in range(N)])/100 - .5)

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
        dims_to_search = np.arange(d_min,dim_max)
        searched_dims = np.zeros(num_dims,dtype=bool)
        searched_dims[-1] = True
        incomplete_pairs = np.repeat(((mask.any(axis=2)).any(axis=0))[:,np.newaxis],N,axis=1).T
        print(f'Checking mask sizes: hub_mask={hub_mask.shape}, pvalue mask: {incomplete_pairs.shape}')
        incomplete_pairs &= hub_mask

        if save:
            np.save(save_path+'significant_hub_nodes.npy', np.argwhere(np.any(incomplete_pairs,axis=0))[:,0])


        #print(f'Significant Pairs: {pandas.DataFrame(np.array(incomplete_pairs,dtype=int))}')
        found_nodes = np.zeros((N,N),dtype=bool)

        reconstruction_accuracies = np.zeros((kfolds,num_dims,N,N,lags.shape[0]))*np.nan
        #corr_accuracies = np.zeros((num_dims,N,N,lags.shape[0]))*np.nan
        #corr_accuracies[:,-1,:,:,:] = corr_accuracy
        reconstruction_accuracies[:,-1,:,:,:] = efcf

        # Run the parallel like this:
        # Build the array of eFCF values for the significant pairs and compute the eFCF over each dim
        # With it parallelized for computing both dim = d_min+i and d_max - i, and a middle dim
        # This generates a curve filled out from both sides and the center, find the elbow of this curve over each i and when it has not changed for a few iterations (heuristic choice) 
        # declare that the early stoping criteria
        tested_optimium_dims = np.ones((dim_max-d_min,N,N))*np.nan #np.zeros((dims_to_search.shape[0],N,N))
        tested_optimium_vals = np.ones((dim_max-d_min,N,N))*np.nan
        actual_optimum_dims = np.ones((N,N))*np.nan
        actual_optimum_vals = np.ones((N,N))*np.nan

        opt_num = 0
        
        while dims_to_search.shape[0] > 0:
            if not incomplete_pairs.any():
                break            
            dims = np.array([dims_to_search[0],dims_to_search[dims_to_search.shape[0] // 2],dims_to_search[-1]])
            dims = np.unique(dims)
            dims_to_search = dims_to_search[~np.isin(dims_to_search,dims)]
            searched_dims[dims-d_min] = True

            print(f'Computing CCM for dims: {dims} and {np.argwhere(incomplete_pairs).shape[0]} pairs')
            
            job_num = 0
            to_check = [[]]*N
            for i,j in np.argwhere(incomplete_pairs):
                to_check[i] += [j]
                job_num += 1
            columns = []
            afferents = np.argwhere(incomplete_pairs.any(axis=1))[:,0]

            for d in range(dims.shape[0]):
                print(f"Computing FCFs for dim: {dims[d]}")
                process_outputs = tqdm.tqdm([p.apply_async(FullCCMColumn,args=((afferents[i],k,d),rolled_delay_vectors[k],afferents[i],train_indices,test_indices,lags)) for k in range(kfolds) for i in range(len(afferents))])
                for iK,proc in enumerate(process_outputs):
                    item = proc.get()
                    aff,kfold,d_curr = item[0]
                    curr_fcfs = item[1]
                    if transform == 'fisher':
                        curr_fcfs = np.arctanh(curr_fcfs)
                    reconstruction_accuracies[kfold,d_curr-1,aff,:,:] = curr_fcfs
                

            y = np.array([np.nanmean(reconstruction_accuracies[:,searched_dims,i,j,maxed_fcf_lags[i,j]],axis=0) for i,j in np.argwhere(incomplete_pairs)])[:,0,:]
            #y = np.array([np.nanmax(np.nanmean(reconstruction_accuracies,axis=0)[searched_dims,i,j,:],axis=1) for i,j in np.argwhere(incomplete_pairs)])
            curr_opts = FindFCFOptimums(test_dims[searched_dims],y,fullrange=False)
            curr_pair = 0
            for i,j in np.argwhere(incomplete_pairs):
                tested_optimium_dims[opt_num,i,j] = curr_opts[0][curr_pair]
                tested_optimium_vals[opt_num,i,j] = curr_opts[1][curr_pair]
                curr_pair += 1

            for i,j in np.argwhere(incomplete_pairs):
                check, count_max = repeated_value_early_stop(tested_optimium_dims[:,i,j])
                incomplete_pairs[i,j] = ~check
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
            np.save(save_path+'current_optimum_dimensions.npy',tested_optimium_vals)


        if save:
            np.save(save_path+'almost_all_computed_fcfs.npy',reconstruction_accuracies)
            np.save(save_path+'final_optimum_dimensions.npy',actual_optimum_dims)

        stopped_too_early = np.logical_and(~np.isnan(actual_optimum_dims),np.isnan(actual_optimum_vals))

        last_nns = [[[[[]]*N]*N]*kfolds]
        
        if (np.argwhere(stopped_too_early).shape[0]) > 0:
            print(f'Still need to compute fcf for pairs: {np.argwhere(stopped_too_early)}')
            process_outputs = tqdm.tqdm([p.apply_async(remote_build_nn_single, args = ((i,j,k),rolled_delay_vectors[k][:,:int(actual_optimum_dims[i,j]),i],
                                                                                        train_indices,test_indices,int(actual_optimum_dims[i,j])+1)) 
                                                                                        for k in range(kfolds) for i,j in np.argwhere(stopped_too_early)])

            for ik,out in enumerate(process_outputs):
                vals = out.get()
                i,j,k = vals[0]
                last_nns[k][i][j] = vals[1]
            
            process_outputs = tqdm.tqdm([p.apply_async(indexed_SequentialLaggedReconstruction,args =([k,i,j],targets[k][:,:int(actual_optimum_dims[i,j]),j],lib_targets[k][:,:int(actual_optimum_dims[i,j]),j],last_nns[k][i][j],test_indices,lags))
                                            for k in range(kfolds) for i,j in np.argwhere(stopped_too_early)])
            
            for ik, out in enumerate(process_outputs):
                vals = out.get()
                k,i,j = vals[0]
                curr_fcf = vals[1]
                d_opt = int(actual_optimum_dims[i,j])
                if transform == 'fisher':
                    curr_fcf = np.arctanh(curr_fcf)
                reconstruction_accuracies[k,d_opt-1,i,j,:] = curr_fcf
                actual_optimum_vals[i,j] = np.nanmean(reconstruction_accuracies[:,d_opt-1,i,j,maxed_fcf_lags[i,j]])
            
    all_averaged_accuracies = np.nanmean(reconstruction_accuracies,axis=0)
    
    if save:
        np.save(save_path+'all_computed_fcfs.npy',reconstruction_accuracies)

    total_time = time.time_ns()-time_start

    print(f'Total Time taken: {total_time*(10**-9)} s')
    np.save(save_path+'time_elapsed.npy', np.array([total_time]))

    """noded_pairs = np.array([[i,j,final_optimums[i,j],max_averaged_accuracies[final_optimums[i,j],i,j],np.argmax(all_averaged_accuracies,axis=3)] for i,j in np.argwhere(found_nodes)])

    if save:
        np.save(save_path+'noded_pairs.npy',noded_pairs)"""

    return averaged_accuracy, pvals, efcf, surrogate_fcfs, surrogates, lags, test_indices, train_indices, rolled_delay_vectors

# %%

class TimeSeriesData:
    def __init__(self, X, d = 1, kfolds = 1):
        self.X = X
        
    

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