# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:40:06 2021

@author: Amin
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import matplotlib as mpl
import numpy as np
import pylab
import DelayEmbedding.DelayEmbedding as DE
from matplotlib.ticker import LinearLocator
from matplotlib import cm

def visualize_matrix(J,pval=None,titlestr='',fontsize=30,save=False,file=None,cmap='coolwarm',vlims=None, elemignores = None):
    """Visualize a matrix using a pre-specified color map
    
    Args:
        J (numpy.ndarray): 2D (x,y) numpy array of the matrix to be visualized
        pval (numpy.ndarray): a binary matrix with the same size as J corresponding
            to the significane of J's elements; significant elements will be 
            shown by a dot in the middle
        titlestr (string): Title of the plot
        fontsize (integer): Fontsize of the plot
        save (bool): Whether to save the plot or not
        file (string): File address to save the plot
        cmap (matplotlib.cmap): Colormap used in the plot
        
    """
    
    plt.figure(figsize=(10,8))
    if vlims is None:
        vlims = (np.nanmin(J),np.nanmax(J))
    current_cmap = mpl.cm.get_cmap(cmap)
    current_cmap.set_bad(color='white')
    #plt.pcolor(J,cmap=current_cmap,vmin=vlims[0],vmax=vlims[1])
    im = plt.imshow(J,cmap=current_cmap,vmin=vlims[0],vmax=vlims[1])
    
    if pval is not None:
        plotval = pval
        if elemignores is not None:
            plotval = np.logical_and(plotval, np.logical_not(elemignores))
        x = np.linspace(0, plotval.shape[0]-1, plotval.shape[0])
        y = np.linspace(0, plotval.shape[1]-1, plotval.shape[1])
        X, Y = np.meshgrid(x, y)
        pylab.scatter(X,Y,s=20*plotval, c='k')
    
    plt.axis('off')
    plt.colorbar(im)
    plt.xlabel('Neurons',fontsize=fontsize)
    plt.ylabel('Neurons',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    
    if save:
        plt.savefig(file+'.eps',format='eps')
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        pylab.show()
        
def visualize_signals(t, signals, labels, spktimes=None, stim=None, t_range=None, stim_t=None, fontsize=30, save=False, file=None, stim_inp = None):
    """Visualize a multidimensional signal in time with spikes and a stimulation
        pattern
    
    Args:
        t (numpy.ndarray): 1D numpy array of the time points in which the 
            signals are sampled
        signals (array): An array of multi-dimensional signals where each 
            element is an NxT numpy array; different elements are shown in 
            different subplots
        labels (array): Array of strings where each string is the label of 
            one of the subplots
        spktimes (array): Array of arrays where each element corresponds to
            the spike times of one channel
        stim (array): Stimulation pattern represented as a binary matrix 
            Nxt_stim where N is the number of channels and t_stim is the timing 
            of stimulation
        t_range ((float,float)): Time range used to limit the x-axis
        t_stim (numpy.ndarray): Time points in which the stimulation pattern
            is sampled
        save (bool): Whether to save the plot or not
        file (string): File address to save the plot

    """

    plt.figure(figsize=(10,signals[0].shape[0]/2))
    
    for i in range(len(signals)):
        c = signals[i]
        plt.subplot(1,len(signals),i+1)
        
        plt.title(labels[i],fontsize=fontsize)
        
        offset = np.append(0.0, np.nanmax(c[0:-1,:],1)-np.nanmin(c[0:-1,:],1))
        s = (c-np.nanmin(c,1)[:,None]+np.cumsum(offset,0)[:,None]).T
        
        plt.plot(t, s)
        plt.yticks(s[0,:],[str(signal_idx) for signal_idx in range(s.shape[1])])
        
        if spktimes is not None:
            for k in range(s.shape[1]):
                plt.scatter(spktimes[i][k],np.nanmax(s[:,k])*np.ones((len(spktimes[i][k]),1)),s=10,marker='|')
                
        if stim is not None:
            for k in range(stim.shape[1]):
                inp = interpolate.interp1d(stim_t,(stim[:,k]).T,kind='nearest',fill_value='extrapolate',bounds_error=False)
                plt.fill_between(t, t*0+np.nanmin(s[:,k]), t*0+np.nanmax(s[:,k]), where=inp(t)>0,
                                 facecolor='red', alpha=0.1)
        
        if stim_inp is not None:
            plt.fill_between(t, t*0+np.nanmin(s[:,k]), t*0+np.nanmax(s[:,k]), where=inp(t)>0,
                            facecolor='red', alpha=0.1)

        if t_range is not None:
            plt.xlim(t_range[0],t_range[1])
            
    if save:
        plt.savefig(file+'.eps',format='eps')
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()



def SurfacePlot(F,n1,n2,opts=None,xlabel='X',ylabel='Y',zlabel='Z'):
    """Plots a matrix as a surface R2->R with a color indicator, opts is a matrix of optimum values that can be displayed"""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    Y = np.arange(2, 2+F.shape[0])
    X = np.arange(2, 2+F.shape[1])
    X, Y = np.meshgrid(X, Y)
    Z =  F[:,:,n1,n2]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(0, 2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'{zlabel} p(neuron {n2}|neuron {n1})')

    if opts is not None:
        ax.text(opts[n1,n2,0]+2,opts[n1,n2,1]+2,opts[n1,n2,2],'X',color='black')
    plt.show()


def DiagonalDistancePartition(FCFMatrix,binsize=1):
    varnum = int(FCFMatrix.shape[0]/binsize)
    binnum = 2*varnum-1
    partition_averages = np.zeros((binnum,3))
    for i in range(varnum):
        k = i+1
        for j in range(FCFMatrix.shape[2]):
            marginal = FCFMatrix[:,:,j]
            partition_averages[i,j] = np.average(marginal.diagonal(offset=k))
            partition_averages[i+varnum-1,j] = np.average(marginal.diagonal(offset=-k))
    return partition_averages

def DistancefromDiagPlot(Matrix,index=0):
    DistSequence = DiagonalDistancePartition(Matrix)
    fig = plt.figure()
    a = plt.axes()
    a.plot(DistSequence[:23,index],label='Super Diagonal')
    a.plot(DistSequence[23:,index],label='Sub Diagonal')
    plt.xlabel('Diagonal Distance')
    plt.ylabel('FCF')
    plt.legend()
    plt.show()

def DelayPlot3D(y,tau):
    """
    plots the 3-delay embedding of a signal with separation tau
    args:
        y is the signal
        tau is the separation
        i is the index of the neuron of interest
    """
    vecs = DE.create_delay_vector(y,tau,3)
    fig = plt.figure(figsize = (14,12))
    a = plt.axes(projection ='3d')
    a.plot3D(vecs[:,0],vecs[:,1],vecs[:,2],linewidth =.5)
    plt.show()

def MultiDelayPlot3D(Y,tau,cmap='plasma',figsize=(14,12)):
    """
    plots multiple 3d delay embeddings
    args:
        Y: TxN set of N time series of time length T > 3*tau
        tau: int representing delay length
        cmap: str for the color map for ploting the different curves
    """
    color_map = cm.get_cmap(cmap)
    colors = color_map(np.linspace(0,1,Y.shape[1]))
    vecs = [DE.create_delay_vector(Y[:,i],tau,3) for i in range(Y.shape[1])]
    fig = plt.figure(figsize = figsize)
    a = plt.axes(projection ='3d')
    for i in range(Y.shape[1]):
        a.plot3D(vecs[i][:,0],vecs[i][:,1],vecs[i][:,2],linewidth =.5,label=f'Variable {i}',color=colors[i])
    plt.legend()
    plt.show()

def ScatterRegress(x,y,order=1,fn = None,average_equals=True):
    xsortargs = np.argsort(x)
    x1 = x[xsortargs]
    y1 = y[xsortargs]
    if average_equals:
        xvals, inverse = np.unique(x1,return_inverse=True)
        yvals = np.zeros(len(xvals))
        for i in range(len(xvals)):
            yvals[i] = np.average(y1[x1==xvals[i]])
        y1 = yvals
        x1 = xvals
    if fn is None:
        A = np.ones(len(x1))
        for i in range(order):
            A = np.vstack([x1**(i+1),A])
        A = A.T
        regress = np.linalg.lstsq(A,y1,rcond = None)
        yhat = A@regress[0]
    return x1, A@regress[0], 1-np.var(yhat-y1)/np.var(y1), regress[0]

def autocorrelate(y, neurons=[0],windowsize = 100, showgraph = True, showcrossgraph = False, maxshowbins = 0, shift = True):
    bin_num = int(y.shape[0]/windowsize)
    x = np.zeros((len(neurons),2*windowsize,bin_num,bin_num))
    time_shift = np.linspace(-windowsize/3,windowsize/3,num=windowsize*2)
    #paddedys = np.zeros((len(neurons)s,y.shape[0]*2))
    
    for j in np.arange(len(neurons)):
        indexes1 = np.arange(windowsize)
        for k in np.arange(bin_num):
            indexes2 = np.arange(windowsize)
            for l in np.arange(bin_num):
                bin1 = np.roll(np.append(y[indexes1,neurons[j]],np.zeros(windowsize)),windowsize>>1)
                bin2 = np.roll(np.append(y[indexes2,neurons[j]],np.zeros(windowsize)),windowsize>>1)
                x[j,:,k,l] = np.correlate(bin1,bin2,mode='same')/np.sqrt(np.sum(bin1**2)*np.sum(bin2**2))
                if shift:
                    x[j,:,k,l] = x[j,:,k,l] - np.average(bin1)*np.average(bin2)/np.sqrt(np.sum(bin1**2)*np.sum(bin2**2))
                indexes2 = indexes2 + windowsize
            indexes1 = indexes1 + windowsize
    
    centermat = np.zeros((len(neurons),2*windowsize,bin_num))
    for j in np.arange(bin_num):
            centermat[:,:,j] = x[:,:,j,j]
                
    
    if showgraph:
        if maxshowbins == 0:
            maxshowbins = bin_num
        for i in np.arange(len(neurons)):
            fig = plt.figure(figsize=(14,12))
            a = plt.axes()
            averageaut = np.average(centermat[i,:,:],axis=1)
            a.plot(time_shift,centermat[i,:,0],color = (0,.05,0.1),alpha=3/bin_num, label = 'individual correlations')
            for k in np.arange(bin_num-1):
                if k < maxshowbins:
                    a.plot(time_shift,centermat[i,:,k+1],color = (0,.05,0.1),alpha=3/bin_num)
            a.plot(time_shift,averageaut,'r-', label = f'Averaged Neuron {neurons[i]}',alpha = 1)
            maxind = np.argmax(averageaut)
            halfmax = np.argmax((averageaut[maxind:] <= np.max(averageaut/2)))
            plt.axvline(time_shift[halfmax+maxind],color = 'k',label = f'Half max at {round(time_shift[halfmax+windowsize],2)} or {round(time_shift[halfmax+windowsize]*3,0)} ticks')
            plt.axvline(time_shift[-halfmax+maxind], color = 'k')
            #print(avgedx)
            if showcrossgraph:
                cross = np.zeros(2*windowsize)
                for j in np.arange(bin_num):
                    for k in np.arange(bin_num):
                        if j != k:
                            cross += x[i,:,j,k]/((bin_num-1)**2)
                a.plot(time_shift,cross,'b--',label = 'Shuffled Threshold')
                a.plot(time_shift,-cross,'b--')
                intercept1 = np.argmax((averageaut[windowsize:]-cross[windowsize:])<0)
                #plt.axvline(x=time_shift[intercept1+windowsize],color = 'k',label = f'Crossing point at dt = {time_shift[intercept1+windowsize]}, or {time_shift[intercept1+windowsize]*3} Ticks')
                #plt.axvline(x=time_shift[windowsize-intercept1],color = 'k')
            std = np.std(np.average(np.abs(centermat[i,:,:]),axis=1))
            #stdindneg = np.argmin()
            plt.axhline(0,color = 'k')
            plt.xlabel('ms')
            plt.ylabel('Corr[y,y](t)')
            plt.legend()
            plt.title(f'Autocorrelation of neuron {neurons[i]} for bin sizes of {round(windowsize/3,0)} ms')
            plt.show()

        return x, time_shift

def PlotCrossedECCM(fcfs,low_lag,indices, plot_error_bands = False, surrogate_hists = None):
    n_kfolds = fcfs.shape[0]
    lags = fcfs.shape[3]

    fig, axs = plt.subplots(ncols=1,nrows=len(indices),figsize=(10,14))
    count = 0
    for i in indices:
        x = np.arange(lags)+low_lag
        axs[count].plot(x,np.average(fcfs[:,i[0],i[1],:],axis=0),label = f'{i[1]} | {i[0]}',color=[0,0,1,1])
        axs[count].plot(x,np.average(fcfs[:,i[1],i[0],:],axis=0),label = f'{i[0]} | {i[1]}',color=[1,0,0,1])
        axs[count].title.set_text(f'crossmap {i}')
        for k in range(n_kfolds):
            if plot_error_bands and surrogate_hists is not None:
                n_samples = surrogate_hists[:,i[0],i[1]].shape[0]
                min_p = np.max([1/n_samples,0.01])
                s_upper10 = np.percentile(surrogate_hists[:,k,i[1],i[0]],(1-min_p)*100)*np.ones(lags)
                s_upper01 = np.percentile(surrogate_hists[:,k,i[0],i[1]],(1-min_p)*100)*np.ones(lags)
                s_lower10 = np.percentile(surrogate_hists[:,k,i[1],i[0]],100*min_p)*np.ones(lags)
                s_lower01 = np.percentile(surrogate_hists[:,k,i[0],i[1]],100*min_p)*np.ones(lags)
                if k == 0:
                    axs[count].fill_between(x,s_upper01,s_lower01,color=[1,0,0,0.05],label=f'Significance band {i[0]} | {i[1]}, p_min = {min_p}')
                    axs[count].fill_between(x,s_upper10,s_lower10,color=[0,0,1,0.05],label=f'Significance band {i[1]} | {i[0]}, p_min = {min_p}')
                else:
                    axs[count].fill_between(x,s_upper01,s_lower01,color=[1,0,0,0.05])
                    axs[count].fill_between(x,s_upper10,s_lower10,color=[0,0,1,0.05])

            #axs[count].axvline(np.argmax(fcfs[k,i[0],i[1],:])+low_lag,color=[0,0,1,.3],linestyle='--',label=f'best lag = {np.argmax(fcfs[k,i[0],i[1],:])+low_lag}')
            axs[count].plot(x, fcfs[k,i[0],i[1],:],color=[0,0,1,0.1])
            #print(np.argmax(fcfs[i[0],i[1],:])+low_lag)
            #axs[count].axvline(np.argmax(fcfs[k,i[1],i[0],:])+low_lag,color=[1,0,0,.3],linestyle='--',label=f'best lag = {np.argmax(fcfs[k,i[1],i[0],:])+low_lag}')
            axs[count].plot(x, fcfs[k,i[1],i[0],:],color=[1,0,0,0.1])
        axs[count].axvline(np.argmax(np.average(fcfs[:,i[0],i[1],:],axis=0))+low_lag,color =[0,0,1,1],
                           label=f'best lag = {np.argmax(np.average(fcfs[:,i[0],i[1],:],axis=0))+low_lag}')
        axs[count].axvline(np.argmax(np.average(fcfs[:,i[1],i[0],:],axis=0))+low_lag,color =[1,0,0,1],
                           label=f'best lag = {np.argmax(np.average(fcfs[:,i[1],i[0],:],axis=0))+low_lag}')
        axs[count].legend()
        count+=1
    plt.xlabel('Lag')
    plt.ylabel('FCF')
    plt.show()

def eFCFMatrixComparison(efcf,efcfpvals,lowlag=0, show_pvals = False, show_lag_maxes = False):
    nvars = efcf.shape[0]
    maxlags = np.argmax(efcf,axis=2)
    efcfmaxes = np.max(efcf,axis=2)
    pvals = np.ones((nvars,nvars))
    for i in range(nvars):
        for j in range(nvars):
            pvals[i,j] = efcfpvals[i,j,maxlags[i,j]]
    if show_pvals:
        visualize_matrix(pvals,titlestr='p-values')
    if show_lag_maxes:
        visualize_matrix(maxlags+lowlag,titlestr='optimum lag',vlims=[lowlag,-lowlag])
    elemignores= (maxlags+lowlag < 0)
    visualize_matrix(efcfmaxes,pval=(pvals <= 0.01),titlestr='Maximum EFCF, p < 0.01 \n Lag < 0 ignored', elemignores=elemignores)

def PlotDelay(TS,d1 = 1, d2 = 2, pointsize = 0.1):
    """PlotDelay(TS, d1=1, d2=2, pointsize = 0.1):
    quickly plots the 3-space delay projection of a time series, TS[i] -> [TS[i],TS[i-d1],TS[i-d2]]"""
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(TS[:-(d1+d2)],TS[d1:-d2],TS[d2:-d1],s=pointsize)
    ax.set_xlabel('Z[t]')
    ax.set_ylabel(f'Z[t-{d1}]')
    ax.set_zlabel(f'Z[t-{d2}]')

    plt.show()


def ScanningWindowCorrelation(TS1, TS2, min_size_ratio = 0.01, max_size_ratio = 0.5):
    """ Based on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6513676/  doi: 10.1016/j.neuroimage.2019.02.001"""
    L1 = TS1.shape
    L2 = TS2.shape
    L = np.min([L1,L2])
    maxSize = int(L*max_size_ratio)
    minSize = int(L*min_size_ratio)



