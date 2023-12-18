import numpy as np
import scipy as sci

def GeneralizedLorenz(time_length,X0 = [],M=3,H=1,nu=1,r=28,PrandltNumber=10,b=8/3,timescale = 1,ilims=[-1,1],time_step = 0):
    """ Simulates a GLS as defined in https://doi.org/10.1142/S0218127419500378
        time_length (int): number of steps in simulation
        M (odd int >= 5): number of modes of system

    """
    y0 = X0
    N = int((M-3)/2)
    ## setting up dj and betaj sequences
    dj = np.arange(N)
    dj = dj + 1
    j = np.arange(N)
    betaj = ((dj)**2)*b
    dj = (2*dj+1)**2
    dj = dj + 1/2
    dj = dj/(3/2)
    kappa= PrandltNumber*nu
    b = 4/(3/2)
    timescale *= kappa*(3/2)*(np.pi/H)
    if len(y0) != M:
        y0 = np.random.uniform(ilims[0],ilims[1],M)
    TimeSeries = np.zeros((time_length,M))
    TimeSeries[0,:] = y0
    #Compute actual differential, refactored for possible speed increase
    def differential(t,X):
        dX = np.zeros(X.shape[0])
        dX[0] = PrandltNumber*(-X[0]+X[1])
        dX[1] = -X[0]*X[2]+r*X[0]-X[1]
        dX[2] = X[0]*X[1]-b*X[2]
        if N > 0:
            dX[2] -= (j[0]+1)*X[0]*X[3]
        dX[3::2] = X[0]*(j*X[2:-2:2]-(j+1)*X[4::2])-dj*X[3::2] #dY_j
        dX[4::2] += -(j+1)*X[0]*X[3::2] #dZ_j
        dX[4::2] += (j+1)*X[3::2]-betaj*X[4::2] #dZ_j
        return dX
    t = None
    if time_step > 0:
        t = np.linspace(0,time_length, int(time_length/time_step))
    sol = sci.integrate.solve_ivp(differential,y0=y0,t_span=[0,time_length],t_eval = t)
    return (sol.t)*timescale, sol.y.T

def GeneralizedRossler(time_length, X0 = [], dimension = 3, a_factor = 0.2, b = 0.2, c = 5.7, d=1, step_size = 0, stimulation_pattern = [],res_freq=1.5):
    """ Generates a generalized Rossler Time series based on this paper: 
        https://journals.aps.org/pre/pdf/10.1103/PhysRevE.56.5069
        The main reasons I want to look at this one as well: GLorenz does not seem to have an increase in dimension, 
        this system is a known hyperchaotic system (has multiple positive lyuponov exponents), 
        and it is simpler to compute"""
    Amatrix = np.zeros((dimension,dimension))
    
    Amatrix[0,0] = a_factor

    if len(X0) != dimension:
        X0 = np.random.uniform(-3,3,dimension)
    
    for i in range(dimension-2):
        Amatrix[i+1,i] = 1
        Amatrix[i,i+1] = -1

    inp = StimPatternToFunc(stimulation_pattern=stimulation_pattern,dimension=dimension,Timelength=time_length,frequency=res_freq)

    def differential(t,X):
        dX = np.zeros(len(X))
        dX += inp(t)
        dX += np.matmul(Amatrix,X)
        dX[-2] -= X[-1]
        dX[-1] += b + d*X[-1]*(X[-2]-c)
        return dX
    if step_size > 0:
        t = np.linspace(0,time_length, int(time_length/step_size))
    sol = sci.integrate.solve_ivp(differential,y0=X0,t_span=[0,time_length],t_eval = t)
    return sol.t, sol.y.T

def RandomStimPattern(nodes,max_time_length, max_num_stims, stim_start=1000, stim_length = 20, stim_rest = 50, stim_chance = 2, amp_min=1,amp_max=4, stim_stop = 2000):
    pattern = np.zeros((max_num_stims,4)) # array with a[i,:] = [stimulated_node,time_start, time_length]
    i = 0
    t = stim_start
    nodes_ = nodes
    stimmed_node = 0
    stim_num = 0
    while i < max_num_stims and t < max_time_length:
        t+=stim_rest+2*stim_length
        if np.random.randint(0,stim_chance) == 0:
            if(len(nodes_) == 0):
                nodes_ = nodes
            j = nodes[stimmed_node] #not sure about this but luca thinks it's what to do
            stimmed_node += 1
            stimmed_node %= len(nodes)
            t0 = t
            t += stim_length
            pattern[i] = [j,t0,t,np.random.uniform(amp_min,amp_max)]
            t += stim_rest
            i += 1
            stim_num +=1
    return pattern, stim_num

def StimPatternToFunc(stimulation_pattern,dimension,Timelength,frequency = 0):
    stim_matrix = np.zeros((dimension,Timelength))
    if len(stimulation_pattern) > 0:
        stim_matrix = np.zeros((dimension,Timelength))
        for i in range(stimulation_pattern.shape[0]):
            stim_matrix[int(stimulation_pattern[i,0]),int(stimulation_pattern[i,1]):int(stimulation_pattern[i,2])] += stimulation_pattern[i,3]
    inp = sci.interpolate.interp1d(np.arange(Timelength),stim_matrix,kind='nearest',bounds_error=False)
    def inpfunc(t):
        return inp(t)*np.cos(frequency*t)
    return inpfunc

def ParticipationRatio(X):
    cov_mat = np.cov(X.T)
    return np.trace(cov_mat)**2/np.trace(cov_mat@cov_mat)

def RosslerInfluencedByRosslers(Timelength, X0 = [],a_factor = 0.2, b = 0.2, c = 5.7, d=1, ExternalTimeSeries=[], DriveMatrix=[]):
    Amatrix = np.zeros((3,3))
    Amatrix[0,0] = a_factor
    Amatrix[1,0] = 1
    Amatrix[0,1] = -1
    if len(X0) != 3:
        X0 = np.random.uniform(-3,3,3)

    if len(ExternalTimeSeries) == 0:
        print('ERROR: External time series should be non_empty, Use the general rossler simulator instead')

    externaldim = 0
    for i in range(len(ExternalTimeSeries)):
        externaldim += ExternalTimeSeries[i].shape[1]

    if len(DriveMatrix) == 0:
        v = np.random.normal(0,1,)

    def differential(t,X):
        dX = np.zeros(len(X))
        dX += np.matmul(Amatrix,X)
        dX[-2] -= X[-1]
        dX[-1] += b + d*X[-1]*(X[-2]-c)
        return dX
