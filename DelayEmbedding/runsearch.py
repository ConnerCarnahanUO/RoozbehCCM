import warnings
warnings.filterwarnings('ignore')
import datetime as date
from tracemalloc import start
from turtle import st
from DelayEmbedding import DelayEmbedding as DE
from Causality import Causality as RA
from Simulation.Simulator import Simulator
import visualizations as V
import numpy as np

# %% Resting State

parameters = {}
parameters['T'] = 4000 # Duration of stimulation
parameters['alpha'] = .2 # Parameter for rossler
parameters['beta'] = .2 # Parameter for rossler
parameters['gamma'] = 5.7 # Parameter for rossler
parameters['bernoulli_p'] = .8 # Downstream connectivity probability
parameters['g_i'] = .1 # Input connectivity strength
parameters['g_r'] = 3. # Recurrent connectivitys strength
parameters['lambda'] = 1. # Parameter for recurrent dynamics
parameters['N'] = 100 # Number of downstream neurons


t,y,J = Simulator.rossler_downstream(parameters)
recorded = np.arange(10)

X = y[:,recorded]
params = {}
params["dims"] = np.arange(39)+1
params["scales"] = np.linspace(-10,10)
params["delays"] = np.arange(20)+1

FCFs = DE.connectivity_parameter_search(X,params = params)

np.save(f"FCF_search_{date.date.today()}",FCFs)