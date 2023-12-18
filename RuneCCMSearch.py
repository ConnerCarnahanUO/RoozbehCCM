import ray
import DelayEmbedding.DelayEmbedding as DE
import yaml
import RoozbehProject.dataloader as dataloader
import numpy as np
import DelayEmbedding.optimizedDE as optDE
from pathlib import Path

initpars = yaml.safe_load(Path('/home/ccarnahan/CCMtests/RoozbehProject/initparams.yaml').read_text())

if initpars['max_memory'] <= 0:
    ray.init(num_cpus = initpars['num_cpus'], num_gpus = initpars['num_gpus'])
else:
    # TODO, possibly limit memory to try to make this horrible package possibly safe
    ray.init(num_cpus = initpars['num_cpus'], num_gpus = initpars['num_gpus'])

datapars = dataloader.LoadNumpyParams('/home/ccarnahan/CCMtests/RoozbehProject/dataparams.yaml')

# For testing
import scipy.io as scio
from pathlib import Path
import data_loader as datl

FIRA = scio.loadmat('/home/ccarnahan/CCMtests/DataSets/Niels/N20191210b.FIRA.resting.ustim.PAG.mat')['FIRA'][0].tolist()
conf = yaml.safe_load(Path('/home/ccarnahan/FCF-master-Amin/example_configs/G20191106cp.yaml').read_text())
g1 = datl.RoozbehLabDataset(pm = conf)
rate_rest,t_rest,out_rest = g1.load_rest(pm=conf)

time_series = rate_rest

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

eccmtest = optDE.ParallelFullECCM(rate_rest,dim_max=30,normal_pval=True,save_path='/home/ccarnahan/CCMtests/testingparallel/roozbeh1/',node_ratio=0.25)

#eccms = DE.ParallelFullECCM(datapars['TS'],datapars['d_min'],datapars['d_max'],datapars['kfolds'],datapars['delay'],
#                           np.arange(datapars['low_lag'],datapars['high_lag']+1,datapars['lag_step']),n_surrogates=datapars['n_surrogates'],save=True,path = '/home/ccarnahan/CCMtests/RoozbehImpossibleProject')

