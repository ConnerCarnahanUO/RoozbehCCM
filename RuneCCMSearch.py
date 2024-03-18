import yaml
import RoozbehProject.eCCMdataloader as eCCMdataloader
import numpy as np
import DelayEmbedding.optimizedDE as optDE
from pathlib import Path

initpars = yaml.safe_load(Path('./RoozbehProject/initparams.yaml').read_text())

datapars = eCCMdataloader.LoadNumpyParams('./RoozbehProject/dataparams.yaml')

eccm = optDE.ParallelFullECCM(datapars['TS'],delay = datapars['delay'],dim_max=datapars['d_max'],d_min=datapars['d_min'],normal_pval=datapars['normal_pval'],
                              save_path=datapars['out_path'],node_ratio=datapars['node_ratio'],lags=np.arange(datapars['low_lag'],datapars['high_lag']+1),
                              kfolds=datapars['kfolds'],compute_pvalue=datapars['compute_pvalue'],n_surrogates=datapars['n_surrogates'],pval_threshold=datapars['pval_threshold'],
                              retain_test_set=datapars['retain_test_set'],early_stop=datapars['early_stop'],min_pairs=datapars['min_pairs'],
                              only_hubs=datapars['only_hubs'], find_optimum_dims=datapars['find_optimum_dims'],max_processes=initpars['num_cpus'])


with open(datapars['out_path']+'data_pars.yaml','w') as outfile:
    savepars = yaml.safe_load(Path('./RoozbehProject/dataparams.yaml').read_text())
    yaml.dump(savepars,outfile,default_flow_style=False)

"""# For testing
import scipy.io as scio
from pathlib import Path
import data_loader as datl

FIRA = scio.loadmat('/home/ccarnahan/CCMtests/DataSets/Niels/N20191210b.FIRA.resting.ustim.PAG.mat')['FIRA'][0].tolist()
conf = yaml.safe_load(Path('/home/ccarnahan/FCF-master-Amin/example_configs/G20191106cp.yaml').read_text())
g1 = datl.RoozbehLabDataset(pm = conf)
rate_rest,t_rest,out_rest = g1.load_rest(pm=conf)

time_series = rate_rest

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) """