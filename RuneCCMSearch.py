import DelayEmbedding.DelayEmbedding as DE
import yaml
import RoozbehProject.eCCMdataloader as eCCMdataloader
import numpy as np
import DelayEmbedding.optimizedDE as optDE
from pathlib import Path

initpars = yaml.safe_load(Path('./RoozbehProject/initparams.yaml').read_text())

datapars = eCCMdataloader.LoadNumpyParams('./RoozbehProject/dataparams.yaml')

eccm = optDE.ParallelFullECCM(datapars['TS'],dim_max=datapars['d_max'],d_min=datapars['d_min'],normal_pval=datapars['normal_pval'],
                              save_path=datapars['out_path'],node_ratio=datapars['node_ratio'],lags=np.arange(datapars['low_lag'],datapars['high_lag']+1),
                              kfolds=datapars['kfolds'],n_surrogates=datapars['n_surrogates'],pval_threshold=datapars['pval_threshold'],
                              retain_test_set=datapars['retain_test_set'],early_stop=datapars['early_stop'],min_pairs=datapars['min_pairs'],only_hubs=datapars['only_hubs'])



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