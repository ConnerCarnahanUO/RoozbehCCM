import yaml
import RoozbehProject.eCCMdataloader as eCCMdataloader
import numpy as np
import DelayEmbedding.optimizedDE as optDE
from multiprocessing import Process, freeze_support
from pathlib import Path

initpars = yaml.safe_load(Path('./RoozbehProject/initparams.yaml').read_text())

datapars = eCCMdataloader.LoadNumpyParams('./RoozbehProject/dataparams.yaml')

if __name__ == '__main__':
    freeze_support()
    eccm = optDE.ParallelFullECCM(datapars['TS'],delay = datapars['delay'],dim_max=datapars['d_max'],d_min=datapars['d_min'],normal_pval=datapars['normal_pval'],
                              save_path=datapars['out_path'],node_ratio=datapars['node_ratio'],lags=np.arange(datapars['low_lag'],datapars['high_lag']+1),
                              kfolds=datapars['kfolds'],compute_pvalue=datapars['compute_pvalue'],n_surrogates=datapars['n_surrogates'],pval_threshold=datapars['pval_threshold'],
                              retain_test_set=datapars['retain_test_set'],early_stop=datapars['early_stop'],min_pairs=datapars['min_pairs'],
                              only_hubs=datapars['only_hubs'], find_optimum_dims=datapars['find_optimum_dims'],max_processes=initpars['num_cpus'])




with open(datapars['out_path']+'data_pars.yaml','w') as outfile:
    savepars = yaml.safe_load(Path('./RoozbehProject/dataparams.yaml').read_text())
    yaml.dump(savepars,outfile,default_flow_style=False)
