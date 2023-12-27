import numpy as np
import yaml
import data_loader as datl
import scipy.io as scio
from pathlib import Path

def LoadNumpyParams(file):
    pars = yaml.safe_load(Path(file).read_text())
    TS = None

    if pars['data_type'] == 'Numpy':
        TS = np.load(pars['path'])
    if pars['data_type'] == "FIRA":
        FIRA = scio.loadmat(pars['path'])['FIRA'][0].tolist()
        conf = yaml.safe_load(Path(pars['config_path']).read_text())
        g1 = datl.RoozbehLabDataset(pm = conf)
        pars['FIRA_Load'] = g1
        rate_rest,t_rest,out_rest = g1.load_rest(pm=conf)
        TS = rate_rest
    
    pars['TS'] = TS

    return pars