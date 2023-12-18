import numpy as np
import yaml
from pathlib import Path

def LoadNumpyParams(file):
    pars = yaml.safe_load(Path(file).read_text())
    TS = np.load(pars['path'])
    pars['TS'] = TS

    return pars