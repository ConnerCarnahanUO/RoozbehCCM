import yaml
import numpy as np
import DelayEmbedding.optimizedDErefactor2 as optDE
from pathlib import Path
import visualizations as vis
import RoozbehDataLoader as rdatl
import os
import sys
import argparse

parse = argparse.ArgumentParser(description='Compute ECCM on an Input FIRA Dataset')
parse.add_argument('FIRA',metavar='p',type=str,help='filepath to FIRA Dataset')
parse.add_argument('-output',metavar='o',type=str,default='./outputs/',help='filepath to send ECCM Results to, default is ./outputs/')
parse.add_argument('--bad_channels',metavar='bc',type=int,nargs='+',help='List of bad channels to ignore during computations IN MATLAB INDEXING')
parse.add_argument('-filetype',metavar='bc',type=str,default='m.v7.3',help='Indicate the filetype if matlab v7 use m.v7, default assumes matlab v7.3')

args = parse.parse_args()
sessionname = os.path.split(args.FIRA)[1].split('.')[0]

if os.path.isfile('./badchannels.yaml'):
    bad_channel_yaml = yaml.safe_load(Path('./badchannels.yaml').read_text())
else:
    bad_channel_yaml = {}
if args.bad_channels is not None:
    bad_channel_yaml[sessionname] = args.bad_channels
    with open('./badchannels.yaml','w') as outfile:
            yaml.dump(bad_channel_yaml,outfile,default_flow_style=False)
    bad_channels_npy = np.array(args.bad_channels)-1
else:
    bad_channels_npy = None

outpath = args.output+f'{sessionname}/'
print(outpath)

if not os.path.isfile(args.FIRA):
    print(f'Could not find the fira file: {args.FIRA}, exiting early.')
    sys.exit()

if not os.path.isdir(outpath):
    os.makedirs(outpath)

conf = yaml.safe_load(Path('./StandardECCMConfig.yaml').read_text())
conf['FIRA_path'] = args.FIRA
if args.bad_channels is not None:
    conf['bad_channels'] = args.bad_channels

data = rdatl.RoozbehDataset(conf,savelocation=outpath,old_data=(args.filetype == 'm.v7'))
np.save(outpath+'/datetime.npy',np.array(str(data.data_dict['datetime'])))
if int(conf['remove_bad_spikes']) == 1:
    rate_rest,t_rest = rdatl.RoozbehDataset.SpikeTimesToRates(data.data_dict['spikes'],pm=conf['fs'],filter='exponential')
else:
    rate_rest,t_rest = rdatl.RoozbehDataset.SpikeTimesToRates(data.data_dict['spikes'],binwidth=conf['fs'],pm=conf,bad_spike_interval=None,bad_spike_ratio=None,filter='exponential')
np.save(f'{outpath}/spike_counts.npy',rate_rest)
np.save(f'{outpath}/time.npy',t_rest)
if args.bad_channels is not None:
    np.save(f'{outpath}/bad_channels.npy',args.bad_channels)
lags = np.array(conf['lags'])
d_min = conf['d_min']
d_max = conf['d_max']
tau=conf['delay']
lagstep=conf['lag_step']
pvalue=conf['compute_pvalue']
kfolds=conf['kfolds']
n_surrs=conf['n_surrogates']
max_processes = conf['max_processes']
projection = conf['random_projection']

optDE.DimensionParameterSearchLargeDatasetOpt2(rate_rest,dim_min=d_min,dim_max=d_max,delay=tau,kfolds=kfolds,lags=lags,only_hubs=True,find_optimum_dims=False,lagstep=lagstep,
                               retain_test_set=True,compute_pvalue=pvalue,normal_pval=True,save_surrogates=True,save_path=outpath,n_surrogates=n_surrs,max_processes=max_processes,random_projection=projection,
                               bad_channels=bad_channels_npy)
