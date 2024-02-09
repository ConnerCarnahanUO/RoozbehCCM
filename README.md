# CCM
Methods for simulating neural data and computing causality and prediction of time series data based on convergent cross mapping.

# Installation
```
git clone https://github.com/amin-nejat/CCM.git
cd CCM

conda install --file requirements.txt 
OR 
pip install -r requirements.txt

pip install -e .
```
This will install "from source".

# ModernCCM

## RuneCCMSearch.py

This file is designed to determine the hub channels and significant connections of a set of Time Series.

### Workflow

Step 1:

The program takes an input file and converts it into a python array that extended Functional Causal Flow (eFCF)  will actually be calculated on. How this is done is defined in the "data_loader.py" file, and in the loading param file (RoozbehProject/dataparams.yaml) it is currently set to get a FIRA file as the input. (TODO: maybe add some more ways to do this).

Step 2a:

The program will compute the best delay that should be used for the total delay embedding, this will be the fastest part of the program but it's significance to the data isn't really known (to us at least) right now.

Step 2b:

The eFCF tensor is computed on the input data set for a (semi-)arbitrary delay dimension. This give an approximation of the "optimum" set of values. The tentitive hub nodes are then found by finding the sum of eFCF values over the set of afferents (hubness). There will be a print statement that gives the ranking of each channel. In an experimental setup, to a good approximation, these will be the channels that should be stimulated for a maximum effect over all other channels and if you want to start working with the stimulation of the monkey neurons as quickly as possible these are what you will want to use.

Step 2c:

The significance of each connection will be computed for a fraction of the channels (defined in "dataparams.yaml") which have the greatest hubness using the twin surrogate procedure. When this is finished computing it will tell you which connections between channels and hubnodes should experience an effect. This will be a very long step with the current implementation, but will probably get better over the time as I update the implementation.

Step 3 (Dimension Search):

This will also take forever. Each connection has an "optimum dimension" that we believe is the best dimension to compute the eFCF for. As we do not have a complete theory for this data set we basically just have to search over all dimensions until we find what we think is best. So this really is just a really long parameter search that you should let run while you are working with the monkey.

Done.

### Parameter Files

There are two parameter files that define how the computation is done "initparams.yaml" and "dataparams.yaml". They are stored in "yaml" files since they are pretty common and have the most natural syntax (to me at least).

1. "initparams.yaml"
num_cpus , int: the max number of processes that the program will take up during the computation, this should always be less than the number of cpu cores that the computer being used has so that you don't get any terrible problems when it has run out of available cpus.
num_gpus , int: Currently unused but will define how many available gpus there are when we I implement gpu usage
max_memory , int: Currently does nothing but was used for bug finding when everything was constantly crashing

2. "dataparams.yaml"
data_type , string: Tells the program how the data should be loaded based on the file type, currently only supports "FIRA" and "numpy"
path , string: The file path where the data currently is
config_path , string: Where the configuration file is for data loading. Depending on the data type there are parameters that are required to bin data and make sure everything is loaded properly. I will have an explanation of all the required fields for FIRA files but if you look at the example configs file it should be fairly explanitory.

d_min , int: the minimum dimension used for the parameter search

d_max , int: The dimension that will first be computed and is the maximum for the parameter search.

delay , int: The delay used for embedding, if set to 0 it will find the best one using a Mutual Information method (maybe I'll explain this later but it's currently not necessary)

low_lag , int: The minimum lag of causality to be computed for each pair, this should always be <= 0

high_lag , int: The maximum lag of causality to be computed for each pair, this should always be >= 0

lag_step , int: The number of steps over the lags that will be computed, this is useful if the data is very slowly varying over time since many lags will be redundant to compute

In general you don't want to compute that many lags, and for our data you should keep them pretty close to 0, but if you go out passed +- 10 then just adjust the step size so that fewer lags are computed on.

kfolds , int: The number of cross validation sections that will be computed on. This should be >= 2, with 5 being what I recommend. This is required for noisy data since the noise generaly gets averaged out.

n_surrogates , int: The number of surrogates generated for the twin surrogate significance test
node_ratio , float: The ratio of channels that will be considered for hub compuations, this should be 0 <= node_ratio <= 1. With what I have currently computed on previous datasets it looks like about 20-26% of connections are significant with pval <= 0.01 so for speed considerations it makes sense to have this ratio around .2

retain_test_set , bool: Whether we should make sure the test set of the eFCF computation is maintained in the twin surrogate test, this is so the data is not corrupted by the fact we have messed up our test set / switched information between partitions.

normal_pval , bool: Whether we should assume that the surrogate eFCFs are normally distributed. This is very useful for keeping the computation length down since we can use far fewer surrogates to approximate how significant each connection is.

pval_threshold , float: The maximum p-value we should consider when determining whether a connection is significant. If any connection is > this value we will ignore it for the parameter search.

min_pairs , int: The minimum number of pairs we want to include in the parameter search. This overrides the pval_threshold and will sucessively increase the threshold until we satisfy significant pairs > min_pairs. Basically this is a check so that we didn't use too low of a threshold and the new threshold will be saved.

early_stop , bool: Whether in the dimension parameter search we should remove a computation when we obtain the same optimum repeatedly when computing the eFCF in the parameter search (this will reduce computation time)