import numpy as np
import DelayEmbedding.DelayEmbedding as DE


import ray

ray.init(num_cpus = 64)

TSgen = np.random.normal(size = (10000,100))

eccm = DE.ParallelFullECCM(TSgen,d_max=20,max_mem=0)

ray.shutdown()