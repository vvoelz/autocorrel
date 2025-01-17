# Tools and helper function for autocorrelation functions
# NOTE: requires python3

import numpy as np

def autocorr(x):
    tau = x.size
    mu = x.mean()
    var = x.var()  # Variance for normalization
    g = np.correlate((x-mu), (x-mu), mode='full')[tau-1:]
    g = g / var
    n = np.arange(tau,0,-1)
    return g/n



###########################


import numpy as np
# from multiprocessing import Pool


def multi_autocorr(x_list, max_tau=None):
    """Given a list of 1D trajectories x (each could be different lengths), compute an autocorrelation.
    INPUTS
    x_list  - a list of np.array objects. Each could be different lengths .

    PARAMETERS
    max_tau - the maximum frame to compute the autocorrelation.  If None, use the whole range. Default: None.
    """


    from tqdm import tqdm

    ntraj = len(x_list)
    x = np.concatenate(x_list)
    mu = x.mean(axis=0)
    var = x.var(axis=0)

    max_trajlength = max( [ x_list[i].shape[0] for i in range(ntraj) ])
    
    if max_tau == None:
        max_tau = max_trajlength
        
    elif max_tau > max_trajlength:
        print("Error: max_tau must be less or equal to the maximum trajectory length. Setting max_tau = max_trajlength") 
        max_tau = max_trajlength
        
    result = np.zeros( max_tau )
    n = np.zeros( (max_tau,) )
        
    for i in tqdm(range(ntraj)):

        this_tau = x_list[i].shape[0]
        print(f'Analyzing traj {i} of {ntraj} ({this_tau} frames)...')
        if this_tau > max_tau:
            ### NOTE:   np.correlate(...., mode='full') returns a 2N-1 length cross-correl of all possible shifts
            T = 2*this_tau-1
            result[0:max_tau] += np.correlate((x_list[i] - mu), (x_list[i] - mu), mode='full')[this_tau-1:this_tau-1+max_tau]
            n[0:max_tau] += np.arange(this_tau,0,-1)[0:max_tau]
        else:
            result[0:this_tau] += np.correlate((x_list[i] - mu), (x_list[i] - mu), mode='full')[this_tau-1:]
            n[0:this_tau] += np.arange(this_tau,0,-1)
                
    result = result / var
    return result/n



                                                       
