#!/usr/bin/env python

"""


"""

import numpy as np
from glob import glob
from scipy.optimize import minimize, differential_evolution
import sys
from costfun import *


if __name__ == '__main__':

    # get command-line args -> data path and seed
    tmp_path = sys.argv[1]
    seed = int(sys.argv[2])

    initfile = np.load(tmp_path, allow_pickle=True)

    initdict = {key: value for key, value in initfile.items()}

    savepath = str(initdict['save_path'])
    savename_prefix = str(initdict['savename_prefix'])

    xcrit_vals = initdict['xcrit']

    x = initdict['x']
    y = initdict['y']

    x, y = pad_vals(x, y)

    sizelist = initdict['sizelist']

    popsize0 = int(initdict['popsize0'])
    maxiter0 = int(initdict['maxiter0'])
    bounds = initdict['bounds']
    
    critical_point = str(initdict['critical_point_model'])
    rescaling_function = str(initdict['rescaling_function'])
    # optimization

    optRes = differential_evolution(minimization_fun,
                                    bounds,
                                    args=(x, y, sizelist,
                                          critical_point,
                                          rescaling_function,
                                          False),
                                    popsize=popsize0,
                                    maxiter=maxiter0,
                                    workers=int(initdict['workers']),
                                    seed=seed)

    # save the results if minimization terminated successfully

    if optRes.success:
        # auxiliary data
        savefile = {
            'params': optRes.x,
            'costfun_value': optRes.fun,
            'nfev': optRes.nfev,
            'nit': optRes.nit,
            'critical_point': critical_point,
            'rescaling_function': rescaling_function,
            'popsize0': popsize0,
            'maxiter0': maxiter0,
            'bounds': bounds,
            'xcrit': xcrit_vals,
        }
        print('Saving results to disk!')
        np.savez((f'{savepath}/{savename_prefix}_'
                  f'rescale_{critical_point}'
                  f'_{rescaling_function}_{seed}.npz'), **savefile)
