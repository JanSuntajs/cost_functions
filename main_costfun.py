#!/usr/bin/env python

"""


"""

import numpy as np
from glob import glob
from scipy.optimize import minimize, differential_evolution
import sys
from functions.costfun import *


if __name__ == '__main__':

    # get command-line args -> data popsize0ath and seed
    tmp_path = sys.argv[1]
    seed = int(sys.argv[2])

    initfile = np.load(tmp_path, allow_pickle=True)

    initdict = {key: value for key, value in initfile.items()}

    savepath = str(initdict['save_path'])
    savename_prefix = str(initdict['savename_prefix'])

    xcrit_vals = initdict['xcrit']

    x = initdict['x_prep']
    y = initdict['y']

    x, y = pad_vals(x, y)
    # count the number of points
    sizelist = initdict['sizelist']

    popsize0 = int(initdict['popsize0'])
    maxiter0 = int(initdict['maxiter0'])
    bounds = initdict['bounds']

    critical_point = str(initdict['critical_point_model'])
    rescaling_function = str(initdict['rescaling_function'])
    crit_operation = str(initdict['critical_operation'])
    x_prep_operation = str(initdict['preprocess_xvals'])
    x_prep_prefactor = str(initdict['preprocess_xvals_prefactor'])
    # optimization

    optRes = differential_evolution(minimization_fun,
                                    bounds,
                                    args=(x, y, sizelist,
                                          critical_point,
                                          rescaling_function,
                                          crit_operation,
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
            'npoints': int(initdict['npoints']),
            'critical_point': critical_point,
            'rescaling_function': rescaling_function,
            'critical_point_operation': crit_operation,
            'popsize0': popsize0,
            'maxiter0': maxiter0,
            'bounds': bounds,
            'xcrit': xcrit_vals,
            'x_preprocessing_operation': x_prep_operation,
            'x_preprocessing_prefactor': x_prep_prefactor,
        }
        print('Saving results to disk!')
        np.savez((f'{savepath}/{savename_prefix}_'
                  f'rescale_{critical_point}'
                  f'_{crit_operation}'
                  f'_{rescaling_function}'
                  f'_preprocess_{x_prep_operation}'
                  f'_prefactor_{x_prep_prefactor}'
                  f'_{seed}.npz'), **savefile)
