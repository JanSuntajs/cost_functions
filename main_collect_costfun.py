#!/usr/bin/env python

"""
A module with function(s) to collect
the main results after the differential
evolution jobs have finished for different
seeds. The post_main(...) function collects
the jobs and takes the average of the parameter
values for the minimization outcomes with the
smallest cost function value. The data are
presented in the .txt format which is the
most convenient for subsequent plotting and
quick inspection.

"""

import numpy as np
from glob import glob
import sys
import os


_special_keys = ['params', 'costfun_value', 'nfev', 'nit', 'bounds']
# path to the costfun data
_save_keys = ['savename_prefix', 'critical_point_model',
              'rescaling_function', 'preprocess_xvals',
              'save_path']


def post_main(costfun_path, savedir,
              savename,
              special_keys=_special_keys,
              eps=1e-08):
    """

    """

    costfun_files = glob(costfun_path)
    # load the npz files into memory
    data_objects = [np.load(file, allow_pickle=True)
                    for file in costfun_files]
    nsamples = len(data_objects)

    # prepare/format the data for storing in an external file
    # -> for each run, we list the cost function first, then the
    # parameters
    storevals = np.array([
        np.append(data_object['costfun_value'], data_object['params'])
        for data_object in data_objects])

    # sort the values according to the lowest value of the cost function
    storevals = storevals[storevals[:, 0].argsort()]
    params = storevals[:, 1:]
    costfun_vals = storevals[:, 0]

    # load original & processed data
    orig_path = os.path.split(costfun_path)[0]
    orig_data = f'{orig_path}/orig*'
    orig_data = glob(orig_data)[0]
    orig_data = np.load(orig_data, allow_pickle=True)
    proc_data = f'{orig_path}/proc*'
    proc_data = glob(proc_data)[0]
    proc_data = np.load(proc_data, allow_pickle=True)

    # number of function evaluations and numbers of iterations
    nfev = np.array([data_object['nfev'] for data_object in data_objects])
    nit = np.array([data_object['nit'] for data_object in data_objects])

    # find the minimum costfun, then select the parameters that are
    # within the eps of the minimum costfun
    min_costfun = costfun_vals[0]
    condition = np.abs(costfun_vals - min_costfun) <= eps
    opt_params = np.mean(params[condition], axis=0)
    costfun_val = np.mean(costfun_vals[condition], axis=0)

    # --------------------txt file saving---------------------
    #
    #             prepare txt file for reading
    #
    # --------------------------------------------------------

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    savefile = f'{savedir}/{savename}.txt'
    with open(savefile, 'w') as file:
        file.write(" ".join(map(str, opt_params)))
        file.write(f'\n{costfun_val} \n')

        file.write(f'\n# 1st row: optimization parameters.\n')
        file.write(f'# 2nd row: cost function value.\n')
        file.write(f'# Additional procedure details: \n \n')

        for key in data_objects[0].files:

            if key not in special_keys:
                value = data_objects[0][key]
                file.write(f'# {key}: {value} \n')

        file.write(f'# bounds: \n')
        for bound in data_objects[0]['bounds']:
            file.write(f'# \t{bound} \n')

        file.write(f'# number of samples: {nsamples}\n')

        mean_nit = int(np.mean(nit, axis=0))
        std_nit = int(np.std(nit, axis=0))
        file.write(f'# nit_mean: {mean_nit:d}, nit_std: {std_nit:d} \n')

        mean_nfev = int(np.mean(nfev, axis=0))
        std_nfev = int(np.std(nfev, axis=0))
        file.write(f'# nfev_mean: {mean_nfev:d}, nfev_std: {std_nfev:d} \n')

    savefile_2 = f'{savedir}/{savename}_all_params.txt'

    np.savetxt(savefile_2, storevals)
    return (costfun_val, opt_params, orig_data,
            proc_data, storevals, data_objects)


if __name__ == '__main__':
    # get command-line args -> data popsize0ath and seed
    tmp_path = sys.argv[1]

    initfile = np.load(tmp_path, allow_pickle=True)

    # load into a dictionary
    initdict = {key: value for key, value in initfile.items()}
    # print(initdict)
    loadpath = str(initdict['save_path'])
    loadname_prefix = str(initdict['savename_prefix'])

    savepath = f'{loadpath}/txtfiles/'
    savename_prefix = loadname_prefix
    savename = ('{0}_{4}_crit_point_scaling_'
                '{1}_rescale_fun_'
                '{2}_prep_xvals_{3}').format(*[str(initdict[key])
                                               for key in _save_keys[:-1]],
                                             os.path.split(
                    str(initdict[_save_keys[-1]]))[1])
    print(savename)
    (costfun_val, opt_params, orig_data,
     proc_data, storevals, data_objects) = post_main(
        f'{loadpath}/{loadname_prefix}*', savepath,
        savename)
