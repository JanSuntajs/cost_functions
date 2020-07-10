"""
Define the actual function that prepares
the arguments and organizes the folder
structure for the main function in which
quantities of interest are calculated.

"""

import numpy as np
import subprocess as sp
import os
from glob import glob


def main(data_path, savepath, xcrit, xcol, ycol,
         popsize0, maxiter0, workers, sizelist,
         critical_point_model, rescaling_function,
         critical_operation, bounds, nsamples,
         savename_prefix, queue=False,
         time='00:00:00',
         ntasks=1,
         cputasks=1,
         memcpu=0,
         ):
    """
    A function for preparing the temporary file and
    running the main script, either sequentially or
    in a parallel manner on a SLURM-based cluster.

    Parameters:
    -----------

    data_path: string
               A path to where the data to be analysed
               are stored.
    savepath: string
              A path to the directory where the data should
              be stored. If the directory does not yet
              exist, it is created.
    xcrit: 1D ndarray or array-like
           An array-like object of len(xcrit)==2; contains
           the limits of the x-range interval in which the
           scaling operation should be performed.

    xcol, ycol: int
                Integers specifying which columns of the actual
                numerical data describe the x and y values,
                respectively.

    popsize0, maxiter0, workers: int, parameters for the
                                 scipy.optimize.differential_evolution()
                                 function.
                                 popsize0: size of the population from
                                 which the random samples for the optimal
                                 parameter finding are drawn.
                                 maxiter0: the number of iterations of the
                                 algorithm
                                 workers: the optimization routine allows
                                 for parallelization in which case the
                                 'workers' parameter specifies the number
                                 of parallel threads used.
    sizelist: 1D array or array-like
              An array with system sizes used in the analysis.

    critical_point_model: string
                          What kind of model do we predict for the scaling
                          of the supposed critical point. See
                          functions.crit_functions for the list of available
                          critical point models. Among possible entries would
                          be:
                          'free', 'lin', 'poly', 'inv', 'log' ...

    rescaling_function: string
                          What kind of functional dependence is used in the
                          actual rescaling of the independent variable values.
                          Among the currently available are:
                          'id' (identity), 'pl' (power-law),
                          'kt' (kosterlitz-thouless)
                          ...

    critical_operation: string
                          Whether to subtract the supposed critical values from
                          the actual independent values or to divide by them:
                          Two possibilities:

                          'sub': subtract
                          'div': divide


    bounds: 1D ndarray or 1D array-like; an array of tuples
            An array of 2-element tuples specifying the boundaries in which the
            optimization routine should seek for the optimum parameters.
            The length of the bounds arrray should equal the number of the
            parameters we would like to find.

    nsamples: int
              How many different random realizations of the optimization
              algorithm
              we should perform.

    savename_prefix: string,
              the first part of the name once the data are saved.

    queue: boolean, optional
           Whether to perform a job in parallel (if True) on a SLURM-based
           cluster or not. Defaults to False.
    """
    # create the results folder if it does
    # not exist already
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    # make the slurmlog folder if it does not
    # yet exist
    if not os.path.isdir('slurmlog'):
        os.makedirs('slurmlog')

    if not os.path.isdir('tmp'):
        os.makedirs('tmp')

    # prepare data to be analyzed

    files = np.sort(glob(data_path))
    vals = np.array([np.loadtxt(file) for file in files])

    x = np.array([val[:, int(xcol)] for val in vals])
    y = np.array([val[:, int(ycol)] for val in vals])

    # cutoff
    y = np.array([y_[(x[i] > xcrit[0]) & (x[i] < xcrit[1])] for i, y_
                  in enumerate(y)])
    x = np.array([x_[(x_ > xcrit[0]) & (x_ < xcrit[1])] for x_ in x])
    # save a temporary file in npz format
    tmpdict = {
        'data_path': data_path,
        'x': x,
        'y': y,
        'xcrit': xcrit,
        'popsize0': popsize0,
        'maxiter0': maxiter0,
        'workers': workers,
        'sizelist': sizelist,
        'critical_point_model': critical_point_model,
        'rescaling_function': rescaling_function,
        'critical_operation': critical_operation,
        'bounds': bounds,
        'save_path': savepath,
        'savename_prefix': savename_prefix
    }

    tmpfilename = (f'/tmp/tmpfile_{critical_point_model}_'
                   f'{rescaling_function}.npz')
    np.savez(tmpfilename, **tmpdict)
    np.savez(f'{savepath}/orig_data.npz', **{'x': x, 'y': y,
                                             'sizes': sizelist,
                                             'desc': savename_prefix})

    if not queue:

        for i in range(nsamples):

            sp.check_call(f'python main_costfun.py {tmpfilename} {i}',
                          shell=True)

        sp.check_call(f'rm {tmpfilename}', shell=True)

    # run on slurm
    else:
        with open('utils/sbatch_template.txt', 'r') as file:
            slurmscript = file.read()

        slurmscript = slurmscript.format(time, ntasks, cputasks,
                                         memcpu, savename_prefix, 'slurmlog',
                                         nsamples,
                                         tmpfilename)

        slurmname = (f'slurm_{savename_prefix}_'
                     f'{critical_point_model}_{rescaling_function}')
        with open(slurmname, 'w') as file:

            file.write(slurmscript)

        sp.check_call(f'sbatch {slurmname}', shell=True)
        sp.check_call(f'rm {slurmname}', shell=True)
