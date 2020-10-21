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


def _sort_sizes(data_path, splitkey1, splitkey2,
                type_=int):
    """
    Function for sorting the data according to their
    size. If string sorting is used to sort the data
    corresponding to, say, different system sizes, system
    sizes of different orders of magnitude could be ordered
    inappropriately. For instance, a sequence of:
    ['substring_32', 'substring_128', 'substring_512'] would be sorted as
    ['substring_128', 'substring_32', 'substring_512'] whereas our
    algorithms would require ordering in terms of increasing numerical
    values. To that end, we provide a function that allows for the extraction
    of the system size parameter and the function can then be provided
    as an argument to the sorting function.

    Parameters
    ----------

    data_path: string
               String in a form of a regex specifiying what type (and location)
               of the files to look for.
    splitkey1, splitkey2: string
               substrings enclosing the numerical value to be extracted.
               Example:
               L_40_dim_2
               In case we wish to extract the value next to 'L_' substring,
               we provide:
               splitkey1 = 'L_'
               'splitkey2' = '_dim_'
               Note: in case we wished to extract the parameter next
               to '_dim_', we would have used:
               splitkey1 = '_dim_'
               splitkey2 = ' '
    type_: function, optional
           Datatype of the parameter that is sought after. Defaults to
           int for system sizes.

    Returns:

    files: list
           An ordered list of files storing the data. The files are
           ordered w.r.t. the specified parameter.

    sizelist: list
            An ordered list of system sizes


    """
    def _get_size(filename):
        """
        Obtain the string describing the system size.
        """
        tail = os.path.split(filename)[1]

        size = type_(tail.split(splitkey1)[1].split(splitkey2)[0])
        return size

    files = glob(data_path)
    files = sorted(files, key=_get_size)
    sizelist = [_get_size(file) for file in files]

    return files, sizelist


def _x_val_preprocess(xvals, preprocess_type='none', prefactor=1.):
    """
    Returns processed x values in case preprocessing is
    chosen before the scaling analysis.
    Parameters:
    -----------

    xvals: ndarray
        Data to be processed.


    preprocess_type: string, optional
        Which operation to perform on the data.
        Options are:
        'none' -> no rescaling
        'log' -> natural logarithm
        'log10' -> base 10 logarithm
        'inv' -> inverse, hence x -> 1./x
        'mult' -> multipy x-axis value by a prefactor
    prefactor: float, optional
        Multiplicative prefactor for the x-data values
        before any transformation is performed.

    """
    def _id(x):
        return x

    def _inv(x):
        return 1. / x

    def _mult(x):
        return prefactor * x

    if preprocess_type == 'none':
        fun = _id
    elif preprocess_type == 'log':
        fun = np.log
    elif preprocess_type == 'log10':
        fun = np.log10
    elif preprocess_type == 'inv':
        fun = _inv
    elif preprocess_type == 'mult':
        fun = _mult
    else:
        raise ValueError(f'{preprocess_type} not yet implemented!')

    return np.array([fun(xval) for xval in xvals])


def main(data_path, savepath, xcrit, xcol, ycol,
         popsize0, maxiter0, workers, sizeloc1,
         sizeloc2, sizedtype,
         critical_point_model, rescaling_function,
         critical_operation, bounds, nsamples,
         savename_prefix, queue=False,
         time='00:00:00',
         ntasks=1,
         cputasks=1,
         memcpu=0,
         preprocess_xvals='none',
         preprocess_xvals_prefactor=1.,
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

    sizeloc1, sizeloc2: string
                        substrings needed to locate the string describing
                        the system size in the filename of the data file.
                        This is needed for proper sorting of data.
                        Example:

                        If the filenames are of the form:
                        '*L_40_dim_2_*' and we wish to extract the parameter
                        next to 'L_', we set the following values for the
                        sizeloc1 and sizeloc2 parameters:
                        sizeloc1 = 'L_'
                        sizeloc2 = '_dim_'

    sizedtype: function
               Function/datatype of the size parameter. Most often int is used.


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

    preprocess_xvals: string, optional
           Whether to perform some operation on the xvalues before the scaling
           analysis is performed, such as taking the logarithm of it. Curently
           'none', 'log', 'log10' and 'inv' options are possible.
    preprocess_xvals_prefactor: float, optional
        Multiplicative prefactor for the x-data values
        before any transformation is performed.
    """
    # create the results folder if it does
    # not exist already
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    # make the slurmlog folder if it does not
    # yet exist
    if not os.path.isdir('slurmlog'):
        os.makedirs('slurmlog')

    # dependency script folder
    if not os.path.isdir('slurmlog_dep'):
        os.makedirs('slurmlog_dep')

    if not os.path.isdir('tmp'):
        os.makedirs('tmp')

    # prepare data to be analyzed

    files, sizelist = _sort_sizes(data_path, sizeloc1, sizeloc2, sizedtype)
    vals = np.array([np.loadtxt(file) for file in files])

    x = np.array([val[:, int(xcol)] for val in vals])
    y = np.array([val[:, int(ycol)] for val in vals])

    # cutoff
    y = np.array([y_[(x[i] > xcrit[0]) & (x[i] < xcrit[1])] for i, y_
                  in enumerate(y)])
    x = np.array([x_[(x_ > xcrit[0]) & (x_ < xcrit[1])] for x_ in x])

    # preprocess data
    # 2 steps -> first, perform multiplication, then do the
    # required transformation
    x_prep = _x_val_preprocess(x, 'mult', preprocess_xvals_prefactor)
    x_prep = _x_val_preprocess(x_prep, preprocess_xvals,
                               1.0)

    # save a temporary file in npz format
    tmpdict = {
        'data_path': data_path,
        'x': x,
        'x_prep': x_prep,
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
        'savename_prefix': savename_prefix,
        'preprocess_xvals': preprocess_xvals,
        'preprocess_xvals_prefactor': preprocess_xvals_prefactor,
    }

    tmpfilename = (f'./tmp/tmpfile_{critical_point_model}_'
                   f'{rescaling_function}.npz')
    np.savez(tmpfilename, **tmpdict)
    np.savez(f'{savepath}/orig_data.npz', **{'x': x, 'y': y,
                                             'sizes': sizelist,
                                             'desc': savename_prefix})
    np.savez(f'{savepath}/processed_data_{preprocess_xvals}.npz',
             **{'x': x_prep, 'y': y,
                'x_operation': preprocess_xvals,
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
