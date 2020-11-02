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


_special_keys = ['params', 'costfun_value', 'nfev', 'nit', 'bounds']


def sort_sizes(data_path, splitkey1, splitkey2,
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

    return np.array(files), np.array(sizelist)


def window_stack(sizelist, stepsize=1, width=3):
    """
    This function stacks different parts of a sorted array
    together. The number of stacked values is determined
    by the width parameter and the step in which stacking
    is performed is determined by the stepsize parameter.

    Example:

    sizelist = [7, 8, 9, 10, 11, 12, 13]
    Choosing stepsize = 2
    and width = 3, would return the following
    stacking:

    [7, 8, 9], [9, 10, 11], [11,  12, 13]
    """
    sizelist = np.array(sizelist)
    n = sizelist.shape[0]
    if width < n:
        top = n - width + 1
    else:
        top = n

    stacks = [sizelist[i:i + width]
              for i in range(0, top, stepsize)]

    stacks = [stack for stack in stacks if stack.size >= 2]

    return np.array(stacks, dtype=np.int)


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


def _preprocessing(data_path, files, sizelist, savepath, xcrit, xcol, ycol,
                   popsize0, maxiter0, workers, sizeloc1,
                   sizeloc2, sizedtype,
                   critical_point_model, rescaling_function,
                   critical_operation, bounds, nsamples,
                   savename_prefix, preprocess_xvals,
                   preprocess_xvals_prefactor):
    """

    """
    # create the results folder if it does
    # not exist already
    size_sign = sizeloc1.strip('_')
    savepath = (f'{savepath}/{size_sign}_{sizelist[0]}_to_{size_sign}_'
                f'{sizelist[-1]}')
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

    # get the folder with the data files -> we add
    # the data_folder name to the names of the temporary
    # files
    if os.path.isdir(data_path):
        data_folder = os.path.split(data_path)[1]
    else:
        head, tail = os.path.split(data_path)
        data_folder = os.path.split(head)[1]

    # get a list of indices in case a parameter sweep over different
    # sizes is desired

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
    rnd_num = np.random.randint(0, 10)
    size_sign = sizeloc1.strip('_')
    tmpfilename = (f'./tmp/tmpfile_{data_folder}_{critical_point_model}_'
                   f'{rescaling_function}_rnd_{rnd_num}_'
                   f'{size_sign}_{sizelist[0]}_to_'
                   f'{size_sign}_{sizelist[-1]}.npz')

    # perform a size sweep if needed
    slurmname = (f'./tmp/{{}}_slurm_{data_folder}_{savename_prefix}_'
                 f'{critical_point_model}_{rescaling_function}_rnd_{rnd_num}'
                 f'{size_sign}_{sizelist[0]}_to_'
                 f'{size_sign}_{sizelist[-1]}.npz')
    np.savez(tmpfilename, **tmpdict)

    # ---------------------------------------------------------
    #
    #  SAVE THE ORIGINAL AND PROCESSED DATA
    #
    # ---------------------------------------------------------
    np.savez((f'{savepath}/'
               '/orig_data.npz'), **{'x': x, 'y': y,
                                     'sizes': sizelist,
                                     'desc': savename_prefix})
    np.savez(f'{savepath}/processed_data_{preprocess_xvals}.npz',
             **{'x': tmpdict['x_prep'], 'y': tmpdict['y'],
                'x_operation': tmpdict['preprocess_xvals'],
                'sizes': tmpdict['sizelist'],
                'desc': tmpdict['savename_prefix']})
    return tmpdict, tmpfilename, slurmname, data_folder


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
         finite_size_scaling_analysis=False,
         finite_size_scaling_step=-1,
         finite_size_scaling_width=-1,
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

    # get the dictonary with the temporary job data,
    # name of the file with the temporary dict,
    # a template name for various subsequent files
    # and the name of the folder containing the data

    # prepare data to be analyzed
    files, sizelist = sort_sizes(data_path, sizeloc1, sizeloc2, sizedtype)

    # check if finite-size scaling needs to be performed
    # check some entries to avoid errors
    if finite_size_scaling_analysis:
        if finite_size_scaling_width == -1:
            raise ValueError(f('{} cannot be set to True '
                               f'with {finite_size_scaling_width} '
                               'equal to -1!'))
        if finite_size_scaling_step == -1:
            raise ValueError(f('{} cannot be set to True '
                               f'with {finite_size_scaling_step} '
                               'equal to -1!'))
        stacks_width = finite_size_scaling_width
        stacks_step = finite_size_scaling_step
    else:
        stacks_width = len(sizelist)
        stacks_step = stacks_width

    size_stacks = window_stack(np.arange(len(sizelist)),
                               stacks_step,
                               stacks_width)

    print(size_stacks)
    for size_stack in size_stacks:
        files_ = files[size_stack]
        sizelist_ = sizelist[size_stack]
        print(sizelist_)
        tmpdict, tmpfilename, slurname, data_folder = _preprocessing(
            data_path, files_, sizelist_, savepath, xcrit, xcol, ycol,
            popsize0, maxiter0, workers, sizeloc1,
            sizeloc2, sizedtype,
            critical_point_model, rescaling_function,
            critical_operation, bounds, nsamples,
            savename_prefix, preprocess_xvals,
            preprocess_xvals_prefactor)
        if not queue:

            for i in range(nsamples):

                sp.check_call(f'python main_costfun.py {tmpfilename} {i}',
                              shell=True)

            sp.check_call(f'python main_collect_costfun.py {tmpfilename}',
                          shell=True)

            sp.check_call(f'rm {tmpfilename}', shell=True)

        # run on slurm
        else:
            # prepare a dependency file which does the minimization
            # jobs, collects the results once everything concludes
            # and then removes the temporary file at the very end
            with open('utils/sbatch_template.txt', 'r') as file:
                slurmscript_run = file.read()

            # runner script -> write it down with the relevant
            # parameters
            slurmscript_run = slurmscript_run.format(time, ntasks, cputasks,
                                                     memcpu, savename_prefix,
                                                     'slurmlog',
                                                     nsamples,
                                                     tmpfilename)

            # prepare the collect script -> the one that collects
            # the results after the first step has completed
            with open('utils/sbatch_collect_template.txt', 'r') as file:
                slurmscript_collect = file.read()

            slurmscript_collect = slurmscript_collect.format(tmpfilename)

            with open('utils/sbatch_remove_template.txt', 'r') as file:
                slurmscript_remove = file.read()

            sbatchlist = []
            for pair in [(slurmscript_run, 'main'),
                         (slurmscript_collect, 'collect')]:
                with open(slurmname.format(pair[1]), 'w') as file:

                    file.write(pair[0])
                    sbatchlist.append(slurmname.format(pair[1]))

            slurmscript_remove = slurmscript_remove.format(
                tmpfilename, *sbatchlist)
            with open(slurmname.format('remove'), 'w') as file:
                file.write(slurmscript_remove)
                sbatchlist.append(slurmname.format('remove'))

            with open('utils/sbatch_dep_template.txt', 'r') as file:
                slurmscript_dep = file.read()

            slurmscript_dep = slurmscript_dep.format(*sbatchlist)

            # prepare the main dependency script and run it
            slurmname_dep = slurmname.format('dep')
            with open(slurmname_dep, 'w') as file:

                file.write(slurmscript_dep)
            sp.check_call(f'sbatch {slurmname_dep}', shell=True)
            sp.check_call(f'rm {slurmname_dep}', shell=True)
