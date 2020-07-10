import numpy as np
import subprocess as sp
import os
from glob import glob

queue = False

# population size for random
# sampling in the differential
# evolution scheme
popsize0 = 100
# maximum number of iterations
# of the differential evolution
# algorithm
maxiter0 = 1000
# number of parallel workers
workers = 4
# system sizes
sizelist = [12, 14, 16, 18, 20]

# where to find x and y values (in which columns)
xcol = 0
ycol = 1

# where to cutoff the non-rescaled data -> which
# interval should we analyse
xcrit = (10**-5, 2.)

# scaling of the critical point
critical_point_model = 'free'

# rescaling_function
rescaling_function = 'kt'

# where the data are stored
data_path = './data/*'
# where to store the results
savepath = (f'results/crit_point_scaling_'
            f'{critical_point_model}/rescale_{rescaling_function}/')

savename_prefix = 'r_collapse'
# number of parallel runs
nsamples = 3

# bounds for parameters to be determined

bounds = [(0., 5.) for i in range(len(sizelist) + 1)]
bounds[-1] = (0.1, 4)

# SLURM PARAMS
time = '00:59:59'
ntasks = 1
cputasks = 4
memcpu = 4096


# --------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------
if __name__ == '__main__':
    # create the results folder if it does
    # not exist already
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    # make the slurmlog folder if it does not
    # yet exist
    if not os.path.isdir('slurmlog'):
        os.makedirs('slurmlog')

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
        'bounds': bounds,
        'save_path': savepath,
        'savename_prefix': savename_prefix
    }

    tmpfilename = (f'tmpfile_{critical_point_model}_'
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
        with open('sbatch_template.txt', 'r') as file:
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
