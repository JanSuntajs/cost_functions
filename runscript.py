import numpy as np

from utils.preprun import main, sort_sizes

queue = False

# population size for random
# sampling in the differential
# evolution scheme
popsize0 = 100
# maximum number of iterations
# of the differential evolution
# algorithm
maxiter0 = 100
# number of parallel workers
workers = 1
# system sizes

# where to find x and y values (in which columns)
xcol = 0
ycol = 1

# where to cutoff the non-rescaled data -> which
# interval should we analyse
xcrit = (10**-5, 2.)


critical_operation = 'sub'
# scaling of the critical point
critical_point_model = 'lin'

# rescaling_function
rescaling_function = 'kt'
# preprocess x vals
x_val_prep = 'none'


# where the data are stored
data_path = './data/r_sweep*'
# where to store the results
savepath = (f'results/test_code/crit_point_scaling_'
            f'{critical_point_model}/rescale_{rescaling_function}/'
            f'prep_xval_{x_val_prep}_{critical_operation}/')

#
sizeloc1 = 'L_'
sizeloc2 = '_nu_'
sizedtype = int
*_, sizelist = sort_sizes(data_path, sizeloc1, sizeloc2, sizedtype)

savename_prefix = 'r_collapse'
# number of parallel runs
nsamples = 4

# bounds for parameters to be determined

bounds = [(0., 5.) for i in range(3)]

# SLURM PARAMS
time = '00:59:59'
ntasks = 1
cputasks = 4
memcpu = 4096


# --------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------
if __name__ == '__main__':

    main(data_path, savepath, xcrit, xcol, ycol,
         popsize0, maxiter0, workers, sizeloc1, sizeloc2, sizedtype,
         critical_point_model,
         rescaling_function, critical_operation, bounds, nsamples,
         savename_prefix, queue, time, ntasks, cputasks, memcpu, x_val_prep,
         finite_size_scaling_analysis=True,
         finite_size_scaling_step=1,
         finite_size_scaling_width=3)
