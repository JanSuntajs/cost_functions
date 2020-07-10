import numpy as np

from utils.preprun import main

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
sizelist = [12, 14, 16]

# where to find x and y values (in which columns)
xcol = 0
ycol = 1

# where to cutoff the non-rescaled data -> which
# interval should we analyse
xcrit = (10**-5, 2.)


critical_operation = 'div'
# scaling of the critical point
critical_point_model = 'free'

# rescaling_function
rescaling_function = 'id'

# where the data are stored
data_path = '/home/jan/costfun_single_rivr_fig3/*'
# where to store the results
savepath = (f'results/heis_sing_fig3_costfun/crit_point_scaling_'
            f'{critical_point_model}/rescale_{rescaling_function}/')

savename_prefix = 'r_collapse'
# number of parallel runs
nsamples = 30

# bounds for parameters to be determined

bounds = [(0., 5.) for i in range(len(sizelist))]

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
         popsize0, maxiter0, workers, sizelist, critical_point_model,
         rescaling_function, critical_operation, bounds, nsamples,
         savename_prefix, queue, time, ntasks, cputasks, memcpu)
