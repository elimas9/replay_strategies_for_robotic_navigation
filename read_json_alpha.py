import json
import numpy as np

# set how many individuals and the type of replay
n_individuals = 100
data_dict_repl = ['no replay', 'backward replay', 'random replay']


# *** DETERMINISTIC CASE *** #
# open the simulation results needed (.json file)
with open(f'data_files/perf_alpha_ind{n_individuals}_detTrue_NEW.json', 'rb') as perftruefile:
    perftruestats = json.load(perftruefile)["stat"]

# convert the data in np.array
detsum = None
for k in data_dict_repl:
    if detsum is None:
        detsum = np.array([d[0][1] for d in perftruestats[k]])
    else:
        detsum = detsum + np.array([d[0][1] for d in perftruestats[k]])

# identify the alpha value for the minimal number of iterations to get to the reward
alphaidx = np.argmin(detsum)
alpha = perftruestats[data_dict_repl[0]][alphaidx][2]
print(f"    Best alpha (deterministic): {alpha}")


# *** STOCHASTIC CASE *** #
# open the simulation results needed (.json file)
with open(f'data_files/perf_alpha_ind{n_individuals}_detFalse_NEW.json', 'rb') as perffalsefile:
    perffalsestats = json.load(perffalsefile)["stat"]

# convert the data in np.array
nondetsum = None
for k in data_dict_repl:
    if nondetsum is None:
        nondetsum = np.array([d[0][1] for d in perffalsestats[k]])
    else:
        nondetsum = nondetsum + np.array([d[0][1] for d in perffalsestats[k]])

# identify the alpha value for the minimal number of iterations to get to the reward
alphaidx = np.argmin(nondetsum)
alpha = perffalsestats[data_dict_repl[0]][alphaidx][2]
print(f"Best alpha (non deterministic): {alpha}")


# *** COMMON ANALYSES *** #
# identify the alpha value for the minimal number of iterations to get to the reward (summing the deterministi and the
# stochastic case)
alphaidx = np.argmin(detsum + nondetsum)
alpha = perffalsestats[data_dict_repl[0]][alphaidx][2]
print(f"              Best alpha (sum): {alpha}")
