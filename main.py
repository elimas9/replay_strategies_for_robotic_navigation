import numpy as np
import random as r
import matplotlib.pyplot as plt
import matplotlib_latex_bridge as mlb
import json
import copy

from functions import reading_transitions, define_transitions_matrix, cv_criterion, run_sim, NpEncoder

# set-up figures parameters
mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)
mlb.figure_textwidth()

# initialize random seeds
np.random.seed(9)
r.seed(3)

# **************************************************************************** #
# SET THE SIMULATION PARAMETERS
save_new_data_file_results = False  # to save new .json file with the statistics of the results
save_new_data_file_results_all_individuals = False  # to save new .json file with the results regarding all individuals
save_all_ind_q_map_trial3 = True  # to save new .json files with the results regarding the q values at iterearion #3 for
# all the individuals
saving_folder = '.'  # path to save the figures
deterministic_world = False  # True if you want a deterministic environment, False for a stochastic one, as computed in
# Gazebo
learning = True  # True if you want to perform the learning phase
test_alpha = False  # True if you want to test the learning performance of the algorithms for different values of alpha
beta = 15.
alpha = 0.78
epsilon = 0.5
gamma = 0.9
reward_state = [22, 4]
starting_point = 35
mem_size_rep = 90  # length of replayed sequences
n_repr_repl = 20  # how many sequences or transitions are replayed for offline learning phase
n_trials = 50
n_individuals = 100
replay_types = ['no replay', 'backward replay', 'shuffled replay']
possible_actions = 8
# **************************************************************************** #

# continuous coordinate of all the states discovered by the robot
states_ = [(0.00320002,0.0059351),(0.310727,0.0241474), (0.593997,0.152759),(0.724483, -0.118826),(0.956215, 0.0761724),
         (-0.0546479, 0.308712),(-0.359653, 0.3537),(-0.465333, 0.651306),(-0.636187 ,0.920479),(-0.325808 ,0.96138),
         (-0.0262775, 0.936864),(-0.12853 ,0.608485),(0.275655, 0.947168),(0.164439 ,1.22664),(-0.588397, 1.21822),
         (-0.779392, 0.652651),(-0.669064, 0.253532),(-0.863761 ,0.0191329),(-0.924563, -0.277856),(-1.08771, 0.22059),
         (-1.23553, 0.490936),(-1.09893 ,0.758705),(-0.293732, 1.30145),(0.208853, 0.454286),(0.522142, 0.755471),
         (0.69987, 0.513679),(-0.0588719, -0.289781),( 0.0889117 ,-0.583929),(-0.120961, -0.804133),(-0.939397, 1.014),
         ( 0.367168, -0.274708),( -0.329299 ,-0.156936),(0.39444, -0.660872),(-0.539525, -0.381784),
         (-1.22956, -0.263396),(-0.504464 ,-0.834182)]

x_states = []  # list of x coordinates of states
y_states = []  # list of y coordinates of states

for s in range(36):  # 36 because this is the number of the common discovered states among the exploration in Gazebo
    x, y = states_[s]
    x_states.append(x)
    y_states.append(y)

# reading the real transition matrix from Gazebo
PATH = "data_files/transitions.txt"
max_len_tMatrix = 60  # max possible number of discovered states (in the Gazebo exploration they are max 43s)
Trans_matrix = reading_transitions(PATH, max_len_tMatrix)

len_tMatrix = 36  # reset the number of states
Trans_matrix = Trans_matrix[0:len_tMatrix, :, 0:len_tMatrix]  # crop the M matrix to the number of
# "common-to-all-states"
Trans_matrix, forbidden_state_action = define_transitions_matrix(Trans_matrix, len_tMatrix, deterministic_world)
# define the transitions matrix (deterministic world is a boolean to set the properties of the environment)

# organised all the parameters in a dictionary
params_sims = {'alpha': alpha, 'n_trials': n_trials, 'mem_size_rep': mem_size_rep, 'n_repr_repl': n_repr_repl,
               'starting_point': starting_point, 'beta': beta, 'gamma': gamma, 'alpha': alpha,
               'len_tMatrix': len_tMatrix, 'epsilon': epsilon, 'reward_state': reward_state,
               'forbidden_state_action': forbidden_state_action, 'Trans_matrix': Trans_matrix}

mean_action = []
replays_ = []
Q_ = {}  # stores the Q table table at the end of each simulation to use it for the generalisation

# learning phase
if learning:
    fig = plt.figure()
    fig.set_size_inches((11, 8.5), forward=False)

    print('LEARNING...')

    mean_act_along_trial = {}
    std_act_along_trial = {}
    all_ind_n_actions = {}
    stat_act_along_trial = {}
    all_tr_q_map_trial = {}
    all_ind_q_map_3trial = {}

    for replay in range(0, len(replay_types)):
        all_ind_n_actions[replay] = []
        all_ind_q_map_3trial[replay_types[replay]] = []

        Q_[f'repl_{replay + 1}'] = {}
        for k in range(n_individuals):
            print('individual nb ' + str(k))
            n_actions = []
            Q_saved, L_global, n_actions = run_sim(params_sims, replay, n_actions)
            Q_[f'repl_{replay + 1}'][f'ind_{k + 1}'] = Q_saved

            # check convergene
            cv_ind, actions_cv, cumu_actions_cv = cv_criterion(np.array(n_actions), 0.7, 5)

            print(f'num trial when convergence happens: {cv_ind}, num of action in the trial when convergence happnes:'
                  f'{actions_cv}, cum act before convergence: {cumu_actions_cv}')
            mean_action.append(np.mean(n_actions))
            replays_.append(replay)  # type of replays

            all_ind_n_actions[replay].append(n_actions)

            if k == (n_individuals/2) - 1:  # save q-value propagation map for individual 25
                q_map_3trial_50ind = copy.deepcopy(Q_saved[2])

            # register q_values for all indivials, trial 3
            all_ind_q_map_3trial[replay_types[replay]].append(copy.deepcopy(Q_saved[2]))

        mean_act_along_trial[replay] = np.mean(all_ind_n_actions[replay], axis=0)
        std_act_along_trial[replay] = np.std(all_ind_n_actions[replay], axis=0)
        stat_act_along_trial[replay] = np.percentile(all_ind_n_actions[replay], [25, 50, 75], axis=0)

        # register q_values for indivial 25, trial 3
        all_tr_q_map_trial[replay_types[replay]] = q_map_3trial_50ind

    if save_new_data_file_results:
        with open(f'data_files/q_values_det{deterministic_world}_ind{n_individuals}_alpha{alpha}_trial_3_ind_50.json',
                  'w') as qv_ot:
                  json.dump(all_tr_q_map_trial, qv_ot, cls=NpEncoder)

        with open(f'data_files/results_det{deterministic_world}_ind{n_individuals}_alpha{alpha}_NEWalpha.json', 'w')\
                as rr:
                  json.dump({'mean': mean_act_along_trial, 'std': std_act_along_trial, 'stat': stat_act_along_trial}, rr,
                            cls=NpEncoder)

    if save_new_data_file_results_all_individuals:
        with open(f'data_files/all_results_det{deterministic_world}_ind{n_individuals}_alpha{alpha}.json', 'w') as ar:
            json.dump(all_ind_n_actions, ar, cls=NpEncoder)

    if save_all_ind_q_map_trial3:
        with open(f'data_files/all_ind_q_valuesdet{deterministic_world}_ind{n_individuals}_alpha{alpha}.json', 'w')\
                as aq:
            json.dump(all_ind_q_map_3trial, aq, cls=NpEncoder)


if test_alpha:
    alpha_ = np.linspace(0, 1, 10)
    mean_nb_actions_ = {}
    std_mean_nb_actions_ = {}
    stat_nb_actions_ = {}
    all_nb_actions = {}

    for replay in range(0, len(replay_types)):
        mean_nb_actions_[replay_types[replay]] = []
        std_mean_nb_actions_[replay_types[replay]] = []
        stat_nb_actions_[replay_types[replay]] = []
        all_nb_actions[replay_types[replay]] = []
        print('replay '+str(replay))

        for k in range(len(alpha_)):
            alpha = alpha_[k]
            params_sims['alpha'] = alpha_[k]
            print('alpha '+str(params_sims['alpha']))
            mean_actions = []

            for ind in range(n_individuals):
                print('ind '+str(ind))
                n_actions = []
                Q_saved, L_global, n_actions = run_sim(params_sims, replay, n_actions)
                cv_ind, actions_cv, cumu_actions_cv = cv_criterion(np.array(n_actions), 0.7, 5)
                mean_actions.append(np.mean(n_actions))

            mean_nb_actions_[replay_types[replay]].append((np.mean(mean_actions), replay_types[replay], alpha))
            std_mean_nb_actions_[replay_types[replay]].append((np.std(mean_actions), replay_types[replay], alpha))
            stat_nb_actions_[replay_types[replay]].append((np.percentile(mean_actions, [25, 50, 75], axis=0),
                                                           replay_types[replay], alpha))
            all_nb_actions[replay_types[replay]].append((mean_actions, replay_types[replay], alpha))

    if save_new_data_file_results:
        with open(f'data_files/perf_alpha_ind{n_individuals}_det{deterministic_world}_NEW.json', 'w') as naa:
            json.dump({'mean': mean_nb_actions_, 'std': std_mean_nb_actions_, 'stat': stat_nb_actions_}, naa, cls=NpEncoder)

    if save_new_data_file_results_all_individuals:
        with open(f'data_files/all_perf_alpha_ind{n_individuals}_det{deterministic_world}_NEW.json', 'w') as anaa:
            json.dump(all_nb_actions, anaa, cls=NpEncoder)