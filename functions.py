import numpy as np
from os import path
import matplotlib.pyplot as plt
import random as r
import json
import copy


def split_function(string, sep):
    """
    Split the elements of the lines of the .txt file containing the transitions

    :param string: string (line)
    :param sep: string with the criterion for splitting
    :return: list with the separated elements of the line
    """
    n = len(string)
    l = []
    i = 0
    while i < n:
        new_seq = ''
        while i < n and string[i] != sep:
            new_seq += string[i]
            i = i + 1
        l.append(new_seq)
        i = i + 1
    return l


def second_largest(numbers):
    '''
    Return the second largest element in a list

    :param numbers: list
    :return: list element
    '''
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def reading_transitions(PATH, max_len_tMatrix):
    '''
    Importing the transition matrix M from the transition.txt file, M first dimension are s0
    (all the states), the second dimension are the 8 actions, and the last dimension are the s1 states (so a list
    of length s1=36 with the probbaility o s0 to take action a and end up in s1)

    :param PATH: path of the transitions file (string)
    :param max_len_tMatrix: possible max numebr of states (int)
    :return: Transition Matrux (numpy array)
    '''

    if path.exists(PATH):
        f = open("data_files/transitions.txt", "r")
        lines = f.readlines()
        n = len(lines)
        Trans_matrix = np.zeros((max_len_tMatrix, 8, max_len_tMatrix))
        for k in range(n - 1):
            l = lines[k]
            l_split = split_function(l, ',')
            l_split[-1] = split_function(l_split[-1], '\n')[0]
            s0 = int(l_split[0])
            a = int(l_split[1])
            s1 = int(l_split[2])
            m = float(l_split[3])
            Trans_matrix[s0, a, s1] = m
    else:
        print("error: path not found")
    return Trans_matrix


def define_transitions_matrix(tMatrix, len_tMatrix, deterministic_world):
    '''
        Define the transitions matrix and output it as a .txt file.

        :param tMatrix: transition matrix from Gazebo (numpy array)
        :param len_tMatrix: len of tMatrix (int)
        :param deterministic_world: bool
        :return: Transition Matrux (numpy array), forbidden_state_action (numpy array)
        '''
    forbidden_state_action = np.zeros((len_tMatrix, 8))  # dead-end states/actions
    for s in range(len_tMatrix):
        for a in range(8):
            if np.sum(tMatrix[s, a, :]) == 0:
                forbidden_state_action[s, a] = 1

    if deterministic_world:
        # defining the deterministic transition matrix by allowing just a with the greatest m, for all s0 (this a will
        # have m=1 the others m=0)
        for s in range(len_tMatrix):  # normalize the M matrix to have transition probabilities
            for a in range(8):
                somme = np.sum(tMatrix[s, a, :])
                if somme != 0:
                    if second_largest(tMatrix[s, a, :]) == max(tMatrix[s, a, :]):
                        idx_chosen_max = int(np.where(tMatrix[s, a, :] == second_largest(tMatrix[s, a, :]))[0][1])
                        tMatrix[s, a, idx_chosen_max] = 1
                        for idx_poss_s1 in range(0, len_tMatrix):
                            if idx_poss_s1 != idx_chosen_max:
                                tMatrix[s, a, idx_poss_s1] = 0
                    else:
                        idx_a_max = np.argmax(tMatrix[s, a, :])
                        tMatrix[s, a, idx_a_max] = 1
                        for idx_poss_s1 in range(0, len_tMatrix):
                            if idx_poss_s1 != idx_a_max:
                                tMatrix[s, a, idx_poss_s1] = 0

        with open('data_files/transitions_deterministic.txt', 'w') as f:
            for s in range(len_tMatrix):
                for a in range(8):
                    for pa in range(len(tMatrix[s, a, :])):
                        f.write(str(s) + ',' + str(a) + ',' + str(pa) + ', ' + str(tMatrix[s, a, pa]))
                        f.write('\n')
    else:
        for s in range(len_tMatrix):  # normalize the M matrix to have transition probabilities
            for a in range(8):
                somme = np.sum(tMatrix[s, a, :])
                if somme != 0:
                    tMatrix[s, a, :] = tMatrix[s, a, :] / somme

        with open('data_files/transitions_stochastic.txt', 'w') as f:
            for s in range(len_tMatrix):
                for a in range(8):
                    for pa in range(len(tMatrix[s, a, :])):
                        f.write(str(s) + ',' + str(a) + ',' + str(pa) + ', ' + str(tMatrix[s, a, pa]))
                        f.write('\n')
    return tMatrix, forbidden_state_action


def proba(s, a, Q, params):
    """
    Softmax function

    :param s: int --> state
    :param a: int --> action
    :param Q: 2D list of Q-values
    :param params: dictionary of the parameters for the simulation
    :return: float with the probability to take action a in state s
    """
    somme = 0.0
    for i in range(8):
        somme = somme + np.exp(params['beta'] * Q[s][i])
    return (np.exp(params['beta'] * Q[s][a])) / somme


def choose_random(L):
    """
    Similar to np.random.choice, will choose the index of L based on the proabilities stored in L.

    :param L: list of all possible actions with their probabilities and other information
    :return: int corresponding to the index of the selected action
    """
    # L has a length of 8 (8 possible actions)
    rand = r.random()
    c = 0.0
    k = 0
    n = len(L)
    while k < n and rand >= c + L[k]:
        c = c + L[k]
        k = k + 1
    if k >= n:
        return n - 1
    else:
        return k


def choose_another_action(forbidden_actions, L):
    """
    Choose another action if the previous one was forbidden.

    :param forbidden_actions: numpy array containing the forbidden states
    :param L: list of probabilities and other information of the 8 neighbour actions from the current state
    :return: the new chose action from L
    """
    kmax = 10
    k = 0
    chosen_action = choose_random(L)
    while k < kmax and (forbidden_actions[chosen_action] == 1):
        k = k + 1
        chosen_action = choose_random(L)
    if forbidden_actions[chosen_action] == 1:
        chosen_action = np.random.choice(np.where(forbidden_actions == 0)[0])
    return chosen_action


def replay1(L_global, Q, params):
    '''
        Reply type 1: backward sequences

        :param L_global: list storing all the transitions (curent_state,action,next_state,reward) since the beginning of
        the simulation
        :param Q: Q-value matrix (numpy array)
        :param params: dictonary of parameters for the simulation
        :return: Updated Q-value matrix (numpy array)
    '''
    l = len(L_global)
    n = min(l, params['mem_size_rep'])  # in case the memory is shorter than the length of replayed sequences
    for k in range(params['n_repr_repl']):
        for i in range(n):
            current_state, a, next_state, reward = L_global[-(i + 1)]
            m = max(Q[next_state])
            Q[current_state][a] = Q[current_state][a] + params['alpha'] * (reward + params['gamma'] * m -
                                                                           Q[current_state][a])
    return Q


def replay2(L_global, Q, params):
    """
    MF shuffled replay

    :param L_global: list storing all the transitions
    :param Q: 2D list of Q-values
    :param params: dictionary of the parameters for the simulation
    :return: 2D list of Q-values
    """
    l = len(L_global)
    n = min(l, params['mem_size_rep'])  # in case the memory is shorter than the length of replayed sequences
    for k in range(params['n_repr_repl']):
        id_repl_states = r.sample(list(range(0, n)), n)
        for i in id_repl_states:
            current_state, a, next_state, reward = L_global[-i]
            m = max(Q[next_state])
            Q[current_state][a] = Q[current_state][a] + params['alpha'] * (reward + params['gamma'] * m -
                                                                           Q[current_state][a])
    return Q


def trial_Q_learning(params, reward_state, type_of_replay, L_global, Q):
    """
    Run a single learning trial

    :param params: dictionary of the parameters for the simulation
    :param reward_state: int defining the number of the rewarding state
    :param type_of_replay: int defining the type of replay (1 backward and 2 shuffled)
    :param L_global: list storing all the transitions
    :param Q: 2D list of Q-values
    :return: Q, list of transitions, number of actions
    """
    L = []  # stores the memory of the trial (transitions)
    current_state = params['starting_point']
    count_actions = 0  # count the nb of actions taken in the trial
    while current_state != reward_state:
        L_aux = []  # stores probabilities of actions
        for j in range(8):
            L_aux.append(proba(current_state, j, Q, params))
        a = choose_random(L_aux)  # equivalent to np.random.choice
        if params['forbidden_state_action'][current_state, a] == 1:
            a = choose_another_action(params['forbidden_state_action'][current_state, :], L_aux)
        next_state = np.random.choice(a=params['len_tMatrix'], p=params['Trans_matrix'][current_state, a, :])
        reward = int(next_state == reward_state)
        m = max(Q[next_state])
        Q[current_state][a] = Q[current_state][a] + params['alpha'] * (reward + params['gamma'] * m -
                                                                       Q[current_state][a])
        L.append((current_state, a, next_state, reward))
        count_actions += 1
        current_state = next_state
    if type_of_replay == 1:
        Q = replay1(L_global + L, Q, params)
    if type_of_replay == 2:
        Q = replay2(L_global + L, Q, params)
    return Q, L, count_actions


def run_sim(params, type_of_replay, n_actions, Q=None):
    """
    Run a whole simulation for an individual

    :param params: dictionary of the parameters for the simulation
    :param type_of_replay:  int defining the type of replay (1 backward and 2 shuffled)
    :param n_actions: list with the number of actions taken to get to the reward, per trial
    :param Q: 2D list of Q-values
    :return: tuple of Q-values at trial 3-5-20, list storing all the transitions, list storing the number of actions
    taken to get to the reward for each trial of the individual
    """
    if Q is None:
        Q = np.zeros((params['len_tMatrix'], 8))

    # global Q
    L_global = []

    for trial in range(params['n_trials']):

        if type(params['reward_state']) == list:
            if trial >= (params['n_trials'] / 2) - 1:
                reward_state = params['reward_state'][1]
            else:
                reward_state = params['reward_state'][0]
        else:
            reward_state = params['reward_state']

        Q, L, count_actions = trial_Q_learning(params, reward_state,type_of_replay, L_global, Q)

        if trial == 3:
            Q_3 = copy.deepcopy(Q)
        elif trial == 5:
            Q_5 = copy.deepcopy(Q)
        elif trial == 20:
            Q_20 = copy.deepcopy(Q)

        n_actions.append(count_actions)
        L_global += L

    return (Q_5, Q_20, Q_3), L_global, n_actions


def cv_criterion(series, perc, window):
    """
    Compute learning convergence.

    :param series: numpy array containing the number of actions to get to the reward for each trial of the individual
    :param perc: float indicating the percentage of Q-values to be considered to compute the convergence
    :param window: int indicating how many trials to consider for convergence
    :return: a numpy array with the trial index of convergence, the number of actions in that trial and the cumulative
    number of actions until that trial
    else:
    """

    b = False
    n = len(series)
    k = 0
    while (k < n) and not b:
        x = series[k]
        l = min((n - k - 1), window)
        sup = x * (1 + perc)
        inf = x * (1 - perc)
        i = 1
        while i <= l and sup >= series[k + i] >= inf:
            i = i + 1
        b = (i == l + 1)
        k = k + 1
    k = k - 1
    if k != (n - 1):
        a = series[k]
        b = np.sum(series[0:k + 1])
        return np.array([k, a, b])
    else:
        return np.array([None, None, None])


class NpEncoder(json.JSONEncoder):
    """
    Class used to open .json file
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)