import numpy as np
import copy
import matplotlib_latex_bridge as mlb
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial as scispa
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.image as mpimg
import json

from functions import reading_transitions, define_transitions_matrix, NpEncoder
mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)

# **************************************************************************** #
# SET THE SIMULATION PARAMETERS
plot_qvalue_map = False  # plot the map of the max q-values propagation
plot_hist = False  # plot histogram of the max q-values distribution
save_max_optimal_qvalues = True  # True if you want to save the optimal qvalues in a .json file
deterministic_world = False  # True if you want a deterministic environment, False for a stochastic one, as computed in
# Gazebo
gamma = 0.9
reward_state = 22
n_bins = 10
cdf = False
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
params_sims = {'gamma': gamma, 'len_tMatrix': len_tMatrix, 'reward_state': reward_state, 'Trans_matrix': Trans_matrix}


def get_transitions(params, state, acton):
    '''
    Get all the information regarding the transition matrix and the reward for a give state and action

    :param params: dictionary with the parameters of the simulation
    :param state: int representing the starting state
    :param acton: int representing the global action
    :return: list of tuples containing the arrival state, the probability to get the arrival state and the reward of the
    arrival state
    '''
    if state == reward_state:
        return []
    else:
        reward = [0] * len_tMatrix
        reward[params['reward_state']] = 1

        transitions = []
        for id_next_state in range(len(params['Trans_matrix'][state][acton])):
            transitions.append((id_next_state, params['Trans_matrix'][state][acton][id_next_state], reward[id_next_state]))

        return transitions


def value_iteration(params, Qvalues, eps=0.01):
    '''
    Value iteration to obtain the optimal policy

    :param params: dictionary with the parameters of the simulation
    :param Qvalues: np.array with the starting q-values
    :param eps: the threshold to compute convergence
    :return: np.array with the optimal q-values
    '''
    iter = 0
    while True:
        oldQvalues = copy.deepcopy(Qvalues)

        for id_st in range(len(params['Trans_matrix'])):
            for id_act in range(len(params['Trans_matrix'][id_st])):
                Qvalues[id_st][id_act] = 0

                for s1, p, r in get_transitions(params, id_st, id_act):
                    Qvalues[id_st][id_act] += p * (r + max(Qvalues[s1]) * params['gamma'])

        delta = np.max(np.abs(np.array(oldQvalues) - Qvalues))
        iter += 1
        if delta < eps:
            break

    return Qvalues


def create_voronoid(ax=None, q_values=None):
    """
    Plots on the "ax" axes the voronoi structure of the environment fill with the "replay_type" max Q-values for
    each state.

    "ax" must be the matplotlib.pyplot.figure.subplots axes for the figure.
    "replay_type" must be a string describing thr replay_type, i.e. the key of the dictionary contained the
    q-values.

    Returns the plt.cm.ScalarMappable instance plotted on the figure.
    """
    # coordinates of the centre of all the states
    states_ = [(0.00320002, 0.0059351), (0.310727, 0.0241474), (0.593997, 0.152759), (0.724483, -0.118826),
               (0.956215, 0.0761724), (-0.0546479, 0.308712), (-0.359653, 0.3537), (-0.465333, 0.651306),
               (-0.636187, 0.920479), (-0.325808, 0.96138), (-0.0262775, 0.936864), (-0.12853, 0.608485),
               (0.275655, 0.947168), (0.164439, 1.22664), (-0.588397, 1.21822), (-0.779392, 0.652651),
               (-0.669064, 0.253532), (-0.863761, 0.0191329), (-0.924563, -0.277856), (-1.08771, 0.22059),
               (-1.23553, 0.490936), (-1.09893, 0.758705), (-0.293732, 1.30145), (0.208853, 0.454286),
               (0.522142, 0.755471), (0.69987, 0.513679), (-0.0588719, -0.289781), (0.0889117, -0.583929),
               (-0.120961, -0.804133), (-0.939397, 1.014), (0.367168, -0.274708), (-0.329299, -0.156936),
               (0.39444, -0.660872), (-0.539525, -0.381784), (-1.22956, -0.263396), (-0.504464, -0.834182)]

    # save the maximum values of these q-values just opened for normalisation later
    max_q_value = {}
    for ids, st in enumerate(q_values):  # state
        max_q_value[str(ids)] = max(st)

    qvalues_min = 0
    qvalues_max = 1

    normalizer = matplotlib.colors.Normalize(vmin=min(max_q_value.values()), vmax=max(max_q_value.values()))
    # normalizer = matplotlib.colors.Normalize(vmin=0, vmax=1))

    centre_states = np.array(states_)
    centre_states = np.append(centre_states, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)  # add 4
    # distant dummy points

    # create Voronoi
    vor = scispa.Voronoi(centre_states)

    # Fill each state with a color corresponding to the normalized Q-value
    print(max_q_value)
    for r in range(len(vor.point_region)):  # state
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=plt.cm.viridis(normalizer(max_q_value[str(r)])), zorder=0)

    # states and legend
    plot_states = np.array(states_)
    ax.scatter(plot_states[35, 0], plot_states[35, 1], c="green", label="35 - initial state")
    ax.scatter(plot_states[22, 0], plot_states[22, 1], c="purple", label="22 - first reward state")
    ax.scatter(plot_states[4, 0], plot_states[4, 1], c="orange", label="4 - second reward state")

    index_state = []
    for i in range(0, 36):
        if i != 4 and i != 22 and i != 35:
            index_state.append(i)

    return plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=normalizer)


def mask_outside_polygon(poly_verts, ax=None):
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.

    "poly_verts" must be a list of tuples of the verticies in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    """
    if ax is None:
        ax = plt.gca()

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Verticies of the plot boundaries in clockwise order
    bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), (xlim[1], ylim[1]), (xlim[1], ylim[0]), (xlim[0], ylim[0])]

    # A series of codes (1 and 2) to tell matplotlib whether to draw a line or
    # move the "pen" (So that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

    # Plot the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none', zorder=2)
    patch = ax.add_patch(patch)

    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return patch


def create_map(map_path="map1.pgm", scale=1.0, offset=np.zeros(2), ax=None):
    """
    Plots the map of the environment, given the contour "map_path" .pgm, the "scale" and the "offset", on "ax" axes.
    """

    img = mpimg.imread(map_path)

    # get list of coordinates for the border
    coords = []
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                coords.append((i * scale, j * scale))
    coords = np.array(coords)

    # center data and adjust offset
    coords[:, 0] -= np.mean(coords[:, 0])
    coords[:, 1] -= np.mean(coords[:, 1])
    coords = coords + offset

    # compute the contour
    hull = ConvexHull(coords)
    hull_points_x = [coords[v, 0] for v in hull.vertices]
    hull_points_x.append(coords[hull.vertices[0], 0])
    hull_points_y = [coords[v, 1] for v in hull.vertices]
    hull_points_y.append(coords[hull.vertices[0], 1])

    # draw contour and white patch outside
    hull_points = list(zip(hull_points_x, hull_points_y))
    mask_outside_polygon(hull_points, ax=ax)
    ax.plot(hull_points_x, hull_points_y, 'k-')
    ax.axis("equal")
    ax.axis("off")


def main():

    Qvalues = np.zeros((params_sims['len_tMatrix'], 8))
    optimal_qvalues = value_iteration(params_sims, Qvalues)

    max_q_value = []
    for st in optimal_qvalues:  # states
        # compute the max q-values for each state
        max_q_value.append(max(st))

    if plot_hist:
        # define the histogram of the sum of the max Q-values along the individual
        hist_qvalues = np.histogram(max_q_value, bins=n_bins)

        mlb.figure(width=6.97522 * 0.8, height=6.97522 * 0.8)

        if not cdf:
            # plot histogram of Q-values divided in bins
            plt.bar(np.arange(0, len(hist_qvalues[0])), hist_qvalues[0], color='red', label='value iteration')
        else:
            # plot the cumulative histogram of the Q-values
            plt.hist(hist_qvalues[0], n_bins, density=True, histtype='step', cumulative=True, color='red',
                     label='value iteration')

        plt.legend()
        plt.show()

    if save_max_optimal_qvalues:
        with open(f'data_files/max_opt_qvalues_det{deterministic_world}.json', 'w') as oqv:
            json.dump(max_q_value, oqv, cls=NpEncoder)

    if plot_qvalue_map:
        # set the figure parameters
        fig = mlb.figure(width=6.97522 * 0.8, height=6.97522 * 0.8)

        ax = plt.gca()
        colormap = create_voronoid(ax=ax, q_values=optimal_qvalues)
        create_map(map_path="data_files/map1.pgm", scale=0.08, offset=np.array([-0.2, 0.2]), ax=ax)
        ax.set_title('MB - Value iteration')

        # fix range of plot
        ax.set_xlim(- 2, 1.5)
        ax.set_ylim(- 1.5, 2)

        # set a common colorbar
        cbar = fig.colorbar(colormap, ax=ax)
        cbar.set_label('maximum Q-value\nin each state (a.u.)')

        plt.show()


if __name__ == '__main__':
    main()
