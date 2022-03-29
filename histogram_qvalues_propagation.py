import matplotlib_latex_bridge as mlb
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as stats
import operator

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)


def main():
    # *** to be set based on the desired plot *** #

    # ---- JUST ONE OF THESE BELOW SHOULD BE TRUE PER SIMULATION ----
    plot_qv2dist = False  # histograms of normalized maximum q-values wrt to the distance to the rewarding state
    plot_qv_ord = False  # histograms of normalized maximum q-values ordered in increasing order
    plot_qvalues_bin = False  # bin distribution ofthe normalized maximum q-values
    plot_all_ind_qvalues_bin = True  # bin distribution ofthe normalized maximum q-values (ALL INDIVIDUALS)
    # ---------------------------------------------------------------

    n_bins = 10  # number of bins for the histograms and cdf
    cdf = False  # select to plot the cdf
    print_stat_world = True  # print the statistics about the difference types of replay wrt to the environemnt
    print_stat_repl = True  # print the statistics about the difference types of replay in the same environemnt
    compare2best_propagation = True

    deterministic_world = [True, False]

    # set the relay types and their color
    label_repl = ['no replay', 'backward replay', 'shuffled replay']
    label_repl_old = ['no replay', 'backward replay', 'random replay']
    color_repl = ['blue', 'orange', 'green']

    states_ = [(0.00320002, 0.0059351), (0.310727, 0.0241474), (0.593997, 0.152759), (0.724483, -0.118826),
               (0.956215, 0.0761724), (-0.0546479, 0.308712), (-0.359653, 0.3537), (-0.465333, 0.651306),
               (-0.636187, 0.920479), (-0.325808, 0.96138), (-0.0262775, 0.936864), (-0.12853, 0.608485),
               (0.275655, 0.947168), (0.164439, 1.22664), (-0.588397, 1.21822), (-0.779392, 0.652651),
               (-0.669064, 0.253532), (-0.863761, 0.0191329), (-0.924563, -0.277856), (-1.08771, 0.22059),
               (-1.23553, 0.490936), (-1.09893, 0.758705), (-0.293732, 1.30145), (0.208853, 0.454286),
               (0.522142, 0.755471), (0.69987, 0.513679), (-0.0588719, -0.289781), (0.0889117, -0.583929),
               (-0.120961, -0.804133), (-0.939397, 1.014), (0.367168, -0.274708), (-0.329299, -0.156936),
               (0.39444, -0.660872), (-0.539525, -0.381784), (-1.22956, -0.263396), (-0.504464, -0.834182)]

    fig = mlb.figure_textwidth(widthp=1, height=3)
    axs = fig.subplots(2, 3)

    # analysis and histograms for all the individuals
    if plot_all_ind_qvalues_bin:
        data_stat_det_world = {}
        data_stat_NOdet_world = {}

        for idx, det_world in enumerate(deterministic_world):  # environment
            # open the .json files with the results
            with open(f"data_files/all_ind_q_valuesdet{det_world}_ind100_alpha0.78.json", "rb") as qvs_a:
                all_q_values_det_sing = json.load(qvs_a)

            print(f'** ENVIRONMENT DETERMINISTIC: {det_world} **')

            max_q_value = {}
            hist_qvalues = {}
            sum_hist_qvalues = {}
            hist_sum_qvalues = {}
            all_by_label = {}

            for idy, tre in enumerate(all_q_values_det_sing.keys()):  # replay types
                print(f'-----> replay type: {tre} -----')

                ax = axs[idx, idy]

                max_q_value[tre] = {}
                hist_qvalues[tre] = {}
                sum_hist_qvalues[tre] = {}

                for id_ind in range(len(all_q_values_det_sing[tre])):  # individual
                    max_q_value[tre][str(id_ind + 1)] = {}

                    for st in range(len(all_q_values_det_sing[tre][id_ind])):  # states
                        # compute the max q-values for each state
                        max_q_value[tre][str(id_ind + 1)][str(st)] =\
                            max(all_q_values_det_sing[tre][id_ind][st])

                # sum the max q-values for each state for each individuals
                sum_hist_qvalues[tre] = []
                for sttt in max_q_value[tre].keys():
                    sum_hist_qvalues[tre].append(np.sum(list(max_q_value[tre][sttt].values())))

                # compute earth mover’s distance to optimal q-values propagation:
                if compare2best_propagation:
                    with open(f"data_files/max_opt_qvalues_det{det_world}.json", "rb") as moqv:
                        max_opt_qvalues = json.load(moqv)

                    dist2opt = stats.wasserstein_distance(sum_hist_qvalues[tre], max_opt_qvalues)
                    print(f'earth mover’s distance to optimal q-values propagation: {dist2opt}')

                # define the histogram of the sum of the max Q-values along the individual
                hist_sum_qvalues[tre] = np.histogram(sum_hist_qvalues[tre], bins=n_bins)

                if not cdf:
                    # plot histogram of Q-values divided in bins
                    ax.bar(np.arange(0, len(hist_sum_qvalues[tre][0])), hist_sum_qvalues[tre][0], color=color_repl[idy],
                           label=label_repl[idy])
                else:
                    # plot the cumulative histogram of the Q-values
                    ax.hist(hist_sum_qvalues[tre][0], n_bins, density=True, histtype='step', cumulative=True,
                            color=color_repl[idy], label=label_repl[idy])

                # set the same limits for the y-axis
                plt.setp(ax, ylim=axs[0, 0].get_ylim())

                # set the labels
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                all_by_label.update(by_label)

            if print_stat_repl:
                print(f'**** DETERMINISTIC {det_world} ****')

            for ttr in label_repl_old:  # replay
                if det_world is True:
                    data_stat_det_world[ttr] = sum_hist_qvalues[ttr]
                else:
                    data_stat_NOdet_world[ttr] = sum_hist_qvalues[ttr]

                for ttr2 in label_repl_old:  # replay
                    # compute the Two-sample Kolmogorov–Smirnov test for the different types of replay
                    st, p_val = stats.ks_2samp(sum_hist_qvalues[ttr], sum_hist_qvalues[ttr2])
                    # print the results
                    if print_stat_repl:
                        print(f'{ttr} vs {ttr2}')
                        print(f'STAT {st}')
                        print(f'P-VALUE {p_val}')

        if print_stat_world:
            for ttr in label_repl_old:  # replay
                print(f"REPLAY: {ttr}")
                # compute the Two-sample Kolmogorov–Smirnov test between the stochastic and the deterministic
                # environment
                st, p_val = stats.ks_2samp(data_stat_det_world[ttr], data_stat_NOdet_world[ttr2])
                # print the results
                print(f'STAT {st}')
                print(f'P-VALUE {p_val}')

        # title
        fig.supxlabel('maximum Q-value cumulative distribution (10 bins)')
        fig.supylabel('sum (over all individuals)\nof maximum Q-value\nin each state (a.u.)')

        # legend
        # axs[0, 0].legend(all_by_label.values(), all_by_label.keys())
        axs[1, 0].legend(all_by_label.values(), all_by_label.keys(), loc='lower right')

    # analysis and histograms for individual 50
    else:
        for idx, det_world in enumerate(deterministic_world):

            # open the .json files with the results
            with open(f"data_files/q_values_det{det_world}_ind100_alpha0.78_trial_3_ind_50.json", "rb") as qvs:
                q_values_det_sing = json.load(qvs)

            max_q_value = {}
            dist_reward = {}
            ord_states2rew = {}
            norm_max_qvalues = {}
            ord_qvalues2dist = {}
            all_by_label = {}
            ord_states2Qv = {}
            hist_qvalues = {}

            for idy, tre in enumerate(q_values_det_sing.keys()):  # replay types
                ax = axs[idx, idy]

                max_q_value[tre] = {}
                dist_reward[tre] = {}
                ord_states2rew[tre] = {}
                norm_max_qvalues[tre] = {}
                ord_states2Qv[tre] = {}
                for st in range(len(q_values_det_sing[tre])):  # states
                    # compute the max q-values for each state
                    # compute the state distance to the reward state
                    if st != 22:  # reward state not considered
                        max_q_value[tre][str(st)] = max(q_values_det_sing[tre][st])
                        dist_reward[tre][str(st)] = np.linalg.norm(np.array(states_[22]) - np.array(states_[st]))

                # Normalize the max Q-value for each state
                min_all_max_qvalues = min(max_q_value[tre].values())
                max_all_max_qvalues = max(max_q_value[tre].values())
                for st in range(len(q_values_det_sing[tre])):  # states
                    if st != 22:  # reward state not considered
                        norm_max_qvalues[tre][str(st)] = (1 - 0) / (max_all_max_qvalues - min_all_max_qvalues) *\
                                                         (max_q_value[tre][str(st)] - max_all_max_qvalues) + 1

                if plot_qv2dist:
                    # order the states in increasing order (per distance to the reward state)
                    ord_states2rew[tre] = dict(sorted(dist_reward[tre].items(), key=operator.itemgetter(1)))
                    ord_qvalues2dist[tre] = {k: norm_max_qvalues[tre][k] for k in ord_states2rew[tre].keys()}

                    # plot histogram of q-values as a function of the distance to the reward position
                    ax.bar(np.arange(0, len(ord_states2rew[tre])), ord_qvalues2dist[tre].values(), color=color_repl[idy],
                           label=label_repl[idy])

                if plot_qv_ord:
                    # order the states in increasing order of Q-values
                    ord_states2Qv[tre] = dict(sorted(norm_max_qvalues[tre].items(), key=operator.itemgetter(1)))

                    # plot histogram of Q-values in an increasing order
                    ax.bar(np.arange(0, len(ord_states2Qv[tre])), ord_states2Qv[tre].values(), color=color_repl[idy],
                           label=label_repl[idy])

                if plot_qvalues_bin:
                    # define an histogram of the Q-values
                    hist_qvalues[tre] = np.histogram(list(norm_max_qvalues[tre].values()), bins=n_bins)

                    if not cdf:
                        # plot histogram of Q-values divided in bins
                        ax.bar(np.arange(0, len(hist_qvalues[tre][0])), hist_qvalues[tre][0], color=color_repl[idy],
                               label=label_repl[idy])
                    else:
                        # plot the cumulative histogram of the Q-values
                        ax.hist(hist_qvalues[tre][0], n_bins, density=True, histtype='step', cumulative=True,
                                color=color_repl[idy], label=label_repl[idy])

                if plot_qv2dist:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                if plot_qvalues_bin:
                    # set the same limits for the y-axis
                    plt.setp(ax, ylim=axs[0, 0].get_ylim())

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                all_by_label.update(by_label)

            if plot_qv2dist:
                fig.supxlabel('distance to the rewarding state')
            if plot_qv_ord:
                fig.supxlabel('states ordered by increasing normalized maximum Q-value')
            if plot_qvalues_bin:
                fig.supxlabel('normalized maximum Q-value distribution bins')

            fig.supylabel('maximum Q-value\nin each state (a.u.)')

            axs[0, 0].legend(all_by_label.values(), all_by_label.keys())

    plt.show()


if __name__ == '__main__':
    main()


