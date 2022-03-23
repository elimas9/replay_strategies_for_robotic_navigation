import scikit_posthocs as sp
import matplotlib_latex_bridge as mlb
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from matplotlib.lines import Line2D
from functions import NpEncoder

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11, usetex=False)


def main():
    # *** to be set based on the desired plot *** #
    all_plots_stats_tables = False  # global plot of the significancy of the different results
    legend = False  # activate p-values legend for the plot above
    x_axis = True  # activate x axis for the plot above
    save_sign_pvalues = False  # save .json file with the statistical results
    plot_overall_statistics = True  # plot all the confusion matrices (not so useful atm)

    deterministic_world = [True, False]
    types_replay = ['MF-RL no replay', 'MF-RL backward replay', 'MF-RL shaffled replay']

    fig = mlb.figure_textwidth(widthp=1, height=1.5)
    axs = fig.subplots(1, 2)

    for idx, det_world in enumerate(deterministic_world):
        ax = axs[idx]

        # open the .json files with the results
        with open(f"data_files/all_results_det{det_world}_ind100_alpha0.78.json", "rb") as qvs:
            all_results = json.load(qvs)

        # create the proper data structure for the statistical analysis
        stat_sign = {}
        for trial in range(len(all_results[str(0)][0])):  # trials
            trial_ds_data = []
            trial_ds_label = []
            for rt in all_results.keys():  # replay types
                for id_ind in range(len(all_results[rt])):  # individuals
                    trial_ds_data.append(all_results[rt][id_ind][trial])
                    if rt == '0':
                        trial_ds_label.append('MF-RL no replay')
                    elif rt == '1':
                        trial_ds_label.append('MF-RL backward replay')
                    else:
                        trial_ds_label.append('MF-RL shaffled replay')

            trial_df = pd.DataFrame({'ind': trial_ds_data, 'replay_type': trial_ds_label})

            # Conover posthoc test
            trial_c_ph_t = sp.posthoc_conover(trial_df, val_col='ind', group_col='replay_type')  # , p_adjust='holm')

            # print the result of the statistical analysis
            # print(f"trial #{trial + 1} --> Conover posthoc test: {trial_c_ph_t}")

            # organise the significant p-values
            stat_sign[str(trial + 1)] = {}
            for tr1 in types_replay:  # first replay types
                stat_sign[str(trial + 1)][tr1] = {}
                for tr2 in types_replay:  # second replay types
                    if trial_c_ph_t[tr1][tr2] < 0.001:
                        stat_sign[str(trial + 1)][tr1][tr2] = '***'
                    elif 0.01 > trial_c_ph_t[tr1][tr2] >= 0.001:
                        stat_sign[str(trial + 1)][tr1][tr2] = '**'
                    elif 0.05 > trial_c_ph_t[tr1][tr2] >= 0.01:
                        stat_sign[str(trial + 1)][tr1][tr2] = '*'
                    else:
                        stat_sign[str(trial + 1)][tr1][tr2] = 'NS'

            if all_plots_stats_tables:
                mlb.figure_textwidth()
                heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                                'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3, ]}
                plt.rc('figure.constrained_layout', use=True)
                sp.sign_plot(trial_c_ph_t, **heatmap_args)
                plt.show()

            if save_sign_pvalues:
                with open(f'data_files/stats_det{det_world}.json', 'w') as st:
                    json.dump(stat_sign, st, cls=NpEncoder)

        if plot_overall_statistics:
            colo = {}
            all_colors = {'backward replay - shaffled replay': [], 'no replay - backward replay': [],
                          'no replay - shaffled replay': []}

            for tri in stat_sign.keys():  # trial
                colo[tri] = {}

                # no replay vs backward replay
                if stat_sign[tri]['MF-RL no replay']['MF-RL backward replay'] == "***":
                    colo[tri]['no replay - backward replay'] = "purple"
                elif stat_sign[tri]['MF-RL no replay']['MF-RL backward replay'] == "**":
                    colo[tri]['no replay - backward replay'] = "red"
                elif stat_sign[tri]['MF-RL no replay']['MF-RL backward replay'] == "*":
                    colo[tri]['no replay - backward replay'] = "pink"
                else:
                    colo[tri]['no replay - backward replay'] = "white"

                all_colors['no replay - backward replay'].append(colo[tri]['no replay - backward replay'])

                # no replay vs shaffled replay
                if stat_sign[tri]['MF-RL no replay']['MF-RL shaffled replay'] == "***":
                    colo[tri]['no replay - shaffled replay'] = "purple"
                elif stat_sign[tri]['MF-RL no replay']['MF-RL shaffled replay'] == "**":
                    colo[tri]['no replay - shaffled replay'] = "red"
                elif stat_sign[tri]['MF-RL no replay']['MF-RL shaffled replay'] == "*":
                    colo[tri]['no replay - shaffled replay'] = "pink"
                else:
                    colo[tri]['no replay - shaffled replay'] = "white"

                all_colors['no replay - shaffled replay'].append(colo[tri]['no replay - shaffled replay'])

                # backward replay vs shaffled replay
                if stat_sign[tri]['MF-RL backward replay']['MF-RL shaffled replay'] == "***":
                    colo[tri]['backward replay - shaffled replay'] = "purple"
                elif stat_sign[tri]['MF-RL backward replay']['MF-RL shaffled replay'] == "**":
                    colo[tri]['backward replay - shaffled replay'] = "red"
                elif stat_sign[tri]['MF-RL backward replay']['MF-RL shaffled replay'] == "*":
                    colo[tri]['backward replay - shaffled replay'] = "pink"
                else:
                    colo[tri]['backward replay - shaffled replay'] = "white"

                all_colors['backward replay - shaffled replay'].append(colo[tri]['backward replay - shaffled replay'])

            # plot
            ax.scatter(np.arange(len(stat_sign)), [2] * len(stat_sign),
                        c=all_colors['backward replay - shaffled replay'], marker='_',
                       label='backward replay vs shaffled replay')
            ax.scatter(np.arange(len(stat_sign)), [4] * len(stat_sign),
                        c=all_colors['no replay - shaffled replay'], marker='_', label='no replay vs shaffled replay')
            ax.scatter(np.arange(len(stat_sign)), [6] * len(stat_sign),
                        c=all_colors['no replay - backward replay'], marker='_', label='no replay vs backward replay')

            # adding labels
            if idx == 0:
                ax.text(1, 2.5, 'backward replay vs shaffled replay')
                ax.text(1, 4.5, 'no replay vs shaffled replay')
                ax.text(1, 6.5, 'no replay vs backward replay')

            # customized legend
            if legend:
                custom_lines = [Line2D([0], [0], color="purple", lw=4),
                                Line2D([0], [0], color="red", lw=4),
                                Line2D([0], [0], color="pink", lw=4)]
                plt.legend(custom_lines, ['p-value < 0.001', 'p-value < 0.01', 'p-value < 0.05'], framealpha=1)
                # plt.legend(custom_lines, ['***', '**', '*'], framealpha=1)

            if x_axis:
                ax.axes.get_yaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
            else:
                ax.axis("off")

    plt.show()


if __name__ == '__main__':
    main()

