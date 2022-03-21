import matplotlib_latex_bridge as mlb
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from bioinfokit.analys import stat
import sys

# set-up figures parameters
mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)

# setting the file you want to open
n_individuals = 100
alpha = 0.78
len_exp_beg = 0  # first iteration to be considered for the plot
len_exp_end = 50  # last iteration to be considered for the plot
label_repl = ['MF-RL\nno replay', 'MF-RL\nbackward replay', 'MF-RL\nshuffled replay']
color_repl = ['blue', 'orange', 'green']
data_dict_repl = ['no replay', 'backward replay', 'random replay']
all_tested_alphas = np.linspace(0, 1, 10)
deterministic_world = [True, False]


fig = mlb.figure_textwidth(widthp=1, height=3)
axs = fig.subplots(1, 2)

# generate learning figure
for idx, wo in enumerate(deterministic_world):

    # reading json file for the performances
    with open(f'data_files/results_det{wo}_ind{n_individuals}_alpha{alpha}_NEWalpha.json', 'rb') as rr:
        res_learning = json.load(rr)

    # build a DataFrame to perform statistical analysis
    act2rew = []
    rt = []
    for trep in res_learning['stat'].keys():
        act2rew = act2rew + res_learning['stat'][trep][1]
        rt = rt + [trep] * len(res_learning['stat'][trep][1])

    df_learn_perf = pd.DataFrame({'act2rew': act2rew, 'rt': rt})

    # one-way ANOVA
    res_stat_learn_perf = stat()
    res_stat_learn_perf.anova_stat(df=df_learn_perf, res_var='act2rew', anova_model='act2rew~ C(rt)')
    print(res_stat_learn_perf.anova_summary)

    ax = axs[idx]

    for idy, tr in enumerate(res_learning['stat'].keys()):
        # plot first, median and third quartile of the results
        ax.plot(np.arange(len_exp_beg, len_exp_end), res_learning['stat'][tr][1][len_exp_beg:len_exp_end],
        c=color_repl[idy], label=label_repl[idy])
        ax.fill_between(np.arange(len_exp_beg, len_exp_end), res_learning['stat'][tr][0][len_exp_beg:len_exp_end],
                         res_learning['stat'][tr][2][len_exp_beg:len_exp_end], color=color_repl[idy], alpha=0.1)

    if wo:
        ax.set_ylabel(r"\# model iteration")

    if not wo:
        ax.set_ylim(all_lims)
        ax.legend(loc='upper right')

    ax.set_xlabel('trial')
    ax.set_yscale("log", base=2)
    all_lims = ax.get_ylim()

plt.show()


# generate comparing alpha figure
fig = mlb.figure_textwidth(widthp=1, height=3)
label_repl = ['MF-RL no replay', 'MF-RL backward replay', 'MF-RL shuffled replay']


def plot_line(data, label, color, ax=None):
    """
    Plot first, median and third quartile of the results

    :param data: 2D list with the data to plot the line
    :param label: string defining the name or the line for the legend in the plot
    :param color: string defining the color of the line in the plot
    :param ax: matplotlib.pyplot.subplots axes for the plot
    """

    alpha = [d[2] for d in data]
    v = [d[0][1] for d in data]
    vstdmin = [d[0][0] for d in data]
    vstdmax = [d[0][2] for d in data]

    ax.plot(alpha, v, label=label, c=color)
    ax.fill_between(alpha, vstdmin, vstdmax, color=color, alpha=0.1)

    ax.set_xlabel(r"$\alpha$")


axs = fig.subplots(1, 2)

for idx, wo in enumerate(deterministic_world):

    # reading json file for the alpha selection
    with open(f'data_files/perf_alpha_ind{n_individuals}_det{wo}_NEW.json', 'rb') as pa:
        alpha_data = json.load(pa)

    stat = alpha_data["stat"]

    ax = axs[idx]

    for i in range(3):
        plot_line(stat[data_dict_repl[i]], label_repl[i], color_repl[i], ax=ax)

        if wo:
            ax.set_ylabel(r"\# model iteration")

        if not wo:
            ax.set_ylim(all_lims)
            ax.legend()

        all_lims = ax.get_ylim()

plt.show()
