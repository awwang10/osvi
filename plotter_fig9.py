import argparse
import os
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.markers import MarkerStyle
import numpy as np
import yaml
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from utils.utilities import setup_problem, get_exp_dir
from utils.rl_utilities import get_optimal_policy_mdp, get_policy_value_mdp

ROOT_OUTPUT_DIR = "./output"

def get_custom_output_dir(exp_dir, exp_name, plot_name, alg_name, smoothing_param):
    if smoothing_param is not None:
        default_exp_dir = f'{alg_name}_{smoothing_param}'
    else:
        default_exp_dir = f'{alg_name}'
    return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{plot_name}/{default_exp_dir}"

####################################
# Plots control, sample
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config1', help='Path of config file')
    parser.add_argument('config2', help='Path of config file')
    parser.add_argument('plot_name1', help='Path of folder')
    parser.add_argument('plot_name2', help='Path of folder')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    parser.add_argument('--num_trials', default=1, help='Number of trials to run')
    parser.add_argument('--plot_every', default=1, help='Interval with which to plot')
    args = parser.parse_args()


    # Plotting
    params = {'font.size': 26,
              'axes.labelsize': 26, 'axes.titlesize': 22, 'legend.fontsize': 22,
              'xtick.labelsize': 22, 'ytick.labelsize': 22, 'lines.linewidth': 3, 'axes.linewidth': 2}
    plt.rcParams.update(params)

    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(28)
    fig.set_figheight(8)
    fig.set_dpi(300)


################################
# Control Sample 1
################################

    with open(args.config1, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_control_sample"
    exp_dir = get_exp_dir(config, "exp_control_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    num_iterations = config["exp_control_sample_num_iterations"]
    alpha_vals = config["exp_control_sample_alphas"]
    num_trials = int(args.num_trials)
    plot_every = int(args.plot_every)

    if config["exp_control_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_control_sample_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        assert False

    optimal_policy = get_optimal_policy_mdp(mdp)
    true_V = get_policy_value_mdp(mdp, optimal_policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V

    cmap = plt.get_cmap("tab10")
    i = 0

    ################################
    # V_pi_k
    ################################

    value_per_trial = []
    for trial in range(num_trials):
        expected_qlearning_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                       plot_name=args.plot_name1,
                                                       alg_name="qlearning_control", smoothing_param=None)
        if os.path.isdir(expected_qlearning_dir):
            with open(f'{expected_qlearning_dir}/value_trace_{trial}.npy', 'rb') as f:
                value_trace = np.load(f)
            value_per_trial.append(value_trace.dot(mdp.initial_dist()))

    mean = np.array(value_per_trial).mean(axis=0)
    stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

    for i in range(len(alpha_vals) // 2):
        axs[0].scatter(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    axs[0].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=r"Q-Learning",
                color=cmap(0), linestyle="dotted")
    for i in range(len(alpha_vals) - len(alpha_vals) // 2 - 1):
        axs[0].scatter(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    axs[0].fill_between(x=np.arange(num_iterations)[::plot_every],
                        y1=(mean - stderr)[::plot_every],
                        y2=(mean + stderr)[::plot_every],
                        alpha=0.1,
                        color=cmap(0)
                        )
    i += 1

    for idx, alpha in enumerate(alpha_vals):
        expected_dyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="dyna_control",
                                                  plot_name=args.plot_name1,
                                                  smoothing_param=alpha)
        value_per_trial = []
        for trial in range(num_trials):
            if os.path.isdir(expected_dyna_dir):
                with open(f'{expected_dyna_dir}/value_trace_{trial}.npy', 'rb') as f:
                    value_trace = np.load(f)
                value_per_trial.append(value_trace.dot(mdp.initial_dist()))

        mean = np.array(value_per_trial).mean(axis=0)
        stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

        axs[0].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every],
                    label=r"Dyna ($\lambda={{{}}}$)".format(alpha), color=cmap(idx+1), linestyle="dashed")
        axs[0].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1,
                            color=cmap(idx+1))
        i += 1

    for idx, alpha in enumerate(alpha_vals):
        expected_osdyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                    plot_name=args.plot_name1,
                                                    alg_name="osdyna_control", smoothing_param=alpha)
        value_per_trial = []
        for trial in range(num_trials):
            if os.path.isdir(expected_osdyna_dir):
                with open(f'{expected_osdyna_dir}/value_trace_{trial}.npy', 'rb') as f:
                    value_trace = np.load(f)
                    value_per_trial.append(value_trace.dot(mdp.initial_dist()))

        mean = np.array(value_per_trial).mean(axis=0)
        stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

        axs[0].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every],
                    label=r"OS-Dyna  ($\lambda={{{}}}$)".format(alpha),
                    color=cmap(idx+1))
        axs[0].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1,
                            color=cmap(idx+1))
        i += 1

    axs[0].set_xlabel("Environment Samples (t)")
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[0].set_ylabel(r'$V^{\pi_t}(0)$')
    axs[0].grid()




################################
# Control Sample 2
################################


    with open(args.config2, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_control_sample"
    exp_dir = get_exp_dir(config, "exp_control_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    num_iterations = config["exp_control_sample_num_iterations"]
    alpha_vals = config["exp_control_sample_alphas"]
    num_trials = int(args.num_trials)
    plot_every = int(args.plot_every)

    if config["exp_control_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_control_sample_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        assert False

    optimal_policy = get_optimal_policy_mdp(mdp)
    true_V = get_policy_value_mdp(mdp, optimal_policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V

    cmap = plt.get_cmap("tab10")
    i = 0

    ################################
    # V_pi_k
    ################################

    value_per_trial = []
    for trial in range(num_trials):
        expected_qlearning_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                       plot_name=args.plot_name2,
                                                       alg_name="qlearning_control", smoothing_param=None)
        if os.path.isdir(expected_qlearning_dir):
            with open(f'{expected_qlearning_dir}/value_trace_{trial}.npy', 'rb') as f:
                value_trace = np.load(f)
            value_per_trial.append(value_trace.dot(mdp.initial_dist()))

    mean = np.array(value_per_trial).mean(axis=0)
    stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

    axs[1].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=r"Q-Learning",
                color=cmap(0), linestyle="dotted")
    axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                        y1=(mean - stderr)[::plot_every],
                        y2=(mean + stderr)[::plot_every],
                        alpha=0.1,
                        color=cmap(0)
                        )
    i += 1

    for idx, alpha in enumerate(alpha_vals):
        expected_dyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="dyna_control",
                                                  plot_name=args.plot_name2,
                                                  smoothing_param=alpha)
        value_per_trial = []
        for trial in range(num_trials):
            if os.path.isdir(expected_dyna_dir):
                with open(f'{expected_dyna_dir}/value_trace_{trial}.npy', 'rb') as f:
                    value_trace = np.load(f)
                value_per_trial.append(value_trace.dot(mdp.initial_dist()))

        mean = np.array(value_per_trial).mean(axis=0)
        stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

        axs[1].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every],
                    label=r"Dyna ($\lambda={{{}}}$)".format(alpha), color=cmap(idx+1), linestyle="dashed")
        axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1,
                            color=cmap(idx+1))
        i += 1

    for idx, alpha in enumerate(alpha_vals):
        expected_osdyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                    plot_name=args.plot_name2,
                                                    alg_name="osdyna_control", smoothing_param=alpha)
        value_per_trial = []
        for trial in range(num_trials):
            if os.path.isdir(expected_osdyna_dir):
                with open(f'{expected_osdyna_dir}/value_trace_{trial}.npy', 'rb') as f:
                    value_trace = np.load(f)
                    value_per_trial.append(value_trace.dot(mdp.initial_dist()))

        mean = np.array(value_per_trial).mean(axis=0)
        stderr = np.array(value_per_trial).std(axis=0) / np.sqrt(num_trials)

        axs[1].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every],
                    label=r"OS-Dyna  ($\lambda={{{}}}$)".format(alpha),
                    color=cmap(idx+1))
        axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1,
                            color=cmap(idx+1))
        i += 1

    axs[1].set_xlabel("Environment Samples (t)")
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[1].grid()


    handles, labels = axs[0].get_legend_handles_labels()
    order = []
    for i in range(len(alpha_vals)):
        order = order + [i + len(alpha_vals), i + 2 * len(alpha_vals), i]
    legend = fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper center", ncol=5,
                        bbox_to_anchor=(0.5, -0.05))

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)

    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSDyna-Control-Left=ConstantDelay-Right=RescaledLinear-ModifiedCliffwalk.pdf", bbox_inches="tight")
    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSDyna-Control-Left=ConstantDelay-Right=RescaledLinear-ModifiedCliffwalk.png", bbox_inches="tight")