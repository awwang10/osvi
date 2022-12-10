import argparse
import os
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.markers import MarkerStyle
import numpy as np
import yaml
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
# Plots control, PE
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

    plot_every = int(args.plot_every)

################################
# PE Sample
################################

    with open(args.config1, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_pe_sample"
    exp_dir = get_exp_dir(config, "exp_pe_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    policy = get_optimal_policy_mdp(mdp)
    num_trials = int(args.num_trials)
    plot_every = int(args.plot_every)
    num_iterations = config["exp_pe_sample_num_iterations"]
    alpha_vals = config["exp_pe_sample_alphas"]


    true_V = get_policy_value_mdp(mdp, policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V

    cmap = plt.get_cmap("tab10")
    i = 0

    value_errors_per_trial = []
    for trial in range(num_trials):
        expected_tdlearning_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                        plot_name=args.plot_name1,
                                                       alg_name="tdlearning_pe", smoothing_param=None)
        if os.path.isdir(expected_tdlearning_dir):
            with open(f'{expected_tdlearning_dir}/V_trace_{trial}.npy', 'rb') as f:
                v_trace = np.load(f)

        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / mdp.num_states()
        value_errors = value_errors / (np.linalg.norm(true_trace, ord=1, axis=1)[0] / mdp.num_states())
        value_errors_per_trial.append(value_errors)

    mean = np.array(value_errors_per_trial).mean(axis=0)
    stderr = np.array(value_errors_per_trial).std(axis=0) / np.sqrt(num_trials)

    for i in range(len(alpha_vals) // 2):
        axs[0].scatter(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    axs[0].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=f"TD-Learning", color=cmap(0), linestyle="dotted")
    for i in range(len(alpha_vals) - len(alpha_vals) // 2 - 1):
        axs[0].scatter(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    
    axs[0].fill_between(x=np.arange(num_iterations)[::plot_every],
                     y1=(mean - stderr)[::plot_every],
                     y2=(mean + stderr)[::plot_every],
                     alpha=0.1, color=cmap(0))
    i += 1

    for idx, alpha in enumerate(alpha_vals):
        value_errors_per_trial = []
        for trial in range(num_trials):
            expected_dyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                      plot_name=args.plot_name1,
                                                       alg_name="dyna_pe", smoothing_param=alpha)
            if os.path.isdir(expected_dyna_dir):
                with open(f'{expected_dyna_dir}/V_trace_{trial}.npy', 'rb') as f:
                    v_trace = np.load(f)

            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / mdp.num_states()
            value_errors = value_errors / (np.linalg.norm(true_trace, ord=1, axis=1)[0] / mdp.num_states())
            value_errors_per_trial.append(value_errors)
        mean = np.array(value_errors_per_trial).mean(axis=0)
        stderr = np.array(value_errors_per_trial).std(axis=0) / np.sqrt(num_trials)
        axs[0].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=r"Dyna ($\lambda = {{{}}}$)".format(alpha),
                 color=cmap(idx+1), linestyle="dashed")
        axs[0].fill_between(x=np.arange(num_iterations)[::plot_every],
                         y1=(mean - stderr)[::plot_every],
                         y2=(mean + stderr)[::plot_every],
                         alpha=0.1, color=cmap(idx+1))
        i += 1

    for idx, alpha in enumerate(alpha_vals):
        value_errors_per_trial = []
        for trial in range(num_trials):
            expected_osdyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                        plot_name=args.plot_name1,
                                  alg_name="osdyna_pe", smoothing_param=alpha)

            if os.path.isdir(expected_osdyna_dir):
                with open(f'{expected_osdyna_dir}/V_trace_{trial}.npy', 'rb') as f:
                    v_trace = np.load(f)
            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / mdp.num_states()
            value_errors = value_errors / (np.linalg.norm(true_trace, ord=1, axis=1)[0] / mdp.num_states())
            value_errors_per_trial.append(value_errors)

        mean = np.array(value_errors_per_trial).mean(axis=0)
        stderr = np.array(value_errors_per_trial).std(axis=0) / np.sqrt(num_trials)
        axs[0].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=r"OS-Dyna PE ($\lambda = {{{}}}$)".format(alpha),
                 color=cmap(idx+1))
        axs[0].fill_between(x=np.arange(num_iterations)[::plot_every],
                         y1=(mean - stderr)[::plot_every],
                         y2=(mean + stderr)[::plot_every],
                         alpha=0.1, color=cmap(idx+1))
        i += 1

    axs[0].set_xlabel("Environment Samples (t)")
    axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[0].set_yscale("log")
    axs[0].grid()
    axs[0].set_ylabel(r'Normalized $||V_t-V^\pi||_1$')


################################
# PE Sample 2
################################

    with open(args.config2, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_pe_sample"
    exp_dir = get_exp_dir(config, "exp_pe_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    policy = get_optimal_policy_mdp(mdp)
    num_trials = int(args.num_trials)
    num_iterations = config["exp_pe_sample_num_iterations"]
    alpha_vals = config["exp_pe_sample_alphas"]

    true_V = get_policy_value_mdp(mdp, policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V

    cmap = plt.get_cmap("tab10")
    i = 0

    value_errors_per_trial = []
    for trial in range(num_trials):
        expected_tdlearning_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                        plot_name=args.plot_name2,
                                                        alg_name="tdlearning_pe", smoothing_param=None)
        if os.path.isdir(expected_tdlearning_dir):
            with open(f'{expected_tdlearning_dir}/V_trace_{trial}.npy', 'rb') as f:
                v_trace = np.load(f)

        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / mdp.num_states()
        value_errors = value_errors / (np.linalg.norm(true_trace, ord=1, axis=1)[0] / mdp.num_states())
        value_errors_per_trial.append(value_errors)

    mean = np.array(value_errors_per_trial).mean(axis=0)
    stderr = np.array(value_errors_per_trial).std(axis=0) / np.sqrt(num_trials)
    axs[1].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=f"TD-Learning", color=cmap(0), linestyle="dotted")
    axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                        y1=(mean - stderr)[::plot_every],
                        y2=(mean + stderr)[::plot_every],
                        alpha=0.1, color=cmap(0))
    i += 1

    for idx, alpha in enumerate(alpha_vals):
        value_errors_per_trial = []
        for trial in range(num_trials):
            expected_dyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                      plot_name=args.plot_name2,
                                                      alg_name="dyna_pe", smoothing_param=alpha)
            if os.path.isdir(expected_dyna_dir):
                with open(f'{expected_dyna_dir}/V_trace_{trial}.npy', 'rb') as f:
                    v_trace = np.load(f)

            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / mdp.num_states()
            value_errors = value_errors / (np.linalg.norm(true_trace, ord=1, axis=1)[0] / mdp.num_states())
            value_errors_per_trial.append(value_errors)
        mean = np.array(value_errors_per_trial).mean(axis=0)
        stderr = np.array(value_errors_per_trial).std(axis=0) / np.sqrt(num_trials)
        axs[1].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=r"Dyna ($\lambda = {{{}}}$)".format(alpha),
                    color=cmap(idx+1), linestyle="dashed")
        axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1, color=cmap(idx+1))
        i += 1

    for idx, alpha in enumerate(alpha_vals):
        value_errors_per_trial = []
        for trial in range(num_trials):
            expected_osdyna_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name,
                                                        plot_name=args.plot_name2,
                                                        alg_name="osdyna_pe", smoothing_param=alpha)

            if os.path.isdir(expected_osdyna_dir):
                with open(f'{expected_osdyna_dir}/V_trace_{trial}.npy', 'rb') as f:
                    v_trace = np.load(f)
            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1) / mdp.num_states()
            value_errors = value_errors / (np.linalg.norm(true_trace, ord=1, axis=1)[0] / mdp.num_states())
            value_errors_per_trial.append(value_errors)

        mean = np.array(value_errors_per_trial).mean(axis=0)
        stderr = np.array(value_errors_per_trial).std(axis=0) / np.sqrt(num_trials)
        axs[1].plot(np.arange(num_iterations)[::plot_every], mean[::plot_every], label=r"OS-Dyna PE ($\lambda = {{{}}}$)".format(alpha),
                    color=cmap(idx+1))
        axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1, color=cmap(idx+1))
        i += 1

    axs[1].set_xlabel("Environment Samples (t)")
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[1].set_yscale("log")
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

    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSDyna-PE-Left=Constant-Right=RescaledLinear-ModifiedCliffwalk.pdf", bbox_inches="tight")
    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSDyna-PE-Left=Constant-Right=RescaledLinear-ModifiedCliffwalk.png", bbox_inches="tight")