import argparse
import os
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from utils.utilities import setup_problem, get_exp_dir
from utils.rl_utilities import get_optimal_policy_mdp, get_policy_value, get_policy_value_mdp, get_optimal_policy

from tueplots import figsizes, fontsizes

ROOT_OUTPUT_DIR = "./output"

def get_custom_output_dir_sample(exp_dir, exp_name, plot_name, alg_name, smoothing_param):
    if smoothing_param is not None:
        default_exp_dir = f'{alg_name}_{smoothing_param}'
    else:
        default_exp_dir = f'{alg_name}'
    return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{plot_name}/{default_exp_dir}"

def get_custom_output_dir_vector(exp_dir, exp_name, plot_name, config_name, alg_name, smoothing_param, seed=None):
    if seed is None:
        if smoothing_param is not None:
            default_exp_dir = f'{alg_name}_{smoothing_param}'
        else:
            default_exp_dir = f'{alg_name}'
        return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{config_name}/{default_exp_dir}"
    else:
        if smoothing_param is not None:
            default_exp_dir = f'{alg_name}_{smoothing_param}'
        else:
            default_exp_dir = f'{alg_name}'
        return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{config_name}/{default_exp_dir}"


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
    parser.add_argument('--num_alphas', default=3, type=int, help='Number of alphas to plot')
    args = parser.parse_args()

# Plotting
    plt.rcParams.update(figsizes.neurips2022(height_to_width_ratio=0.7, ncols=2))
    plt.rcParams.update(fontsizes.neurips2022())
    plt.rcParams.update({'legend.fontsize': 6, 'lines.linewidth': 1, 'axes.labelsize': 7})

    fig, axs = plt.subplots(1, 2)

################################
# Control Vector
################################

    with open(args.config1, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_control_vector"
    exp_dir = get_exp_dir(config, "exp_control_vector", exp_dir=args.exp_dir)
    mdp = setup_problem(config)

    num_iterations = config["exp_control_exact_num_iterations"]
    alpha_vals = config["exp_control_exact_alphas"][:args.num_alphas]
    alpha_vals_sensitivity = config["exp_control_exact_alphas_sensitivity"]
    alpha_val_sensitivity = list(np.around(np.arange(0, 1, alpha_vals_sensitivity), decimals=2))
    all_alphas = sorted(list(set(alpha_vals + alpha_val_sensitivity)))

    if config["exp_control_exact_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_control_exact_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        assert False

    cmap = plt.get_cmap("tab10")
    i = 0

    # Get true V for mdp
    optimal_policy = get_optimal_policy_mdp(mdp)
    true_V = get_policy_value_mdp(mdp, optimal_policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V

    ## VI
    expected_vi_dir = get_custom_output_dir_vector(exp_dir=args.exp_dir, exp_name=exp_name, plot_name=args.plot_name1, alg_name="vi_control",
                                            config_name=args.plot_name1,
                                            smoothing_param=None)


    if os.path.isdir(expected_vi_dir):
        with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
            v_trace = np.load(f)
        with open(f'{expected_vi_dir}/policy_trace.npy', 'rb') as f:
            policy_trace = np.load(f)
            
    vpi_trace = np.zeros(policy_trace.shape)
    for k in range(num_iterations):
        vpi_trace[k, :] = get_policy_value(mdp.P(), mdp.R(), mdp.discount(), policy_trace[k, :].astype(int))

    plot_end_iter = 30
    value_errors = np.linalg.norm(vpi_trace - true_trace, ord=1, axis=1)
    value_errors = value_errors / np.linalg.norm(true_V, ord=1)
    axs[0].plot(np.arange(num_iterations)[:plot_end_iter], value_errors[:plot_end_iter], label="VI",
                linestyle="dotted", color=cmap(0))

    ## OSVI
    Vs_by_alpha = []
    for i, alpha in enumerate(range(len(all_alphas))):
        expected_osvi_dir = get_custom_output_dir_vector(exp_dir=args.exp_dir, exp_name=exp_name, plot_name=args.plot_name1, alg_name="osvi_control",
                                                  config_name=args.plot_name1,
                                                  smoothing_param=all_alphas[i])


        if os.path.isdir(expected_osvi_dir):
            with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
            with open(f'{expected_osvi_dir}/policy_trace.npy', 'rb') as f:
                policy_trace = np.load(f)

        vpi_trace = np.zeros(policy_trace.shape)
        for k in range(num_iterations):
            vpi_trace[k, :] = get_policy_value(mdp.P(), mdp.R(), mdp.discount(), policy_trace[k, :].astype(int))                
        Vs_by_alpha.append(vpi_trace)

    ## Model VI
    model_VI_by_alpha = []
    for i in range(len(all_alphas)):
        Phat = model_class.get_P_hat_using_P(mdp.P(), all_alphas[i])
        optimal_policy_for_model = get_optimal_policy(Phat, mdp.R(), mdp.discount(), mdp.num_states(),
                                                      mdp.num_actions())
        value = get_policy_value(mdp.P(), mdp.R(), mdp.discount(), optimal_policy_for_model)
        model_VI_by_alpha.append(value)

    cmap = plt.get_cmap("tab10")

    color = 0
    for i, alpha in enumerate(range(len(all_alphas))):
        if all_alphas[i] in alpha_vals:
            axs[0].plot(np.arange(num_iterations)[:plot_end_iter],
                        (np.linalg.norm(Vs_by_alpha[i]-true_trace, ord=1, axis=1) / np.linalg.norm(true_V, ord=1)) [:plot_end_iter],
                        label=r"OS-VI ($\lambda$ = {alpha})".format(alpha=all_alphas[i]), color=cmap(color+1))

            y = np.linalg.norm(model_VI_by_alpha[i] - true_V, ord=1) / np.linalg.norm(true_V, ord=1)
            axs[0].axhline(y=y,
                           label=r"Model $(\lambda = {alpha})$".format(alpha=all_alphas[i]),
                           linestyle="dashed", color=cmap(color+1))
            color += 1

    handles, labels = axs[0].get_legend_handles_labels()
    order = [2 * i for i in range(args.num_alphas + 1)] + [2 * i + 1 for i in range(args.num_alphas)]
    legend = axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], )

    axs[0].set_xlabel("Iterations ($k$)")
    axs[0].set_xticks(np.arange(0, 30, 5))
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"Normalized $|| V^{\pi_k} - V^* ||_1$")
    axs[0].set_ylim(bottom=1e-4)
    axs[0].grid(alpha=0.3)
    
################################
# Control Sample
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
    alpha_vals = config["exp_control_sample_alphas"][:args.num_alphas]
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
        expected_qlearning_dir = get_custom_output_dir_sample(exp_dir=args.exp_dir, exp_name=exp_name,
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
        expected_dyna_dir = get_custom_output_dir_sample(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="dyna_control",
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
        expected_osdyna_dir = get_custom_output_dir_sample(exp_dir=args.exp_dir, exp_name=exp_name,
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
                    color=cmap(idx+1), linewidth='1')
        axs[1].fill_between(x=np.arange(num_iterations)[::plot_every],
                            y1=(mean - stderr)[::plot_every],
                            y2=(mean + stderr)[::plot_every],
                            alpha=0.1,
                            color=cmap(idx+1))
        i += 1

    handles, labels = axs[1].get_legend_handles_labels()
    order = [i for i in range(2 * args.num_alphas + 1)]
    legend = axs[1].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    axs[1].set_xlabel("Environment Samples ($t$)")
    axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}$'.format(x/1000) + 'k'))
    axs[1].set_ylabel(r'$V^{\pi_t}(0)$')
    axs[1].set_ylim(bottom=-125)
    axs[1].grid(alpha=0.3)

    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSVIAndOSDyna-Control-Left=Vector-Right=SampleConstantDelay-ModifiedCliffwalk.pdf")

    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSVIAndOSDyna-Control-Left=Vector-Right=SampleConstantDelay-ModifiedCliffwalk.png")