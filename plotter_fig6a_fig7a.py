import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import yaml
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from utils.utilities import setup_problem, get_exp_dir
from utils.rl_utilities import get_optimal_policy_mdp, get_policy_value_mdp

ROOT_OUTPUT_DIR = "./output"


def get_custom_output_dir(exp_dir, exp_name, config_name, alg_name, smoothing_param, seed=None):
    if seed is None:
        if smoothing_param is not None:
            default_exp_dir = f'{alg_name}_{smoothing_param}'
        else:
            default_exp_dir = f'{alg_name}'
        return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{config_name}/{exp_name}0/{default_exp_dir}"
    else:
        if smoothing_param is not None:
            default_exp_dir = f'{alg_name}_{smoothing_param}'
        else:
            default_exp_dir = f'{alg_name}'
        return f"{ROOT_OUTPUT_DIR}/{exp_dir}/{config_name}/{exp_name}{seed}/{default_exp_dir}"

####################################
# Plots policy evaluation (||V^k - V^*|| vs lambda)
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config1', help='Path of config file')
    parser.add_argument('config2', help='Path of config file')
    parser.add_argument('config3', help='Path of config file')
    parser.add_argument('config1_dir', help='Path of folder')
    parser.add_argument('config2_dir', help='Path of folder')
    parser.add_argument('config3_dir', help='Path of folder')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    parser.add_argument('--output_file', default=None, help='Output filename')
    parser.add_argument('--alpha_start', default=1, help='Start k to plot alpha plot')
    parser.add_argument('--alpha_stop', default=10, help='End k to plot alpha plot')
    parser.add_argument('--alpha_step', default=2, help='Interval at which to plot alpha plot')
    parser.add_argument('--num_trials', default=1, help='Number of trials to run')
    parser.add_argument('--plot_every', default=1, help='Interval with which to plot')
    args = parser.parse_args()


    # Plotting
    params = {'font.size': 26,
              'axes.labelsize': 26, 'axes.titlesize': 22, 'legend.fontsize': 22,
              'xtick.labelsize': 22, 'ytick.labelsize': 22, 'lines.linewidth': 3, 'axes.linewidth': 2}
    plt.rcParams.update(params)

    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_figwidth(28)
    fig.set_figheight(8)
    fig.set_dpi(300)


###################################
# Plot Maze
###################################

    with open(args.config1, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_pe_vector"
    exp_dir = get_exp_dir(config, "exp_pe_vector", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    policy = get_optimal_policy_mdp(mdp)
    num_iterations = config["exp_pe_exact_num_iterations"]
    alpha_vals = config["exp_pe_exact_alphas"]
    alpha_vals_sensitivity = config["exp_pe_exact_alphas_sensitivity"]
    alpha_vals_sensitivity = list(np.around(np.arange(0, 1, alpha_vals_sensitivity), decimals=2))
    all_alphas = sorted(list(set(alpha_vals + alpha_vals_sensitivity)))

    if config["exp_pe_exact_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_pe_exact_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        assert False

    true_V = get_policy_value_mdp(mdp, policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V
    cmap = plt.get_cmap("tab10")


    start, stop, step = int(args.alpha_start), int(args.alpha_stop), int(args.alpha_step)
    num_plots = 0
    for idx, k in enumerate(range(start, stop, step)):
        plot = []
        for i in range(len(all_alphas)):
            if all_alphas[i] in alpha_vals_sensitivity:
                expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, config_name=args.config1_dir, alg_name="osvi_pe", smoothing_param=all_alphas[i])
                if os.path.isdir(expected_osvi_dir):
                    with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                        v_trace = np.load(f)
                value_errors = np.linalg.norm(true_trace[k, :] - v_trace[k, :], ord=1)
                value_errors = value_errors / np.linalg.norm(true_V, ord=1)
                plot.append(value_errors)

        num_plots += 1
        axs[0].plot(alpha_vals_sensitivity, plot, label=r"$||V_{{{}}} - V^\pi||_1$".format(k),
                 color=cmap(idx))
        plot_every = 5
        axs[0].scatter(alpha_vals_sensitivity[::plot_every], plot[::plot_every],
                    label=r"$||V_{{{}}} - V^\pi||_1$".format(k),
                    color=cmap(idx), marker='o')

    axs[0].set_yscale('log')
    if config["exp_pe_exact_model_type"] == "LocallySmoothed":
        axs[0].set_xlabel(r"Model Smoothing Factor ($\lambda$)")
    else:
        axs[0].set_xlabel(r"Model Self-loop Factor ($\lambda$)")
    axs[0].set_ylabel(r"Normalized $||V_k - V^\pi||_1$")
    axs[0].set_ylim((1e-6, 3))
    axs[0].grid()


###################################
# Plot Garnet
###################################

    with open(args.config2, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    num_instances = config["garnet_instances"]
    plot_all = []
    for num_instance in range(num_instances):

        exp_name = "exp_pe_vector"
        exp_dir = get_exp_dir(config, "exp_pe_vector", exp_dir=args.exp_dir)
        mdp = setup_problem(config, seed=num_instance)
        policy = get_optimal_policy_mdp(mdp)
        num_iterations = config["exp_pe_exact_num_iterations"]
        alpha_vals = config["exp_pe_exact_alphas"]
        alpha_vals_sensitivity = config["exp_pe_exact_alphas_sensitivity"]
        alpha_vals_sensitivity = list(np.around(np.arange(0, 1, alpha_vals_sensitivity), decimals=2))
        all_alphas = sorted(list(set(alpha_vals + alpha_vals_sensitivity)))

        if config["exp_pe_exact_model_type"] == "LocallySmoothed":
            model_class = LocallySmoothedModel
        elif config["exp_pe_exact_model_type"] == "IdentitySmoothed":
            model_class = IdentitySmoothedModel
        else:
            assert False

        true_V = get_policy_value_mdp(mdp, policy)
        true_trace = np.zeros((num_iterations, mdp.num_states()))
        true_trace[np.arange(num_iterations), :] = true_V
        cmap = plt.get_cmap("tab10")

        start, stop, step = int(args.alpha_start), int(args.alpha_stop), int(args.alpha_step)
        num_plots = 0

        plot_per_k = []
        for idx, k in enumerate(range(start, stop, step)):
            plot_per_alpha = []
            for i in range(len(all_alphas)):
                if all_alphas[i] in alpha_vals_sensitivity:
                    expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, config_name=args.config2_dir, alg_name="osvi_pe", smoothing_param=all_alphas[i], seed=num_instance)
                    if os.path.isdir(expected_osvi_dir):
                        with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                            v_trace = np.load(f) #(num iterations, num_states)
                    else:
                        assert False
                    value_errors = np.linalg.norm(true_trace[k, :] - v_trace[k, :], ord=1)
                    value_errors = value_errors / np.linalg.norm(true_V, ord=1)
                    plot_per_alpha.append(value_errors)
            plot_per_k.append(plot_per_alpha)
        plot_all.append(plot_per_k)

    mean = np.array(plot_all).mean(axis=0) #(num_k, num_alpha)
    stderr = np.array(plot_all).std(axis=0) / np.sqrt(num_instances)
    for idx, k in enumerate(range(start, stop, step)):
        axs[1].plot(alpha_vals_sensitivity, mean[idx], label=r"$||V_{{{}}} - V^\pi||_1$".format(k),
                    color=cmap(idx))
        axs[1].scatter(alpha_vals_sensitivity[::plot_every], mean[idx][::plot_every],
                    label=r"$||V_{{{}}} - V^\pi||_1$".format(k),
                    color=cmap(idx), marker='o')            
        axs[1].fill_between(x=alpha_vals_sensitivity,
                            y1=mean[idx] - stderr[idx],
                            y2=mean[idx] + stderr[idx],
                            alpha=0.2, color=cmap(idx))

    axs[1].set_yscale('log')
    if config["exp_pe_exact_model_type"] == "LocallySmoothed":
        axs[1].set_xlabel(r"Model Smoothing Factor ($\lambda$)")
    else:
        axs[1].set_xlabel(r"Model Self-loop Factor ($\lambda$)")
    axs[1].set_ylim(bottom=1e-6)
    axs[1].grid()

###################################
# Plot Cliffwalk
###################################

    with open(args.config3, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_pe_vector"
    exp_dir = get_exp_dir(config, "exp_pe_vector", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    policy = get_optimal_policy_mdp(mdp)
    num_iterations = config["exp_pe_exact_num_iterations"]
    alpha_vals = config["exp_pe_exact_alphas"]
    alpha_vals_sensitivity = config["exp_pe_exact_alphas_sensitivity"]
    alpha_vals_sensitivity = list(np.around(np.arange(0, 1, alpha_vals_sensitivity), decimals=2))
    all_alphas = sorted(list(set(alpha_vals + alpha_vals_sensitivity)))

    if config["exp_pe_exact_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_pe_exact_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        assert False

    true_V = get_policy_value_mdp(mdp, policy)
    true_trace = np.zeros((num_iterations, mdp.num_states()))
    true_trace[np.arange(num_iterations), :] = true_V
    cmap = plt.get_cmap("tab10")


    start, stop, step = int(args.alpha_start), int(args.alpha_stop), int(args.alpha_step)
    num_plots = 0
    for idx, k in enumerate(range(start, stop, step)):
        plot = []
        for i in range(len(all_alphas)):
            if all_alphas[i] in alpha_vals_sensitivity:
                expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, config_name=args.config3_dir, alg_name="osvi_pe", smoothing_param=all_alphas[i])
                if os.path.isdir(expected_osvi_dir):
                    with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                        v_trace = np.load(f)
                value_errors = np.linalg.norm(true_trace[k, :] - v_trace[k, :], ord=1)
                value_errors = value_errors / np.linalg.norm(true_V, ord=1)
                plot.append(value_errors)

        num_plots += 1
        axs[2].plot(alpha_vals_sensitivity, plot, label=r"$||V_{{{}}} - V^\pi||_1$".format(k),
                 color=cmap(idx))
        plot_every = 5
        axs[2].scatter(alpha_vals_sensitivity[::plot_every], plot[::plot_every],
                    label=r"$||V_{{{}}} - V^\pi||_1$".format(k),
                    color=cmap(idx), marker='o')

    if config["exp_pe_exact_model_type"] == "LocallySmoothed":
        axs[2].set_xlabel(r"Model Smoothing Factor ($\lambda$)")
    else:
        axs[2].set_xlabel(r"Model Self-loop Factor ($\lambda$)")
    
    axs[2].set_yscale('log')
    axs[2].set_ylim(bottom=1e-6)
    axs[2].grid()

    handles, labels = axs[0].get_legend_handles_labels()
    order = [2 * i for i in range(num_plots)]
    legend = fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper center", ncol=5,
                        bbox_to_anchor=(0.5, -0.05))


    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)



    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/{args.output_file}", bbox_inches="tight")