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
# Plots control (||V^k - V^*|| vs lambda)
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

    exp_name = "exp_control_vector"
    exp_dir = get_exp_dir(config, "exp_control_vector", exp_dir=args.exp_dir)
    mdp = setup_problem(config)

    num_iterations = config["exp_control_exact_num_iterations"]
    alpha_vals = config["exp_control_exact_alphas"]
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
    expected_vi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="vi_control",
                                            config_name=args.config1_dir,
                                            smoothing_param=None)
    if os.path.isdir(expected_vi_dir):
        with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
            v_trace = np.load(f)
        with open(f'{expected_vi_dir}/policy_trace.npy', 'rb') as f:
            policy_trace = np.load(f)

    plot_end_iter = 30
    ## OSVI
    Vs_by_alpha = []
    for i, alpha in enumerate(range(len(all_alphas))):
        expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="osvi_control",
                                                  config_name=args.config1_dir,
                                                  smoothing_param=all_alphas[i])
        if os.path.isdir(expected_osvi_dir):
            with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
            with open(f'{expected_osvi_dir}/policy_trace.npy', 'rb') as f:
                policy_trace = np.load(f)

        Vs_by_alpha.append(v_trace)


    start, stop, step = int(args.alpha_start), int(args.alpha_stop), int(args.alpha_step)
    num_plots = 0
    for idx, k in enumerate(range(start, stop, step)):
        plot = []
        for i in range(len(all_alphas)):
            result = np.linalg.norm(true_trace[k, :] - Vs_by_alpha[i][k, :], ord=1) / np.linalg.norm(true_trace[k, :], ord=1)
            plot.append(result)
        axs[0].plot(all_alphas, plot, label=r"$||V_{{{k}}}-V^*||_1$".format(k=k), color=cmap(idx))
        num_plots += 1
        marker_every = 4
        axs[0].scatter(all_alphas[::marker_every], plot[::marker_every],
                    label=r"$||V^{{\pi_{{{k}}}}}-V^*||_1$".format(k=k), color=cmap(idx), marker='o')

    axs[0].set_yscale('log')
    axs[0].grid()
    if config["exp_control_exact_model_type"] == "LocallySmoothed":
        axs[0].set_xlabel(r"Model Smoothing Factor ($\lambda$)")
    else:
        axs[0].set_xlabel(r"Model Self-loop Factor ($\lambda$)")
    # axs[0].set_ylabel(r'Normalized Error of $V_k$')
    axs[0].set_ylabel(r'Normalized $||V_k-V^*||_1$')
    axs[0].set_ylim((1e-6, 3))


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

        exp_name = "exp_control_vector"
        exp_dir = get_exp_dir(config, "exp_control_vector", exp_dir=args.exp_dir)
        mdp = setup_problem(config, seed=num_instance)

        num_iterations = config["exp_control_exact_num_iterations"]
        alpha_vals = config["exp_control_exact_alphas"]
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

        ## OSVI
        Vs_by_alpha = []
        for i, alpha in enumerate(range(len(all_alphas))):
            expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="osvi_control",
                                                      config_name=args.config2_dir,
                                                      smoothing_param=all_alphas[i],
                                                      seed=num_instance)
            if os.path.isdir(expected_osvi_dir):
                with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                    v_trace = np.load(f)
                with open(f'{expected_osvi_dir}/policy_trace.npy', 'rb') as f:
                    policy_trace = np.load(f)
            else:
                assert False

            Vs_by_alpha.append(v_trace)

        start, stop, step = int(args.alpha_start), int(args.alpha_stop), int(args.alpha_step)
        plot_per_k = []
        for idx, k in enumerate(range(start, stop, step)):
            plot = []
            for i in range(len(all_alphas)):
                result = np.linalg.norm(true_trace[k, :] - Vs_by_alpha[i][k, :], ord=1) / np.linalg.norm(true_trace[k, :], ord=1)
                plot.append(result)
            plot_per_k.append(plot)
        plot_all.append(plot_per_k)

    mean = np.array(plot_all).mean(axis=0)  # (num_k, num_alpha)
    stderr = np.array(plot_all).std(axis=0) / np.sqrt(num_instances)

    for idx, k in enumerate(range(start, stop, step)):
        axs[1].plot(all_alphas, mean[idx], label=r"$||V_{{{k}}}-V^*||_1$".format(k=k), color=cmap(idx))
        axs[1].scatter(all_alphas[::marker_every], mean[idx][::marker_every],
                    label=r"$||V^{{\pi_{{{k}}}}}-V^*||_1$".format(k=k), color=cmap(idx), marker='o')

        axs[1].fill_between(x=all_alphas,
                            y1=mean[idx] - stderr[idx],
                            y2=mean[idx] + stderr[idx],
                            alpha=0.2, color=cmap(idx))


    axs[1].set_yscale('log')
    axs[2].set_ylim(bottom=1e-6)
    axs[1].grid()
    if config["exp_control_exact_model_type"] == "LocallySmoothed":
        axs[1].set_xlabel(r"Model Smoothing Factor ($\lambda$)")
    else:
        axs[1].set_xlabel(r"Model Self-loop Factor ($\lambda$)")
    

###################################
# Plot Cliffwalk
###################################

    with open(args.config3, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    exp_name = "exp_control_vector"
    exp_dir = get_exp_dir(config, "exp_control_vector", exp_dir=args.exp_dir)
    mdp = setup_problem(config)

    num_iterations = config["exp_control_exact_num_iterations"]
    alpha_vals = config["exp_control_exact_alphas"]
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
    expected_vi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="vi_control",
                                            config_name=args.config3_dir,
                                            smoothing_param=None)
    if os.path.isdir(expected_vi_dir):
        with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
            v_trace = np.load(f)
        with open(f'{expected_vi_dir}/policy_trace.npy', 'rb') as f:
            policy_trace = np.load(f)

    plot_end_iter = 30
    ## OSVI
    Vs_by_alpha = []
    for i, alpha in enumerate(range(len(all_alphas))):
        expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="osvi_control",
                                                  config_name=args.config3_dir,
                                                  smoothing_param=all_alphas[i])
        if os.path.isdir(expected_osvi_dir):
            with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
            with open(f'{expected_osvi_dir}/policy_trace.npy', 'rb') as f:
                policy_trace = np.load(f)

        Vs_by_alpha.append(v_trace)


    start, stop, step = int(args.alpha_start), int(args.alpha_stop), int(args.alpha_step)
    num_plots = 0
    for idx, k in enumerate(range(start, stop, step)):
        plot = []
        for i in range(len(all_alphas)):
            result = np.linalg.norm(true_trace[k, :] - Vs_by_alpha[i][k, :], ord=1) / np.linalg.norm(true_trace[k, :], ord=1)
            plot.append(result)
        axs[2].plot(all_alphas, plot, label=r"$||V_{{{k}}}-V^*||_1$".format(k=k), color=cmap(idx))
        num_plots += 1
        marker_every = 4
        axs[2].scatter(all_alphas[::marker_every], plot[::marker_every],
                    label=r"$||V^{{\pi_{{{k}}}}}-V^*||_1$".format(k=k), color=cmap(idx), marker='o')

    axs[2].set_yscale('log')
    axs[2].grid()
    if config["exp_control_exact_model_type"] == "LocallySmoothed":
        axs[2].set_xlabel(r"Model Smoothing Factor ($\lambda$)")
    else:
        axs[2].set_xlabel(r"Model Self-loop Factor ($\lambda$)")
    axs[2].set_ylim(bottom=1e-6)




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