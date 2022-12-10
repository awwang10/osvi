import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import yaml
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from utils.utilities import setup_problem, get_exp_dir
from utils.rl_utilities import get_optimal_policy_mdp, get_policy_value, get_policy_value_mdp, get_optimal_policy

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
# Plots control (||V^k - V^*|| vs iterations)
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
    value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
    value_errors = value_errors / np.linalg.norm(true_V, ord=1)
    for i in range(len(alpha_vals) // 2):
        axs[0].scatter(np.arange(num_iterations), value_errors[:plot_end_iter], label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    axs[0].plot(np.arange(num_iterations)[:plot_end_iter], value_errors[:plot_end_iter], label="VI",
                linestyle="dotted", color=cmap(0))
    for i in range(len(alpha_vals) - len(alpha_vals) // 2 - 1):
        axs[0].scatter(np.arange(num_iterations), value_errors[:plot_end_iter], label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    
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

    ## Model VI
    model_VI_by_alpha = []
    for i in range(len(all_alphas)):
        Phat = model_class.get_P_hat_using_P(mdp.P(), all_alphas[i])
        optimal_policy_for_model = get_optimal_policy(Phat, mdp.R(), mdp.discount(), mdp.num_states(),
                                                      mdp.num_actions())
        value = get_policy_value(Phat, mdp.R(), mdp.discount(), optimal_policy_for_model)
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

    axs[0].set_xlabel("Iteration (k)")
    axs[0].set_xticks(np.arange(0, 30, 5))
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"Normalized $||V_k - V^*||_1$")
    axs[0].set_ylim((1e-4, 2))
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
    vi_value_errors_all = []
    osvi_value_errors_all = []
    model_value_errors_all = []
    for num_instance in range(num_instances):
        ## VI
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

        expected_vi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="vi_control",
                                                config_name=args.config2_dir,
                                                smoothing_param=None, seed=num_instance)
        if os.path.isdir(expected_vi_dir):
            with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
            with open(f'{expected_vi_dir}/policy_trace.npy', 'rb') as f:
                policy_trace = np.load(f)
        else:
            assert False

        plot_end_iter = 25
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        value_errors = value_errors / np.linalg.norm(true_V, ord=1)
        vi_value_errors_all.append(value_errors)

        ## OSVI
        Vs_by_alpha = []
        osvi_errors_by_alpha = []
        for i, alpha in enumerate(range(len(all_alphas))):
            expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="osvi_control",
                                                      config_name=args.config2_dir,
                                                      smoothing_param=all_alphas[i], seed=num_instance)
            if os.path.isdir(expected_osvi_dir):
                with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                    v_trace = np.load(f)
                with open(f'{expected_osvi_dir}/policy_trace.npy', 'rb') as f:
                    policy_trace = np.load(f)
            else:
                assert False

            Vs_by_alpha.append(v_trace)
            error = (np.linalg.norm(v_trace-true_trace, ord=1, axis=1) / np.linalg.norm(true_V, ord=1)) #(num_iterations,)
            osvi_errors_by_alpha.append(error)

        osvi_value_errors_all.append(osvi_errors_by_alpha)

        ## Model VI
        model_VI_by_alpha = []
        model_value_errors_by_alpha = []
        for i in range(len(all_alphas)):
            Phat = model_class.get_P_hat_using_P(mdp.P(), all_alphas[i])
            optimal_policy_for_model = get_optimal_policy(Phat, mdp.R(), mdp.discount(), mdp.num_states(),
                                                          mdp.num_actions())
            value = get_policy_value(mdp.P(), mdp.R(), mdp.discount(), optimal_policy_for_model)
            model_VI_by_alpha.append(value)

            model_value_error = np.linalg.norm(model_VI_by_alpha[i] - true_V, ord=1) / np.linalg.norm(true_V, ord=1)
            model_value_errors_by_alpha.append(model_value_error)

        model_value_errors_all.append(model_value_errors_by_alpha)


    cmap = plt.get_cmap("tab10")
    # Plot VI
    mean = np.array(vi_value_errors_all).mean(axis=0)
    stderr = np.array(vi_value_errors_all).std(axis=0) / np.sqrt(num_instances)
    axs[1].plot(np.arange(num_iterations)[:plot_end_iter], mean[:plot_end_iter], label="VI",
                linestyle="dotted", color=cmap(0))

    axs[1].fill_between(x=np.arange(num_iterations),
                        y1=mean[:plot_end_iter] - stderr[:plot_end_iter],
                        y2=mean[:plot_end_iter] + stderr[:plot_end_iter],
                        alpha=0.2, color=cmap(0))

    # Plot OSVI
    mean = np.array(osvi_value_errors_all).mean(axis=0)  # (num_alphas, num_iterations (t))
    stderr = np.array(osvi_value_errors_all).std(axis=0) / np.sqrt(num_instances)
    color = 0
    for i, alpha in enumerate(range(len(all_alphas))):
        if all_alphas[i] in alpha_vals:
            axs[1].plot(np.arange(num_iterations)[:plot_end_iter], mean[i][:plot_end_iter],
                        label=r"OS-VI ($\lambda$ = {alpha})".format(alpha=all_alphas[i]), color=cmap(color+1))
            axs[1].fill_between(x=np.arange(num_iterations)[:plot_end_iter],
                                y1=mean[i][:plot_end_iter] - stderr[i][:plot_end_iter],
                                y2=mean[i][:plot_end_iter] + stderr[i][:plot_end_iter],
                                alpha=0.2, color=cmap(color+1))
            color += 1

    # Plot Model
    mean = np.array(model_value_errors_all).mean(axis=0)
    stderr = np.array(model_value_errors_all).std(axis=0) / np.sqrt(num_instances)
    color = 0
    for i, alpha in enumerate(range(len(all_alphas))):
        if all_alphas[i] in alpha_vals:
            axs[1].plot(np.arange(num_iterations),
                           [mean[i]] * num_iterations,
                           label=r"Model $(\lambda = {alpha})$".format(alpha=all_alphas[i]),
                           linestyle="dashed", color=cmap(color+1))
            axs[1].fill_between(x=np.arange(num_iterations),
                                y1=mean[i] - stderr[i],
                                y2=mean[i] + stderr[i],
                                alpha=0.2, color=cmap(color+1))
            color += 1


    axs[1].set_xlabel("Iteration (k)")
    axs[1].set_xticks(np.arange(0, 30, 5))
    axs[1].set_yscale("log")
    axs[1].set_ylim(bottom=1e-4)
    axs[1].grid()


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
    value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
    value_errors = value_errors / np.linalg.norm(true_V, ord=1)
    axs[2].plot(np.arange(num_iterations)[:plot_end_iter], value_errors[:plot_end_iter], label="VI",
                linestyle="dotted", color=cmap(0))

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
            axs[2].plot(np.arange(num_iterations)[:plot_end_iter],
                        (np.linalg.norm(Vs_by_alpha[i]-true_trace, ord=1, axis=1) / np.linalg.norm(true_V, ord=1)) [:plot_end_iter],
                        label=r"OS-VI ($\lambda$ = {alpha})".format(alpha=all_alphas[i]), color=cmap(color+1))

            y = np.linalg.norm(model_VI_by_alpha[i] - true_V, ord=1) / np.linalg.norm(true_V, ord=1)
            axs[2].axhline(y=y,
                           label=r"Model $(\lambda = {alpha})$".format(alpha=all_alphas[i]),
                           linestyle="dashed", color=cmap(color+1))
            color += 1

    axs[2].set_xlabel("Iteration (k)")
    axs[2].set_xticks(np.arange(0, 30, 5))
    axs[2].set_yscale("log")
    # axs[2].set_ylabel(r"Normalized Error of $V^k$")
    axs[2].set_ylim(bottom=1e-4)
    axs[2].grid()

    handles, labels = axs[0].get_legend_handles_labels()
    order = []
    for i in range(len(alpha_vals)):
        order = order + [2*i + len(alpha_vals), 2*i+1 + len(alpha_vals), i]
    legend = fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper center", ncol=5,
                        bbox_to_anchor=(0.5, -0.05))


    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.1)

    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSVI-Control-Smooth-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifedCliffwalk.pdf", bbox_inches="tight")
    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSVI-Control-Smooth-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifedCliffwalk.png", bbox_inches="tight")