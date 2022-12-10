import argparse, os, yaml
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from utils.utilities import setup_problem, get_exp_dir
from utils.rl_utilities import get_optimal_policy_mdp, get_policy_value, get_policy_value_mdp

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
# Plots policy evaluation (||V^k - V^*|| vs iterations)
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

    ## VI
    expected_vi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, config_name=args.config1_dir, alg_name="vi_pe",
                                            smoothing_param=None)
    if os.path.isdir(expected_vi_dir):
        with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
            v_trace = np.load(f)
    value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
    value_errors = value_errors / np.linalg.norm(true_V, ord=1)
    for i in range(len(alpha_vals) // 2):
        axs[0].scatter(np.arange(num_iterations), value_errors, label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    axs[0].plot(np.arange(num_iterations), value_errors, label="VI", linestyle="dotted", color=cmap(0))
    for i in range(len(alpha_vals) - len(alpha_vals) // 2 - 1):
        axs[0].scatter(np.arange(num_iterations), value_errors, label=" ", marker=MarkerStyle(marker=" ", fillstyle="none"))
    
    ## OSVI_PE
    for i in range(len(alpha_vals)):
        expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, config_name=args.config1_dir, alg_name="osvi_pe",
                                                  smoothing_param=alpha_vals[i])
        if os.path.isdir(expected_osvi_dir):
            with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        value_errors = value_errors / np.linalg.norm(true_V, ord=1)
        axs[0].plot(np.arange(num_iterations), value_errors, label=r"OS-VI ($\lambda$ = {})".format(alpha_vals[i]),
                    color=cmap(i+1))

    ## Model VI
    for i in range(len(alpha_vals)):
        Phat = model_class.get_P_hat_using_P(mdp.P(), alpha_vals[i])
        value = get_policy_value(Phat, mdp.R(), mdp.discount(), policy)
        value_error = np.linalg.norm(value - true_V, ord=1)
        value_error = value_error / np.linalg.norm(true_V, ord=1)
        axs[0].axhline(y=value_error, label=r"Model ($\lambda$ = {})".format(alpha_vals[i]),
                       linestyle="dashed", color=cmap(i+1))

    axs[0].set_xlabel("Iteration (k)")
    axs[0].set_ylabel(r"Normalized $||V_k - V^\pi||_1$")
    axs[0].set_yscale("log")
    axs[0].set_ylim(bottom=1e-4, top=1e1)
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

        ## VI
        expected_vi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=f"{exp_name}", config_name=args.config2_dir,
                                                alg_name="vi_pe",
                                                smoothing_param=None, seed=num_instance)
        if os.path.isdir(expected_vi_dir):
            with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
        else:
            assert False

        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        value_errors = value_errors / np.linalg.norm(true_V, ord=1)

        vi_value_errors_all.append(value_errors)

        ## OSVI_PE
        osvi_value_errors = []
        for i in range(len(alpha_vals)):
            expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=f"{exp_name}", alg_name="osvi_pe",
                                                      config_name=args.config2_dir,
                                                      smoothing_param=alpha_vals[i], seed=num_instance)
            if os.path.isdir(expected_osvi_dir):
                with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                    v_trace = np.load(f)
            else:
                assert False
            value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
            value_errors = value_errors / np.linalg.norm(true_V, ord=1)
            osvi_value_errors.append(value_errors)
        osvi_value_errors_all.append(osvi_value_errors)

        ## Model VI
        model_value_errors = []
        for i in range(len(alpha_vals)):
            Phat = model_class.get_P_hat_using_P(mdp.P(), alpha_vals[i])
            value = get_policy_value(Phat, mdp.R(), mdp.discount(), policy)
            value_error = np.linalg.norm(value - true_V, ord=1)
            value_error = value_error / np.linalg.norm(true_V, ord=1)
            model_value_errors.append(value_error)

        model_value_errors_all.append(model_value_errors)

    #VI plot
    mean = np.array(vi_value_errors_all).mean(axis=0)
    stderr = np.array(vi_value_errors_all).std(axis=0) / np.sqrt(num_instances)
    axs[1].plot(np.arange(num_iterations), mean, label="VI", linestyle="dotted", color=cmap(0))
    axs[1].fill_between(x=np.arange(num_iterations),
                        y1=mean - stderr,
                        y2=mean + stderr,
                        alpha=0.2, color=cmap(0))

    # OSVI plot
    mean = np.array(osvi_value_errors_all).mean(axis=0)
    stderr = np.array(osvi_value_errors_all).std(axis=0) / np.sqrt(num_instances)
    for i in range(len(alpha_vals)):
        axs[1].plot(np.arange(num_iterations), mean[i], label=r"OS-VI ($\lambda$ = {})".format(alpha_vals[i]),
                        color=cmap(i+1))
        axs[1].fill_between(x=np.arange(num_iterations),
                            y1=mean[i] - stderr[i],
                            y2=mean[i] + stderr[i],
                            alpha=0.2, color=cmap(i+1))

    # Model plot
    mean = np.array(model_value_errors_all).mean(axis=0)
    stderr = np.array(model_value_errors_all).std(axis=0) / np.sqrt(num_instances)
    for i in range(len(alpha_vals)):
        axs[1].axhline(y=mean[i], label=r"Model ($\lambda$ = {})".format(alpha_vals[i]),
                       linestyle="dashed", color=cmap(i+1))
        axs[1].fill_between(x=np.arange(num_iterations),
                            y1=mean[i] - stderr[i],
                            y2=mean[i] + stderr[i],
                            alpha=0.2, color=cmap(i+1))

    axs[1].set_xlabel("Iteration (k)")
    axs[1].set_yscale("log")
    axs[1].set_ylim(bottom=1e-4, top=1e1)
    axs[1].grid()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.2)


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

    ## VI
    expected_vi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="vi_pe",
                                            config_name=args.config3_dir,
                                            smoothing_param=None)
    if os.path.isdir(expected_vi_dir):
        with open(f'{expected_vi_dir}/V_trace.npy', 'rb') as f:
            v_trace = np.load(f)
    value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
    value_errors = value_errors / np.linalg.norm(true_V, ord=1)
    axs[2].plot(np.arange(num_iterations), value_errors, label="VI", linestyle="dotted", color=cmap(0))

    ## OSVI_PE
    for i in range(len(alpha_vals)):
        expected_osvi_dir = get_custom_output_dir(exp_dir=args.exp_dir, exp_name=exp_name, alg_name="osvi_pe",
                                                  config_name=args.config3_dir,
                                                  smoothing_param=alpha_vals[i])
        if os.path.isdir(expected_osvi_dir):
            with open(f'{expected_osvi_dir}/V_trace.npy', 'rb') as f:
                v_trace = np.load(f)
        value_errors = np.linalg.norm(v_trace - true_trace, ord=1, axis=1)
        value_errors = value_errors / np.linalg.norm(true_V, ord=1)
        axs[2].plot(np.arange(num_iterations), value_errors, label=r"OS-VI ($\lambda$ = {})".format(alpha_vals[i]),
                    color=cmap(i+1))

    ## Model VI
    for i in range(len(alpha_vals)):
        Phat = model_class.get_P_hat_using_P(mdp.P(), alpha_vals[i])
        value = get_policy_value(Phat, mdp.R(), mdp.discount(), policy)
        value_error = np.linalg.norm(value - true_V, ord=1)
        value_error = value_error / np.linalg.norm(true_V, ord=1)
        axs[2].axhline(y=value_error, label=r"Model ($\lambda$ = {})".format(alpha_vals[i]),
                       linestyle="dashed", color=cmap(i+1))

    axs[2].set_xlabel("Iteration (k)")
    axs[2].set_yscale("log")
    axs[2].set_ylim(bottom=1e-4, top=1e1)
    axs[2].grid()

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

    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSVI-PE-Smooth-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifedCliffwalk.pdf", bbox_inches="tight")
    plt.savefig(f"{ROOT_OUTPUT_DIR}/{exp_dir}/OSVI-PE-Smooth-Maze-Garnet(runs=100, S=50, A=4, bp=3, br=5)-ModifedCliffwalk.png", bbox_inches="tight")