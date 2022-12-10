import argparse
import numpy as np
import yaml
import shutil
import random
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from multiprocessing import Pool
from algorithms.QLearning import QLearning
from algorithms.OSDyna_Control import OSDyna_Control
from algorithms.Dyna_Control import Dyna_Control
from utils.LearningRate import LearningRate
from utils.utilities import setup_problem, setup_alg_output_dir, get_exp_dir, get_default_alg_output_dir

ROOT_OUTPUT_DIR = "./output"

def run_qlearning(inputs):
    mdp, config, config_path, num_iterations, trial, exp_dir, model_class = inputs["mdp"], inputs["config"], inputs["config_path"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    qlearning_pe = QLearning(mdp)
    qlearning_out_dir = get_default_alg_output_dir(config, "exp_control_sample", "qlearning_control", smoothing_param=None,
                                             exp_dir=exp_dir)

    lr_scheduler = LearningRate(config["exp_control_lr_type"],
                                config["exp_control_qlearning_lr"],
                                config["exp_control_qlearning_delay"],
                                config["exp_control_qlearning_gamma"])

    qlearning_pe.run(num_iterations,
                     lr_scheduler=lr_scheduler,
                     policy_filename=f"{qlearning_out_dir}/policy_trace.npy",
                     value_filename=f"{qlearning_out_dir}/value_trace_{trial}.npy")
    shutil.copyfile(src=config_path, dst=f"{qlearning_out_dir}/config.yaml")


def run_dyna_control(inputs):
    mdp, config, config_path, alpha_vals, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for alpha in alpha_vals:
        dyna_control = Dyna_Control(mdp, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        dyna_out_dir = get_default_alg_output_dir(config, "exp_control_sample", "dyna_control", smoothing_param=alpha,
                                            exp_dir=exp_dir)
        dyna_control.run(num_iterations, policy_filename=f"{dyna_out_dir}/policy_trace_{trial}.npy",
                         value_filename=f"{dyna_out_dir}/value_trace_{trial}.npy")
        shutil.copyfile(src=config_path, dst=f"{dyna_out_dir}/config.yaml")


def run_osdyna_control(inputs):
    mdp, config, config_path, alpha_vals, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i, alpha in enumerate(alpha_vals):
        osdyna_control = OSDyna_Control(mdp, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        osdyna_out_dir = get_default_alg_output_dir(config, "exp_control_sample", "osdyna_control", smoothing_param=alpha,
                                            exp_dir=exp_dir)
        lr_scheduler = LearningRate(config["exp_control_lr_type"],
                                    config["exp_control_osdyna_lr"][i],
                                    config["exp_control_osdyna_delays"][i],
                                    config["exp_control_osdyna_gamma"][i])
        osdyna_control.run(num_iterations,
                           lr_scheduler=lr_scheduler,
                           policy_filename=f"{osdyna_out_dir}/policy_trace_{trial}.npy",
                           value_filename=f"{osdyna_out_dir}/value_trace_{trial}.npy")
        shutil.copyfile(src=config_path, dst=f"{osdyna_out_dir}/config.yaml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path of config file')
    parser.add_argument('alg_name', help='Algorithm to run. "ALL" to run all, "None" to just plot')
    parser.add_argument('--num_trials', default=1, help='Number of trials to run')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    parser.add_argument('--num_procs', default=24, type=int, help='Number of Processes')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
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

    if config["exp_control_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_control_sample_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        raise Exception("Incorrect Model Class Given")

    # Running Algorithms
    if args.alg_name in ["ALL", "qlearning"]:
        tdlearning_out_dir = setup_alg_output_dir(config, "exp_control_sample", "qlearning_control", smoothing_param=None,
                                                  exp_dir=args.exp_dir)
        with Pool(args.num_procs) as p:
            inputs = []
            for trial in range(num_trials):
                inputs.append({"mdp": mdp,
                               "config": config,
                               "config_path": args.config,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_qlearning, inputs)

    if args.alg_name in ["ALL", "dyna_control"]:
        for alpha in alpha_vals:
            dyna_out_dir = setup_alg_output_dir(config, "exp_control_sample", "dyna_control", smoothing_param=alpha,
                                                exp_dir=args.exp_dir)
        with Pool(args.num_procs) as p:
            inputs = []
            for trial in range(num_trials):
                inputs.append({"mdp": mdp,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_dyna_control, inputs)

    if args.alg_name in ["ALL", "osdyna_control"]:
        for alpha in alpha_vals:
            osdyna_out_dir = setup_alg_output_dir(config, "exp_control_sample", "osdyna_control", smoothing_param=alpha, exp_dir=args.exp_dir)

        with Pool(args.num_procs) as p:

            inputs = []
            for trial in range(num_trials):
                inputs.append({"mdp": mdp,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_osdyna_control, inputs)

