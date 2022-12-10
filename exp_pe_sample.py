import argparse
import numpy as np
import yaml
import shutil
import random
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from multiprocessing import Pool
from algorithms.OSDyna_PE import OSDyna_PE
from algorithms.Dyna_PE import Dyna_PE
from algorithms.TDLearning_PE import TDLearning_PE
from utils.rl_utilities import get_optimal_policy_mdp
from utils.utilities import setup_problem, setup_alg_output_dir, get_exp_dir, get_default_alg_output_dir
from utils.LearningRate import LearningRate

ROOT_OUTPUT_DIR = "./output"

def run_td_learning(inputs):
    mdp, policy, config, config_path, num_iterations, trial, exp_dir, model_class = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    tdlearning_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "tdlearning_pe", smoothing_param=None,
                                              exp_dir=exp_dir)
    tdlearning_pe = TDLearning_PE(mdp, policy)
    lr_scheduler = LearningRate(config["exp_pe_lr_type"],
                                config["exp_pe_td_learning_pe_lr"],
                                config["exp_pe_td_learning_pe_delay"],
                                config["exp_pe_td_gamma"])
    tdlearning_pe.run(num_iterations, lr_scheduler=lr_scheduler,
                      output_filename=f"{tdlearning_out_dir}/V_trace_{trial}.npy")
    shutil.copyfile(src=config_path, dst=f"{tdlearning_out_dir}/config.yaml")

def run_dyna_pe(inputs):
    mdp, policy, config, config_path, alpha_vals, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for alpha in alpha_vals:
        dyna_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "dyna_pe", smoothing_param=alpha, exp_dir=exp_dir)
        dyna_pe = Dyna_PE(mdp, policy, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        dyna_pe.run(num_iterations, output_filename=f"{dyna_out_dir}/V_trace_{trial}.npy")
        shutil.copyfile(src=config_path, dst=f"{dyna_out_dir}/config.yaml")

def run_osdyna_pe(inputs):
    mdp, policy, config, config_path, alpha_vals, num_iterations, trial, exp_dir, model_class  = inputs["mdp"], inputs["policy"], inputs["config"], inputs["config_path"], inputs["alpha_vals"], inputs["num_iterations"], inputs["trial"], inputs["exp_dir"], inputs["model_class"]
    random.seed(trial)
    np.random.seed(trial)
    for i, alpha in enumerate(alpha_vals):
        osdyna_out_dir = get_default_alg_output_dir(config, "exp_pe_sample", "osdyna_pe", smoothing_param=alpha, exp_dir=exp_dir)
        osdyna_pe = OSDyna_PE(mdp, policy, model_class(mdp.num_states(), mdp.num_actions(), alpha))
        lr_scheduler = LearningRate(config["exp_pe_lr_type"],
                                    config["exp_pe_osdyna_lr"][i],
                                    config["exp_pe_osdyna_lr_delay"],
                                    config["exp_pe_osdyna_gamma"][i])
        osdyna_pe.run(num_iterations, lr_scheduler=lr_scheduler,
                      output_filename=f"{osdyna_out_dir}/V_trace_{trial}.npy")
        shutil.copyfile(src=config_path, dst=f"{osdyna_out_dir}/config.yaml")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path  of config file')
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

    exp_name = "exp_pe_sample"
    exp_dir = get_exp_dir(config, "exp_pe_sample", exp_dir=args.exp_dir)
    mdp = setup_problem(config)
    policy = get_optimal_policy_mdp(mdp)
    num_trials = int(args.num_trials)
    num_iterations = config["exp_pe_sample_num_iterations"]
    alpha_vals = config["exp_pe_sample_alphas"]

    if config["exp_pe_sample_model_type"] == "LocallySmoothed":
        model_class = LocallySmoothedModel
    elif config["exp_pe_sample_model_type"] == "IdentitySmoothed":
        model_class = IdentitySmoothedModel
    else:
        raise Exception("Incorrect Model Class Given")

    # Running Algorithms
    if args.alg_name in ["ALL", "tdlearning_pe"]:
        tdlearning_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "tdlearning_pe", smoothing_param=None,
                                                  exp_dir=args.exp_dir)
        with Pool(args.num_procs) as p:
            inputs = []
            for trial in range(num_trials):
                inputs.append({"mdp": mdp,
                                "policy": policy,
                                "config": config,
                                "config_path": args.config,
                                "num_iterations": num_iterations,
                                "trial": trial,
                                "exp_dir": args.exp_dir,
                                "model_class": model_class,
                                })
            p.map(run_td_learning, inputs)

    if args.alg_name in ["ALL", "dyna_pe"]:
        for alpha in alpha_vals:
            dyna_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "dyna_pe", smoothing_param=alpha,
                                                exp_dir=args.exp_dir)
        with Pool(args.num_procs) as p:
            inputs = []
            for trial in range(num_trials):
                inputs.append({"mdp": mdp,
                               "policy": policy,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_dyna_pe, inputs)

    if args.alg_name in ["ALL", "osdyna_pe"]:
        for alpha in alpha_vals:
            osdyna_out_dir = setup_alg_output_dir(config, "exp_pe_sample", "osdyna_pe", smoothing_param=alpha,
                                                  exp_dir=args.exp_dir)
        with Pool(args.num_procs) as p:
            inputs = []
            for trial in range(num_trials):
                inputs.append({"mdp": mdp,
                               "policy": policy,
                               "config": config,
                               "config_path": args.config,
                               "alpha_vals": alpha_vals,
                               "num_iterations": num_iterations,
                               "trial": trial,
                               "exp_dir": args.exp_dir,
                               "model_class": model_class,
                               })
            p.map(run_osdyna_pe, inputs)

