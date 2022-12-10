import argparse
import numpy as np
import yaml
import shutil
from model.LocallySmoothedModel import LocallySmoothedModel
from model.IdentitySmoothedModel import IdentitySmoothedModel
from algorithms.OSVI_Control import OSVI_Control
from algorithms.VI_Control import VI_Control
from utils.utilities import setup_problem, setup_alg_output_dir, get_exp_dir

ROOT_OUTPUT_DIR = "./output"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs tabular experiments')
    parser.add_argument('config', help='Path  of config file')
    parser.add_argument('alg_name', help='Algorithm to run. "ALL" to run all, "None" to just plot')
    parser.add_argument('--exp_dir', default=None, help='Subdirectory containing data')
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if config["env"] == "garnet":
        iterations = config["garnet_instances"]
    else:
        iterations = 1

    for i in range(iterations):

        exp_name = "exp_control_vector"
        exp_dir = get_exp_dir(config, "exp_control_vector", exp_dir=args.exp_dir)
        mdp = setup_problem(config, seed=i)

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
            raise Exception("Incorrect Model Class Given")

        # Running Algorithms
        if args.alg_name in ["ALL", "vi_control"]:
            vi_control = VI_Control(mdp)
            vi_out_dir = setup_alg_output_dir(config, "exp_control_vector{}".format(i), "vi_control",
                                                smoothing_param=None, exp_dir=args.exp_dir)
            vi_control.run(num_iterations, f"{vi_out_dir}/policy_trace.npy", f"{vi_out_dir}/V_trace.npy")
            shutil.copyfile(src=args.config, dst=f"{vi_out_dir}/config.yaml")


        if args.alg_name in ["ALL", "osvi_control"]:
            for alpha in all_alphas:
                Phat = model_class.get_P_hat_using_P(mdp.P(), alpha)
                osvi_control = OSVI_Control(mdp, Phat)
                osvi_out_dir = setup_alg_output_dir(config, "exp_control_vector{}".format(i), "osvi_control",
                                                    smoothing_param=alpha, exp_dir=args.exp_dir)
                osvi_control.run(num_iterations, f"{osvi_out_dir}/policy_trace.npy", f"{osvi_out_dir}/V_trace.npy")
                shutil.copyfile(src=args.config, dst=f"{osvi_out_dir}/config.yaml")
