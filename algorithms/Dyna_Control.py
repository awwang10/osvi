import numpy as np
from utils.rl_utilities import get_optimal_policy, sample_one_uniform_from_mdp, get_policy_value_mdp
from tqdm import tqdm


class Dyna_Control:

    def __init__(self, mdp, model):
        self.mdp = mdp
        self.model = model

    def train(self, num_iteration):
        discount = self.mdp.discount()
        num_states, num_actions = self.mdp.num_states(), self.mdp.num_actions()
        self.policy_trace = np.zeros((num_iteration, num_states))
        self.V_pi_trace = np.zeros((num_iteration, num_states))
        with tqdm(iter(range(num_iteration)), desc="Dyna", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                P_hat = self.model.get_P_hat()
                r_hat = self.model.get_r_hat()
                optimal_policy = get_optimal_policy(P_hat, r_hat, discount, num_states, num_actions, err=1e-6, max_iterations=100000)
                self.policy_trace[k, :] = optimal_policy
                self.V_pi_trace[k, :] = get_policy_value_mdp(self.mdp, optimal_policy, err=1e-6, max_iteration=100000)
                (s, a, r, next_s) = sample_one_uniform_from_mdp(self.mdp)
                self.model.update(np.array([(s, a, r, next_s)]))

    def run(self, num_iteration, policy_filename, value_filename):
        self.train(num_iteration)
        np.save(value_filename, self.V_pi_trace)
        np.save(policy_filename, self.policy_trace)