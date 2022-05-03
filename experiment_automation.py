import numpy as np
import torch
import random
from privacy_estimation import Evaluation
from adversarial_training import ARL


seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def alpha_sens():
    eps = 10.0
    delta = 0.1
    eval_size = 100
    alphas = [0.9999, 0.999, 0.99, 0.9, 0.0]
    radii = [0.01, 0.1, 0.2, 0.3, 0.4, 1.0]
    for alpha in alphas:
        arl_config = {"alpha": alpha, "obf_in": 8, "obf_out": 5}
        arl = ARL(arl_config)
        arl.load_state()
        for radius in radii:
            eval_config = {"epsilon": eps, "delta": delta,
                           "radius": radius, "eval_size": eval_size}
            print("-"*50, "Starting new experiment ", "-"*50)
            print(arl_config)
            print(eval_config, "\n")
            eval = Evaluation(arl, eval_config)
            eval.test_local_sens()
            print("-"*50, "Experiment over", "-"*50)

def eps_acc():
    max_upper_bound_radius = 2.
    delta = 0.2 # delta=0.2 means 0.1 effectively
    eval_size = 100
    eps_list = [1, 5, 10, 20]
    alphas = [0.901, 0.999, 0.99, 0.9, 0.5, 0.0]
    proposed_bounds = [18, 5, 18, 16, 19, 22]
    radius = 0.3
    for i, alpha in enumerate(alphas):
        proposed_bound = proposed_bounds[i]
        arl_config = {"alpha": alpha, "obf_in": 8, "obf_out": 5}
        arl = ARL(arl_config)
        arl.load_state()
        for eps in eps_list:
            eval_config = {"epsilon": eps, "delta": delta,
                           "radius": radius, "eval_size": eval_size,
                           "proposed_bound": proposed_bound,
                           "max_upper_bound": max_upper_bound_radius}
            print("-"*50, "Starting new experiment ", "-"*50)
            eval = Evaluation(arl, eval_config)
            eval.create_logger()
            eval.logger.log_console(arl_config)
            eval.logger.log_console(eval_config)
            eval.test_local_sens()
            eval.test_ptr()
            print("-"*50, "Experiment over", "-"*50)

if __name__ == '__main__':
    eps_acc()
