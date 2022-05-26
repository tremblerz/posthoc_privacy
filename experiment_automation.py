import pdb
import numpy as np
import torch
import random
from privacy_estimation import Evaluation
from adversarial_training import ARL

from torch.distributions import Laplace

seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def add_noise(eps_list, z, sens):
    e = 1e-12
    variance = sens / eps_list + e
    scale = variance.unsqueeze(0).tile((z.shape[0], 1)).unsqueeze(2).tile(1, 1, z.shape[2])
    return z + Laplace(torch.zeros_like(z), scale).sample()


def multi_eps():

    # Encoder
    # proposed_bound = 1.5
    # arl_config = {'dset': 'mnist', 'alpha': 0.0, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': False, 'margin': 0., 'lambda': 0.0}
    # proposed_bound = 2.5
    # arl_config = {'dset': 'fmnist', 'alpha': 0.0, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25., 'lambda': 50.0}
    proposed_bound = 0.94
    arl_config = {'alpha': 0.00, 'tag': 'gender', 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 25.0}

    # ARL
    # proposed_bound = 0.9
    # arl_config = {'alpha': 0.9, 'dset': 'mnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 1.3
    # arl_config = {'alpha': 0.9, 'dset': 'fmnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 0.5
    # arl_config = {'alpha': 0.99, 'tag': 'gender', 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 25.0}
    
    # N
    # proposed_bound = 2.0
    # arl_config = {'alpha': 0.001, 'dset': 'mnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.1, 'siamese_reg': False, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 4.2
    # arl_config = {'alpha': 0.001, 'dset': 'fmnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.1, 'siamese_reg': False, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 0.6
    # arl_config = {'tag': 'gender', 'alpha': 0.0, 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.05, 'siamese_reg': False, 'margin': 25, 'lambda': 25.0}

    # C
    # proposed_bound = 0.6
    # arl_config = {'dset': 'mnist', 'alpha': 0.001, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 100.0}
    # proposed_bound = 3.9
    # arl_config = {'alpha': 0.001, 'dset': 'fmnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 0.72
    # arl_config = {'tag': 'gender', 'alpha': 0.0, 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 25.0}

    # ARL-N
    # proposed_bound = 6.0
    # arl_config = {'alpha': 0.99, 'dset': 'mnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 1.1
    # arl_config = {'dset': 'fmnist', 'alpha': 0.99, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25., 'lambda': 50.0}
    # proposed_bound = 0.38
    # arl_config = {'alpha': 0.99, 'tag': 'gender', 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 25.0}

    # ARL-C
    # proposed_bound = 1.3
    # arl_config = {'dset': 'mnist', 'alpha': 0.99, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 100.0}
    # proposed_bound = 2.2
    # arl_config = {'alpha': 0.9, 'dset': 'fmnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 50.0}
    # proposed_bound = 0.2
    # arl_config = {'alpha': 0.99, 'tag': 'gender', 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 25.0}

    # CN
    # proposed_bound = 0.6
    # arl_config = {'alpha': 0.001, 'dset': 'mnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 30, 'lambda': 50.0}
    # proposed_bound = 4.9
    # arl_config = {'alpha': 0.001, 'dset': 'fmnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 30, 'lambda': 50.0}
    # proposed_bound = 0.88
    # arl_config = {'alpha': 0.0, 'tag': 'gender', 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 25.0}

    # ARL-CN
    # proposed_bound = 0.8
    # arl_config = {'alpha': 0.99, 'dset': 'mnist', 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 100.0}
    # proposed_bound = 0.3
    # arl_config = {'dset': 'fmnist', 'alpha': 0.995, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25., 'lambda': 50.0}
    # proposed_bound = 0.56
    # arl_config = {'alpha': 0.95, 'tag': 'gender', 'dset': 'utkface', 'obf_in': 10, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 1.0}

    arl_obj = ARL(arl_config)
    arl_obj.load_state()
    arl_obj.vae.eval()
    arl_obj.pred_model.eval()
    radius, num_samples = 0.3, 0
    eps_list = [0.1, 2, 5, 10, torch.inf]
    effective_eps_list = torch.tensor([eps / radius for eps in eps_list]).cuda()
    noisy_pred_correct = torch.zeros(len(eps_list)).cuda()
    for data, labels, _ in arl_obj.test_loader:
        data, labels = data.cuda(), labels.cuda()
        z, _ = arl_obj.vae.encode(data)#.view(-1, 784))
        z_tilde = arl_obj.obfuscator(z)
        z_tilde = z_tilde.unsqueeze(1).tile(1, len(eps_list), 1)
        z_hat = add_noise(effective_eps_list, z_tilde, proposed_bound)
        noisy_preds = []
        for i in range(z_hat.shape[1]):
            # can't be vectorized due to batch norm
            z_i = z_hat[:, i, :]
            noisy_preds.append(arl_obj.pred_model(z_i).argmax(dim=1).unsqueeze(1))
        #pdb.set_trace()
        noisy_preds = torch.cat(noisy_preds, 1).cuda()
        noisy_pred_correct += (noisy_preds == labels.unsqueeze(1).tile(1, noisy_preds.shape[1])).sum(0)
        num_samples += data.shape[0]
    print(noisy_pred_correct / num_samples)


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
    delta = 0.1 # delta=0.1 means 0.05 effectively
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
    multi_eps()
