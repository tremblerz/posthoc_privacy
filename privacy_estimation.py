# add calibrated noise
# measure performance on noisy data
# measure perfromance on original data
import pdb
import time
import numpy as np
import utils
import torch
import random
from lipmip.hyperbox import Hyperbox
from lipmip.lipMIP import LipMIP
from embedding import get_dataloader
from torch.distributions import Laplace
from adversarial_training import ARL


seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class Evaluation():
    def __init__(self, arl_obj: ARL, config) -> None:
        self.epsilon = config["epsilon"]
        self.delta = config["delta"]
        self.radius = config["radius"]
        self.max_upper_bound_radius = config["max_upper_bound"]
        self.proposed_bound = config["proposed_bound"]
        self.eval_size = config["eval_size"]
        self.arl_obj = arl_obj

        self.max_time = 20 # seconds
        self.margin = 0.001

        # For now we are using center of 0 and l_inf norm of size 1 around the center to define the output space
        self.out_domain = 'l1Ball1'
        self.dset = 'mnist'
        self.setup_data()

        self.arl_obj.on_cpu()

        self.base_dir = "./experiments/"

    def create_logger(self):
        dset = self.arl_obj.dset
        alpha = self.arl_obj.alpha
        eps = self.epsilon
        delta = self.delta
        radius = self.radius
        proposed_bound = self.proposed_bound
        self._log_path = "eval_{}_alpha{}_eps{}_delta{}_radius{}_proposal{}".format(dset, alpha, eps, delta, radius, proposed_bound)
        self._log_path = self.base_dir + self._log_path
        utils.check_and_create_path(self._log_path)
        utils.save_object(self, self._log_path)
        self.logger = utils.Logger(self._log_path)

    def setup_data(self):
        self.train_loader, self.test_loader = get_dataloader(self.dset, batch_size=1)

    def get_noise(self, z_tilde, sens):
        variance = sens / self.epsilon
        return Laplace(torch.zeros_like(z_tilde), variance).sample()

    def test_ptr(self):
        self.arl_obj.vae.eval()
        sample_size, unanswered_samples = 0, 0
        noisy_pred_correct, noiseless_pred_correct = 0, 0
        for batch_idx, (data, labels) in enumerate(self.test_loader):
            data = data.cuda()
            mu, log_var = self.arl_obj.vae.encoder(data.view(-1, 784))
            center = self.arl_obj.vae.sampling(mu, log_var).cpu()

            lower_bound_s = self.radius
            upper_bound_s = self.max_upper_bound_radius

            simple_domain = Hyperbox.build_linf_ball(center[0], lower_bound_s)
            cross_problem = LipMIP(self.arl_obj.obfuscator.cpu(), simple_domain,
                                   'l1Ball1', num_threads=8, verbose=True)
            cross_problem.compute_max_lipschitz()
            lip_val = cross_problem.result.value
            self.logger.log_console(lip_val)
            if lip_val > self.proposed_bound:
                self.logger.log_console("bot before starting")
                unanswered_samples += 1
            else:
                start_time = time.perf_counter()
                while upper_bound_s >= lower_bound_s + self.margin:
                    radius = (lower_bound_s + upper_bound_s) / 2
                    simple_domain = Hyperbox.build_linf_ball(center[0], radius)
                    cross_problem = LipMIP(self.arl_obj.obfuscator.cpu(), simple_domain,
                                        'l1Ball1', num_threads=8, verbose=False)
                    cross_problem.compute_max_lipschitz()
                    lip_val = cross_problem.result.value
                    if self.proposed_bound > lip_val:
                        lower_bound_s = radius
                    else:
                        upper_bound_s = radius
                    time_elapsed = time.perf_counter() - start_time
                    if time_elapsed > self.max_time:
                        self.logger.log_console("timeout")
                        break
                # Dividing by 2 because that's the actual lower bound as shown in the paper
                real_lower_bound_s = lower_bound_s / 2
                noisy_lower_bound_s = real_lower_bound_s + self.get_noise(torch.zeros(1), self.radius)
                if noisy_lower_bound_s < np.log(1 / self.delta) * (self.radius / self.epsilon):
                    self.logger.log_console("bot {:.4f}".format(noisy_lower_bound_s.item()))
                    unanswered_samples += 1
                else:
                    # pass it through obfuscator
                    z_tilde = self.arl_obj.obfuscator(center)
                    # This is not private yet since the noise added is based on private data
                    # TODO: Switch to PTR version
                    z_hat = z_tilde + self.get_noise(z_tilde, lip_val)
                    # pass obfuscated z through pred_model
                    noisy_preds = self.arl_obj.pred_model(z_hat)
                    noisy_pred_correct += (noisy_preds.argmax(dim=1) == labels).sum()

                    noiseless_preds = self.arl_obj.pred_model(z_tilde)
                    noiseless_pred_correct += (noiseless_preds.argmax(dim=1) == labels).sum()
                    sample_size += 1
            if unanswered_samples + sample_size > self.eval_size:
                break
        assert sample_size + unanswered_samples - 1 == batch_idx
        noisy_pred_acc = noisy_pred_correct.item() / sample_size
        noiseless_pred_acc = noiseless_pred_correct.item() / sample_size
        self.logger.log_console('====> Unanswered_samples {}, Noisy pred acc {:.2f}, Noiseless pred acc {:.2f}'.format(unanswered_samples, noisy_pred_acc, noiseless_pred_acc))

    def test_local_sens(self):
        self.arl_obj.vae.eval()
        sample_size, lip_vals = 0, []
        noisy_pred_correct, noiseless_pred_correct = 0, 0
        for batch_idx, (data, labels) in enumerate(self.test_loader):
            data = data.cuda()
            # get sample embedding from the VAE
            mu, log_var = self.arl_obj.vae.encoder(data.view(-1, 784))
            center = self.arl_obj.vae.sampling(mu, log_var).cpu()
            # batch size is 1 and we are removing the 0th dimension for Lip estimation
            simple_domain = Hyperbox.build_linf_ball(center[0], self.radius)
            cross_problem = LipMIP(self.arl_obj.obfuscator.cpu(), simple_domain,
                                   'l1Ball1', num_threads=8, verbose=True)
            cross_problem.compute_max_lipschitz()
            lip_val = cross_problem.result.value
            lip_vals.append(lip_val)
            # pass it through obfuscator
            z_tilde = self.arl_obj.obfuscator(center)
            # This is not private yet since the noise added is based on private data
            # TODO: Switch to PTR version
            z_hat = z_tilde + self.get_noise(z_tilde, lip_val)
            # pass obfuscated z through pred_model
            noisy_preds = self.arl_obj.pred_model(z_hat)
            noisy_pred_correct += (noisy_preds.argmax(dim=1) == labels).sum()

            noiseless_preds = self.arl_obj.pred_model(z_tilde)
            noiseless_pred_correct += (noiseless_preds.argmax(dim=1) == labels).sum()
            sample_size += 1
            if sample_size > self.eval_size:
                break
        assert sample_size - 1 == batch_idx
        noisy_pred_acc = noisy_pred_correct.item() / sample_size
        noiseless_pred_acc = noiseless_pred_correct.item() / sample_size
        self.logger.log_console('====> Noisy pred acc {:.2f}, Noiseless pred acc {:.2f}'.format(noisy_pred_acc, noiseless_pred_acc))
        self.logger.log_console('====> mean = {:.4f}, std = {:.4f}'.format(np.array(lip_vals).mean(), np.array(lip_vals).std()))


if __name__ == '__main__':
    arl_config = {"alpha": 0.9999, "obf_in": 8, "obf_out": 5}

    eps = 20.0
    delta = 0.2

    proposed_bound = 1.5
    max_upper_bound_radius = 2.
    radius = 0.3
    eval_size = 100
    eval_config = {"epsilon": eps, "delta": delta, "radius": radius, "eval_size": eval_size,
                   "proposed_bound": proposed_bound, "max_upper_bound": max_upper_bound_radius}
    arl = ARL(arl_config)
    arl.load_state()
    eval = Evaluation(arl, eval_config)
    eval.create_logger()
    eval.logger.log_console(arl_config)
    eval.logger.log_console(eval_config)
    eval.test_local_sens()
    eval.test_ptr()

