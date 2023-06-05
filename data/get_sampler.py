from functools import partial

import torch
from ml_collections import ConfigDict

from imle.noise import GumbelDistribution
from imle.target import TargetDistribution
from imle.wrapper import imle
from training.gumbel_scheme import GumbelSampler
from training.imle_scheme import IMLEScheme
from training.simple_scheme import EdgeSIMPLEBatched


def get_sampler(imle_configs: ConfigDict,
                sample_configs: ConfigDict,
                device: torch.device):
    if imle_configs is None:
        return None, None, None

    if imle_configs.sampler == 'imle':
        imle_scheduler = IMLEScheme(sample_configs.sample_policy,
                                    sample_configs.sample_k)

        @imle( target_distribution=TargetDistribution(alpha=1.0, beta=imle_configs.beta),
               noise_distribution=GumbelDistribution(0., imle_configs.noise_scale, device),
               nb_samples=imle_configs.num_train_ensemble,
               input_noise_temperature=1.,
               target_noise_temperature=1., )
        def imle_train_scheme(logits: torch.Tensor):
            return imle_scheduler.torch_sample_scheme(logits)

        train_forward = imle_train_scheme

        @imle(target_distribution=None,
              noise_distribution=GumbelDistribution(0., imle_configs.noise_scale, device),
              nb_samples=imle_configs.num_val_ensemble,
              input_noise_temperature=1. if imle_configs.num_val_ensemble > 1 else 0.,
              # important
              target_noise_temperature=1., )
        def imle_val_scheme(logits: torch.Tensor):
            return imle_scheduler.torch_sample_scheme(logits)

        val_forward = imle_val_scheme
        sampler_class = imle_scheduler
    elif imle_configs.sampler == 'gumbel':
        gumbel_sampler = GumbelSampler(sample_configs.sample_k,
                                       tau=imle_configs.tau,
                                       policy=sample_configs.sample_policy)
        train_forward = partial(gumbel_sampler.forward,
                                train_ensemble=imle_configs.num_train_ensemble)
        val_forward = partial(gumbel_sampler.validation,
                              val_ensemble=imle_configs.num_val_ensemble)
        sampler_class = gumbel_sampler
    elif imle_configs.sampler == 'simple':
        simple_sampler = EdgeSIMPLEBatched(sample_configs.sample_k,
                                           device,
                                           val_ensemble=imle_configs.num_val_ensemble,
                                           train_ensemble=imle_configs.num_train_ensemble,
                                           policy=sample_configs.sample_policy,
                                           logits_activation=imle_configs.logits_activation)
        train_forward = simple_sampler
        val_forward = simple_sampler.validation
        sampler_class = simple_sampler
    else:
        raise ValueError

    return train_forward, val_forward, sampler_class
