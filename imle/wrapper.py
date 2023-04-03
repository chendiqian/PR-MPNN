# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor

from imle.noise import BaseNoiseDistribution
from imle.target import BaseTargetDistribution, TargetDistribution

from typing import Callable, Optional

import logging

logger = logging.getLogger(__name__)


def imle(function: Callable[[Tensor], Tensor] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         input_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The input function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there is no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example::

        >>> from imle.wrapper import imle
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>       input_noise_temperature=input_noise_temperature, target_noise_temperature=5.0)
        >>> def imle_solver(weights_batch: Tensor) -> Tensor:
        >>>     return torch_solver(weights_batch)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise sammples
        input_noise_temperature (float): noise temperature for the input distribution
        target_noise_temperature (float): noise temperature for the target distribution
    """
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(imle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 input_noise_temperature=input_noise_temperature,
                                 target_noise_temperature=target_noise_temperature)

    @functools.wraps(function)
    def wrapper(input: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input: Tensor, *args):

                if noise_distribution is None:
                    noise = torch.zeros(size=input.shape)
                else:
                    noise = noise_distribution.sample(input.shape)

                input_noise = noise * input_noise_temperature

                perturbed_input = input + input_noise

                perturbed_output = function(perturbed_input)
                ctx.save_for_backward(input, noise, perturbed_output)

                return perturbed_output

            @staticmethod
            def backward(ctx, dy, *args):
                input, noise, perturbed_output = ctx.saved_variables

                target_input = target_distribution.params(input, dy)

                target_noise = noise * target_noise_temperature

                perturbed_target_input = target_input + target_noise

                target_output = function(perturbed_target_input)

                gradient = (perturbed_output - target_output)
                gradient = gradient / target_distribution.beta
                return gradient

        return WrappedFunc.apply(input, *args)
    return wrapper
