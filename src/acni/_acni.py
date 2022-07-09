from typing import Iterable, Optional, Union

import torch


class ACNI:
    """Apply anticorrelated noise injection to a set of parameters.

    Similar to `torch.optim.Optimizer`, this class operates on a set of given parameters and must be stepped in order
    to update the parameters. This should be run after each step of the optimizer.

    Args:
        parameters: The parameters to apply noise to.
        std: The standard deviation of the noise.
        generator: An optional torch.Generator to use for random number generation. Defaults to None.

    References:
        1. Orvieto, A., Kersting, H., Proske, F., Bach, F.R., & Lucchi, A. (2022).
        Anticorrelated Noise Injection for Improved Generalization. ArXiv, abs/2202.02831.

    TODO: add support for parameter groups like torch.optim.Optimizer
    TODO: proper input validation like torch.optim.Optimizer
    TODO: state management (e.g., state_dict, load_state_dict)
    """

    def __init__(
        self,
        parameters: Union[Iterable[torch.Tensor], torch.Tensor],
        std: float,
        generator: Optional[torch.Generator] = None,
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        self.std = std
        self.generator = generator
        self.parameters = list(parameters)
        self.state = {}

    @torch.no_grad()
    def step(self):
        for p in self.parameters:
            if p not in self.state:
                self.state[p] = torch.empty_like(p).normal_(0, self.std, generator=self.generator)
            noise = torch.empty_like(p).normal_(0, self.std, generator=self.generator)
            diff = noise - self.state[p]
            p.data.add_(diff)
            self.state[p] = noise


__all__ = ["ACNI"]
