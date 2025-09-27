"""Better Optimization-based Adversarial Attacks

A library for gradient-based adversarial attacks including GCG implementations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main attack functions from package
from .betteroptsattack.algorithms import gcg
from .betteroptsattack.algorithms.gcg import (
    og_gcg_signal,
    neg_gcg_signal,
    rand_gcg_signal,
    custom_gcg,
    weakly_universal_gcg,
    average_target_logprobs_signal,
    check_argmax_match,
)

# Import utilities from package
from .betteroptsattack.utils import attack_utility
from .betteroptsattack.utils import experiment_logger

__all__ = [
    "gcg",
    "og_gcg_signal",
    "neg_gcg_signal",
    "rand_gcg_signal",
    "custom_gcg",
    "weakly_universal_gcg",
    "average_target_logprobs_signal",
    "check_argmax_match",
    "attack_utility",
    "experiment_logger",
]