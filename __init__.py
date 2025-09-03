"""Better Optimization-based Adversarial Attacks

A library for gradient-based adversarial attacks including GCG implementations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main attack functions
from .algorithms import gcg
from .algorithms.gcg import (
    og_gcg_signal,
    neg_gcg_signal,
    rand_gcg_signal,
    custom_gcg,
    weakly_universal_gcg,
    average_target_logprobs_signal,
)

# Import utilities
from .utils import attack_utility
from .utils import experiment_logger

__all__ = [
    "gcg",
    "og_gcg_signal",
    "neg_gcg_signal", 
    "rand_gcg_signal",
    "custom_gcg",
    "weakly_universal_gcg",
    "average_target_logprobs_signal",
    "attack_utility",
    "experiment_logger",
]