"""
BetterOptsAttack - Enhanced GCG Attack Implementation
=====================================================

A modern implementation of Greedy Coordinate Gradient (GCG) attacks
for evaluating and improving the robustness of language models.
"""

__version__ = "0.1.0"

from betteroptsattack.algorithms.gcg import (
    custom_gcg,
    og_gcg_signal,
)
from betteroptsattack.algorithms.gcg_enhanced import (
    custom_gcg as custom_gcg_enhanced,
)
from betteroptsattack.utils.attack_utility import (
    initialize_adversarial_strings,
    string_masks_with_retry,
    target_logprobs,
    ADV_PREFIX_INDICATOR,
    ADV_SUFFIX_INDICATOR,
    DEFAULT_TEXT_GENERATION_CONFIG,
)
from betteroptsattack.utils.experiment_logger import ExperimentLogger

__all__ = [
    "custom_gcg",
    "custom_gcg_enhanced",
    "og_gcg_signal",
    "initialize_adversarial_strings",
    "string_masks_with_retry",
    "target_logprobs",
    "ExperimentLogger",
    "ADV_PREFIX_INDICATOR",
    "ADV_SUFFIX_INDICATOR",
    "DEFAULT_TEXT_GENERATION_CONFIG",
]