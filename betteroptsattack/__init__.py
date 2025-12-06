"""
BetterOptsAttack - Enhanced GCG Attack Implementation
=====================================================

A modern implementation of Greedy Coordinate Gradient (GCG) attacks
for evaluating and improving the robustness of language models.
"""

__version__ = "0.1.0"

from betteroptsattack.algorithms.gcg import (
    custom_gcg,
    check_argmax_match,
)
from betteroptsattack.algorithms.signals import (
    og_gcg_signal,
    rand_gcg_signal,
)
from betteroptsattack.algorithms.universal_gcg import (
    weakly_universal_gcg,
)
from betteroptsattack.utils.attack_utility import (
    initialize_adversarial_strings,
    string_masks_with_retry,
    target_logprobs,
    ADV_PREFIX_INDICATOR,
    ADV_SUFFIX_INDICATOR,
    DEFAULT_TEXT_GENERATION_CONFIG,
)

__all__ = [
    "custom_gcg",
    "og_gcg_signal",
    "rand_gcg_signal",
    "check_argmax_match",
    "weakly_universal_gcg",
    "initialize_adversarial_strings",
    "string_masks_with_retry",
    "target_logprobs",
    "ADV_PREFIX_INDICATOR",
    "ADV_SUFFIX_INDICATOR",
    "DEFAULT_TEXT_GENERATION_CONFIG",
]

