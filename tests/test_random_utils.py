import random

import numpy as np

from unity_wheel.utils.random_utils import set_seed


def test_set_seed_reproducibility() -> None:
    """Random and numpy sequences should be reproducible after seeding."""
    set_seed(42)
    r1 = random.random()
    n1 = np.random.rand()

    set_seed(42)
    assert random.random() == r1
    assert np.random.rand() == n1
