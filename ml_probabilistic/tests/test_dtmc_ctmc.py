import numpy as np

from src.core.config import BASE_GENERATOR, BASE_TRANSITION, build_generator
from src.models.ctmc import transition_matrix, validate_generator
from src.models.dtmc import expected_steps_to_absorption, validate_transition_matrix


def test_dtmc_matrix_valid():
    validate_transition_matrix(BASE_TRANSITION)


def test_absorption_time_positive():
    vals = expected_steps_to_absorption(BASE_TRANSITION, [0, 1, 2, 3])
    assert np.all(vals > 0)


def test_ctmc_generator_and_transition():
    a = build_generator(BASE_GENERATOR)
    validate_generator(a)
    p = transition_matrix(a, 1.0)
    assert p.shape == BASE_TRANSITION.shape
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
