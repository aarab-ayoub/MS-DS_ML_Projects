import numpy as np

from src.core.config import BASE_TRANSITION
from src.models.hmm_discrete import forward_filter, viterbi


def test_hmm_shapes():
    observations = [
        {
            "temperature": 65.0,
            "ecc_count": 0,
            "xid_code": 0,
            "utilization": 85.0,
            "power_usage": 580.0,
            "retired_pages": 0,
        }
        for _ in range(10)
    ]
    pi0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    post = forward_filter(observations, BASE_TRANSITION, pi0)
    path = viterbi(observations, BASE_TRANSITION, pi0)

    assert post.shape == (10, 5)
    assert len(path) == 10
    assert np.allclose(post.sum(axis=1), 1.0, atol=1e-6)
