"""
Unit tests for softmax, ER-N, strategic user, and metrics.
Run with: python -m pytest tests/test_implementation.py
"""
import numpy as np
from src.ern import softmax, ERNLearner
from src.strategic_users import StrategicUser
import src.metrics as metrics


def test_softmax_basic():
    logits = np.array([1.0, 2.0, 3.0])
    p = softmax(logits)
    assert abs(p.sum() - 1.0) < 1e-8
    assert np.all(p >= 0)
    assert p[2] > p[1] > p[0]


def test_ern_policy_and_update():
    learner = ERNLearner(n_items=10, beta=0.1, sigma=0.01, eta0=0.1, seed=123)
    policy = learner.get_policy()
    assert abs(policy.sum() - 1.0) < 1e-8
    old_Q = learner.Q.copy()
    learner.update(0, 1.0)
    assert learner.Q[0] > old_Q[0]
    assert np.all(learner.Q[1:] == old_Q[1:])


def test_strategic_user():
    utils = np.array([0.1, 0.9, 0.3])
    su = StrategicUser(utils, alpha=0.2)
    truthful = su.report_feedback(1, strategy='truthful')
    greedy = su.report_feedback(1, strategy='greedy')
    assert greedy >= truthful
    nonfav_truth = su.report_feedback(0, strategy='truthful')
    nonfav_greedy = su.report_feedback(0, strategy='greedy')
    assert nonfav_greedy <= nonfav_truth


def test_metrics_manipulation_gain():
    strat = np.array([0.9, 0.8, 0.7])
    truth = np.array([0.6, 0.8, 0.5])
    gain = metrics.compute_manipulation_gain(strat, truth)
    assert abs(gain - np.mean(strat - truth)) < 1e-9
