"""
Unit tests for ER-N implementation.
Run with: pytest tests/test_implementation.py -v
"""
import numpy as np
import pytest
from src.ern import softmax, ERNLearner
from src.baselines import BaselineSoftmaxModel
from src.coordinated_manipulation import CoordinatedAttack


def test_softmax_basic():
    """Test softmax normalization and ordering."""
    logits = np.array([1.0, 2.0, 3.0])
    p = softmax(logits)
    
    # Should sum to 1
    assert abs(p.sum() - 1.0) < 1e-8
    
    # Should be non-negative
    assert np.all(p >= 0)
    
    # Should preserve ordering
    assert p[2] > p[1] > p[0]


def test_softmax_numerical_stability():
    """Test softmax handles large values without overflow."""
    logits = np.array([1000.0, 1001.0, 1002.0])
    p = softmax(logits)
    
    assert abs(p.sum() - 1.0) < 1e-8
    assert np.all(np.isfinite(p))


def test_ern_initialization():
    """Test ERNLearner initializes correctly."""
    learner = ERNLearner(n_items=10, beta=0.1, sigma=0.01, eta0=0.1, seed=123)
    
    assert learner.n_items == 10
    assert learner.beta == 0.1
    assert learner.sigma == 0.01
    assert learner.eta0 == 0.1
    assert learner.t == 0
    assert len(learner.Q) == 10
    assert np.all(learner.Q == 0.0)


def test_ern_policy():
    """Test ERNLearner policy is valid probability distribution."""
    learner = ERNLearner(n_items=10, beta=0.1, sigma=0.01, eta0=0.1, seed=123)
    policy = learner.get_policy()
    
    # Valid probability distribution
    assert abs(policy.sum() - 1.0) < 1e-8
    assert np.all(policy >= 0)
    assert len(policy) == 10


def test_ern_update():
    """Test ERNLearner update increases Q-value for rewarded action."""
    learner = ERNLearner(n_items=10, beta=0.1, sigma=0.01, eta0=0.1, seed=123)
    old_Q = learner.Q.copy()
    
    # Update action 0 with reward 1.0
    learner.update(0, 1.0)
    
    # Q[0] should increase
    assert learner.Q[0] > old_Q[0]
    
    # Other Q-values unchanged
    assert np.all(learner.Q[1:] == old_Q[1:])
    
    # Time counter incremented
    assert learner.t == 1


def test_ern_batch_update():
    """Test ERNLearner batch update."""
    learner = ERNLearner(n_items=10, beta=0.1, sigma=0.01, eta0=0.1, seed=123)
    
    actions = [0, 1, 2]
    rewards = [1.0, 0.5, 0.8]
    
    learner.update(actions, rewards)
    
    assert learner.t == 3
    assert learner.Q[0] > 0  # Should have increased
    assert learner.Q[1] > 0
    assert learner.Q[2] > 0


def test_bsm_baseline():
    """Test BaselineSoftmaxModel works correctly."""
    learner = BaselineSoftmaxModel(n_items=10, beta=1.0, eta0=0.1, seed=123)
    
    # Initial policy
    policy = learner.get_policy()
    assert abs(policy.sum() - 1.0) < 1e-8
    
    # Update
    old_Q = learner.Q.copy()
    learner.update(0, 1.0)
    assert learner.Q[0] > old_Q[0]


def test_coordinated_attack_initialization():
    """Test CoordinatedAttack initializes correctly."""
    attack = CoordinatedAttack(
        n_users=100,
        collusion_rate=0.05,
        target_item=10,
        alpha=0.8,
        seed=42
    )
    
    assert attack.n_users == 100
    assert attack.collusion_rate == 0.05
    assert attack.target_item == 10
    assert attack.alpha == 0.8
    assert len(attack.colluding_users) == 5  # 5% of 100


def test_coordinated_attack_feedback_truthful():
    """Test truthful feedback returns true reward."""
    attack = CoordinatedAttack(
        n_users=100,
        collusion_rate=0.05,
        target_item=10,
        alpha=0.8,
        seed=42
    )
    
    true_reward = 0.7
    feedback = attack.get_feedback(0, 5, true_reward, strategy='truthful')
    
    # Truthful should return exact reward
    assert feedback == true_reward


def test_coordinated_attack_feedback_strategic():
    """Test strategic feedback manipulates for target item."""
    attack = CoordinatedAttack(
        n_users=100,
        collusion_rate=0.05,
        target_item=10,
        alpha=0.8,
        seed=42
    )
    
    colluder = list(attack.colluding_users)[0]
    
    # Recommending target to colluder -> inflated feedback
    true_reward = 0.5
    feedback_target = attack.get_feedback(colluder, 10, true_reward, strategy='coordinated')
    assert feedback_target > true_reward  # Should be inflated
    
    # Recommending non-target to colluder -> deflated feedback
    feedback_other = attack.get_feedback(colluder, 5, true_reward, strategy='coordinated')
    assert feedback_other < true_reward  # Should be deflated


def test_ern_reset():
    """Test ERNLearner reset works."""
    learner = ERNLearner(n_items=10, beta=0.1, sigma=0.01, eta0=0.1, seed=123)
    
    # Make some updates
    learner.update(0, 1.0)
    learner.update(1, 0.5)
    
    assert learner.t > 0
    assert np.any(learner.Q != 0)
    
    # Reset
    learner.reset()
    
    assert learner.t == 0
    assert np.all(learner.Q == 0)


def test_ern_sigma_zero_equals_bsm():
    """Test ER-N with sigma=0 behaves like BSM."""
    ern = ERNLearner(n_items=10, beta=1.0, sigma=0.0, eta0=0.1, seed=123)
    bsm = BaselineSoftmaxModel(n_items=10, beta=1.0, eta0=0.1, seed=123)
    
    # Same updates
    for _ in range(10):
        action = 0
        reward = 1.0
        ern.update(action, reward)
        bsm.update(action, reward)
    
    # Q-values should be identical (or very close due to numerical differences)
    assert np.allclose(ern.Q, bsm.Q, rtol=1e-5)
