"""Strategic user simulator"""
import numpy as np

class StrategicUser:
    def __init__(self, true_utilities, alpha=0.1):
        self.true_utilities = np.asarray(true_utilities, dtype=np.float64)
        assert self.true_utilities.ndim == 1
        assert np.all(self.true_utilities >= 0) and np.all(self.true_utilities <= 1)
        self.alpha = float(alpha)
        self.favorite = int(np.argmax(self.true_utilities))

    def get_true_reward(self, action):
        return float(self.true_utilities[action])

    def report_feedback(self, action, strategy='greedy'):
        true = float(self.true_utilities[action])
        if strategy == 'truthful':
            return true
        elif strategy == 'greedy':
            best_item = self.favorite
            if action == best_item:
                inflated = true + self.alpha * (1.0 - true)
                return float(np.clip(inflated, 0.0, 1.0))
            else:
                deflated = true - self.alpha * true
                return float(np.clip(deflated, 0.0, 1.0))
        else:
            raise ValueError(f"Unknown strategy {strategy}")

def generate_user_utilities(n_users, n_items, seed=42):
    rng = np.random.RandomState(seed)
    return rng.beta(2.0, 2.0, size=(n_users, n_items)).astype(np.float64)

def generate_alpha_distribution(n_users, distribution='uniform', seed=42):
    rng = np.random.RandomState(seed)
    if distribution == 'uniform':
        return rng.uniform(0.3, 0.8, size=n_users).astype(np.float64)
    elif distribution == 'beta':
        return (rng.beta(2.0, 5.0, size=n_users) * 0.8).astype(np.float64)
    elif distribution == 'bimodal':
        choices = rng.choice([0.4, 0.8], size=n_users)
        return choices.astype(np.float64)
    elif distribution == 'adversarial':
        return np.ones(n_users, dtype=np.float64)
    else:
        raise ValueError("Unsupported distribution")
