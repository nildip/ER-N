"""ER-N learner implementation and softmax helper"""
import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    max_logit = np.max(logits)
    exps = np.exp(logits - max_logit)
    sum_exps = np.sum(exps)
    if sum_exps == 0:
        return np.ones_like(logits) / logits.shape[0]
    return exps / sum_exps

class ERNLearner:
    def __init__(self, n_items, beta=0.1, sigma=0.01, eta0=0.001, seed=None):
        self.n_items = int(n_items)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.eta0 = float(eta0)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.Q = np.zeros(self.n_items, dtype=np.float64)
        self.t = 0

    def get_policy(self):
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=self.n_items)
        if self.sigma > 0:
            clip_val = 3.0 * self.sigma
            noise = np.clip(noise, -clip_val, clip_val)
        Q_tilde = self.Q + noise
        logits = self.beta * Q_tilde
        p = softmax(logits)
        return p

    def sample_action(self):
        p = self.get_policy()
        return int(self.rng.choice(self.n_items, p=p))

    def update(self, actions, rewards=None):
        """Support scalar or batch updates.

        actions: int or iterable of ints
        rewards: float or iterable of floats (if None and actions is array-like, raises)
        Behavior for single action mirrors previous implementation.
        """
        # normalize to lists
        if np.isscalar(actions):
            action_list = [int(actions)]
            reward_list = [float(rewards)]
        else:
            action_list = list(actions)
            if rewards is None:
                raise ValueError("rewards must be provided for batch updates")
            if np.isscalar(rewards):
                reward_list = [float(rewards)] * len(action_list)
            else:
                reward_list = list(rewards)
            if len(action_list) != len(reward_list):
                raise ValueError("actions and rewards must have same length")

        for reward_action in zip(action_list, reward_list):
            action, reward = reward_action
            self.t += 1
            eta_t = self.eta0 / np.sqrt(self.t)
            self.Q[action] = (1 - eta_t) * self.Q[action] + eta_t * float(reward)

    def reset(self):
        self.Q[:] = 0.0
        self.t = 0
