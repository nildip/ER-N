"""Baseline models: BaselineSoftmaxModel, RobustMF, SASRec, LightGCN (simplified)"""
import numpy as np
from .ern import softmax

class BaselineSoftmaxModel:
    def __init__(self, n_items, beta=1.0, eta0=0.001, seed=None):
        self.n_items = int(n_items)
        self.beta = float(beta)
        self.eta0 = float(eta0)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.Q = np.zeros(self.n_items, dtype=np.float64)
        self.t = 0

    def get_policy(self):
        logits = self.beta * self.Q
        return softmax(logits)

    def sample_action(self):
        p = self.get_policy()
        return int(self.rng.choice(self.n_items, p=p))

    def update(self, action, reward):
        self.t += 1
        eta_t = self.eta0 / (1.0 + 0.1 * np.sqrt(self.t))
        self.Q[action] = (1 - eta_t) * self.Q[action] + eta_t * float(reward)

    def reset(self):
        self.Q[:] = 0.0
        self.t = 0


class RobustMF:
    def __init__(self, n_users, n_items, n_factors=16, lr=0.01, delta=0.5, seed=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.delta = delta
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.user_factors = rng.normal(scale=0.1, size=(n_users, n_factors))
        self.item_factors = rng.normal(scale=0.1, size=(n_items, n_factors))

    def predict(self, user, item):
        return float(np.dot(self.user_factors[user], self.item_factors[item]))

    def get_policy(self, user):
        scores = np.dot(self.user_factors[user], self.item_factors.T)
        return softmax(scores)

    def update(self, user, item, rating):
        pred = self.predict(user, item)
        error = rating - pred
        if abs(error) <= self.delta:
            grad = error
        else:
            grad = self.delta * np.sign(error)
        u = self.user_factors[user]
        v = self.item_factors[item]
        self.user_factors[user] += self.lr * grad * v
        self.item_factors[item] += self.lr * grad * u


class SASRec:
    def __init__(self, n_items, n_factors=32, max_history=50, seed=None):
        self.n_items = n_items
        self.n_factors = n_factors
        self.max_history = max_history
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.item_embeddings = rng.normal(scale=0.1, size=(n_items, n_factors))
        self.user_history = {}

    def get_policy(self, user):
        hist = self.user_history.get(user, [])

        # If no history, uniform policy
        if not hist:
            return np.ones(self.n_items) / self.n_items

        # Take last max_history items
        recent = hist[-self.max_history:]

        # Robust conversion/filtering:
        valid_indices = []
        for idx in recent:
            if idx is None:
                continue
            # Accept numpy integer types as well
            if isinstance(idx, (int, np.integer)):
                i = int(idx)
            else:
                # Try coercing strings like '123' to int; otherwise skip
                try:
                    i = int(idx)
                except Exception:
                    continue
            # Check bounds
            if 0 <= i < self.n_items:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            # fallback: uniform policy
            return np.ones(self.n_items) / self.n_items

        # Convert to numpy array of ints for safe advanced indexing
        recent_idx = np.array(valid_indices, dtype=np.int64)

        # Compute representation and scores
        rep = np.mean(self.item_embeddings[recent_idx], axis=0)
        scores = np.dot(rep, self.item_embeddings.T)
        return softmax(scores)

    def update(self, user, item, rating=None):
        self.user_history.setdefault(user, []).append(item)
        if len(self.user_history[user]) > self.max_history:
            self.user_history[user] = self.user_history[user][-self.max_history:]


class LightGCN:
    def __init__(self, n_users, n_items, n_factors=32, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.user_embeddings = rng.normal(scale=0.1, size=(n_users, n_factors))
        self.item_embeddings = rng.normal(scale=0.1, size=(n_items, n_factors))
        self.interactions = set()

    def get_policy(self, user):
        scores = np.dot(self.user_embeddings[user], self.item_embeddings.T)
        return softmax(scores)

    def update(self, user, item, rating=None):
        self.interactions.add((user, item))
