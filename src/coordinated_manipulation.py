"""Coordinated manipulation: groups of users colluding"""
import numpy as np

class CoordinatedAttack:
    """
    Simulates coordinated manipulation where a group of users colludes
    to promote a specific target item.
    
    Real-world examples:
    - Review bombing (gaming, movies)
    - Astroturfing (paid reviews)
    - Influencer cartels
    - Bot farms
    """
    
    def __init__(self, n_users, collusion_rate, target_item, alpha=0.8, seed=42):
        """
        Args:
            n_users: Total number of users
            collusion_rate: Fraction colluding (e.g., 0.05 = 5%)
            target_item: Item index to promote
            alpha: Manipulation intensity (0-1)
            seed: Random seed
        """
        rng = np.random.RandomState(seed)
        
        self.n_users = n_users
        self.collusion_rate = collusion_rate
        self.target_item = target_item
        self.alpha = alpha
        
        # Randomly select colluding users
        n_colluding = int(n_users * collusion_rate)
        self.colluding_users = set(rng.choice(n_users, size=n_colluding, replace=False))
        
        print(f"Coordinated Attack: {n_colluding}/{n_users} users ({collusion_rate*100:.0f}%) "
              f"targeting item {target_item} with alpha={alpha}")
    
    def is_colluding(self, user_id):
        """Check if user is part of colluding group."""
        return user_id in self.colluding_users
    
    def get_feedback(self, user_id, action, true_reward, strategy='coordinated'):
        """
        Get feedback from user (potentially manipulated).
        
        Coordinated strategy:
        - If user is colluding AND action is target: inflate maximally
        - If user is colluding AND action is NOT target: deflate maximally
        - If user is not colluding: report truthfully
        """
        if strategy == 'truthful' or not self.is_colluding(user_id):
            return float(true_reward)
        
        # Coordinated manipulation
        if action == self.target_item:
            # Boost target item
            manipulated = true_reward + self.alpha * (1.0 - true_reward)
        else:
            # Suppress competitors
            manipulated = true_reward - self.alpha * true_reward
        
        return float(np.clip(manipulated, 0.0, 1.0))
