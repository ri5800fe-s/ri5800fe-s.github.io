#src/classes.py

import numpy as np
from scipy.special import comb

class Link: 
    def __init__(self, direction, a=1):
        self.direction = direction
        self.a = a

class RubberBand:
    def __init__(self, N, a=1, kBT=1):
        self.N = N
        self.a = a
        self.kBT = kBT
        self.beta = 1 / kBT
        self.links = []

    def create_random_links_unbiased(self, rng): #without force
        directions = rng.choice([-1, 1], size=self.N)
        self.links = [Link(direction=d, a=self.a) for d in directions]

    def create_random_links_biased(self, rng, f): #with force
        p_plus = np.exp(self.beta * f * self.a)
        p_minus = np.exp(-self.beta * f * self.a)
        prob_plus = p_plus / (p_plus + p_minus)

        directions = rng.choice([-1, 1], size=self.N, p=[1 - prob_plus, prob_plus])
        self.links = [Link(direction=d, a=self.a) for d in directions]

    def length(self): 
        return sum(link.a * link.direction for link in self.links)

    def n_from_L(self, L): # acessory function to the analytical result, from eq.1
        return np.round((self.N + np.asarray(L) / self.a) / 2).astype(int)

    def omega(self, L): # acessory function to the analytical result, from eq.5
        n = self.n_from_L(L)
        valid = (n >= 0) & (n <= self.N)
        out = np.zeros_like(np.asarray(L), dtype=float)
        out[valid] = comb(self.N, n[valid])
        return out

    def P_unbiased(self, L): # P analytical without force
        return self.omega(L) / (2 ** self.N)

    def w_microstate(self, L, f): # from task II
        return np.exp(self.beta * f * np.asarray(L))

    def P_exact_biased(self, L, f): # from task II
        omega_vals = self.omega(L)
        w_vals = self.w_microstate(L, f)
        Z = np.sum(omega_vals * w_vals)
        return omega_vals * w_vals / Z

    def sample_length_unbiased(self, rng): # Monte Carlo without force
        self.create_random_links_unbiased(rng)
        return self.length()

    def sample_length_biased(self, rng, f): # Monte Carlo with force
        self.create_random_links_biased(rng, f)
        return self.length()

    def mu_eff(self, L, f): # introduced in task II, statistics
        w = self.w_microstate(L, f)
        return (np.sum(w) ** 2) / np.sum(w ** 2)

    def avg_L(self, L_vals): # avg_L from Monte Carlo
        return np.mean(L_vals)

    def avg_L_exact(self, L_vals, P_vals): # avg_L from analytical function ie eq.16
        return np.sum(L_vals * P_vals)
    

class CorrelatedRubberBand(RubberBand): # Task 4 nem class
    
    # weight calculation in 3 functions
    def neighbor_sum(self, state): # s sum
        return np.sum(state[:-1] * state[1:])

    def log_w_J(self, state, J): # beta*J*s_sum
        return self.beta * J * self.neighbor_sum(state)

    def weights_J(self, states, J): # exp(beta*J*s_sum)
        S = np.array([self.neighbor_sum(state) for state in states], dtype=np.float64)
        x = self.beta * J * S
        return np.exp(x)

    def mu_eff_J(self, states, J):
        w = self.weights_J(states, J)
        return (np.sum(w) ** 2) / np.sum(w ** 2)
    
    # Same as before, but return array to use J later
    def directions(self): 
        return np.array([link.direction for link in self.links], dtype=np.int64)

    def sample_microstate_biased(self, rng, f): 
        self.create_random_links_biased(rng, f)
        return self.directions()