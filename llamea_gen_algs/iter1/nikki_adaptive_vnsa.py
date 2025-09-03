import random
import math

from kernel_tuner.strategies.wrapper import OptAlg

class AdaptiveVNSA(OptAlg):
    """
    Adaptive Variable-Neighborhood Simulated Annealing with randomized restarts.
    """
    def __init__(self, init_temp=1.0, cooling_rate=0.95):
        self.init_temp = init_temp
        self.cooling_rate = cooling_rate
        self.constraint_aware = True
        self.costfunc_kwargs = {
            "scaling": True,
            "snap": True,
        }

    def __call__(self, func, searchspace):
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()
        evals = 0
        self.f_opt = float('inf')
        self.x_opt = None
        methods = ["strictly-adjacent", "adjacent", "Hamming"]
        method_scores = {m: 1.0 for m in methods}

        # Main loop: restarts until budget exhausted
        while func.budget_spent_fraction <= 1.0:
            # 1) random start
            x_cur = list(random.choice(self.searchspace.get_random_sample(1)))
            if not self.searchspace.is_param_config_valid(tuple(x_cur)):
                x_cur = self.repair(x_cur)
            f_cur = func(x_cur); evals += 1
            if f_cur < self.f_opt:
                self.f_opt, self.x_opt = f_cur, list(x_cur)

            T = self.init_temp
            # 2) inner SA loop
            while func.budget_spent_fraction <= 1.0 and T > 1e-6:
                # choose neighborhood method by adaptive weights
                weights = [method_scores[m] for m in methods]
                m = random.choices(methods, weights=weights, k=1)[0]
                # sample a neighbor
                neighbors = self.searchspace.get_neighbors(tuple(x_cur), neighbor_method=m)
                if not neighbors:
                    T *= self.cooling_rate
                    continue
                x_new = list(random.choice(neighbors))
                if not self.searchspace.is_param_config_valid(tuple(x_new)):
                    x_new = self.repair(x_new)
                f_new = func(x_new); evals += 1
                delta = f_new - f_cur
                # acceptance criterion
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # reward the method if it improved
                    if delta < 0:
                        method_scores[m] += 1.0
                    x_cur, f_cur = x_new, f_new
                    if f_cur < self.f_opt:
                        self.f_opt, self.x_opt = f_cur, list(x_cur)
                T *= self.cooling_rate

        return self.x_opt

    def repair(self, dna):
        """Repair invalid configuration by stepping through neighbor methods."""
        if self.searchspace.is_param_config_valid(tuple(dna)):
            return dna
        for nm in ["strictly-adjacent", "adjacent", "Hamming"]:
            neighbors = self.searchspace.get_neighbors_no_cache(tuple(dna), neighbor_method=nm)
            if neighbors:
                return list(random.choice(neighbors))
        return dna