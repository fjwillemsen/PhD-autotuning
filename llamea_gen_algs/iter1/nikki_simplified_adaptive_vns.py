import numpy as np
import random

from kernel_tuner.strategies.wrapper import OptAlg

class SimplifiedAdaptiveVNS(OptAlg):
    """
    Adaptive two-phase VNS with decaying Levy-driven global jumps,
    Hamming-based local moves and lightweight intensification.
    """
    def __init__(self, p0=0.4, p_min=0.05, alpha0=1.5, alpha1=3.0, intensify_steps=2):
        """
        Args:
            p0 (float): initial probability of global Levy jump
            p_min (float): final (min) probability of global jump
            alpha0 (float): initial Pareto tail index for Levy
            alpha1 (float): final Pareto tail index for Levy
            intensify_steps (int): number of greedy local intensification steps
        """
        self.p0 = float(p0)
        self.p_min = float(p_min)
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)
        self.intensify_steps = int(intensify_steps)
        self.constraint_aware = True
        self.costfunc_kwargs = {
            "scaling": True,
            "snap": True,
        }

    def __call__(self, func, searchspace):
        """
        Optimize black-box func under searchspace constraints.
        Returns best param tuple and its score.
        """
        self.searchspace = searchspace
        eval_count = 0
        visited = set()
        best = None  # (dna_list, score)

        # initialize best with a few random samples
        for _ in range(min(10, round(searchspace.size / 3))):
            cfg = searchspace.get_random_sample(1)[0]
            tup = tuple(cfg)
            if tup in visited or not searchspace.is_param_config_valid(tup):
                continue
            f = func(tup); eval_count += 1
            visited.add(tup)
            if best is None or f < best[1]:
                best = (list(tup), f)
            if func.budget_spent_fraction >= 1.0:
                return tuple(best[0]), best[1]

        # main loop
        while func.budget_spent_fraction < 1.0:
            frac = func.budget_spent_fraction
            p_global = (1 - frac) * self.p0 + frac * self.p_min
            alpha = (1 - frac) * self.alpha0 + frac * self.alpha1

            # choose global or local move
            if random.random() < p_global:
                # global Levy-driven jump
                start = list(best[0])
                L = int(np.random.pareto(alpha) + 1)
                x = start.copy()
                for _ in range(L):
                    nbrs = self.searchspace.get_neighbors(tuple(x), neighbor_method="Hamming")
                    if not nbrs:
                        break
                    x = list(random.choice(nbrs))
                move_type = "global"
            else:
                # single-step local Hamming move
                nbrs = self.searchspace.get_neighbors(tuple(best[0]), neighbor_method="Hamming")
                if nbrs:
                    x = list(random.choice(nbrs))
                else:
                    x = list(self.searchspace.get_random_sample(1)[0])
                move_type = "local"

            x = self._repair(x)
            xt = tuple(x)
            if (xt in visited or
                not self.searchspace.is_param_config_valid(xt)):
                continue

            f = func(xt); eval_count += 1
            visited.add(xt)

            # accept and optionally intensify
            if f < best[1]:
                best = (x, f)
                if move_type == "local":
                    # lightweight greedy intensification around new best
                    for _ in range(self.intensify_steps):
                        nbrs = self.searchspace.get_neighbors(tuple(best[0]), neighbor_method="Hamming")
                        if not nbrs:
                            break
                        y = self._repair(list(random.choice(nbrs)))
                        yt = tuple(y)
                        if yt in visited or not self.searchspace.is_param_config_valid(yt):
                            break
                        fy = func(yt); eval_count += 1
                        visited.add(yt)
                        if fy < best[1]:
                            best = (y, fy)
                        else:
                            break

        return tuple(best[0]), best[1]

    def _repair(self, dna):
        """
        Repair invalid configuration via increasing neighborhood scope.
        """
        if self.searchspace.is_param_config_valid(tuple(dna)):
            return dna
        for method in ["strictly-adjacent", "adjacent", "Hamming"]:
            nbrs = self.searchspace.get_neighbors(tuple(dna), neighbor_method=method)
            valid = [nb for nb in nbrs if self.searchspace.is_param_config_valid(tuple(nb))]
            if valid:
                return list(random.choice(valid))
        return dna