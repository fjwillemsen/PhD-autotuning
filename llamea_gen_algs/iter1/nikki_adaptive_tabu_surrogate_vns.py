import random
import math
import collections

from kernel_tuner.strategies.wrapper import OptAlg

class AdaptiveTabuSurrogateVNS(OptAlg):
    """
    Adaptive Tabu Surrogate Variable‐Neighborhood Search (ATS‐VNS)
    UCB‐driven operator selection among:
      0: global random exploration
      1: local neighborhood search
      2: surrogate‐guided exploitation
      3: directional intensification around current best
    with dynamic tabu memory and reactive restarts.
    """
    def __init__(self, k_nn=5, p_sur=10, beta=1.0, tabu_size=100, restart_patience=50):
        """
        k_nn: number of neighbors to use in surrogate predictions
        p_sur: initial pool size for surrogate sampling
        beta: UCB exploration parameter
        tabu_size: max length of the tabu queue
        restart_patience: number of failed iterations before a restart
        """
        self.k_nn = k_nn
        self.p_sur0 = p_sur
        self.beta = beta
        self.tabu_size = tabu_size
        self.restart_patience = restart_patience
        self.constraint_aware = True
        self.costfunc_kwargs = {
            "scaling": True,
            "snap": True,
        }

    def __call__(self, func, searchspace):
        visited = {}          # dict: cfg_tuple -> function value
        self.x_opt = None
        self.f_opt = float("inf")

        # UCB statistics
        n_ops = 4
        op_count = [0] * n_ops
        op_reward = [0.0] * n_ops
        total_trials = 0

        # Tabu memory
        tabu = collections.deque(maxlen=self.tabu_size)

        # stagnation / restart
        no_improve = 0
        p_sur = self.p_sur0

        # helper to safely evaluate a configuration once
        def try_eval(cfg):
            t = tuple(cfg)
            if func.budget_spent_fraction >= 1.0 or t in visited:
                return None
            if not searchspace.is_param_config_valid(t):
                return None
            f = func(t)
            visited[t] = f
            if f < self.f_opt:
                self.f_opt, self.x_opt = f, t
            return f

        # Weighted kNN surrogate predictor
        def surrogate_predict(candidate):
            # build list of (dist, f) over visited
            dists = []
            for v_cfg, v_f in visited.items():
                d = sum(1 for a,b in zip(v_cfg, candidate) if a != b)
                dists.append((d, v_f))
            if not dists:
                return self.f_opt
            dists.sort(key=lambda x: x[0])
            # take at most k_nn nearest, with distance weighting
            neigh = dists[:self.k_nn]
            num = 0.0
            den = 0.0
            for d, fv in neigh:
                w = 1.0 / (d + 1e-6)  # closer => higher weight
                num += w * fv
                den += w
            return num/den if den > 0 else self.f_opt

        # INITIAL SEEDING: a few random samples
        for _ in range(min(n_ops, round(searchspace.size / 3))):
            sample = list(searchspace.get_random_sample(1))[0]
            try_eval(sample)

        # MAIN OPTIMIZATION LOOP
        while func.budget_spent_fraction <= 1.0:
            total_trials += 1
            # select operator via UCB
            choice = None
            # ensure each op tried once
            for i in range(n_ops):
                if op_count[i] == 0:
                    choice = i
                    break
            if choice is None:
                ucb_vals = []
                for i in range(n_ops):
                    avg_r = op_reward[i] / op_count[i]
                    bonus = self.beta * math.sqrt(math.log(total_trials) / op_count[i])
                    ucb_vals.append(avg_r + bonus)
                choice = max(range(n_ops), key=lambda i: ucb_vals[i])
            op_count[choice] += 1

            parent_f = self.f_opt
            cfg = None

            # Operator 0: global random
            if choice == 0:
                cfg = list(searchspace.get_random_sample(1))[0]

            # Operator 1: local neighborhood
            elif choice == 1 and visited:
                # pick a seed in top 10% best
                topk = sorted(visited.items(), key=lambda x: x[1])
                cutoff = max(1, len(topk)//10)
                seed_cfg, seed_f = random.choice(topk[:cutoff])
                parent_f = seed_f
                # random neighbor
                nbrs = searchspace.get_neighbors(seed_cfg, neighbor_method=random.choice(
                    ["strictly-adjacent","adjacent","Hamming"]))
                if nbrs:
                    c = random.choice(nbrs)
                    cfg = list(c)
                else:
                    cfg = list(searchspace.get_random_sample(1))[0]
            # Operator 2: surrogate‐guided pool exploitation
            elif choice == 2:
                pool_size = min(p_sur, searchspace.size-len(visited))
                # sample candidates (some may be repeats or invalid)
                pool = list(searchspace.get_random_sample(pool_size))
                best_pred = float("inf")
                best_c = None
                for c in pool:
                    t = tuple(c)
                    if t in visited or not searchspace.is_param_config_valid(t):
                        continue
                    pred = surrogate_predict(c)
                    if pred < best_pred:
                        best_pred, best_c = pred, c
                cfg = list(best_c) if best_c is not None else list(searchspace.get_random_sample(1))[0]

            # Operator 3: directional intensification around current best
            else:
                if self.x_opt is not None:
                    # fetch 1‐hop neighbors and filter unseen valid
                    nbrs = searchspace.get_neighbors(self.x_opt, neighbor_method="Hamming")
                    candidates = []
                    for c in nbrs:
                        if tuple(c) not in visited and searchspace.is_param_config_valid(tuple(c)):
                            candidates.append(c)
                    if candidates:
                        # choose neighbor with best surrogate prediction
                        best_pred = float("inf")
                        best_c = None
                        for c in candidates:
                            pred = surrogate_predict(c)
                            if pred < best_pred:
                                best_pred, best_c = pred, c
                        cfg = list(best_c)
                    else:
                        cfg = list(searchspace.get_random_sample(1))[0]
                else:
                    cfg = list(searchspace.get_random_sample(1))[0]

            # Enforce tabu: avoid recent cycles
            tries = 0
            while tuple(cfg) in tabu and tries < 5:
                cfg = list(searchspace.get_random_sample(1))[0]
                tries += 1

            # Evaluate
            f_new = try_eval(cfg)
            if f_new is not None:
                # compute reward
                reward = max(0.0, parent_f - f_new)
                op_reward[choice] += reward
                tabu.append(tuple(cfg))
                # reset stagnation if improved
                if f_new < parent_f:
                    no_improve = 0
                    p_sur = self.p_sur0
                else:
                    no_improve += 1
            else:
                # invalid or duplicate
                op_reward[choice] += 0.0
                no_improve += 1

            # reactive restart if stuck
            if no_improve >= self.restart_patience and func.budget_spent_fraction <= 1.0:
                # single random injection
                sample = list(searchspace.get_random_sample(1))[0]
                try_eval(sample)
                no_improve = 0
                p_sur = min(p_sur * 2, max(1, self.p_sur0*10))

        return self.x_opt