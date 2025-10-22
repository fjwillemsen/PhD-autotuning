import numpy as np
import random

from kernel_tuner.strategies.wrapper import OptAlg

class AdaptivePSO_SA(OptAlg):
    """
    Simplified Adaptive PSO-SA with targeted local search and temperature annealing.
    """

    def __init__(self, swarm_size=15, inertia_weight=0.7, cognitive_coeff=1.4, social_coeff=1.4,
                 sa_initial_temp=0.5, sa_cooling_rate=0.98, local_search_prob=0.1):
        """Initialize the AdaptivePSO_SA algorithm."""
        self.constraint_aware = True
        self.costfunc_kwargs = {
            "scaling": True,
            "snap": True,
        }
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.local_search_prob = local_search_prob
        self.current_temp = sa_initial_temp
        self.eval_count = 0
        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_values = []
        self.global_best_position = None
        self.global_best_value = np.inf
        self.local_search_success_rate = 0.5  # Initial success rate for local search

    def __call__(self, func, searchspace):
        """Optimize the black-box function using AdaptivePSO_SA."""
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()
        num_params = len(self.tune_params)

        # Initialize the swarm
        for _ in range(self.swarm_size):
            particle = list(searchspace.get_random_sample(1)[0])
            self.particles.append(particle)
            self.velocities.append(np.zeros(num_params))
            assert searchspace.is_param_config_valid(tuple(particle))
            value = func(tuple(particle))
            self.eval_count += 1
            self.personal_best_positions.append(particle[:])
            self.personal_best_values.append(value)

            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = particle[:]

        local_search_attempts = 0
        local_search_successes = 0

        # Iterate until budget is exhausted
        while func.budget_spent_fraction <= 1.0:
            for i in range(self.swarm_size):
                # Update velocity and position - simplified velocity update
                r1 = np.random.rand(num_params)
                r2 = np.random.rand(num_params)
                velocity = (self.inertia_weight * self.velocities[i] +
                            self.cognitive_coeff * r1 * (np.array(self.personal_best_positions[i]) - np.array(self.particles[i])) +
                            self.social_coeff * r2 * (np.array(self.global_best_position) - np.array(self.particles[i])))

                new_particle = np.array(self.particles[i]) + velocity

                # Discrete variables.
                for j in range(num_params):
                    param_name = list(self.tune_params.keys())[j]
                    param_values = self.tune_params[param_name]
                    if isinstance(param_values, list): # Discrete Parameter
                        self.particles[i][j] = param_values[np.argmin(np.abs(np.array(param_values) - new_particle[j]))]

                self.particles[i] = self.repair(self.particles[i], searchspace)

                # Local Search - Targeted to closest personal best.
                if random.random() < self.local_search_prob:
                    local_search_attempts += 1
                    best_particle_index = np.argmin([np.linalg.norm(np.array(self.particles[i]) - np.array(p)) for p in self.personal_best_positions])
                    neighbor = self.get_better_neighbor(self.particles[i], func, searchspace, self.personal_best_values[best_particle_index]) #Pass best value

                    if neighbor is not None:
                        self.particles[i] = neighbor
                        local_search_successes += 1


                # Evaluate the new position
                assert searchspace.is_param_config_valid(tuple(self.particles[i]))
                value = func(tuple(self.particles[i]))
                self.eval_count += 1

                # Simplified Simulated Annealing acceptance criterion and update personal best
                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i][:]
                elif random.random() < np.exp(-(value - self.personal_best_values[i]) / self.current_temp):
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i][:]
                self.velocities[i] = velocity #Store velocity

            # Check if any particle is better than the current global best
            best_particle_index = np.argmin(self.personal_best_values)
            if self.personal_best_values[best_particle_index] < self.global_best_value:
                self.global_best_value = self.personal_best_values[best_particle_index]
                self.global_best_position = self.personal_best_positions[best_particle_index][:]

            # Cooling and Adaptive Local Search
            self.current_temp *= self.sa_cooling_rate
            if local_search_attempts > 5:  # Only adjust if there were enough local search attempts
                 self.local_search_success_rate = local_search_successes / local_search_attempts
                 self.local_search_prob = min(1.0, self.local_search_success_rate)  #Increase if its effective
            else:
                 self.local_search_prob = min(1.0, self.local_search_prob + 0.01) #Otherwise increase probablity
            local_search_attempts = 0
            local_search_successes = 0


            if func.budget_spent_fraction >= 1.0:
                break
        return self.global_best_position, self.global_best_value

    def repair(self, dna, searchspace):
        """Repair invalid configurations."""
        if not searchspace.is_param_config_valid(tuple(dna)):
            neighbors = searchspace.get_neighbors_no_cache(tuple(dna), neighbor_method="adjacent")
            if neighbors:
                return list(random.choice(neighbors))
        return dna

    def get_better_neighbor(self, dna, func, searchspace, best_value_nearby):
        """Find a neighbor with a better value than best_value_nearby."""
        neighbors = searchspace.get_neighbors_no_cache(tuple(dna), neighbor_method="adjacent")
        best_neighbor = None
        best_value = np.inf
        for neighbor in neighbors:
            if func.budget_spent_fraction >= 1.0:
                break
            assert searchspace.is_param_config_valid(tuple(neighbor))
            neighbor_value = func(tuple(neighbor))
            self.eval_count += 1
            if neighbor_value < best_value_nearby:  #Compare to the best value nearby.
                best_value = neighbor_value
                best_neighbor = list(neighbor)
                break #Take the first better neighbour found
        return best_neighbor
