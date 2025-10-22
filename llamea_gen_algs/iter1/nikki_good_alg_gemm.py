import os
import numpy as np
import random
import re
import json
import time
import traceback
import math
import traceback

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc

from kernel_tuner.strategies.wrapper import OptAlg

import numpy as np
import random

class AdaptiveSimulatedAnnealing(OptAlg):
    """
    Enhanced Adaptive Simulated Annealing with Dynamic Parameter Importance and Fine-Grained Temperature Control, refining search direction using parameter sensitivities and dynamically adjusting temperature gradients.
    """

    def __init__(self, initial_temp=1.0, cooling_rate=0.95, temp_adaptation_rate=0.1, param_importance=None, importance_update_rate=0.1, neighbor_decay_rate=0.05):
        """
        Initialize the Adaptive Simulated Annealing algorithm.

        Args:
            initial_temp (float): The initial temperature for simulated annealing.
            cooling_rate (float): The cooling rate for simulated annealing.
            temp_adaptation_rate (float): Rate at which the temperature is adapted.
            param_importance (dict): Dictionary of parameter names and their importance scores.
            importance_update_rate (float): Rate at which parameter importance is updated.
            neighbor_decay_rate (float): Rate at which the neighbor exploration radius decays.
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.history = {}
        self.success_counter = 0
        self.failure_counter = 0
        self.temp_adaptation_rate = temp_adaptation_rate
        self.param_importance = param_importance or {}
        self.importance_update_rate = importance_update_rate
        self.neighbor_decay_rate = neighbor_decay_rate
        self.neighbor_radius = 1.0  # Initial exploration radius
        self.constraint_aware = False
        self.costfunc_kwargs = {
            "scaling": False,
            "snap": False,
        }

    def __call__(self, func, searchspace):
        """
        Optimize the black-box function `func` using the given `searchspace` and evaluation budget.

        Args:
            func (callable): The black-box function to optimize.
            searchspace (SearchSpace): The search space object.

        Returns:
            tuple: The best solution found and its corresponding function value.
        """
        self.budget = searchspace.size
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()
        self.param_names = list(self.tune_params.keys())

        if not self.param_importance:
            self.param_importance = {name: 1.0 for name in self.param_names}

        self.f_opt = np.Inf
        self.x_opt = None
        eval_count = 0

        # Initialize solution
        current_solution = list(self.searchspace.get_random_sample(1)[0])
        current_fitness = self.evaluate(func, current_solution)
        eval_count += 1

        self.f_opt = current_fitness
        self.x_opt = current_solution

        temp = self.initial_temp

        # Main optimization loop
        while eval_count < self.budget:
            # Generate neighbor
            neighbor = self.get_neighbor(current_solution, temp)
            neighbor_fitness = self.evaluate(func, neighbor)
            eval_count += 1

            # Acceptance criterion (Simulated Annealing)
            delta = neighbor_fitness - current_fitness
            if delta < 0 or random.random() < np.exp(-delta / temp):
                # Update solution
                current_solution = neighbor
                current_fitness = neighbor_fitness

                # Update best solution
                if current_fitness < self.f_opt:
                    self.f_opt = current_fitness
                    self.x_opt = current_solution[:]  # Store a copy
                    self.success_counter += 1
                    self.update_parameter_importance(neighbor, current_solution) #Update importance if improvement
                else:
                    self.failure_counter += 1

            else:
                self.failure_counter += 1

            # Cooling
            temp *= self.cooling_rate

            # Adapt temperature
            temp = self.adapt_temperature(temp)

            # Decay the neighbor radius
            self.neighbor_radius *= (1 - self.neighbor_decay_rate)
            self.neighbor_radius = max(0.01, self.neighbor_radius) # Ensure radius doesn't go too low

        return self.x_opt, self.f_opt

    def evaluate(self, func, dna):
        """Evaluate the fitness of an individual."""
        config_tuple = tuple(dna)
        if config_tuple in self.history:
            return self.history[config_tuple]
        else:
            if not self.searchspace.is_param_config_valid(config_tuple):
                dna = self.repair(dna)
                config_tuple = tuple(dna)
            if config_tuple not in self.history:
                fitness = func(dna)
                self.history[config_tuple] = fitness
                return fitness
            else:
                return self.history[config_tuple]

    def get_neighbor(self, solution, temperature):
        """Get a neighbor with parameter-specific exploration."""
        neighbor = solution[:]
        num_params_to_change = max(1, int(random.gauss(1, 0.5)))  # Sample number of parameters to change
        indices = random.sample(range(len(neighbor)), min(num_params_to_change, len(neighbor)))

        for i in indices:
            parameter_name = self.param_names[i]
            possible_values = self.tune_params[parameter_name]
            # Use parameter importance to influence the choice of new value
            probabilities = np.array([self.param_importance[parameter_name] for _ in possible_values])
            probabilities = np.exp(probabilities / temperature) #Temperature scaling to influence probabilities
            probabilities /= np.sum(probabilities)  # Normalize probabilities

            neighbor[i] = random.choices(possible_values, weights=probabilities, k=1)[0]

        if not self.searchspace.is_param_config_valid(tuple(neighbor)):
            neighbor = self.repair(neighbor)
        return neighbor


    def repair(self, dna):
        """Repair an invalid configuration."""
        if not self.searchspace.is_param_config_valid(tuple(dna)):
            neighbor_methods = ["strictly-adjacent", "adjacent", "Hamming"]
            random.shuffle(neighbor_methods)
            for neighbor_method in neighbor_methods:
                neighbors = self.searchspace.get_neighbors_no_cache(tuple(dna), neighbor_method=neighbor_method)
                if len(neighbors) > 0:
                    return list(random.choice(neighbors))
        return dna


    def adapt_temperature(self, temp):
        """Adapt the temperature based on acceptance rate."""
        if self.success_counter + self.failure_counter == 0:
            acceptance_rate = 0.5  # Start with a neutral adjustment
        else:
            acceptance_rate = self.success_counter / (self.success_counter + self.failure_counter)
        temp_change = (acceptance_rate - 0.5) * self.temp_adaptation_rate
        temp += temp_change
        temp = max(0.01, temp)  # Ensure temperature doesn't go too low
        return temp

    def update_parameter_importance(self, new_solution, old_solution):
        """Update parameter importance based on successful move."""
        for i, param_name in enumerate(self.param_names):
            if new_solution[i] != old_solution[i]:
                self.param_importance[param_name] += self.importance_update_rate
            else:
                self.param_importance[param_name] *= (1 - self.importance_update_rate * 0.1)  # Slightly reduce importance if not changed
            self.param_importance[param_name] = max(0.1, self.param_importance[param_name])  # Ensure importance doesn't go too low
