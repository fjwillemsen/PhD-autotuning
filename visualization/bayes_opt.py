from copy import deepcopy
from math import sin, cos, pi, sqrt
from random import randint
from random import uniform as randuni
import itertools
import progressbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

parameters = {
    'x': np.asarray(range(1, 20)),
    'y': np.array(range(1, 20)),
}


class ParameterConfig():

    def __init__(self, param_dict: dict, index: int):
        self.__param_config = param_dict
        self.__observed = False
        self.__observation = None
        self.__valid_observation = None
        self.index = index

    @property
    def param_config(self) -> dict:
        return self.__param_config

    @param_config.setter
    def param_config(self, value: dict):
        self.__param_config = value

    def get_as_list(self) -> list:
        return list(self.__param_config.values())

    @property
    def observation(self) -> float:
        # print("Observed: {}, valid: {}, observation: {}".format(self.__observed, self.__valid_observation, self.__observation))
        if self.__observed and self.__valid_observation:
            return self.__observation
        return None

    def get_observation_safe(self) -> float:
        if self.__observed and self.__valid_observation:
            return self.__observation
        elif not self.__observed:
            raise ValueError("Parameter configuration has not been evaluated")
        elif not self.__valid_observation:
            raise ValueError("Parameter configuration is invalid")

    @observation.setter
    def observation(self, value: float):
        self.__observed = True
        self.__observation = value
        if value == None or value <= 0:
            self.__valid_observation = False
        else:
            self.__valid_observation = True

    @property
    def valid_observation(self) -> bool:
        return self.__valid_observation

    @valid_observation.setter
    def valid_observation(self, value: bool):
        self.__valid_observation = value


class SearchSpace():

    def __init__(self, params_dict: dict):
        self.__parameters = params_dict
        self.__search_space = self.cartesian_product(params_dict)
        self.visited_indices = []
        self.unvisited_indices = list(range(len(self.__search_space)))

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, value: dict):
        self.__parameters = value

    @property
    def search_space(self):
        return self.__search_space

    @search_space.setter
    def search_space(self, value):
        self.__search_space = value

    def set_visited(self, x: ParameterConfig):
        self.__search_space[x.index] = x
        if x.index not in self.visited_indices:
            self.visited_indices.append(x.index)
            self.unvisited_indices.remove(x.index)

    def visited(self) -> list:
        return list(self.__search_space[index] for index in self.visited_indices)

    def unvisited(self) -> list:
        return list(self.__search_space[index] for index in self.unvisited_indices)

    def cartesian_product(self, params_dict: dict) -> list:
        cp = list((dict(zip(params_dict.keys(), x)) for x in itertools.product(*params_dict.values())))
        return list(ParameterConfig(x, index) for index, x in enumerate(cp))

    def size(self) -> int:
        return len(self.__search_space)

    def num_dimensions(self) -> int:
        return len(self.__parameters.keys())

    def dimensions(self) -> list:
        return self.__parameters.values()

    def get_parameter_config(self, index: int) -> ParameterConfig:
        return self.__search_space[index]

    def get_parameter_dict(self, index: int) -> dict:
        return self.__search_space[index].param_config

    def find_parameter_config_index(self, param_list: list) -> int:
        for index, x in enumerate(self.__search_space):
            # print("{} == {}".format(param_list, x.get_as_list()))
            if param_list == x.get_as_list():
                return index
        return None

    def draw_random_sample(self) -> (ParameterConfig, int):
        """ Draws a random sample from the search space, returns the sample and index """
        sample, index = self.draw_random_samples(num_samples=1)
        return sample[0], index[0]

    def draw_random_samples(self, num_samples=1) -> (list, list):
        """ Draws a random set of unique parameters from the search space, returns the samples and their indices """
        if self.size() < num_samples:
            raise ValueError("Can't sample more than the size of the search space")
        samples = []
        samples_index = []
        while len(samples) < num_samples:
            index = randint(0, self.size() - 1)
            if index not in samples_index:
                samples_index.append(index)
                samples.append(self.__search_space[index])
        return samples, samples_index

    def draw_latin_hypercube_samples(self, num_samples) -> list:
        """ Draws a LHS-distributed sample from the search space """
        from skopt.sampler import Lhs
        if self.size() < num_samples:
            raise ValueError("Can't sample more than the size of the search space")
        # based on https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
        lhs = Lhs(criterion="maximin", iterations=10000)
        return lhs.generate(self.dimensions(), num_samples)


# define objective function
def evaluate_objective_function(x: ParameterConfig) -> ParameterConfig:
    x_i = x.param_config['x']
    # y_i = x_i
    y_i = x.param_config['y'] if 'y' in x.param_config.keys() else x_i
    # x.observation = (x_i**2 * sin(5 * pi * x_i)**6.0)
    # noise = randuni(-0.25, 0.25)
    noise = 0
    x.observation = None if x_i % 10 == 0 else 5 + 0.2 * sin(y_i) + 2 * cos(sqrt(x_i)) + noise
    return x


class SurrogateModel():

    def __init__(self, searchspace: SearchSpace, acquisition_function, acquisition_function_parameters=None, num_initial_samples=1):
        self.searchspace = searchspace
        self.__af = acquisition_function
        self.__af_params = acquisition_function_parameters
        self.__observations = []
        self.__model = GaussianProcessRegressor()
        self.initial_sample(num_initial_samples)

    def initial_sample(self, num_samples):
        """ Draws an initial sample using Latin Hypercube Sampling. Invalid samples are resampled randomly up to 10 times. """
        if num_samples <= 0:
            raise ValueError("At least one initial sample is required")
        samples = self.searchspace.draw_latin_hypercube_samples(num_samples)
        # params = []
        # observations_values = np.array([])
        for sample in samples:
            sample_index = self.searchspace.find_parameter_config_index(sample)
            param_config = self.searchspace.get_parameter_config(sample_index)
            obs_param_config = evaluate_objective_function(param_config)
            # if a sample is invalid, retake a random sample until valid instead
            if not obs_param_config.valid_observation:
                max_attempts = 10
                for attempt in range(max_attempts):
                    param_config, sample_index = self.searchspace.draw_random_sample()
                    sample = param_config.get_as_list()
                    obs_param_config = evaluate_objective_function(param_config)
                    if obs_param_config.valid_observation:
                        continue
                    if attempt == max_attempts - 1:
                        raise ValueError("Could not find a valid configuration in the searchspace within {} attemps.".format(max_attempts))
            self.searchspace.set_visited(obs_param_config)
        # obs_values = np.array(self.get_observations_values(must_be_valid=True))
        # init_mean = np.mean(obs_values)
        # init_std = np.std(obs_values)
        # print("Initial mean: {}, std: {}".format(init_mean, init_std))
        self.update_model()

    def get(self, x: ParameterConfig) -> (float, float):
        """ Returns the value estimated by the surrogate model for a parameter configuration """
        # print("X: {}, list: {}".format(x, [x.get_as_list()]))
        return self.__model.predict([x.get_as_list()], return_std=True)

    def update_model(self):
        """ Update the model based on the current list of observations """
        params = self.get_observations_params(must_be_valid=True)
        observations = self.get_observations_values(must_be_valid=True)
        self.__model.fit(params, observations)

    def do_next(self) -> ParameterConfig:
        """ Find the next best candidate configuration, execute it and update the model accordingly """
        candidate = self.get_candidate()
        est_mu, est_std = self.get(candidate)
        observation = evaluate_objective_function(candidate)
        # print("{} estimate: {} ({} std), observed: {}".format(observation.get_as_list(), est_mu, est_std, observation.observation))
        self.searchspace.set_visited(observation)
        self.update_model()
        return observation

    def get_observations(self):
        """ Get a list of all observations """
        return self.searchspace.visited()

    def get_valid_observations(self):
        """ Get a list of all valid observations """
        return list(obs for obs in self.searchspace.visited() if obs.valid_observation)

    def get_observations_values(self, must_be_valid=False):
        """ Get a list of all observation values """
        observations = self.get_valid_observations() if must_be_valid else self.get_observations()
        return list(obs.observation for obs in observations)

    def get_observations_params(self, must_be_valid=False):
        """ Get a list of all observation parameters """
        observations = self.get_valid_observations() if must_be_valid else self.get_observations()
        return list(obs.get_as_list() for obs in observations)

    def get_best_observation(self) -> ParameterConfig:
        """ Get the best observed configuration so far """
        index = np.argmin(self.get_observations_values(must_be_valid=True))
        return self.get_valid_observations()[index]

    def get_best_observation_value(self) -> float:
        """ Get the best observed configuration value so far """
        return min(self.get_observations_values(must_be_valid=True))

    def get_candidate(self) -> ParameterConfig:
        """ Get the next candidate observation """
        if len(self.get_observations()) >= self.searchspace.size():
            raise ValueError("The search space has been fully observed")
        # assert len(model.searchspace.unvisited()) + len(model.searchspace.visited()) == model.searchspace.size()
        return self.__af(self, self.__af_params)


def af_probability_of_improvement(model: SurrogateModel, params: dict) -> ParameterConfig:
    """ Acquisition function Probability of Improvement (PI) """
    if params is None:
        params = {
            'explorationfactor': 0
        }
    fplus = model.get_best_observation_value() + params['explorationfactor']
    best_prob = None
    best_found = None
    # iterate over the entire unvisited model space to find the optimal
    unvisited = model.searchspace.unvisited()
    prob_of_improvement = lambda x_mu, x_std: norm.cdf((x_mu - fplus) / (x_std + 1E-9))
    highest_pi = np.argmin(list(prob_of_improvement(*model.get(x)) for x in unvisited))
    return unvisited[highest_pi]
    # for x in model.searchspace.unvisited():
    #     x_mu, x_std = model.get(x)
    #     z_value = (x_mu - fplus) / (x_std + 1E-9)
    #     prob_improvement = norm.cdf(z_value)
    #     if best_prob is None or prob_improvement <= best_prob:
    #         best_prob = prob_improvement
    #         best_found = x
    # return best_found


def af_expected_improvement(model: SurrogateModel, params: dict) -> ParameterConfig:
    """ Acquisition function Expected Improvement (EI) """
    if params is None:
        params = {
            'explorationfactor': 0
        }
    fplus = model.get_best_observation_value()
    best_prob = None
    best_found = None
    # iterate over the entire unvisited model space to find the optimal
    for x in model.searchspace.unvisited():
        x_mu, x_std = model.get(x)

        prob_improvement = norm.cdf((x_mu - fplus) / (x_std + 1E-9))
        if best_prob is None or prob_improvement <= best_prob:
            best_prob = prob_improvement
            best_found = x
        # found_best = min(found_best, )
    return best_found


# visualize the objective function and surrogate model
def visualize():
    ss = SearchSpace(parameters)
    ss_list = list(x.get_as_list() for x in ss.search_space)
    # brute-force the objective function
    objective_func = []
    for x in ss.search_space:
        x = evaluate_objective_function(x)
        objective_func.append(x.observation)
    # print(objective_func)

    # apply bayesian optimization
    model = SurrogateModel(ss, af_probability_of_improvement)
    for _ in range(10):
        model.do_next()
    bo_mean = []
    bo_std = []
    for x in ss.search_space:
        mean, std = model.get(x)
        # mean = mean[:, 0]
        bo_mean.append(mean[0])
        bo_std.append(std[0])

    # visualize
    bo_mean = np.array(bo_mean)
    bo_std = np.array(bo_std)
    plt.plot(objective_func, marker='.', linestyle="-", label='Objective function')
    plt.plot(range(len(ss_list)), bo_mean, marker='o', linestyle='--', label='Surrogate model')
    plt.fill_between(range(len(ss_list)), bo_mean - bo_std, bo_mean + bo_std, alpha=0.2, antialiased=True)
    plt.xlabel("Parameter config index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    # # #first make some fake data with same layout as yours
    # # data = pd.DataFrame(np.random.randn(100, 10), columns=['x1', 'x2', 'x3',\
    # #                     'x4','x5','x6','x7','x8','x9','x10'])

    # # #now plot using pandas
    # # scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')


def visualize_animated():
    ss = SearchSpace(parameters)
    ss_list = list(x.get_as_list() for x in ss.search_space)
    # brute-force the objective function
    objective_func = []
    for x in ss.search_space:
        x = evaluate_objective_function(x)
        objective_func.append(x.observation)
    model = SurrogateModel(ss, af_probability_of_improvement)

    # visualize
    x_data = range(len(ss_list))
    fig, ax = plt.subplots()
    ax.plot(objective_func, marker='o', linestyle="-", label='Objective function')
    line, = ax.plot([], [], marker='.', linestyle=':', label='Surrogate model')
    dot, = ax.plot(0, 0, 'ro', label='Best')

    def animation_init():
        ax.set_xlabel("Parameter config index")
        ax.set_ylabel("Value")
        ax.legend()
        return line, dot,

    def update(frame_number):
        # apply bayesian optimization
        model.do_next()
        bo_mean = []
        bo_std = []
        for x in ss.search_space:
            mean, std = model.get(x)
            # mean = mean[:, 0]
            bo_mean.append(mean[0])
            bo_std.append(std[0])
        bo_mean = np.array(bo_mean)
        bo_std = np.array(bo_std)
        # get the current best
        current_best = model.get_best_observation()
        # visualize
        ax.collections.clear()
        err = plt.fill_between(x_data, bo_mean - bo_std, bo_mean + bo_std, alpha=0.2, antialiased=True)
        line.set_data(x_data, bo_mean)
        dot.set_data(current_best.index, current_best.observation)
        # line, = ax.plot(x_data, bo_mean, marker='o', linestyle='--', label='Surrogate model')
        # return line, err,
        return line, err, dot,

    _ = FuncAnimation(fig, update, frames=89, interval=200, init_func=animation_init, blit=False, repeat=False)
    ax.legend()
    plt.show()


def visualize_af_performance(repeats=7, max_evaluations=50):
    """ Visualize the performance of acquisition functions against the objective function """
    fig, ax = plt.subplots(figsize=(20, 10))
    acquisition_functions = [(af_expected_improvement, None)]
    # acquisition_functions = [(af_probability_of_improvement, None), (af_probability_of_improvement, dict(explorationfactor=0.5)), (af_expected_improvement, None), (af_expected_improvement, dict(explorationfactor=1))]
    ss = SearchSpace(parameters)
    # brute-force the objective function to find the optimal
    best_objective_observation = min(evaluate_objective_function(x).observation for x in ss.search_space if evaluate_objective_function(x).valid_observation)
    # best_objective_observation = ss.search_space[np.argmin(objective_values)].get_observation_safe()
    # obtain a list of best observation per number of evaluations per acquisition function
    for af, af_params in acquisition_functions:
        label_name = "{}{}".format(af.__name__, ', ' + str(af_params) if af_params is not None else '')
        print(label_name)
        collected_best_distance_over_time = []
        for _ in progressbar.progressbar(range(repeats), redirect_stdout=True):
            best_over_time = []
            model = SurrogateModel(deepcopy(ss), af, af_params)
            best_found = False
            evaluation_counter = 0
            while not best_found and evaluation_counter < max_evaluations:
                evaluation_counter += 1
                try:
                    model.do_next()
                except ValueError:
                    break
                current_best = model.get_best_observation()
                best_over_time.append(current_best)
                best_found = current_best.observation == best_objective_observation
            collected_best_distance_over_time.append(list(x.observation - best_objective_observation for x in best_over_time if x.valid_observation))
        # fill the remaining indices with Nones
        collected_max_length = max(len(collected_best_distance_over_time[r]) for r in range(repeats))
        for r in range(repeats):
            collected_best_distance_over_time[r] += list(None for _ in range(collected_max_length - len(collected_best_distance_over_time[r])))
        collected_lengths = np.array(list(len(collected_best_distance_over_time[r]) for r in range(repeats)))
        assert np.all(collected_lengths == collected_max_length), "Lengths should all be {}, yet are {}".format(collected_max_length, collected_lengths)
        # calculate mean and standard deviation over distances per number of evaluations
        mean = np.array([])
        std = np.array([])
        x_data = range(collected_max_length)
        for index in x_data:
            distance_per_index = np.array([])
            for r in range(repeats):
                value = collected_best_distance_over_time[r][index]
                distance_per_index = np.append(distance_per_index, np.nan if value is None else value)
            mean = np.append(mean, np.nanmean(distance_per_index))
            std = np.append(std, np.nanstd(distance_per_index))
        # visualize
        ax.plot(mean, label=label_name)
        ax.fill_between(x_data, mean - std, mean + std, alpha=0.2, antialiased=True)
    # plot settings
    ax.set_xlabel("Number of evaluations used")
    ax.set_ylabel("Distance from global objective optimal ({})".format(round(best_objective_observation, 3)))
    ax.legend()
    plt.show()


# visualize()
visualize_animated()
# visualize_af_performance()
