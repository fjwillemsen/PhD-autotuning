from copy import deepcopy
from math import sin, cos, pi, sqrt
from random import randint
from random import uniform as randuni
import itertools
import progressbar
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.colors
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# parameters = {
#     'x': np.asarray(range(1, 5)),
#     'y': np.array(range(1, 3)),
# }

# parameters = {
#     'x': np.linspace(-1.5, 1.5, num=40),
#     'y': np.linspace(-1.5, 1.5, num=40),
# }

parameters = {
    'x': np.linspace(-10, 0, num=100),
    'y': np.linspace(-6.5, 0, num=100),
}

# parameters = {
#     'x': np.linspace(-2 * np.pi, 2 * np.pi, num=50),
#     'y': np.linspace(-2 * np.pi, 2 * np.pi, num=50),
# }

num_initial_samples = 5


class ParameterConfig():

    def __init__(self, param_dict: dict, index: int):
        self.__param_config = param_dict
        self.__observed = False
        self.__observation = np.NaN
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
        return np.NaN

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
        if value == np.NaN:
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

    def search_space_grid(self) -> list:
        """ N-dimensional representation of the search space instead of 1-dimensional """
        shape = tuple(len(params) for params in self.__parameters.values())
        return np.reshape(self.__search_space, shape).tolist()

    def search_space_grid_observations(self) -> list:
        """ N-dimensional representation of the search space observations instead of 1-dimensional """
        return list(list(x.observation for x in lst) for lst in self.search_space_grid())

    def search_space_grid_observations_normalized(self, obs_min: float, obs_max: float) -> list:
        """ N-dimensional representation of the normalized search space observations instead of 1-dimensional """
        norm_const = obs_max - obs_min
        return list(
            list(1 - ((x.observation - obs_min) / norm_const) if x.valid_observation else x.observation for x in lst) for lst in self.search_space_grid())

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
    # noise = randuni(-0.25, 0.25)
    # noise = 0

    # Rosenbrock function constrained to a disk
    # value = np.NaN if (x_i**2 + y_i**2) > 2 else ((1 - x_i)**2 + 100 * (y_i - x_i**2)**2) + noise

    # Mishra's bird function
    # value = sin(x_i) * np.exp((1 - cos(y_i))**2) + cos(y_i) * np.exp((1 - sin(x_i))**2) + (x_i - y_i)**2

    # Mishra's bird function (constrained)
    value = np.NaN if (x_i + 5)**2 + (y_i + 5)**2 < 25 else sin(x_i) * np.exp((1 - cos(y_i))**2) + cos(y_i) * np.exp((1 - sin(x_i))**2) + (x_i - y_i)**2

    # value = (x_i**2 * sin(5 * pi * x_i)**6.0) + noise
    # value = np.NaN if x_i % 10 == 0 else 5 + 0.2 * sin(y_i) + 2 * cos(sqrt(x_i)) + noise
    x.observation = value
    return x


class SurrogateModel():

    def __init__(self, searchspace: SearchSpace, acquisition_function, acquisition_function_parameters=None, num_initial_samples=num_initial_samples):
        self.__best_observed_value = np.inf
        self.searchspace = searchspace
        self.__af = acquisition_function
        self.__af_params = acquisition_function_parameters
        self.__observations = []
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        self.__model = GaussianProcessRegressor(kernel=kernel)
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
                log_list = []
                for attempt in range(max_attempts):
                    param_config, sample_index = self.searchspace.draw_random_sample()
                    sample = param_config.get_as_list()
                    obs_param_config = evaluate_objective_function(param_config)
                    if obs_param_config.valid_observation:
                        break
                    log_list.append(sample)
                    if attempt == max_attempts - 1:
                        raise ValueError("No valid configuration found in {} attemps, last was {} using {}. All attempted: {}.".format(
                            max_attempts, obs_param_config.observation, str(sample), log_list))
            self.searchspace.set_visited(obs_param_config)
        # obs_values = np.array(self.get_observations_values(must_be_valid=True))
        # init_mean = np.mean(obs_values)
        # init_std = np.std(obs_values)
        # print("Initial mean: {}, std: {}".format(init_mean, init_std))
        self.update_model()

    def predict(self) -> (list, list):
        """ Returns a list of values predicted by the surrogate model for the parameter configurations """
        return self.__model.predict(list(x.get_as_list() for x in self.searchspace.search_space), return_std=True)

    def get(self, x: ParameterConfig) -> (float, float):
        """ Returns the value estimated by the surrogate model for a parameter configuration """
        # print("X: {}, list: {}".format(x, [x.get_as_list()]))
        return self.__model.predict([x.get_as_list()], return_std=True)

    def update_model(self):
        """ Update the model based on the current list of observations """
        params = self.get_observations_params(must_be_valid=True)
        observations = self.get_observations_values(must_be_valid=True)
        self.__model.fit(params, observations)

    def do_next(self) -> (ParameterConfig, int, list):
        """ Find the next best candidate configuration, execute it and update the model accordingly """
        candidate, candidate_index, list_of_acquisition_values = self.get_candidate()
        if len(list_of_acquisition_values) > 0: list_of_acquisition_values = np.concatenate(list_of_acquisition_values)
        # est_mu, est_std = self.get(candidate)
        observation = evaluate_objective_function(candidate)
        if observation.valid_observation and observation.observation < self.get_best_observation_value():
            self.__best_observed_value = observation.observation
        # print("{} estimate: {} ({} std), observed: {}".format(observation.get_as_list(), est_mu, est_std, observation.observation))
        self.searchspace.set_visited(observation)
        self.update_model()
        return observation, candidate_index, list_of_acquisition_values

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

    def get_best_observation_value(self, allow_cached=True) -> float:
        """ Get the best observed configuration value so far """
        if not allow_cached:
            return min(self.get_observations_values(must_be_valid=True))
        return self.__best_observed_value

    def get_candidate(self) -> (ParameterConfig, int, list):
        """ Get the next candidate observation """
        if len(self.get_observations()) >= self.searchspace.size():
            raise ValueError("The search space has been fully observed")
        # assert len(model.searchspace.unvisited()) + len(model.searchspace.visited()) == model.searchspace.size()
        return self.__af(self, self.__af_params)


def af_random(model: SurrogateModel, params: dict) -> (ParameterConfig, int, list):
    """ Acquisition function returning a random candidate for comparison """
    unvisited = model.searchspace.unvisited()
    index = randint(0, len(unvisited) - 1)
    return unvisited[index], index, list()


def af_probability_of_improvement(model: SurrogateModel, params: dict) -> (ParameterConfig, int, list):
    """ Acquisition function Probability of Improvement (PI) """
    if params is None:
        params = {
            'explorationfactor': 0.01
        }
    fplus = model.get_best_observation_value() + params['explorationfactor']
    # iterate over the entire unvisited model space to find the optimal
    unvisited = model.searchspace.unvisited()
    prob_of_improvement = lambda x_mu, x_std: norm.cdf((x_mu - fplus) / (x_std + 1E-9))
    list_prob_of_improvement = list(prob_of_improvement(*model.get(x)) for x in unvisited)
    highest_pi = np.argmin(list_prob_of_improvement)
    return unvisited[highest_pi], highest_pi, list_prob_of_improvement


def af_expected_improvement(model: SurrogateModel, params: dict) -> (ParameterConfig, int, list):
    """ Acquisition function Expected Improvement (EI) """
    if params is None:
        params = {
            'explorationfactor': 0.01
        }
    fplus = model.get_best_observation_value() + params['explorationfactor']
    # iterate over the entire unvisited model space to find the optimal
    unvisited = model.searchspace.unvisited()
    diff = lambda x_mu: x_mu - fplus
    diff_improvement = lambda x_mu, x_std: diff(x_mu) / (x_std + 1E-9)
    exp_improvement = lambda x_mu, x_std: diff(x_mu) * norm.cdf(diff_improvement(x_mu, x_std)) + x_std * norm.pdf(diff_improvement(x_mu, x_std))
    list_exp_improvement = list(exp_improvement(*model.get(x)) for x in unvisited)
    highest_ei = np.argmin(list_exp_improvement)
    return unvisited[highest_ei], highest_ei, list_exp_improvement


def visualize_searchspace_2D(explore="random", resolution=0.0215):
    """ Visualize the search space and objective function """
    fully_explored_ss = SearchSpace(parameters)
    # brute-force the objective function
    for x in fully_explored_ss.search_space:
        x = evaluate_objective_function(x)
    if explore == 'full':
        ss = fully_explored_ss
    else:
        ss = SearchSpace(parameters)
    if explore == 'random':
        # random search over the objective function
        for _ in range(round(ss.size() * resolution)):
            index = randint(0, len(ss.unvisited_indices) - 1)
            x = ss.unvisited()[index]
            ss.set_visited(evaluate_objective_function(x))
    if explore == 'grid':
        # grid search over the objective function
        to_visit = round(ss.size() * resolution)
        step_size = ss.size() // to_visit
        print(ss.size())
        print(to_visit)
        print(step_size)
        for i in range(to_visit):
            index = i * step_size
            x = ss.search_space[index]
            x = evaluate_objective_function(x)

    # normalize
    valid_obs = list(x.observation for x in fully_explored_ss.search_space if x.valid_observation)
    obs_min = np.nanmin(valid_obs)
    obs_max = np.nanmax(valid_obs)
    for x in ss.search_space:
        if x.valid_observation:
            x.observation = 1 - (x.observation - obs_min) / (obs_max - obs_min)
    observations = ss.search_space_grid_observations()

    # visualize
    current_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "silver", "slategrey", "gold"])
    current_cmap.set_bad(color='darkred')
    current_cmap.set_under(color='white')
    x_params = ss.parameters['x']
    y_params = ss.parameters['y']
    extent = [min(x_params), max(x_params), min(y_params), max(y_params)]
    # change x and y axis to be positive
    if min(x_params) < 0 or min(y_params) < 0:
        extent = [0, max(x_params) - min(x_params), 0, max(y_params) - min(y_params)]
    plt.imshow(observations, extent=extent, cmap=current_cmap, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Gold content (normalized)')
    plt.xlabel('x parameter (kilometers)')
    plt.ylabel('y parameter (kilometers)')
    plt.show()


def visualize_searchspace_2D_exploration_animated(acq_func=af_expected_improvement, max_evaluations=-1, save=False):
    """ Visualize the search space and objective function """
    # brute-force the objective function
    fully_explored_ss = SearchSpace(parameters)
    for x in fully_explored_ss.search_space:
        x = evaluate_objective_function(x)
    if max_evaluations < 0:
        max_evaluations = fully_explored_ss.size() - 1

    # normalize
    valid_obs = list(x.observation for x in fully_explored_ss.search_space if x.valid_observation)
    obs_min = np.nanmin(valid_obs)
    obs_max = np.nanmax(valid_obs)

    # prepare search space to be searched
    ss = SearchSpace(parameters)
    observations = ss.search_space_grid_observations()
    model = SurrogateModel(ss, acq_func)

    # visualize
    fig, axes = plt.subplots()
    current_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "silver", "slategrey", "gold"])
    current_cmap.set_bad(color='darkred')
    current_cmap.set_under(color='white')
    x_params = ss.parameters['x']
    y_params = ss.parameters['y']
    extent = [min(x_params), max(x_params), min(y_params), max(y_params)]
    # change x and y axis to be positive
    if min(x_params) < 0 or min(y_params) < 0:
        extent = [0, max(x_params) - min(x_params), 0, max(y_params) - min(y_params)]
    plot = plt.imshow(observations, extent=extent, cmap=current_cmap, interpolation='nearest')

    def animation_init():
        plt.clim(0, 1)
        cb = plt.colorbar()
        cb.set_label('Gold content (normalized)')
        plt.xlabel('x parameter (kilometers)')
        plt.ylabel('y parameter (kilometers)')
        return plot,

    def update(frame_number):
        # apply bayesian optimization
        if frame_number % 10 == 0:
            print(frame_number)
        _, _, _ = model.do_next()
        observations = ss.search_space_grid_observations_normalized(obs_min, obs_max)
        plot.set_data(observations)
        # print("After {}: ".format(frame_number))
        return plot,

    animation = FuncAnimation(fig, update, frames=max_evaluations, interval=100, init_func=animation_init, blit=False, repeat=False)
    if save:
        animation.save('animation_2d_exploration.gif', writer='imagemagick', fps=5)
    else:
        plt.show()


def visualize():
    """ Visualize the objective function and surrogate model """
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


def visualize_animated(acq_func=af_expected_improvement, plot_acquisition_values=True, max_evaluations=500, save=False):
    """ Visualize the objective function, surrogate model, distance to objective optimal, and acquisition function (optional) animated over time """
    if acq_func == af_random:
        plot_acquisition_values = False
    ss = SearchSpace(parameters)
    ss_list = list(x.get_as_list() for x in ss.search_space)
    # brute-force the objective function
    objective_func = []
    for x in ss.search_space:
        x = evaluate_objective_function(x)
        objective_func.append(x.observation)
    best_objective_observation = min(x for x in objective_func if x is not np.NaN)
    model = SurrogateModel(ss, acq_func)

    # visualize
    x_data = range(len(ss_list))
    number_of_columns = 3 if plot_acquisition_values else 2
    fig, axes = plt.subplots(figsize=(20, 8), ncols=number_of_columns, gridspec_kw={ 'width_ratios': [2, 2, 1] })
    if plot_acquisition_values:
        ax_main, ax_aqfunc, ax_distance_over_time = axes
        acq_func_line, = ax_aqfunc.plot([], [], marker='.', linestyle=':', label='Acquisition values')
        acq_func_candidate_line, = ax_aqfunc.plot([30, 30], [0, 2], marker='', linestyle='-', label='Next sampling point')
    else:
        ax_main, ax_distance_over_time = axes
    ax_main.plot(objective_func, marker='o', linestyle="-", label='Objective function')
    line, = ax_main.plot([], [], marker='.', linestyle=':', label='Surrogate model')
    dot, = ax_main.plot(0, 0, 'ro', label='Best')
    distance_over_time_line, = ax_distance_over_time.plot([], [], marker='.', linestyle=':')
    distance_over_time_list = []

    def animation_init():
        ax_main.set_xlabel("Parameter config index")
        ax_main.set_ylabel("Value")
        ax_main.legend()
        ax_distance_over_time.set_xlabel("Number of evaluations used")
        ax_distance_over_time.set_ylabel("Distance from global objective optimal ({})".format(round(best_objective_observation, 3)))
        ax_distance_over_time.set_xlim(1, max_evaluations)
        if plot_acquisition_values:
            ax_aqfunc.set_ylabel("Acquisition value")
            ax_aqfunc.set_xlabel("Unvisited parameter config index")
            ax_aqfunc.set_xlim(0, len(x_data))
            ax_aqfunc.legend()
        return line, dot, distance_over_time_line,

    def update(frame_number):
        if frame_number % 10 == 0:
            print(frame_number)
        # apply bayesian optimization
        _, candidate_index, list_acq_values = model.do_next()
        bo_mean, bo_std = model.predict()
        bo_mean = np.array(bo_mean)
        bo_std = np.array(bo_std)
        # get the current best
        current_best = model.get_best_observation()
        distance_over_time_list.append(current_best.observation - best_objective_observation)
        # print(distance_over_time_list)
        # visualize
        ax_main.collections.clear()
        err = ax_main.fill_between(x_data, bo_mean - bo_std, bo_mean + bo_std, alpha=0.2, antialiased=True)
        line.set_data(x_data, bo_mean)
        dot.set_data(current_best.index, current_best.observation)
        ax_distance_over_time.set_ylim(0, max(distance_over_time_list) + 1E-9)
        distance_over_time_line.set_data(range(1, frame_number + 2), distance_over_time_list)
        # visualize acquisition values if applicable
        if plot_acquisition_values:
            max_acq_value = np.nanmax(list_acq_values)
            if not np.isnan(max_acq_value):
                acq_func_line.set_data(range(len(list_acq_values)), list_acq_values)
                acq_func_candidate_line.set_data([candidate_index], [0, max_acq_value])
                ax_aqfunc.set_ylim(0, max_acq_value)
        return line, err, dot, distance_over_time_line,

    animation = FuncAnimation(fig, update, frames=max_evaluations, interval=100, init_func=animation_init, blit=False, repeat=False)
    if save:
        animation.save('animation_bayesian_optimization.gif', writer='imagemagick', fps=1.5)
    else:
        plt.show()


def visualize_af_performance(repeats=7, max_evaluations=300):
    """ Visualize the performance of acquisition functions against the distance to the objective function """
    fig, ax = plt.subplots(figsize=(20, 10), num="N={} initial_samples={}".format(repeats, num_initial_samples))
    acquisition_functions = [(af_probability_of_improvement, None), (af_expected_improvement, None), (af_random, None)]
    # acquisition_functions = [(af_probability_of_improvement, None), (af_probability_of_improvement, dict(explorationfactor=0.5)),
    #                          (af_expected_improvement, None), (af_expected_improvement, dict(explorationfactor=0.5)), (af_random, None)]
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
            collected_best_distance_over_time[r] += list(0 for _ in range(collected_max_length - len(collected_best_distance_over_time[r])))
        collected_lengths = np.array(list(len(collected_best_distance_over_time[r]) for r in range(repeats)))
        assert np.all(collected_lengths == collected_max_length), "Lengths should all be {}, yet are {}".format(collected_max_length, collected_lengths)
        # calculate mean and standard deviation over distances per number of evaluations
        mean = np.array([])
        std = np.array([])
        x_data = range(collected_max_length)
        x_axis_data = range(1, collected_max_length + 1)
        for index in x_data:
            distance_per_index = np.array([])
            for r in range(repeats):
                value = collected_best_distance_over_time[r][index]
                distance_per_index = np.append(distance_per_index, np.nan if value is None else value)
            mean = np.append(mean, np.nanmean(distance_per_index))
            std = np.append(std, np.nanstd(distance_per_index))
        # visualize
        ax.plot(x_axis_data, mean, marker='.', linestyle=':', label=label_name)
        ax.fill_between(x_axis_data, mean - std, mean + std, alpha=0.2, antialiased=True)
    # plot settings
    ax.set_xlabel("Number of evaluations used")
    ax.set_ylabel("Distance from global objective optimal ({})".format(round(best_objective_observation, 3)))
    ax.legend()
    plt.show()


# visualize()
# visualize_animated(save=True)
visualize_af_performance()
# visualize_searchspace_2D()
# visualize_searchspace_2D_exploration_animated()
