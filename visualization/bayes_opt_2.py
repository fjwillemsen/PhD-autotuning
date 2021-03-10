from math import sin, cos, pi, sqrt
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

parameters = {
    # 'x': [1, 2, 4, 5, 7, 10, 12],
    'x': np.asarray(range(1, 100)),
    # 'x': [5, 8, 9],
    'y': np.array(range(1, 100, 2)),
}


class ParameterConfig():

    def __init__(self, param_dict: dict):
        self.__param_config = param_dict
        self.__observed = False
        self.__observation = None
        self.__valid_observation = None

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

    def cartesian_product(self, params_dict: dict) -> list:
        cp = list((dict(zip(params_dict.keys(), x)) for x in itertools.product(*params_dict.values())))
        return list(ParameterConfig(x) for x in cp)

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

    def find_parameter_config_index(self, param_list) -> int:
        for index, x in enumerate(self.__search_space):
            # print("{} == {}".format(param_list, x.get_as_list()))
            if param_list == x.get_as_list():
                return index
        return None


# define objective function
def evaluate_objective_function(x: ParameterConfig) -> ParameterConfig:
    # observed_value = 0
    # for _, value in x.param_config.items():
    #     observed_value += value
    # if observed_value % 5 == 0:
    #     observed_value = 0
    # x.observation = observed_value
    # return x
    x_i = x.param_config['x']
    # x.observation = (x_i**2 * sin(5 * pi * x_i)**6.0)
    x.observation = None if x_i % 10 == 0 else 5 + 0.2 * sin(x_i) + 2 * cos(sqrt(x_i))
    return x


class SurrogateModel():

    def __init__(self, searchspace: SearchSpace, acquisition_function, num_initial_samples=5):
        self.searchspace = searchspace
        self.__af = acquisition_function
        self.__observations = []
        self.__model = GaussianProcessRegressor()
        self.initial_sample(num_initial_samples)

    def initial_sample(self, num_samples):
        from skopt.sampler import Lhs
        # based on https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
        lhs = Lhs(criterion="maximin", iterations=10000)
        samples = lhs.generate(self.searchspace.dimensions(), num_samples)
        params = []
        observations = np.array([])
        for sample in samples:
            print(sample)
            ss_index = self.searchspace.find_parameter_config_index(sample)
            param_config = self.searchspace.get_parameter_config(ss_index)
            obs_param_config = evaluate_objective_function(param_config)
            params.append(sample)
            observations = np.append(observations, obs_param_config.observation)
            self.__observations.append(obs_param_config)
            # TODO: retake samples that are invalid, as .fit throws "ValueError: array must not contain infs or NaNs"
        params = np.array(params)
        # params = params.reshape(len(params), len(samples[0]))
        observations = observations.reshape(len(observations), 1)
        self.__model.fit(params, observations)

    def get(self, x: ParameterConfig) -> float:
        print("X: {}, list: {}".format(x, [x.get_as_list()]))
        return self.__model.predict([x.get_as_list()], return_std=True)

    # update the model based on a new observation
    def add(self, x: ParameterConfig):
        pass

    def get_valid_observations(self):
        return list(obs.observation for obs in self.__observations if obs.valid_observation)

    # get the best observation value so far
    def get_best_observation(self) -> float:
        return min(obs.observation for obs in self.__observations)

    # get the next candidate observation
    def get_candidate(self) -> ParameterConfig:
        return self.__af(self)


# acquisition function PI
def af_probability_of_improvement(model: SurrogateModel, explorationfactor=0) -> ParameterConfig:
    fplus = model.get_best_observation() + explorationfactor
    found_best = np.infty
    for x in model.searchspace:
        gx = model.get(x)
        found_best = min()
    return found_best


# visualize the objective function and surrogate model
def visualize():
    ss = SearchSpace(parameters)
    # brute-force the objective function
    objective_func = []
    for x in ss.search_space:
        x = evaluate_objective_function(x)
        objective_func.append(x.observation)
    # print(objective_func)
    # apply bayesian optimization
    model = SurrogateModel(ss, af_probability_of_improvement)
    print(model.get(ss.search_space[0]))
    # # visualize
    # plt.plot(objective_func, marker='.', linestyle="-", label='Objective function')
    # plt.xlabel("Parameter config index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.show()
    # # #first make some fake data with same layout as yours
    # # data = pd.DataFrame(np.random.randn(100, 10), columns=['x1', 'x2', 'x3',\
    # #                     'x4','x5','x6','x7','x8','x9','x10'])

    # # #now plot using pandas
    # # scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')


def visualize_animated():
    fig, ax = plt.subplots()
    # brute-force the objective function
    ss = SearchSpace(parameters)
    objective_func = []
    for x in ss.search_space:
        x = evaluate_objective_function(x)
        objective_func.append(x.observation)
    ax.plot(objective_func, marker='.', label='Objective function')
    # visualize
    xdata, ydata = [], []
    line, = plt.plot([], [], 'ro')

    def animation_init():
        ax.set_xlabel("Parameter config index")
        ax.set_ylabel("Value")
        ax.legend()
        # ax.set_xlim(0, 2 * np.pi)
        # ax.set_ylim(-1, 1)
        return line,

    def update(frame_number):
        # call
        xdata.append(frame_number)
        ydata.append(6 + 0.5 * np.sin(frame_number))
        line.set_data(xdata, ydata)
        return line,

    animation = FuncAnimation(fig, update, frames=np.linspace(0, 6 * np.pi, 60), init_func=animation_init, blit=True)
    plt.show()


visualize()
# visualize_animated()
