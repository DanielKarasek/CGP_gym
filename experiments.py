import logging
import os
from itertools import product
from typing import Any, Callable, Dict, List, Tuple

import cgp
import gym
import numpy as np
from tqdm import tqdm

from functions import sat_add, sat_sub, cgp_min, cgp_max, greater_than, sat_mul, scale_up, scale_down, multiplexor, \
  continous_or, continous_and
from lunar_lander import CGPGymWrapper


def generate_experiments_from_settings(settings: Dict, experiment_names: List[str] = []):
  experiment_varying_vars = []
  for key, value in settings.items():
    if isinstance(value, list):
      experiment_varying_vars.append(key)
      print(key)
  assert (len(experiment_varying_vars) < 3 or
          len(experiment_varying_vars) > 0 or
          "No more than two parameters should be varied at the same time")
  varying_params = []
  if len(experiment_varying_vars) == 2:
    for first_setting, second_setting in product(settings[experiment_varying_vars[0]],
                                                 settings[experiment_varying_vars[1]]):
       varying_params.append({experiment_varying_vars[0]: first_setting,
                              experiment_varying_vars[1]: second_setting})
  elif len(experiment_varying_vars) == 1:
    varying_params = [{experiment_varying_vars[0]: setting} for
                      setting in settings[experiment_varying_vars[0]]]
  else:
    varying_params = [{}]  # No varying parameters
  for key in experiment_varying_vars:
    del settings[key]
  if experiment_names:
    experiments = (Experiment(**settings,
                              **varying_params_group,
                              experiment_name=experiment_names[i],
                              experimented_values=varying_params_group) for
                   i, varying_params_group in enumerate(varying_params))
  else:
    print("No experiment names provided, using default name")
    experiments = (Experiment(**settings,
                              **varying_params_group,
                              experimented_values=varying_params_group) for
                   varying_params_group in varying_params)

  return experiments


class Experiment:
  def __init__(self, parents: int, n_inputs: int, n_outputs: int, n_columns: int,
               n_rows: int, levelsback: int, primitives: Tuple, noffsprings: int,
               mutationrate: float, generations: int, tournament_size: float, terminal_fitness: int, fitness_share_params: Dict[Any, Any],
               experimented_values: Dict[str, Any], task_wrapper: CGPGymWrapper, use_logger: bool = True,
               experiment_name: str = "", experiment_type: str = "regression", *args, **kwargs):
    self.experiment_type = experiment_type
    self.experiment_name = experiment_name
    self.repetition_id = 0
    self.end_log_function = lambda exp_logger, pop: None
    self.objective = task_wrapper.objective
    self.task_wrapper = task_wrapper
    self.experimented_values = experimented_values
    self.use_logger = use_logger
    self.population_params = {"n_parents": parents, "seed": np.random.randint(0, 1e7)}
    self.terminal_fitness = terminal_fitness
    self.genome_params = {"n_inputs": n_inputs,
                          "n_outputs": n_outputs,
                          "n_columns": n_columns,
                          "n_rows": n_rows,
                          "levels_back": levelsback,
                          "primitives": primitives}
    self.ea_params = {"n_offsprings": noffsprings,
                      "mutation_rate": mutationrate,
                      "n_processes": 64,
                      "tournament_size": tournament_size,
                      **fitness_share_params}

    self.max_generations = generations

  def __repr__(self):
    return f"Experiment with experiment values: {self.experimented_values}"

  def __str__(self):
    return f"Experiment with experiment values: {self.experimented_values}"

  def add_end_log_function(self, end_log_function: Callable):
    self.end_log_function = end_log_function

  def _init_logger(self):
    file_path = f"./logs_{self.experiment_type}"
    for key in self.experimented_values.keys():
      file_path = f"{file_path}_{key}"
    try:
      os.mkdir(file_path)
    except FileExistsError as e:
      pass
    file_path = f"{file_path}/"
    if self.experiment_name and len(self.experimented_values.items()) > 0:
      file_path = f"{file_path}{self.experiment_name}"
    else:
      for key, value in self.experimented_values.items():
        file_path = f"{file_path}{key}_{value}"
    file_path = f"{file_path}_repetition_{self.repetition_id}"

    experiment_logger = logging.getLogger("experiment_logger")
    for handler in experiment_logger.handlers[:]:
      experiment_logger.removeHandler(handler)
    file_handler = logging.FileHandler(f"{file_path}.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    experiment_logger.addHandler(file_handler)
    experiment_logger.setLevel(logging.INFO)

    print(f"Logging to {file_path}.log")

    return experiment_logger

  def log_generation(self, pop: cgp.Population):
    if self.use_logger:
      best_fitness = pop.champion.shitness
      self.logger.info(f"Generation: {self.pop.generation} Best fitness: {best_fitness}")

  def _run(self) -> cgp.Population:
    self.population_params["seed"] = np.random.randint(1e7)
    self.pop = cgp.Population(**self.population_params, genome_params=self.genome_params)
    ea = cgp.ea.MuPlusLambda(**self.ea_params, repetition_id = self.repetition_id)

    last_best = -99999
    last_improvement_step = 0

    cgp.evolve(self.pop,
               self.objective,
               ea)
    self.log_generation(self.pop)
    mutation_rate_increase_accumulator = 0
    with tqdm(total=100) as pbar:
      for i in np.arange(self.max_generations):
        solved = cgp.hl_api.evolve_continue(self.pop,
                                            self.objective,
                                            ea,
                                            terminal_fitness=self.terminal_fitness)

        self.log_generation(self.pop)

        self.pop.champion.genome.calculate_count_per_function()
        self.pop.generation = 0
        mutation_rate_increase_accumulator += 1
        if self.pop.champion.orig_fitness > last_best:
          last_best = self.pop.champion.orig_fitness
          last_improvement_step = i

          ea._mutation_rate = self.ea_params["mutation_rate"]
          self.pop.n_parents = self.population_params["n_parents"]

          mutation_rate_increase_accumulator = 0

        if mutation_rate_increase_accumulator > (50/self.ea_params["n_offsprings"]):
          ea._mutation_rate = np.clip(ea._mutation_rate * 1.3, 0, self.ea_params["mutation_rate"] * 1.6)
          self.pop.n_parents = np.clip(self.pop.n_parents + 4, self.population_params["n_parents"],
                                       self.population_params["n_parents"]*1.8)
          self.pop.n_parents = int(self.pop.n_parents)
          mutation_rate_increase_accumulator = 0

        pbar.update(100 / self.max_generations)
        pbar.set_description(f"Generation {i} Last improvement: {last_improvement_step}")
        pbar.set_postfix_str(
          f"Best fitness: {last_best}, Mutation rate: {ea._mutation_rate:.2f}, Parents: {self.pop.n_parents}")
        if solved:
          break
    self.task_wrapper.render(self.pop.champion, True, f"{self.repetition_id}.mp4")
    print(cgp.CartesianGraph(self.pop.champion.genome).pretty_str())
    # with open(f"regression_actual_best_{self.experiment_name}.pkl", "wb") as f:
    #   pickle.dump(self.pop.champion, f)
    if self.end_log_function:
      self.end_log_function(self.logger, self.pop)
    return self.pop

  def run(self, repetitions: int):
    for repetition in range(repetitions):
      self.repetition_id = repetition
      if self.use_logger:
        self.logger = self._init_logger()
      self._run()


def regression_experiments():
  # experiment_settings = {
  #   "parents": 7,
  #   "n_outputs": 2,
  #   "n_inputs": 8,
  #   "n_columns": 5,
  #   "n_rows": 5,
  #   "levelsback": 3,
  #   "primitives": (
  #     sat_add, sat_sub,
  #     cgp_min, cgp_max,
  #     greater_than, sat_mul, scale_up, greater_than,
  #     scale_down,
  #     continous_or, continous_and, multiplexor),
  #   "noffsprings": 20,
  #   "mutationrate": 0.07,
  #   "generations": 500,
  #   "experiment_type": "LUNAR_LANDER",
  #   "termination_fitness": 1000,
  #   "tournament_size": 2,
  #   "use_logger": True,
  #   "fitness_share_params": {"fitness_sharing_flag": False, "fitness_sharing_alpha": 2, "fitness_sharing_sigma": 0.3},
  # }

  experiment_settings = {
    "parents": 80,
    "n_outputs": 2,
    "n_inputs": 8,
    "n_columns": 5,
    "n_rows": 5,
    "levelsback": 3,
    "primitives": (
      sat_add, sat_sub,
      cgp_min, cgp_max,
      greater_than, sat_mul, scale_up, greater_than,
      scale_down,
      continous_or, continous_and, multiplexor),
    "noffsprings": 400,
    "mutationrate": 0.05,
    "generations": 41,
    "experiment_type": "LUNAR_LANDER",
    "terminal_fitness": 800,
    "tournament_size": 15,
    "use_logger": True,
    "fitness_share_params": {"fitness_sharing_flag": False, "fitness_sharing_alpha": 2, "fitness_sharing_sigma": 0.3},
  }

  env = gym.make("LunarLander-v2",
                 continuous=True,
                 render_mode="rgb_array",)

  gym_wrapper = CGPGymWrapper(env)
  experiment_settings["task_wrapper"] = gym_wrapper

  experiments = generate_experiments_from_settings(experiment_settings)

  for experiment in experiments:
    experiment.add_end_log_function(gym_wrapper.log_end)
    experiment.run(repetitions=40)

if __name__ == "__main__":
  regression_experiments()

