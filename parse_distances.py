import glob
import json
import os
import re
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def parse_distance_file(file_name: str) -> pd.DataFrame:
  experiment_type = parse_distance_file_name(file_name)
  sharing_functions = parse_sharing_functions(file_name)
  data = np.vstack([sharing_functions, [experiment_type]*len(sharing_functions)]).T
  df = pd.DataFrame(data,
                    columns=["sharing function", "experiment type"])
  return df


def parse_sharing_functions(file_name: str) -> list:
  sharing_functions = []
  with open(file_name, "r") as f:
    lines = f.readlines()
  for line in lines:
    groups = re.match(r"Fitness sharing denominator:\s+([0-9.]+)", line)
    if groups is None:
      break
    sharing_functions.append(float(groups.group(1)))
  return sharing_functions


def parse_distance_file_name(file_name: str) -> str:
  return file_name.split("/")[-1].split("_")[0]


def parse_all_files(folder: str) -> pd.DataFrame:
  files = glob.glob(folder + "/*.log")
  dfs = []
  for file in files:
    dfs.append(parse_distance_file(file))
  df = pd.concat(dfs)
  df["sharing function"] = df["sharing function"].astype(float)
  return df


def parse_file_name(file_name):
  file_name_simple = file_name.split("/")[-1]
  file_split = file_name_simple.split("_")
  filtered_split = filter(lambda x: x != "n", file_split)
  filtered_split = list(filtered_split)
  keys = filtered_split[::2][:-1]
  values = filtered_split[1::2][:-1]
  for i, value in enumerate(values):
    values[i] = float(value)
  return keys, values


def parse_file(file_name):
  keys, values = parse_file_name(file_name)
  with open(file_name, "r") as f:
    lines = f.readlines()
    generations = []
    fitnesses = []
    parse_functions = False
    for i, line in enumerate(lines):
      parsed_line = parse_generation_line(line)
      if parsed_line is False:
        parse_functions = True
        break
      generation, fitness = parsed_line
      generations.append(generation)
      fitnesses.append(fitness)
    if parse_functions:
      func_dict = parse_functions_dictionary(line.replace("'", '"'))
    else:
      func_dict = parse_functions_dictionary("{'sat_add': -1, 'sat_sub': 0, 'cgp_min': 0,"
                                             " 'cgp_max': 0, 'greater_than': 1, 'sat_mul': 1,"
                                             " 'scale_up': 0, 'scale_down': 0, 'continous_or': 0,"
                                             " 'continous_and': 0, 'multiplexor': 0}".replace("'", '"'))

    return keys, values, fitnesses, func_dict


def parse_file_all_generations(file_name):
  keys, values, fitnesses, *rest = parse_file(file_name)
  if "noffsprings" in keys:
    generations = np.linspace(0, 1600, len(fitnesses))
  else:
    generations = np.linspace(0, len(fitnesses), len(fitnesses))
  return keys, np.array([values] * len(fitnesses)).T, fitnesses, generations


def parse_experiment(experiment_folder: str):
  data = []
  keys = parse_functions_dictionary("{'sat_add': -1, 'sat_sub': 0, 'cgp_min': 0,"
                                         " 'cgp_max': 0, 'greater_than': 1, 'sat_mul': 1,"
                                         " 'scale_up': 0, 'scale_down': 0, 'continous_or': 0,"
                                         " 'continous_and': 0, 'multiplexor': 0}".replace("'", '"')).keys()
  print(keys)
  func2pd_dict = {key: [] for key in keys}
  for file_name in os.listdir(experiment_folder):
    keys, values, fitness, func_dict = parse_file(f"{experiment_folder}/{file_name}")
    for key, value in func_dict.items():
      func2pd_dict[key].append(value)
    data.append([*values, fitness[-1]])
  df = pd.DataFrame.from_dict(func2pd_dict)
  df["total functions"] = df.sum(axis=1)
  df = df.astype(int)
  df2 = pd.DataFrame(data, columns=[*keys, "fitness"])
  df2 = pd.concat([df2, df], axis=1)
  return df2


def parse_experiment_all(experiment_folder: str):
  for file_name in os.listdir(experiment_folder):
    keys, values, fitness, generations = parse_file_all_generations(f"{experiment_folder}/{file_name}")
    break
  data = np.empty((len(keys)+2, 0))
  for file_name in os.listdir(experiment_folder):
    keys, values, fitness, generations = parse_file_all_generations(f"{experiment_folder}/{file_name}")

    data = np.hstack([data, [*values, fitness, generations]])
  return pd.DataFrame(data.T, columns=[*keys, "fitness", "generation"])


def parse_generation_line(line):
  matched = re.match(r'Generation:\s+([0-9]+) Best fitness:\s+([-0-9.]+)', line)
  if matched is None:
    return False
  return int(matched.group(1)), float(matched.group(2))



def boxplot(df: pd.DataFrame, x: str, y: str, hue: str, title: str):
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111)
  sns.set_style("whitegrid")
  sns.set_context("paper")
  sns.set_palette("colorblind")
  sns.set(font_scale=1.5)
  sns.boxplot(x=x, y=y, hue=hue, data=df, ax=ax)
  ax.set_title(title)


def lineplot(df: pd.DataFrame, variable: str, error_bar):
  df = df.copy(deep=True)
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(1, 1, 1)
  pallete = sns.color_palette("bright")
  ax.set_title(f"Lineplot of {variable}")
  ax.set_xlabel("Generation")
  ax.set_ylabel("Fitness")
  # df = df[df["mutationrate"] < 0.15]
  sns.lineplot(x="generation", y="fitness", hue=variable, data=df, ax=ax, palette=pallete, errorbar=error_bar)
  # fig.savefig(f"plots_and_images/lineplot_{variable}.png")


def parse_functions_dictionary(line: str) -> Dict[str, int]:
  func_dictionary = json.loads(line)
  return func_dictionary


def t_test_all(df: pd.DataFrame, variable: str):
  """t_test between all pairs of unique values of variable"""
  df = df.copy(deep=True)
  df.sort_values(by=variable, ascending=False, inplace=True)
  unique_values = pd.unique(df[variable])
  for i, value1 in enumerate(unique_values):
    for j, value2 in enumerate(unique_values):
      df1 = df[df[variable] == value1]
      df2 = df[df[variable] == value2]
      res = stats.ttest_ind(df1['fitness'], df2['fitness'], equal_var=False, alternative='greater')
      print(f"{variable} {value1} {value2} {res}")
def barplot(df: pd.DataFrame, x:str, y:str, hue:str = None, title:str = None):
  fig = plt.figure(figsize=(12, 10))
  ax = fig.add_subplot(111)
  sns.set_style("whitegrid")
  sns.set_context("paper")
  sns.set_palette("colorblind")
  sns.set(font_scale=1.5)
  sns.barplot(x=x, y=y, hue=hue, data=df, ax=ax)
  plt.xticks(fontsize=12, rotation=45)
  if title is None:
    ax.set_title(f"Barplot of {x} and {y}")
  else:
    ax.set_title(title)

def countplot(df: pd.DataFrame, x:str, y:str, hue:str = None, title: str = None):
  fig = plt.figure(figsize=(12, 10))
  ax = fig.add_subplot(111)
  sns.set_style("whitegrid")
  sns.set_context("paper")
  sns.set_palette("colorblind")
  sns.set(font_scale=1.5)
  sns.countplot(x=x, y=y, hue=hue, data=df, ax=ax)
  plt.xticks(fontsize=12, rotation=45)
  if title is None:
    ax.set_title(f"Barplot of {x} and {y}")
  else:
    ax.set_title(title)



def nichingtype(log_dir: str):
  df = parse_experiment(log_dir)
  df["fitness"] = df["fitness"].astype(float)
  df["fitness"] = (df["fitness"] - 3000)/3
  df_all = parse_experiment_all(log_dir)
  df_all["fitness"] = df_all["fitness"].astype(float)
  df_all["fitness"] = (df_all["fitness"] - 3000)/3
  df["nichingtype"].replace(1, "vanilla", inplace=True)
  df["nichingtype"].replace(2, "dynamic mutation", inplace=True)
  df["nichingtype"].replace(3, "fitness sharing", inplace=True)
  df_all["nichingtype"].replace(1, "vanilla", inplace=True)
  df_all["nichingtype"].replace(2, "dynamic mutation", inplace=True)
  df_all["nichingtype"].replace(3, "fitness sharing", inplace=True)
  #filter out rows that have "nichingtype" == "vanilla" CANNOT use ["nichingtype" == "vanilla"]
  df_total_filtered = df[df["total functions"] != 0]
  df_total_filtered = df_total_filtered[df_total_filtered["total functions"] != 9]
  df_total_filtered = df_total_filtered[df_total_filtered["total functions"] != 11]
  df_total_filtered = df_total_filtered[df_total_filtered["total functions"] != 8]
  df_total_filtered = df_total_filtered[df_total_filtered["total functions"] != 10]
  df_total_filtered = df_total_filtered[df_total_filtered["total functions"] != 13]
  boxplot(df_total_filtered, "nichingtype", "fitness", None, "Niching type comparison")
  boxplot(df_total_filtered, "total functions", "fitness", None, "Total functions comparison")
  boxplot( df_total_filtered, "nichingtype", "total functions", None, "Total functions comparison")
  barplot(df_total_filtered, "total functions", "fitness", "nichingtype")
  countplot(df_total_filtered, "total functions", None, hue="nichingtype")

  # sum all columns which are int or float and create dataframe from them
  df_func = df[df["sat_add"]!=-1]
  df_func = df_func.select_dtypes(include=['int', 'float']).sum().to_frame()
  df_func.drop(index=["fitness", "total functions"], inplace=True)
  df_func.reset_index(inplace=True)
  df_func.rename({0: "count", "index": "function name"}, axis=1, inplace=True)
  barplot(df_func, "function name", "count")
  # lineplot(df_all, "nichingtype", error_bar=("ci", 90))
  t_test_all(df, "total functions")
  t_test_all(df, "nichingtype")
  plt.show()


def violinplot(df: pd.DataFrame, variable: str):
  fig = plt.figure(figsize=(12, 10))
  ax = fig.add_subplot(111)
  sns.set_style("whitegrid")
  sns.set_context("paper")
  sns.set_palette("colorblind")
  sns.set(font_scale=1.5)
  sns.violinplot(x=variable, y="fitness", data=df, ax=ax)
  plt.xticks(fontsize=12, rotation=0)
  ax.set_title(f"Violinplot of {variable}")

def lots_of_generations():
  df_lot_gen = parse_experiment("experiment_lots_of_generations")
  df_few_gen = parse_experiment("logs_LUNAR_LANDER_parents")
  df_few_gen["fitness"]= df_few_gen["fitness"].astype(float)
  df_few_gen["fitness"] = (df_few_gen["fitness"] - 3000)/3

  df = pd.concat([df_lot_gen, df_few_gen[df_few_gen["nichingtype"]==2]], copy=True)
  df_lot_gen_all = parse_experiment_all("experiment_lots_of_generations")
  df_few_gen_all = parse_experiment_all("logs_LUNAR_LANDER_parents")
  df_few_gen_all["fitness"]= df_few_gen_all["fitness"].astype(float)
  df_few_gen_all["fitness"] = (df_few_gen_all["fitness"] - 3000)/3
  df_all = pd.concat([df_lot_gen_all, df_few_gen_all[df_few_gen_all["nichingtype"]==2]])
  df["budget distribution"] = df["nichingtype"].copy()

  df["budget distribution"].replace(2, "500 gens 20 offspring", inplace=True)
  df["budget distribution"].replace(4, "41 gens 400 offspring", inplace=True)
  df_all["budget distribution"] = df_all["nichingtype"].copy()
  df_all["budget distribution"].replace(2, "500 gens 20 offspring", inplace=True)
  df_all["budget distribution"].replace(4, "41 gens 400 offspring", inplace=True)
  df_all.loc[df_all["nichingtype"]==2,"generation"] = df_all.loc[df_all["nichingtype"]==2,"generation"]*20
  df_all.loc[df_all["nichingtype"]==4,"generation"] = df_all.loc[df_all["nichingtype"]==4,"generation"]*400
  df_all = df_all[df_all["generation"]<(25*400)]


  df["fitness"] = df["fitness"].astype(float)
  boxplot(df, "budget distribution", "fitness", None, "Niching type comparison")
  lineplot(df_all, "budget distribution", error_bar=("ci", 90))
  df_lot_gen["budget distribution"] = df_lot_gen["nichingtype"].copy()
  df_lot_gen["budget distribution"].replace(4, "41 gens 400 offspring", inplace=True)
  violinplot(df_lot_gen, "budget distribution")
  df_lot_gen_filtered = df_lot_gen[df_lot_gen["total functions"] != 9]
  df_lot_gen_filtered = df_lot_gen_filtered[df_lot_gen_filtered["total functions"] != 10]
  barplot(df_lot_gen_filtered, "total functions", "fitness")
  plt.show()

if __name__ == "__main__":
  # VIOLIN PLOT FOR 400 generations
  # nichingtype("logs_LUNAR_LANDER_parents")
  # lots_of_generations()
  # df = parse_all_files("logs_average_distance")
  # print(df)
  # boxplot(df, x="experiment type", y="sharing function", hue=None, title="Sharing function comparison")
  # for file in os.listdir("experiment_lots_of_generations"):
  #   os.rename("experiment_lots_of_generations/" + file, "experiment_lots_of_generations/" + file.replace("_re", "nichingtype_4_re"))

  x = np.linspace(0,15, 1000)
  y = pow((x+1)/13, 3/5)
  y[np.where(x > 12)[0]] = 1
  plt.plot(x, y)
  plt.xlabel("Number of active functions")
  plt.ylabel("Regulating ratio")
  plt.title("Ratio used to upregulate more active functions")
  plt.show()
