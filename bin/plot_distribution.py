# coding=utf=8

import numpy as np
import pandas as pd
from pandas.tools.plotting import bootstrap_plot
import matplotlib.pyplot as plt
from bin import paths
import bin.utils as utils
import matplotlib
matplotlib.style.use('ggplot')

# data_dict = {}
# for index, payoff_file in zip(paths.prefixes, paths.payoff_files):
#     actions_vector = np.load(payoff_file)
#     actions_vector = actions_vector[0][-1000:]
#     data_dict[''.join(['f', str(index)])] = actions_vector
#
# actions_df = pd.DataFrame(data_dict)
# actions_df = actions_df[['f1', 'f2', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f20']]
# # print(actions_df)
#
# actions_df.plot.hist(stacked=True, bins=paths.bins)
# actions_df.diff().hist(color='k', alpha=0.5, bins=paths.bins)
# plt.axhline(0, color='k')
# plt.show()

# actions_files = utils.get_files(paths.root_dir, "actions")
# data = pd.DataFrame(np.load(actions_files['f9'][1])[-1000:].transpose())
# bootstrap_plot(data, size=50, samples=500, color='grey')
# plt.show()


def plot_distribution() -> dict:
    files = utils.get_files(paths.root_dir, "actions")
    return files


def file2dataframe(root_dir: str, expression: str, offset: int) -> pd.DataFrame:
    files = utils.get_files(root_dir, expression)
    data_dict = {}
    for key, values in files.items():
        i = 0
        run_dict = {}
        for file in values:
            run_dict[i] = np.load(file)[0][-offset:]
            i += 1
        data_dict[key] = run_dict

    # idx = pd.Index([i for i in range(0, offset)])

    return pd.DataFrame(data_dict)


def calculate_hist(data: np.ndarray) -> (np.ndarray, np.ndarray):
    hist_data = []
    for i in range(len(data)):
        hist_data.append(np.histogram(data[i], bins=paths.bins, density=True)[0])
    means = np.mean(hist_data, axis=0)
    errors = np.std(hist_data, axis=0)
    return means, errors


def plot_hist_with_errors(means: np.ndarray, errors: np.ndarray, index: list):
    means_df = pd.DataFrame(means, index=pd.Index(index))
    errors_df = pd.DataFrame(errors, index=index)
    fig, ax = plt.subplots()
    means_df.plot(yerr=errors_df, ax=ax, kind='bar')
    plt.show()


def hist_by_tag(data: pd.DataFrame, tag: str, index: list):
    means, errors = calculate_hist(data.eval(tag))
    plot_hist_with_errors(means, errors, index)

# data = file2dataframe(paths.root_dir, "actions", 1000)
# gp3 = data.groupby(by=('f1', 'f2', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'))
# print(data)
# print(gp3.mean())
