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


def calculate_hist(data: np.ndarray, density=False) -> (np.ndarray, np.ndarray):
    hist_data = []
    for i in range(len(data)):
        hist_data.append(np.histogram(data[i], bins=paths.bins, density=density)[0])
    means = np.mean(hist_data, axis=0)
    errors = np.std(hist_data, axis=0)
    return means, errors


def plot_hist_with_errors(means: np.ndarray, errors: np.ndarray, index: list, column: list):
    means_df = pd.DataFrame(means, columns=column)
    errors_df = pd.DataFrame(errors, columns=column)
    fig, ax = plt.subplots()
    means_df.plot(yerr=errors_df, ax=ax, kind='bar')
    plt.savefig(''.join([paths.save_path, column[0], '_actions_hist.svg']))


def hist_by_tag(data: pd.DataFrame, tag: str, index: list, density=False):
    means, errors = calculate_hist(data.eval(tag), density=density)
    plot_hist_with_errors(means, errors, index, [tag])

# data = file2dataframe(paths.root_dir, "actions", 1000)
# gp3 = data.groupby(by=('f1', 'f2', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10'))
# print(data)
# print(gp3.mean())

if __name__ == "__main__":
    idx = [i for i in range(0, 11)]
    data = file2dataframe(paths.root_dir, "actions", 1000)
    for el in paths.prefixes:
        if el == 2: continue
        hist_by_tag(data, ''.join(['f', str(el)]), idx, True)
