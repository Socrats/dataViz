# coding=utf=8

import numpy as np
import pandas as pd
# from pandas.tools.plotting import bootstrap_plot
import matplotlib.pyplot as plt
import dataviz.utils as utils
import matplotlib

matplotlib.style.use('ggplot')


def file2dataframe(root_dir: str, expression: str, offset: int, sort_columns: dict) -> pd.DataFrame:
    """
    gets data with and :offset from files at :root_dir that follow the pattern 
    :expression and :returns a pandas.DataFrame with the data
    """
    files = utils.get_files(root_dir, expression, [key for key, values in sort_columns.items()])
    data_dict = {}
    for key, values in files.items():
        i = 0
        run_dict = {}
        for file in values:
            run_dict[i] = np.load(file)[0][-offset:]
            i += 1
        data_dict[key] = run_dict

    # idx = pd.Index([i for i in range(0, offset)])

    return pd.DataFrame(pd.DataFrame(data_dict), columns=sorted(sort_columns, key=sort_columns.get))


def calculate_hist(data: np.ndarray, bins: list, density=False) -> (np.ndarray, np.ndarray):
    hist_data = []
    for i in range(len(data)):
        hist_data.append(np.histogram(data[i], bins=bins, density=density)[0])
    means = np.mean(hist_data, axis=0)
    errors = np.std(hist_data, axis=0)
    return means, errors


def plot_hist_with_errors(means: np.ndarray, errors: np.ndarray, index: list, column: list, save_path: str):
    means_df = pd.DataFrame(means, columns=column)
    errors_df = pd.DataFrame(errors, columns=column)
    fig, ax = plt.subplots()
    means_df.plot(yerr=errors_df, ax=ax, kind='bar')
    plt.savefig(''.join([save_path, column[0], '_actions_hist.svg']))


def hist_by_tag(data: pd.DataFrame, tag: str, index: list, bins: list, save_path: str, density=False):
    means, errors = calculate_hist(data.eval(tag), bins, density=density)
    plot_hist_with_errors(means, errors, index, [tag], save_path)


def calculate_avg_behavior(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    means_dict = {}
    errors_dict = {}
    for column in data.columns:
        means_dict[column] = []
        errors_dict[column] = []
        for row in data[column]:
            means_dict[column].append(np.mean(row))
            errors_dict[column].append(np.std(row))

    return pd.DataFrame(means_dict, columns=data.columns), pd.DataFrame(errors_dict, columns=data.columns)


def beautiful_box_plot(df: pd.DataFrame, save_path: str):
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                 medians='DarkBlue', caps='Gray')

    ax = df.plot.box(color=color, sym='r+')
    fig = ax.get_figure()
    fig.savefig(''.join([save_path, 'boxplot_actions_avg_behavior.svg']))
