import numpy as np
import pandas as pd

import dataviz.distplot as distplot
import bin.paths as paths


def load_file(path: str, file: str):
    return np.load("{}/{}".format(path, file))


def make_plots(df: pd.DataFrame, bins=np.arange(1, 11)):
    df.plot()
    df.hist(bins=bins)


def plot_dynamics(var: np.array, save_path: str, name: str):
    df = pd.DataFrame(var)
    ax = df.plot(figsize=(10, 5))
    fig = ax.get_figure()
    fig.savefig(''.join([save_path, name]))


def get_adversarial_data(var: str):
    path = "/Users/eliasfernandez/PreplayResults/experiments/Adversarial/AntDictAntRec/" \
           "f10f10b{beta_d}b03k0001k{k_r}m10/data"
    file = "{date}_AntDictAntRec_f10f10b{beta_d}b03k0001k{k_r}m10_{num}_{run}_{var}-9999.npy"
    data_array = load_file(path.format(beta_d=99, k_r=1),
                           file.format(date='17-05-17_17-56-12', beta_d=99, k_r=1, num=9, run=0,
                                       var=var))

    return data_array


if __name__ == "__main__":
    sort_dict = "AntDictAntRec"
    param_num = 3
    file_num = 2
    formt = 'pdf'
    variable = 'action'
    offset = -50
    # dir_check = "{}{}/data/".format(paths.root_dir, paths.tested_params[param_num])
    data = distplot.file2array(paths.root_dir, variable, 1000, paths.tested_params, size=9999)
    tst = data[paths.tested_params[param_num]]
    means, errors = distplot.calculate_hist(tst, paths.bins, density=True)
    distplot.plot_hist_with_errors(means, errors, paths.bins, ["AntDictAntRec"], paths.save_path,
                                   "{}_{}.{}".format(paths.tested_params[param_num], variable, formt))
    # tst = data[paths.tested_params[param_num]]
    # means, errors = distplot.calculate_avg_behavior(data)
    # distplot.beautiful_box_plot(means, paths.save_path,
    #                             name='boxplot_{}_avg_behavior.{}'.format(variable, formt))

    # data = np.load(paths.ADVERSARIAL_FILES[paths.tested_params[param_num]][file_num].format(var=variable))
    # plot_dynamics(data[offset:], paths.save_path,
    #               "{}_{}.{}".format(paths.tested_params[param_num], variable, formt))

    #
    # # Plot dynamics
    # for param_tested in paths.tested_params:
    #     print(data[param_tested])
    #     ax = data[param_tested].plot()
    #     fig = ax.get_figure()
    #     fig.savefig(''.join([paths.save_path, "{}_probabilities.{}".format(param_tested, formt)]))
    #     # plot_dynamics(data[param_tested], paths.save_path,
    #     #               "{}_probabilities.{}".format(param_tested, formt))
