# coding=ISO-8859-1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sympy import latex
from sympy.abc import x

import os


save_dir = os.path.normpath('')
files = [os.path.normpath(''),
         os.path.normpath('')]
save_dir_time = os.path.normpath('')
GRAPH_COMBINATIONS = [['count', 'reproductions'], ['count', 'algae'], ['algae', 'reproductions'], 'count', 'algae',
                      'reproductions']
ALGAE_PROB = [0.1, 0.5, 0.9]
time_dict = {}


def run():
    for graphs in GRAPH_COMBINATIONS:
        if isinstance(graphs, list):
            process_files(graphs)
            save_plot('foram_pop_5000_limit_algae_5000_energy_need_0.5_compare_' + graphs[0] + '_' + graphs[1])
        else:
            fig, ax = plt.subplots()
            process(graphs, ax)
            save_plot('foram_pop_5000_limit_algae_5000_energy_need_0.5_' + graphs + '_algae_prob_compare')


def process(graph_type, ax):
    if graph_type == 'algae':
        ylabel = 'number of algae'
    elif graph_type == 'reproductions':
        ylabel = 'number of reproduction calls'
    else:
        ylabel = 'forams count'

    plot_conf(ax, ylabel, True)
    for algae_prob in ALGAE_PROB:
        x = []
        y = []
        std = []
        graph_file = os.path.normpath('' +
                                      graph_type + '/forams_population_5000_algae_limit_5000/energy_need_0.5/'
                                                   'TEST_RESULTS_FORAMS_' + graph_type + '_100_algae_prob_' +
                                      str(algae_prob) + '.txt')

        f = open(graph_file, 'r')
        for l in f.readlines()[1:]:
            s = l.strip().split(' ')
            x.append(int(s[0]))
            y.append(float(s[1]))
            std.append(float(s[2]))

        f.close()
        legend = (graph_type + ' algae_prob=' + str(algae_prob))
        plot([x, y, std], ax, legend)


def process_files(graphs):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    axis = [ax1, ax2]
    # plt.ylim(0, 10**8)
    for graph, ax in zip(graphs, axis):
        process(graph, ax)


def process_time_file():
    time_types = ['_v1', '_v2', '_3d', '_3d_v2']

    for name in time_types:
        time_dict.clear()
        graph_file = os.path.normpath(save_dir_time + '/time_results' + name + '.log')

        f = open(graph_file, 'r')
        for l in f.readlines():
            s = l.strip().split(' ')
            if name == '_3d' or name == '_3d_v2':
                grid = int(s[0]) ** 3
            else:
                grid = int(s[0]) ** 2
            # grid = int(s[0])
            time = float(s[1])

            add_value_to_dict(time_dict, grid, time)

        f.close()
        plot_time(time_dict, name)
        plot_time_nofit(time_dict, name)
        plot_time_error_bars(time_dict, name)
        plot_time_error_bars_nofit(time_dict, name)
        plot_fit(time_dict, name)


def add_value_to_dict(dict, key, val):
    temp = dict.get(key, [])
    temp.append(val)
    dict[key] = temp


def decimate(values, fact):
    res = []
    if isinstance(values[0], list):
        for array in values:
            array = array[0:len(array):fact]
            res.append(array)
    else:
        array = values[0:len(values):fact]
        return array

    return res


# TODO: use ks-test to find which deg fits better the points
def multi_poly_fit(x_time, y, ax, type_name):
    if type_name == '_3d' or type_name == '_3d_v2':
        coefficients = np.polyfit(x_time, y, deg=2)
        xs = np.arange(10 ** 3, 100 ** 3 + 10 ** 4, 1)
    else:
        coefficients = np.polyfit(x_time, y, deg=2)
        xs = np.arange(10 ** 2, 2000 ** 2 + 4*10 ** 4, 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(xs)

    expr = round(coefficients[0], 15) * x ** 2 + round(coefficients[1], 5) * x + round(coefficients[2], 5)
    show = latex(expr)
    ax.plot(xs, ys, label=r'$Polynomial fit$' + '\n' + r'$y = ' + show + r'$', color='b')


def plot_conf(ax, ylabel, logscale):
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_yscale('log')
    ax.set_ylim(10 ** 2, 10 ** 8)
    plt.xlim(0, 3000)
    plt.xlabel("step")
    plt.hold(True)


def plot_time(dictionary, file_name):
    x = sorted(dictionary.keys())
    y = [np.mean(dictionary[key]) for key in x]

    # df = pd.DataFrame(np.array([dictionary[key] for key in x]).T, columns=x)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    multi_poly_fit(x, y, ax, file_name)
    # Save the default tick positions, so we can reset them...
    locs, labels = plt.xticks()
    if file_name == '_3d' or file_name == '_3d_v2':
        bp = ax.boxplot([dictionary[key] for key in x], positions=x, widths=10 ** 4)
    else:
        bp = ax.boxplot([dictionary[key] for key in x], positions=x, widths=4*10 ** 4)

    ax.errorbar(0, 0, yerr=0, fmt='o', color='black', label=r'$Experimental$ $values$')
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xticklabels(x)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Reset the xtick locations.
    plt.xticks(locs)
    if file_name == '_3d' or file_name == '_3d_v2':
        ax.set_xlim(0, 100 ** 3 + 10 ** 4)
    else:
        ax.set_xlim(0, 2000 ** 2 + 4*10 ** 4)
    ax.set_ylim(0)
    # Finally, add a basic legend
    plt.legend(loc='lower right')
    plt.ylabel('time (s)')
    plt.xlabel("grid size (cells)")
    plt.savefig(os.path.normpath(save_dir_time + '/box&wiskers/time_results' + file_name + '.pdf'))
    plt.close()


def plot_time_nofit(dictionary, file_name):
    x = sorted(dictionary.keys())
    y = [np.mean(dictionary[key]) for key in x]

    # df = pd.DataFrame(np.array([dictionary[key] for key in x]).T, columns=x)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    multi_poly_fit(x, y, ax, file_name)

    # Save the default tick positions, so we can reset them...
    locs, labels = plt.xticks()
    if file_name == '_3d' or file_name == '_3d_v2':
        bp = ax.boxplot([dictionary[key] for key in x], positions=x, widths=10 ** 4)
    else:
        bp = ax.boxplot([dictionary[key] for key in x], positions=x, widths=4*10 ** 4)

    ax.lines.pop(0)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xticklabels(x)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Reset the xtick locations.
    plt.xticks(locs)
    if file_name == '_3d' or file_name == '_3d_v2':
        ax.set_xlim(0, 100 ** 3 + 30000)
    else:
        ax.set_xlim(0, 2000 ** 2 + 100000)
    ax.set_ylim(0)
    # Finally, add a basic legend
    plt.ylabel('time (s)')
    plt.xlabel("grid size (cells)")
    plt.savefig(os.path.normpath(save_dir_time + '/box&wiskers_nofit/time_results' + file_name + '.pdf'))
    plt.close()


def plot_fit(dictionary, file_name):
    x = sorted(dictionary.keys())
    y = [np.mean(dictionary[key]) for key in x]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    multi_poly_fit(x, y, ax, file_name)

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xticklabels(x)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if file_name == '_3d' or file_name == '_3d_v2':
        ax.set_xlim(0, 100 ** 3 + 10 ** 4)
    else:
        ax.set_xlim(0, 2000 ** 2 + 4*10 ** 4)
    # Finally, add a basic legend
    plt.legend(loc='upper left')
    plt.ylabel('time (s)')
    plt.xlabel("grid size (cells)")
    plt.savefig(os.path.normpath(save_dir_time + '/fit/time_results' + file_name + '.pdf'))
    plt.close()


def plot_time_error_bars(dictionary, file_name):
    x = sorted(dictionary.keys())
    y = [np.mean(dictionary[key]) for key in x]
    std = [np.std(dictionary[key]) for key in x]
    f = open(
        os.path.normpath(
            save_dir_time + '/errorbars/time_results' + file_name + ".ssv"),
        'w')
    for xi, yi, stdi in zip(x, y, std):
        f.write(str(xi) + " " + str(yi) + " " + str(stdi) + "\n")
    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    multi_poly_fit(x, y, ax, file_name)
    ax.errorbar(x, y, std, fmt='o', color='OrangeRed', label=r'$Experimental$ $results$')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if file_name == '_3d' or file_name == '_3d_v2':
        ax.set_xlim(0, 100 ** 3 + 10 ** 4)
    else:
        ax.set_xlim(0, 2000 ** 2 + 4*10 ** 4)
    # Finally, add a basic legend
    plt.legend(loc='lower right')
    plt.ylabel('time (s)')
    plt.xlabel("grid size (cells)")
    plt.savefig(os.path.normpath(save_dir_time + '/errorbars/time_results' + file_name + '.pdf'))
    plt.close()


def plot_time_error_bars_nofit(dictionary, file_name):
    x = sorted(dictionary.keys())
    y = [np.mean(dictionary[key]) for key in x]
    std = [np.std(dictionary[key]) for key in x]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.errorbar(x, y, std, fmt='o')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if file_name == '_3d' or file_name == '_3d_v2':
        ax.set_xlim(0, 100 ** 3)
    else:
        ax.set_xlim(0, 2000 ** 2)
    # Finally, add a basic legend
    plt.ylabel('time (s)')
    plt.xlabel("grid size (cells)")
    plt.savefig(os.path.normpath(save_dir_time + '/errorbars_nofit/time_results' + file_name + '.pdf'))
    plt.close()


def plot(line, ax, legend):
    ax_dec = decimate(line, 20)

    ax.errorbar(ax_dec[0], ax_dec[1], yerr=ax_dec[2], fmt='-o', label=legend)
    # ax.errorbar(0, 0, yerr=0, fmt='-or', label=legend)


def save_plot(plotfilename):
    plt.legend(loc='lower right')
    plt.savefig(os.path.normpath(save_dir + '/' + plotfilename + '.pdf'))
    plt.close()


def plot_box_whisker(dict, ylabel, plotfilename):
    x = sorted(dict.keys())
    x_dec = decimate(x, 20)
    df = pd.DataFrame(np.array([dict[key] for key in x_dec]).T, columns=x_dec)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bp = ax.boxplot(df.values, widths=0.5)

    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    plt.ylabel(ylabel)
    plt.xlabel("step")
    plt.savefig(plotfilename + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pdf')
    plt.close()


def mult_plot_box_whisker(multiple_dicts, ylabel, plotfilename, legend, logscale):
    pass


def comp_gen():
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    axis = [ax1, ax2]

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    std1 = []
    std2 = []

    f1 = open(file[0], 'r')
    f2 = open(file[1], 'r')
    for l1, l2 in zip(f1.readlines(), f2.readlines()):
        s1 = l1.strip().split(';')
        s2 = l2.strip().split(';')
        x1.append(int(s1[0]))
        y1.append(float(s1[1]))
        std1.append(float(s1[2]))
        x2.append(int(s2[0]))
        y2.append(float(s2[1]))
        std2.append(float(s2[2]))

    f1.close()
    f2.close()
    plot([x1, y1, std1], ax1, 'emas_grid')
    plot([x1, y1, std1], ax1, 'emas_migration')


if __name__ == '__main__':
    # run()
    process_time_file()