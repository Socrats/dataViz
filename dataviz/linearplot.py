# coding=ISO-8859-1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sympy import latex
from sympy.abc import x

import os

plt.style.use('ggplot')

save_dir = os.path.normpath('../results')
files = [os.path.normpath('../data/im_reg_i00_m00.csv'),
         os.path.normpath('../data/im_reg_i04_m00.csv'),
         os.path.normpath('../data/im_reg_i04_m034.csv'),
         os.path.normpath('../data/im_reg_i04_m05.csv'),
         os.path.normpath('../data/im_reg_i04_m10.csv')]


def gplot(plot_data):
    pass


def run(graphs):
    fig, ax = plt.subplots()
    plt.hold(True)
    plot_data = process_files(graphs, ax)
    for name, line in plot_data.iteritems():
        ax.plot(line['x'], line['y'], label=name)
    plt.legend(loc='upper left')
    plt.show()


def process_files(graphs, ax):
    plot_data = {}
    for file in files:
        name, data = process(file, ax)
        plot_data[name] = data
    return plot_data


def process(graph, ax):
    def getdata2(s):
        data['x'].append(s[0])
        data['y'].append(s[1])

    def getdata3(s):
        data['x'].append(s[0])
        data['y'].append(s[1])
        data['z'].append(s[2])

    data = {}

    # Get number of columns and graph legend
    f = open(graph, 'r')
    l = f.readline()
    s = l.strip().split(';')
    num_dimmensions = int(s[0])
    name = s[1]
    getdata = getdata2 if num_dimmensions == 2 else getdata3
    # Get axis names
    l = f.readline()
    s = l.strip().split(';')
    plt.xlabel(s[0])
    ax.set_ylabel(s[1])
    data['x'] = []
    data['y'] = []
    if num_dimmensions == 3:
        z = s[2]
        data['z'] = []
    for l in f.readlines():
        s = l.strip().split(';')
        getdata(s)

    f.close()

    return name, data


def plot_conf(ax, ylabel, logscale):
    ax.set_ylabel(ylabel)
    if logscale:
        ax.set_yscale('log')
    ax.set_ylim(10 ** 2, 10 ** 8)
    plt.xlim(0, 3000)
    plt.xlabel("step")
    plt.hold(True)


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
        xs = np.arange(10 ** 2, 2000 ** 2 + 4 * 10 ** 4, 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(xs)

    expr = round(coefficients[0], 15) * x ** 2 + round(coefficients[1], 5) * x + round(coefficients[2], 5)
    show = latex(expr)
    ax.plot(xs, ys, label=r'$Polynomial fit$' + '\n' + r'$y = ' + show + r'$', color='b')


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
        ax.set_xlim(0, 2000 ** 2 + 4 * 10 ** 4)
    # Finally, add a basic legend
    plt.legend(loc='upper left')
    plt.ylabel('time (s)')
    plt.xlabel("grid size (cells)")
    plt.savefig(os.path.normpath(save_dir_time + '/fit/time_results' + file_name + '.pdf'))
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
    run(files)
