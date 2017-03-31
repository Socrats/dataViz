# coding=ISO-8859-1
import numpy as np
from sympy import latex
from sympy.abc import x


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
