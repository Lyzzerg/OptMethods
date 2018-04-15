import random
from typing import List
from tkinter import *

import numpy as np
from matplotlib import pyplot as plt


def get_test_function_method_min(n: int, a: List[List[float]], c: List[List[float]],
                                 p: List[List[float]], b: List[float]):
    """
    :param n: количество экстремумов
    :param a: список коэффициентов крутости экстремумов, чем выше значения,
              тем быстрее функция убывает/возрастает и тем уже область экстремума
    :param c: список координат экстремумов
    :param p: список степеней гладкости в районе экстремума
    :param b: список значений функции
    :return: возвращает функцию, которой необходимо передавать одномерный список
координат точки, возвращаемая функция вернет значение тестовой функции в данной точке
    """

    def func(x):
        l = []
        for i in range(n):
            res = 0
            for j in range(len(x)):
                res = res + a[i][j] * np.abs(x[j] - c[i][j]) ** p[i][j]
            res = res + b[i]
            l.append(res)
        res = np.array(l)
        return np.min(res)

    return func


def rand_func_generator(intervals: List[float], extremes_num: int = 1, params_num: int = 1):
    """
    :param intervals:
    :param extremes_num:
    :param params_num:
    :return: function with random params
    """
    global accuracy

    def rand():
        res = random.uniform(intervals[0], intervals[1])
        return res

    extremes_coeff = [[random.random() for _ in range(params_num)] for _ in range(extremes_num)]
    extremes_coord = [[(int(rand() * accuracy) / accuracy) for _ in range(params_num)] for _ in
                      range(extremes_num)]
    smoothness_near_extremes_coeff = [[random.random()*10 for _ in range(params_num)] for _ in range(extremes_num)]
    values = [(1 + i) / accuracy for i in range(extremes_num)]
    res_func = get_test_function_method_min(extremes_num, extremes_coeff, extremes_coord,
                                            smoothness_near_extremes_coeff, values)

    return res_func, extremes_coord


def slice1d(value, func_params: int = 1):
    res = [value for _ in range(func_params)]
    return res


def find_min(func, grid, func_params: int = 1):
    x_min = grid[0]
    y_min = func(slice1d(grid[0], func_params))
    for el in grid:
        y_temp = func(slice1d(el, func_params))
        if y_temp <= y_min:
            x_min = el
            y_min = y_temp

    return x_min, y_min


def enumeration_method(func, steps: List[float], a: float = 0, b: float = 1, n: int = 2,
                       epsilon: float = 0.001, func_params: int = 1):
    """
    :param func_params:
    :param steps:
    :param func: function
    :param a: [A,b]
    :param b: [a,B]
    :param n: steps of grid
    :param epsilon: maximum inaccuracy
    :return:
    """

    grid_x = [(a + i * (b - a) / (n + 1)) for i in range(1, n)]

    x_min_, y_min_ = find_min(func, grid_x, func_params)

    steps.append(x_min_)
    interval = [(x_min_ - (b - a) / (n + 1)), (x_min_ + (b - a) / (n + 1))]
    inaccuracy = (b - a) / (n + 1)
    if inaccuracy > epsilon:
        interval, inaccuracy = enumeration_method(func, steps, interval[0], interval[1], n+1, epsilon, func_params)

    return interval, inaccuracy


def input_and_close_window():
    global ab_interval
    global eps
    global grid_step
    global a
    global b
    global eps1
    global grid_step1
    ab_interval[0] = float(a.get())
    ab_interval[1] = float(b.get())
    eps = float(eps1.get())
    grid_step = int(grid_step1.get())
    root.destroy()


def close_window():
    root.destroy()


ab_interval = [-1, 1]
eps = 0.001
grid_step = 10

root = Tk()
root.title("Input")
a = StringVar()
b = StringVar()
eps1 = StringVar()
grid_step1 = StringVar()
label = Label(root, text="a").grid(row=0, column=0, sticky=E)
label1 = Label(root, text="b").grid(row=1, column=0, sticky=E)
label2 = Label(root, text="eps").grid(row=2, column=0, sticky=E)
label3 = Label(root, text="step").grid(row=3, column=0, sticky=E)
textbox = Entry(root, textvariable=a).grid(row=0, column=1, sticky=E)
textbox1 = Entry(root, textvariable=b).grid(row=1, column=1, sticky=E)
textbox2 = Entry(root, textvariable=eps1).grid(row=2, column=1, sticky=E)
textbox3 = Entry(root, textvariable=grid_step1).grid(row=3, column=1, sticky=E)
btn = Button(root, text="ok", command=input_and_close_window).grid(row=4, column=0, sticky=W)
btn1 = Button(root, text="default", command=close_window).grid(row=4, column=1, sticky=W)
root.mainloop()

random.seed()

accuracy = 100
extremes_count = func_count = 5

test_func, extremes_x = rand_func_generator([-10, 10], extremes_count)
extremes_y = [test_func(x) for x in extremes_x]

X = np.arange(-10, 10, 1 / accuracy)
Y = []

for x in X:
    x_slice = [x for _ in range(1)]
    Y.append(test_func(x_slice))

steps_x = []
interval_x, inaccuracy_ = enumeration_method(test_func, steps_x, ab_interval[0], ab_interval[1], grid_step, eps)

steps_y = [test_func([x for _ in range(1)]) for x in steps_x]
interval_y = [test_func([x for _ in range(1)]) for x in interval_x]

x_min, y_min = find_min(test_func, [i[0] for i in extremes_x])


#
step_points = plt.scatter(steps_x, steps_y, c='b')

last_step_point = plt.scatter(steps_x[len(steps_x)-1], steps_y[len(steps_y)-1], c='k')

extremes_points = plt.scatter(extremes_x, extremes_y, c='y')

extr_point = plt.scatter(x_min, y_min, c='r')

min_point_interval = plt.scatter(interval_x, interval_y, c='g')

plt.axvline(x=ab_interval[0], c='r', linestyle='--')
plt.axvline(x=ab_interval[1], c='r', linestyle='--')

graph = plt.plot(X, Y)
grid1 = plt.grid(True)
plt.show()
