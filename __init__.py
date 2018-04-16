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
    smoothness_near_extremes_coeff = [[random.random() * 10 for _ in range(params_num)] for _ in range(extremes_num)]
    values = [(1 + i) / accuracy for i in range(extremes_num)]
    res_func = get_test_function_method_min(extremes_num, extremes_coeff, extremes_coord,
                                            smoothness_near_extremes_coeff, values)

    return res_func, extremes_coord


def enumeration(func, X, eps: float = 0.1):
    R = [(X[i] - X[i - 1]) for i in range(1, len(X))]
    tmp_max = R[0]
    res = 0
    for i in range(1, len(R)):
        if R[i] >= tmp_max:
            tmp_max = R[i]
            res = i

    X.append((X[res] + X[res + 1]) / 2)
    X.sort()
    if np.max([(X[i] - X[i - 1]) for i in range(1, len(X))]) > eps:
        X = enumeration(func, X, eps)

    return X


def lipsh_const(func, X, r: float = 2):
    M = [(abs((func([X[i]]) - func([X[i - 1]])) / (X[i] - X[i - 1]))) for i in range(1, len(X))]
    M = np.max(M)
    m = 1
    if M > 0:
        m = r * M
    return m


def Strongin(func, X, r: float = 2, eps: float = 0.1):
    L = lipsh_const(func, X, r)
    R = [(L * (X[i] - X[i - 1])) + ((func([X[i]]) + func([X[i - 1]])) ** 2) / (L * (X[i] - X[i - 1])) - (2 * (
        func([X[i]]) + func([X[i - 1]]))) for i in range(1, len(X))]
    tmp_max = R[0]
    res = 0
    for i in range(1, len(R)):
        if R[i] >= tmp_max:
            tmp_max = R[i]
            res = i

    res += 1

    X.append((X[res] - X[res - 1]) / 2 + (func([X[res]]) - func([X[res - 1]])) / (2 * L))
    X.sort()
    if np.min([(X[i] - X[i - 1]) for i in range(1, len(X))]) > eps:
        X = Strongin(func, X, r)
    return X


def Piavian(func, X, r: float = 2, eps: float = 0.1):
    L = lipsh_const(func, X, r)
    R = [0.5 * L * (X[i] - X[i - 1]) - ((func([X[i]]) + func([X[i - 1]])) / 2) for i in range(1, len(X))]
    tmp_max = R[0]
    res = 0
    for i in range(1, len(R)):
        if R[i] >= tmp_max:
            tmp_max = R[i]
            res = i

    res += 1
    X.append((X[res] - X[res - 1]) / 2 + (func([X[res]]) - func([X[res - 1]])) / (2 * L))
    X.sort()
    if np.min([(X[i] - X[i - 1]) for i in range(1, len(X))]) > eps:
        X = Piavian(func, X, r)
    return X


def find_min(func, grid):
    x_min = grid[0]
    y_min = func([x_min])
    for el in grid:
        y_temp = func([el])
        if y_temp <= y_min:
            x_min = el
            y_min = y_temp
    print(x_min, y_min)
    return x_min, y_min


def button_enum():
    global test_func
    global ab_interval
    global eps
    global xe_min

    default_a = -1
    default_b = 1
    default_eps = 0.1

    try:
        ab_interval[0] = float(a.get())
    except ValueError:
        ab_interval[0] = default_a
    try:
        ab_interval[1] = float(b.get())
    except ValueError:
        ab_interval[1] = default_b

    try:
        eps = float(eps1.get())
    except ValueError:
        eps = default_eps
    check_input()
    xe_min, _ = find_min(test_func, enumeration(test_func, ab_interval, eps))
    root.destroy()


def button_strongin():
    global test_func
    global ab_interval
    global r1
    global eps
    global xe_min
    global a
    global b
    global eps1
    global r
    default_a = -1
    default_b = 1
    default_eps = 0.1
    default_r = 2

    try:
        ab_interval[0] = float(a.get())
    except ValueError:
        ab_interval[0] = default_a
    try:
        ab_interval[1] = float(b.get())
    except ValueError:
        ab_interval[1] = default_b

    try:
        r = float(r1.get())
    except ValueError:
        r = default_r
    try:
        eps = float(eps1.get())
    except ValueError:
        eps = default_eps
    check_input()

    xe_min, _ = find_min(test_func, Strongin(test_func, ab_interval, r, eps))

    root.destroy()


def button_piavian():
    global test_func
    global ab_interval
    global r1
    global eps
    global xe_min
    global a
    global b
    global eps1
    global r

    default_a = -1
    default_b = 1
    default_eps = 0.1
    default_r = 2

    try:
        ab_interval[0] = float(a.get())
    except ValueError:
        ab_interval[0] = default_a
    try:
        ab_interval[1] = float(b.get())
    except ValueError:
        ab_interval[1] = default_b

    try:
        r = float(r1.get())
    except ValueError:
        r = default_r
    try:
        eps = float(eps1.get())
    except ValueError:
        eps = default_eps

    check_input()
    xe_min, _ = find_min(test_func, Piavian(test_func, ab_interval, r, eps))
    root.destroy()


def check_input():
    global ab_interval
    global eps
    global r

    default_eps = 0.1
    default_r = 2

    if ab_interval[0] > ab_interval[1]:
        tmp = ab_interval[0]
        ab_interval[0] = ab_interval[1]
        ab_interval[1] = tmp
    if r <= 1:
        r = default_r

    if eps <= 0:
        eps = default_eps


ab_interval = [-1, 1]
eps = 0.1
grid_step = 10
accuracy = 100
r = 2
extremes_count = func_count = 5

random.seed()

test_func, _ = rand_func_generator([-10, 10], extremes_count)

X = np.arange(-10, 10, 1 / accuracy)
Y = []

xe_min = 0

for x in X:
    x_slice = [x for _ in range(1)]
    Y.append(test_func(x_slice))

root = Tk()
root.title("Input")
a = StringVar()
b = StringVar()
eps1 = StringVar()
r1 = StringVar()
grid_step1 = StringVar()
label = Label(root, text="a[-10,10]: ").grid(row=0, column=0, sticky=E)
label1 = Label(root, text="b[-10,10]: ").grid(row=1, column=0, sticky=E)
label2 = Label(root, text="eps(0,inf): ").grid(row=2, column=0, sticky=E)
label3 = Label(root, text="r(1, inf): ").grid(row=3, column=0, sticky=E)
textbox = Entry(root, textvariable=a).grid(row=0, column=1, sticky=E)
textbox1 = Entry(root, textvariable=b).grid(row=1, column=1, sticky=E)
textbox2 = Entry(root, textvariable=eps1).grid(row=2, column=1, sticky=E)
textbox3 = Entry(root, textvariable=r1).grid(row=3, column=1, sticky=E)
btn = Button(root, text="enumer", command=button_enum).grid(row=4, column=0, sticky=W)
btn1 = Button(root, text="strongin", command=button_strongin).grid(row=4, column=1, sticky=W)
btn2 = Button(root, text="piavian", command=button_piavian).grid(row=4, column=2, sticky=W)
root.mainloop()

# extremes_y = [test_func(x) for x in extremes_x]


step_points = plt.scatter(ab_interval, [test_func([i]) for i in ab_interval], c='b')

plt.axvline(x=xe_min, c='r')
plt.axvline(x=ab_interval[0], c='b', linestyle=':')
plt.axvline(x=ab_interval[len(ab_interval) - 1], c='b', linestyle=':')

graph = plt.plot(X, Y)
grid1 = plt.grid(True)
plt.show()
