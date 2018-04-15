import random
from typing import List

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


def rand_func_generator(extremes_num, params_num):
    global accuracy

    def rand():
        res = random.random()
        if random.random() > 0.5:
            res *= random.random() * 10
        return res

    extremes_coeff = [[rand() for _ in range(params_num)] for _ in range(extremes_num)]
    extremes_coord = [[int(random.random()*accuracy)/accuracy for _ in range(params_num)] for _ in range(extremes_num)]
    smoothness_near_extremes_coeff = [[rand() for _ in range(params_num)] for _ in range(extremes_num)]
    values = [(1 + i) / accuracy for i in range(extremes_num)]
    res_func = get_test_function_method_min(extremes_num, extremes_coeff, extremes_coord,
                                            smoothness_near_extremes_coeff, values)

    return res_func, extremes_coord


random.seed()

accuracy = 200
func_params_num = 1
eps = 0.0001
extremes_count = func_count = 10

test_func, extremes_x = rand_func_generator(extremes_count, func_params_num)
extremes_y = [test_func(x) for x in extremes_x]

X = [i / accuracy for i in range(accuracy)]
Y = []

for x in X:
    x_slice = [x for _ in range(func_params_num)]
    Y.append(test_func(x_slice))

extr_points = plt.scatter(extremes_x, extremes_y)
graph = plt.plot(X, Y)
grid1 = plt.grid(True)
plt.show()
