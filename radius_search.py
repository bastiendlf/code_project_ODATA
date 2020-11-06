import numpy as np
from math import sqrt

norms = {"L1": lambda x: np.sum(np.abs(x)),
         "L2": lambda x: np.sum(x ** 2),
         "inf": lambda x: np.max(np.abs(x))
         }


def compute_distance_1d(data, query, norm="L2"):
    norm_function = norms[norm]
    distances = np.zeros(shape=len(data), dtype=np.float32)
    for i, d in enumerate(data):
        distances[i] = norm_function(d - query)
    return distances


def compute_distance_2d(data, q, norm='L2'):
    total_array_q = np.zeros(dtype=object, shape=273)
    for i, person in enumerate(data):
        distances = compute_distance_1d(person, q, norm)
        total_array_q[i] = distances
    return total_array_q


def radius_search_bruteforce(data, q, radius, norm='L2'):
    distance_matrix = compute_distance_2d(data, q, norm)
    nn_list = []
    for i, person in enumerate(distance_matrix):
        for j, dist_pic in enumerate(person):
            if dist_pic <= radius:
                nn_list.append((i, j, dist_pic))
    return nn_list


def radius_opti(data):
    mean_total_list = []
    for i, row in enumerate(data):
        mean_row_list = []
        for j, element in enumerate(row):
            tmp = row
            if element.shape == (150, 150):
                mean_row_list.append(np.mean(compute_distance_1d(np.delete(tmp, j, axis=0), element)))
        mean_row = np.mean(mean_row_list)
        mean_total_list.append(mean_row)
    mean_total = np.mean(mean_total_list)  # moyenne de distance entre deux photos d'une meme pers
    std_total = sqrt(np.var(mean_total_list))  # ecart type
    return mean_total + std_total  # radius opti


def radius_opti_eigen(data):
    mean_person_list = []
    for j, person in enumerate(data):
        mean_row_list = []
        for i, row in enumerate(data[j]):
            tmp = row
            mean_row_list.append(np.mean(compute_distance_1d(np.delete(tmp, i), row[i])))
        mean_person_list.append(np.mean(mean_row_list))
    mean_total = np.mean(mean_person_list)
    std_total = sqrt(np.var(mean_person_list))  # ecart type
    return 10 * mean_total  # radius opti
