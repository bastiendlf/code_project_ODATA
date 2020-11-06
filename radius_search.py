import numpy as np
from math import sqrt

norms = {"L1": lambda x: np.sum(np.abs(x)),
         "L2": lambda x: np.sum(x ** 2),
         "inf": lambda x: np.max(np.abs(x))
         }


def compute_distance_1d(data, query, norm="L2"):
    """Compute distances.
	Computes the distances between the vectors (rows) of a dataset and a
	single query). Three distances are supported:
	  * Manhattan distance ("L1");
	  * squared Euclidean distance ("L2");
	  * Chebyshev distance ("inf").

    :param data: (d)-sized Numpy array (1 dimension)
	:param query: Query vector
	:param norm: indicate the type of distance chosen. L1, L2, inf
	:type data: Numpy array/list of floats
	:type query: (m)-sized Numpy array of floats
    :return: (d)-sized Numpy array of all the distances between elements in data and the query
    """
    norm_function = norms[norm]
    distances = np.zeros(shape=len(data), dtype=np.float32)
    for i, d in enumerate(data):
        distances[i] = norm_function(d - query)
    return distances


def compute_distance_2d(data, q, norm='L2'):
    """Compute distances in 2D.
    Computes the distance between a dataset in 2D (array of array) and a query
    :param data: (n,d)-sized Numpy array (2 dimensions). (n) lines and (d) columns
	:param query: Query vector
	:param norm: indicate the type of distance chosen. L1, L2, inf
	:type data: Numpy array in 2 dimensions
	:type query: (m)-sized Numpy array of floats
    :return: (n,d)-sized Numpy array of all the distances between elements in a row of dataset
    and the query for each rows in dataset
    """
    total_array_q = np.zeros(dtype=object, shape=273)
    for i, person in enumerate(data):
        distances = compute_distance_1d(person, q, norm)
        total_array_q[i] = distances
    return total_array_q


def radius_search_bruteforce(data, q, radius, norm='L2'):
    """Radius search method.
    Search all elements in dataset which have a distance to the query under
    a defined radius
    :param data: (n,d)-sized Numpy array (2 dimensions). (n) lines and (d) columns
    :param q: Query vector
    :param radius: radius selected to accept elements in the list
    :param norm: indicate the type of distance chosen. L1, L2, inf
    :return: List of (3)-sized tuples. A tuple corresponds to one element accepted.
    (id of the row, id of the element in row, distance between element and query)
    """
    distance_matrix = compute_distance_2d(data, q, norm)
    nn_list = []
    for i, person in enumerate(distance_matrix):
        for j, dist_pic in enumerate(person):
            if dist_pic <= radius:
                nn_list.append((i, j, dist_pic))
    return nn_list


def radius_opti(data):
    """
    Optimal radius for dataset with pictures (150,150)
    :param data: (n,d)-sized Numpy array (2 dimensions). (n) lines and (d) columns
    :return: the optimal radius to do a radius search on dataset with pictures (150,150)
    """
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
    print(mean_total)
    print(std_total)
    return mean_total + std_total  # radius opti


def radius_opti_eigen(data):
    """
    Optimal radius for dataset with pictures (1,d). (dataset with ACP reduction)
    :param data: (n,d)-sized Numpy array (2 dimensions). (n) lines and (d) columns
    :return: the optimal radius to do a radius search on dataset with pictures of size(1,d)
    """
    mean_person_list = []
    for j, person in enumerate(data):
        mean_row_list = []
        for i, row in enumerate(data[j]):
            tmp = row
            mean_row_list.append(np.mean(compute_distance_1d(np.delete(tmp, i), row[i])))
        mean_person_list.append(np.mean(mean_row_list))
    mean_total = np.mean(mean_person_list)
    std_total = sqrt(np.var(mean_person_list))  # ecart type
    print(mean_total)
    print(std_total)
    return mean_total + 9 * std_total  # radius opti
