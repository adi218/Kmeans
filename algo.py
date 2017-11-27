import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from random import *


def kmeans(x_f, k_f, r_f):
    c = np.zeros((x_f.shape[0], k_f))
    total_loss = float('Inf')
    cluster_indices = []
    for q in range(r_f):
        for j in range(k_f):
            c[:, j] = x_f[:, randint(0, x_f.shape[1])-1].T
        for t in range(500):
            if t % 2 == 0:
                z = np.zeros((x_f.shape[1], k_f))
                for i in range(x_f.shape[1]):
                    min_d = float('Inf')
                    min_ind = 0
                    for l in range(k_f):
                        temp = np.linalg.norm(x_f[:, i].T - c[:, l].T)
                        if temp < min_d:
                            min_d = temp
                            min_ind = l
                    z[i, min_ind] = 1
                    # print(q, i, min_ind)
            else:
                c = np.zeros((x_f.shape[0], k_f))
                for l in range(k_f):
                    n = 0
                    for j in range(x_f.shape[1]):
                        if z[j, l] == 1:
                            temp = c[:, l] + x_f[:, j].T
                            c[:, l] = temp
                            # print(c.shape)
                            n += 1
                    if n != 0:
                        c[:, l] /= n
        total_loss_q = 0
        cluster_indices_q = []
        for b in range(k_f):
            cluster_indices_q.append([0])
        for l in range(k_f):
            for j in range(x_f.shape[1]):
                if z[j, l] == 1:
                    total_loss_q += np.linalg.norm(c[:, l].T - x_f[:, j].T)
                    cluster_indices_q[l].append(j)
        if total_loss_q <= total_loss:
            total_loss = total_loss_q
            cluster_indices = cluster_indices_q
        print(total_loss, total_loss_q)
    cluster_indices[0].pop(0)
    cluster_indices[1].pop(0)
    return cluster_indices


data = scipy.io.loadmat('dataset1.mat')
Y = np.asmatrix(data['Y'])
c = kmeans(Y, 2, 3)
print(c)
