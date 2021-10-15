import numpy
import math


def softmax(in_matrix):
    m, n = numpy.shape(in_matrix)
    out_matrix = numpy.mat(numpy.zeros((m, n)))
    soft_sum = 0
    for idx in range(0, n):
        out_matrix[0, idx] = math.exp(in_matrix[0, idx])
        soft_sum += out_matrix[0, idx]

    for idx in range(0, n):
        out_matrix[0, idx] = out_matrix[0, idx] / soft_sum

    return out_matrix


a = numpy.array([[1, 2, 1, 2, 1, 1, 3]])
print(softmax(a))
