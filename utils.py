import torch
import math
import tsp


def dist(x1, y1, x2, y2):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    return math.sqrt(x ** 2 + y ** 2)


def from_x_y(xys):
    cost_matrix = torch.zeros(len(xys), len(xys), dtype=torch.float)
    forbidden = torch.zeros(len(xys), len(xys), dtype=torch.float)
    for index1, (x1, y1) in enumerate(xys):
        for index2, (x2, y2) in enumerate(xys):
            cost_matrix[index1, index2] = dist(x1, y1, x2, y2)
            if index1 == index2:
                forbidden[index1, index2] = 1
    return tsp.TSP(cost_matrix, forbidden)
