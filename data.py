import tensorflow as tf
import numpy as np
import cv2
import os


###
# This file concerns itself with data handling, such as loading and
# splitting the data into training, testing, and validation

# This is the same as the constructor in the FFP class. It is used here to
# return the graph and pre-process it based on the file
def construct(fileName):
    file = open(fileName, "r")
    text = file.read()
    tokens = text.split()
    seed = int(tokens.pop(0))
    n = int(tokens.pop(0))
    model = int(tokens.pop(0))
    int(tokens.pop(0))  # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    state = [0] * n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
        b = int(tokens.pop(0))
        state[b] = -1
    graph = []
    for i in range(n):
        graph.append([0] * n)
    while tokens:
        x = int(tokens.pop(0))
        y = int(tokens.pop(0))
        graph[x][y] = 1
        graph[y][x] = 1
    return np.array(graph)  # Only modification: Return as numpy array


# We work on the graph as if it was a single-channel image. We resize it to our
# defined dims in the parameters
def preprocess(graph, dims):
    graph = graph.reshape(np.shape(graph)).astype('float32')
    graph = cv2.resize(graph, dims, interpolation=cv2.INTER_AREA)
    return graph


# Load all data, and save it to a numpy array. Since it takes time to iterate
# through the dataset, it's a trade-off between speed and space. We decided
# we'd rather use more space.
def load_all_data(dims):
    files = os.listdir("instances/GBRL/")
    data = np.zeros((1,) + dims)
    gt = []
    for file in files:
        graph = construct("instances/GBRL/" + file)
        graph = preprocess(graph, dims)
        graph = np.expand_dims(graph, axis=0)
        data = np.append(data, graph, axis=0)
        gt = np.append(gt, get_heuristic(graph))
    files = os.listdir("instances/BBGRL/")
    for file in files:
        graph = construct("instances/BBGRL/" + file)
        graph = preprocess(graph, dims)
        graph = np.expand_dims(graph, axis=0)
        data = np.append(data, graph, axis=0)
        gt = np.append(gt, get_heuristic(graph))
    data = np.delete(data, 0, 0)
    np.save('graphs_matrices', data)
    np.save('graph_gt',gt)


# Split into train/val/test sets
# We have relatively few data points (about 400), so we opted for a 80/10/10
# split.
def generate_splits(data, gt):
    total_samples = np.shape(data)[0]
    total80 = np.floor(total_samples * 0.8).astype('int')
    total10 = np.floor(total_samples * 0.1).astype('int')
    # Split is 80/20/20
    train_x = data[0:total80, :, :]
    train_y = gt[0:total80]
    val_x = data[total80:total80 + total10, :, :]
    val_y = gt[total80:total80 + total10]
    test_x = data[total80 + total10:total80 + total10 * 2, :, :]
    test_y = gt[total80 + total10:total80 + total10 * 2]
    return train_x, train_y, val_x, val_y, test_x, test_y


# TODO: Add the actual function
# Placeholder function for groundtruth generation
def get_heuristic(graph):
    return 1
