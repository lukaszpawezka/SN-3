from neuron_network import NeuronNetwork, Optimalisation
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import random

f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()

tt, vt = training_data
tv, vv = validation_data
# print(np.shape(t), np.shape(v))

# print(training_data)
# print(np.shape(validation_data))
# print(np.shape(test_data))

layers_sig = [
    (784, 60),
    # (60, 30),
    (60, 10)

]


layers_relu = [
    (784, 60),
    # (60, 30),
    (60, 10)
]

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return 1. * (x > 0)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

relu_activation_function = {
    lambda x: relu(x),
    lambda x: relu_derivative(x)
}

sigmoid_activation_function = (
    lambda x: sigmoid(x),
    lambda x: sigmoid_prime(x)
)

net_relu = NeuronNetwork(
    layers_relu,
    0.1,
    relu_activation_function,
    momentum=0.1,
    adadelta_y=0.9,
    negative=True,
    soft_max_output=True,
    optimalisation=Optimalisation.MOMENTUM
)

net_sigmoid = NeuronNetwork(
    layers_sig,
    0.1,
    sigmoid_activation_function,
    momentum=0.1,
    adadelta_y=0.9,
    negative=True,
    soft_max_output=True,
    optimalisation=Optimalisation.MOMENTUM
)

# def singular_run():
#
#     print(np.shape(tt), np.shape(vt))
#
#     start = time.time()
#     nn.train(tt, vt, 3, 200, 0.05)
#     print(round((time.time()-start)*100)/100, "s")
#
#     start = time.time()
#     nn.test(tv, vv)
#     print(round((time.time()-start)*100)/100, "s")
#
#     index = random.randint(0, len(tv))
#     print("Przewidywania dla: ", nn.predict(tv[index]))
#
#     plt.imshow(np.resize(tv[index], (28, 28)))
#     plt.show()
# singular_run()

def plot_run():
    data_for_plot = []
    data_for_plot2 = []
    for i in range(0, 10):
        print("Epoka: ", i)
        net_relu.train(tt, vt, 1, 100, 0.01)
        data_for_plot.append(net_relu.test(tv, vv))
        net_sigmoid.train(tt, vt, 1, 100, 0.01)
        data_for_plot2.append(net_sigmoid.test(tv, vv))
    plt.plot(data_for_plot, 'b', marker='o', label='Relu')
    plt.plot(data_for_plot2, 'r', marker='o', label='Sigmoid')
    plt.legend(loc='lower right')
    plt.suptitle('Wykres procentowego stanu wyuczenia modelu w danej epoce')
    plt.xlabel("Numer epoki")
    plt.ylabel("Procent wyuczenia modelu")
    # plt.xticks(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
               ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.savefig('wykres1')
    plt.show()

plot_run()

# print(np.array([1,4,1,2,5,6])*np.array([2,1,4,6])[:, np.newaxis])
#
#
# inp = np.array([
#     [1, 0, 0],
#     [0, 1, 1],
# ])
#
# neur = np.array([
#     [10, 2, 2, 2],
#     [10, 3, 3, 3]
# ])
#

# print(np.tile(inp[0], (neur.shape[0], 1)))
# print(neur[:, 0])
# print(np.sum(np.tile(inp[0], (neur.shape[0], 1))*neur[:, 1:], axis=1)+neur[:, 0])
# print(np.sum(inp[0]*neur[:, 1:], axis=1)+neur[:, 0])

# delta = np.array([
#     1,2,3,4,5
# ])
#
# prime = np.array([
#     0,0,1,0,0,1,0
# ])
#
# layers = np.array([
#     [1, 0, 0, 0, 0],
#     [1, 1, 0, 0, 0],
#     [1, 0, 1, 0, 0],
#     [1, 0, 0, 1, 0],
#     [1, 0, 0, 0, 1],
#     [1, 0, 0, 1, 0],
#     [1, 0, 1, 0, 0],
# ])


#
# print(
#     np.array([
#         [1,1,2,3,5,2,3,5,2],
#         [1,3,2,3,4,6,2,1,2],
#         [2,4,2,5,6,2,6,7,2]
#     ]).T *
#     np.array([1,0,1])
# )
#
#
# print(
#     np.sum(
#     np.array([
#         [1,1,2,3,5,2,3,5,2],
#         [1,3,2,3,4,6,2,1,2],
#         [2,4,2,5,6,2,6,7,2]
#     ]).T *
#     np.array([1,0,1]), axis=1)
# )

# print(layers*prime[:, np.newaxis])
# print(layers*delta)
#
# print(np.array([1,0,0,0])*np.array([2,2,2,2]))
#
# print(np.shape(layers), np.shape(delta))
# print(np.sum(layers*delta, axis=1))