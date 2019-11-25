from Mathias_test import *
from Dan_test import *  # import the test stuff


# TODO: Det går att skriva saker som måste göras på detta sätt. Då dyker de up under TODO-fliken

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((19, 1)) - 1

    # commit lots of math
    def sigmoid_nn(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x(1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            weight_adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += weight_adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':  # typ våran main tror jag.

    data_segmentation()

    size = 18
    amount = 16 * 54
    rate = 0.1

    weight1 = 2 * np.random.random((size, size)) - 1
    weight2 = 2 * np.random.random(size) - 1

    inputs = training_inputs

    s1 = (size, 3)
    layer1 = np.zeros(s1)  # 0 = value, 1 = error, 2 = w0i

    s2 = (3, 1)
    output = np.zeros(s2)  # 0 = value, 1 = error, 2 = w0i
    output[2] = 2 * np.random.random_sample() - 1

    for i in range(size):  # Setting W0i
        layer1[i, 2] = 2 * np.random.random_sample() - 1

    for R in range(amount):

        for i in range(size):  # Calculating output for layer 1 nodes
            sum1 = layer1[i, 2]  # Constant weight

            for j in range(size):  # Summing all inputs TO an node
                sum1 += inputs[0, 0, j] * weight1[j, i]

            layer1[i, 0] = sigmoid(sum1)  # Value of node

        sum2 = output[2]  # Why does output[2] ever change value??????????????

        for i in range(size):  # Calculating output for layer 2 nodes (total output in this case)

            sum2 += layer1[i, 0] * weight2[i]

        output[0] = sigmoid(sum2)

        output[1] = output[0] * (1 - output[0]) * (training_outputs[math.floor(R / 54), R % 54] - output[0])
        # The floor function is used due to the training data being split in groups of 54

        for i in range(size):
            layer1[i, 1] = layer1[i, 0] * (1 - layer1[i, 0]) * weight2[i] * output[1]  # Delta for layer1 i
            # print(layer1[i,1])
            layer1[i, 2] += rate * layer1[i, 1]  # W0i for Layer1 nodes
            weight2[i] += rate * output[1] * layer1[i, 0]  # New weights for layer2

        for i in range(size):  # Inputs
            for j in range(size):  # Layer1
                weight1[i, j] = rate * layer1[j, 1] * inputs[R % 16, math.floor(R / 16), i]  # New weight1

        # print(weight2)

for i in range(size):  # Calculating output for layer 1 nodes

    layer1[i, 0] += sigmoid(validation_data[0, i] * weight1[i, 0])

for i in range(size):  # Calculating output for layer 2 nodes (total output in this case)

    output[0] += sigmoid(layer1[i, 0] * weight2[i])
