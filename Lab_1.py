from Mathias_test import *
from Dan_test import *  # import the test stuff

# TODO: Det går att skriva saker som måste göras på detta sätt. Då dyker de up under TODO-fliken

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        np.random.seed()
        self.weights = 2 * np.random.random((19, 1)) -1

    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, Input_layer, output_layer, training_iterations):
        for i in range(training_iterations):
            output = self.think(Input_layer)
            error = output_layer - output
            weight_adjustments = np.dot(Input_layer.T, error * self.sigmoid_derivative(output))
            self.weights += weight_adjustments
            print("\n", self.weights)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output


if __name__ == '__main__':  # typ våran main tror jag.

    data_segmentation()  # Imports and sorts data
    NN = NeuralNetwork()

    size = 18
    amount = 16 * 54
    layers = 2
    learning_rate = 0.1

    weight1 = 2 * np.random.random((size, size)) - 1
    weight2 = 2 * np.random.random(size) - 1
    layer1 = np.zeros((size, 3))  # 0 = value, 1 = delta, 2 = w0i
    outputs = np.zeros((2, 1))

    # Ser om jag får samma resultat med matrismultiplikation
if 0:
    for R in range(1):
        for i in range(size):  # Setting W0i
            layer1[i, 2] = 2 * np.random.random_sample() - 1

        for i in range(size):  # Calculating output for layer 1 nodes

            layer1[i, 0] += NN.sigmoid(training_inputs[0, 0, i] * weight1[i, 0])

        for i in range(size):  # Calculating output for layer 2 nodes (total output in this case)

            outputs[0] = NN.sigmoid(layer1[i, 0] * weight2[i])

        outputs[1] = outputs[0] * (1 - outputs[0]) * (training_outputs[0, 0] - outputs[0])  # Delta for output

        for i in range(size):
            layer1[i, 1] = layer1[i, 0] * (1 - layer1[i, 0]) * weight2[i] * outputs[1]  # Delta for layer1 i
            layer1[i, 2] += learning_rate * layer1[i, 1]  # W0i for Layer1 nodes
            weight2[i] += learning_rate * outputs[1] * layer1[i, 0]  # New weights for layer2
            print(weight2[i])

        for i in range(size):  # Inputs
            for j in range(size):  # Layer1
                weight1[i, j] = learning_rate * layer1[j, 1] * training_inputs[0, 0, i]  # New weight1

print("\n", NN.sigmoid(np.dot(training_inputs[0].T, training_outputs[0])))
