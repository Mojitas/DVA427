from Mathias_test import *
#from Dan_test import *
# TODO: Det går att skriva saker som måste göras på detta sätt. Då dyker de up under TODO-fliken

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        self.weights = 2 * np.random.random((19, 1)) - 1 # original weights, too be removed soon
        self.w1 = 2 * np.random.random((19, 9)) -1     # hidden layer 1
        self.w2 = 2 * np.random.random((9,9)) - 1      # hidden layer 2
        self.w3 = 2 * np.random.random((9,1)) - 1      # output layer

    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, input_layer, output_layer, training_iterations):  #does training
        for i in range(training_iterations):

            output = self.think(input_layer)
            error = output_layer - output
            weight_adjustments = np.dot(input_layer.T, error * self.sigmoid_derivative(output))
            self.weights += weight_adjustments


    def forward(self,input_layer): # functions that uses more layers
        l1 = np.dot(input_layer, self.w1)
        print("\nl1: ",l1)
        l2 = np.dot(l1, self.w2)
        print("\nl2: ",l2)
        output = np.dot(l2, self.w3)
        print("\noutput: ",output)


    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))

        return output

    def compare(self, inputs, output):  # func for comparing when training has been done
        inputs = inputs.astype(float)
        outputs = self.think(inputs)
        accuracy=0
        size= inputs.shape[0]
        for i in range(size):
            if abs(output[i]-outputs[i]) < 0.1:
                accuracy += 1

        return accuracy/size



if __name__ == '__main__':

    DM.segmentation()  # Imports and sorts data
    NN = NeuralNetwork()

    size = 18
    amount = 16 * 54
    layers = 2
    learning_rate = 0.1

    weight1 = 2 * np.random.random((size, size)) - 1
    weight2 = 2 * np.random.random(size) - 1
    layer1 = np.zeros((size, 3))  # 0 = value, 1 = delta, 2 = w0i
    outputs = np.zeros((2, 1))

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
                weight1[i, j] = learning_rate * layer1[j, 1] * DM.training_inputs[0, 0, i]  # New weight1

else:
    training_sessions=0
    NN.forward(DM.training_inputs[0,0])
    while 0:
        for i in range(16):
            NN.train(DM.training_inputs[i],DM.training_outputs[i],100)
            accuracy = NN.compare(DM.validation_data, DM.validation_result)
            training_sessions+=100
            if training_sessions%10000 == 0:
                print("stuff\n")

        print("Overall accuracy is: ", accuracy, training_sessions)
