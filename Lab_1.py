from Mathias_test import *


# from Dan_test import *

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        self.w1 = 2 * np.random.random((19, 9)) - 1  # weight matrix 1
        self.w2 = 2 * np.random.random((9, 9)) - 1  # weight matrix 2
        self.w3 = 2 * np.random.random((9, 1)) - 1  # weight matrix 3
        self.learning_rate = 0.1
        self.l1 = np.zeros((1, 9))  # hidden layer 1
        self.l2 = np.zeros((1, 9))  # hidden layer 2

        self.bias1 = 2 * np.random.random((1, 9)) - 1  # Bias for hidden layer 1
        self.bias2 = 2 * np.random.random((1, 9)) - 1  # Bias for hidden layer 2
        self.bias3 = 2 * np.random.random((1, 1)) - 1  # Bias for output layer

        self.best_w1 = 0
        self.best_w2 = 0
        self.best_w3 = 0
        self.best_bias1 = 0
        self.best_bias2 = 0
        self.best_bias3 = 0

    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  #

    def think(self, input_layer):
        self.l1 = self.sigmoid(np.dot(input_layer, self.best_w1) + self.best_bias1)  #
        self.l2 = self.sigmoid(np.dot(self.l1, self.best_w2))  # + self.bbias2
        return self.sigmoid(np.dot(self.l2, self.best_w3))  # + self.bbias3

    def forward(self, input_layer):  # functions that uses more layers
        input_layer = input_layer.astype(float)
        self.l1 = self.sigmoid(np.dot(input_layer, self.w1) + self.bias1) #
        self.l2 = self.sigmoid(np.dot(self.l1, self.w2) + self.bias2) #
        return self.sigmoid(np.dot(self.l2, self.w3) + self.bias3)#

    def backwards(self, input_layer, output_layer):

        input_layer = input_layer.astype(float)
        output = self.forward(input_layer)
        out_error = (output_layer - output)

        # print("Output is: ",output)
        delta_1 = out_error * self.sigmoid_derivative(output)
        # print("shape of l2: {}\nshape of delta1: {}\nshape of w3: {}".format(self.l2.shape, delta_1.shape, self.w3.shape))
        self.bias3 = self.learning_rate * delta_1
        # print("Shape of bias3: {}\nShape of w3: {}".format(self.bias3.shape,self.w3.shape))
        self.w3 += self.learning_rate * np.dot(self.l2.T, delta_1)

        delta_2 = np.multiply(self.sigmoid_derivative(self.l2), self.w3.T) * delta_1
        # print("shape of l2: {}\nshape of delta2: {}\nshape of w2: {}".format(self.l2.shape, delta_2.shape, self.w2.shape))
        self.bias2 = self.learning_rate * delta_2
        self.w2 += self.learning_rate * np.dot(self.l1.T, delta_2)  # Kanske inte klar

        # delta_3 = np.dot(self.sigmoid_derivative(self.l1),self.w2.T) * delta_2
        # TODO Fixa denna for i: dot(sef.w2[i,:], delta2) sen multiply(sigmoid der l1, det)
        # print("\nshape of input: {}\nshape of delta3: {}\nshape of w1: {}".format(input_layer.shape, delta_3.shape,self.w1.shape))
        downstream = np.dot(self.w2, delta_2.T)

        delta_3 = np.multiply(NN.sigmoid_derivative(self.l1).T, downstream)

        self.bias1 = self.learning_rate * delta_3
        # print("shape of input layer: ",input_layer.shape)
        self.w1 += self.learning_rate * np.dot(input_layer.T, delta_3.T)

    def compare(self, inputs, output, mode):  # func for comparing when training has been done
        inputs = inputs.astype(float)

        if mode == 0:
            outputs = self.forward(inputs)  # use during training
        elif mode == 1:
            outputs = self.think(inputs)  # use for final result

        accuracy = 0
        size = inputs.shape[0]
        for i in range(size):

            if abs(output[i] - outputs[i]) < 0.5:
                accuracy += 1

        return accuracy / size


if __name__ == '__main__':
    DM.segmentation()  # Imports and sorts data
    NN = NeuralNetwork()

    iterations = 5000  # Stoppvillkor
    batch_size = 16  # Hur många exempel som vi tränar på i taget
    training_iterations = 0  # Hur många exempel som vi har tränat på

    best_accuracy = 0  # Bästa bedömningen
    training_accuracy = 0
    validation_accuracy = 0

    for i in range(iterations):
        NN.backwards(DM.training_inputs[i % 864:(i % 864) + batch_size ],
                     DM.training_outputs[i % 864:(i % 864) + batch_size])
        #training_accuracy = NN.compare(DM.training_inputs, DM.training_outputs, 0)
        #validation_accuracy = NN.compare(DM.validation_data, DM.validation_result, 0)

        if best_accuracy < validation_accuracy:
            best_accuracy = validation_accuracy
            print("New best result: ", best_accuracy)
            NN.best_w1 = NN.w1
            NN.best_w2 = NN.w2
            NN.best_w3 = NN.w3
            NN.best_bias1 = NN.bias1
            NN.best_bias2 = NN.bias2
            NN.best_bias3 = NN.bias3

        training_iterations += batch_size
        if training_iterations == iterations * batch_size:
            print("Training accuracy is: ", training_accuracy)
            print("Validation accuracy is: ", validation_accuracy)
            print("iterations : ", i)

    print("Best result: ", NN.compare(DM.validation_data, DM.validation_result, 1))
