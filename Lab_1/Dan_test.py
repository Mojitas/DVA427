# coding=utf-8

from Mathias_test import *


class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        self.w1 = 2 * np.random.random((19, 9)) - 1   # weight matrix 1
        self.w2 = 2 * np.random.random((9, 9)) - 1  # weight matrix 2
        self.w3 = 2 * np.random.random((9, 1)) - 1  # weight matrix 3
        self.learning_rate = 0.1
        self.l1 = np.zeros((1, 9))  #hidden layer 1
        self.l2 = np.zeros((1, 9))  # hidden layer 2

        self.bias1 = 2 * np.random.random((1, 9)) - 1  # Bias for hidden layer 1
        self.bias2 = 2 * np.random.random((1, 9)) - 1  # Bias for hidden layer 2
        self.bias3 = 2 * np.random.random((1, 1)) - 1  # Bias for output layer

        self.bw1 = 0
        self.bw2 = 0
        self.bw3 = 0
        self.bbias1 = 0
        self.bbias2 = 0
        self.bbias3 = 0
    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  #

    def forward(self, input_layer):  # functions that uses more layers

        input_layer = input_layer.astype(float)
        self.l1 = self.sigmoid(np.dot(input_layer, self.w1) + self.bias1) #+ self.bias1
        self.l2 = self.sigmoid(np.dot(self.l1, self.w2) + self.bias2) #

        return self.sigmoid(np.dot(self.l2, self.w3) + self.bias3)#

    def backwards(self, input_layer, output_layer, training_iterations):

        for i in range(training_iterations):
            output = self.forward(input_layer)
            out_error = (output_layer - output)

            #print("Output is: ",output)
            delta_1 = out_error*self.sigmoid_derivative(output)     #TODO Vi borde vanda pa delta_1 & delta_3
            #print("shape of l2: {}\nshape of delta1: {}\nshape of w3: {}".format(self.l2.shape, delta_1.shape, self.w3.shape))

            self.bias3 = self.learning_rate * delta_1
            self.bias3 = sum(self.bias3)/self.bias3.shape[0]
            #print("Shape of bias3: {}\nShape of w3: {}".format(self.bias3.shape,self.w3.shape))
            self.w3 += self.learning_rate*np.dot(self.l2.T,delta_1)

            delta_2 = np.multiply(self.sigmoid_derivative(self.l2),self.w3.T)*delta_1
            #print("shape of l2: {}\nshape of delta2: {}\nshape of w2: {}".format(self.l2.shape, delta_2.shape, self.w2.shape))
            self.bias2 = self.learning_rate * delta_2
            self.bias2 = sum(self.bias2) / self.bias2.shape[0]
            self.w2 += self.learning_rate*np.dot(self.l1.T,delta_2)

            #print("\nshape of input: {}\nshape of delta3: {}\nshape of w1: {}".format(input_layer.shape, delta_3.shape,self.w1.shape))
            downstream = np.dot(self.w2, delta_2.T)

            delta_3 = np.multiply(NN.sigmoid_derivative(self.l1).T, downstream)

            self.bias1 = self.learning_rate * delta_3.T
            self.bias1 = sum(self.bias1) / self.bias1.shape[0]
            self.w1 += self.learning_rate*np.dot(input_layer.T,delta_3.T)


    def compare(self, inputs, output):  # func for comparing when training has been done
        inputs = inputs.astype(float)
        outputs = self.forward(inputs)
        accuracy = 0
        size = inputs.shape[0]
        for i in range(size):
            #print("shape of output: {}\nshape of outputs: {}".format(output[i,:].shape,outputs[i,:].shape))
            # TODO fixa rett format
            if output[i] - np.round(outputs[i], 0) == 0:
                accuracy += 1

        return accuracy / size


if __name__ == '__main__':
    DM.segmentation()  # Imports and sorts data
    NN = NeuralNetwork()
    batch_size = 16  # Hur manga exempel som vi trenar pa i taget
    training_sessions = 0  # Hur manga exempel som vi har trenat pa
    iterations = 6000  # Stoppvillkor
    best_accuracy=0     # Besta resultatet
    training_accuracy=0
    validation_accuracy=0

    val_acc = np.zeros((6000, 1))
    epochs = np.zeros((6000, 1))

    j = 0
    i = 0
    while(j < iterations):
        NN.backwards(DM.training_inputs[i%864:(i%864 + batch_size)], DM.training_outputs[i%864:(i%864 + batch_size)], 1)
        training_accuracy = NN.compare(DM.training_inputs,DM.training_outputs)
        validation_accuracy = NN.compare(DM.validation_data, DM.validation_result)

        val_acc[j] = validation_accuracy
        epochs[j] = i

        j += 1
        i += batch_size
        if i % (512*batch_size) == 0:

            print("Iteration: ", j)
            print("Training accuracy is: ", training_accuracy)
            print("Validation accuracy is: ", validation_accuracy)

    plt.plot(epochs, val_acc)

    print('Test accuracy is:', NN.compare(DM.test_data, DM.test_result))


