from Mathias_test import *


# from Dan_test import *
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

class NeuralNetwork():  # class for related functions

    # initialize the weigths
    def __init__(self):
        self.w1 = 2 * np.random.random((19, 9)) - 1  # weight matrix 1
        self.w2 = 2 * np.random.random((9, 9)) - 1  # weight matrix 2
        self.w3 = 2 * np.random.random((9, 1)) - 1  # weight matrix 3
        self.learning_rate = 0.1

        self.l1 = np.zeros((1, 9))  # hidden layer 1
        self.l2 = np.zeros((1, 9))  # hidden layer 2

        self.bias1 = 2 * np.random.random((1, 9)) - 1  # Biases for the different layers
        self.bias2 = 2 * np.random.random((1, 9)) - 1
        self.bias3 = 2 * np.random.random((1, 1)) - 1

        self.train_acc=[]
        self.test_acc=[]
        self.val_acc=[]
        self.epocs=[]
        self.batch_size=8

    # commit lots of math
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  #

    def think(self, input_layer):       #Use when training is complete

        input_layer = input_layer.astype(float)
        self.l1 = self.sigmoid(np.dot(input_layer, self.best_w1))  # + self.best_bias1
        self.l2 = self.sigmoid(np.dot(self.l1, self.best_w2))  # + self.bbias2
        return self.sigmoid(np.dot(self.l2, self.best_w3))  # + self.bbias3

    def forward(self, input_layer):  # Use when training

        input_layer = input_layer.astype(float)
        self.l1 = self.sigmoid(np.dot(input_layer, self.w1))  # + self.bias1
        self.l2 = self.sigmoid(np.dot(self.l1, self.w2))  # + self.bias2
        return self.sigmoid(np.dot(self.l2, self.w3))  # + self.bias3

    def backwards(self, input_layer, output_layer):

        output = self.forward(input_layer)
        error = (output_layer - output)
        error = sum(error)/self.batch_size

        delta_1 = error * self.sigmoid_derivative(output)
        self.bias1 = self.learning_rate * delta_1
        self.w3 += self.learning_rate * np.dot(self.l2.T, delta_1)

        delta_2 = error * self.sigmoid_derivative(self.l2) * self.w3.T * delta_1
        self.bias2 = self.learning_rate * delta_2
        self.w2 += self.learning_rate * np.dot(self.l1.T, delta_2)

        downstream = np.dot(self.w2, delta_2.T)
        delta_3 = error * NN.sigmoid_derivative(self.l1).T * downstream
        self.bias1 = self.learning_rate * delta_3
        self.w1 += self.learning_rate * np.dot(input_layer.T, delta_3.T)

    def compare(self, inputs, output, mode):  # func for comparing when training has been done

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

    iterations = 1080  # Stoppvillkor
    batch_size = NN.batch_size  # Hur många exempel som vi tränar på i taget
    training_iterations = 0  # Hur många exempel som vi har tränat på
    best_accuracy = np.zeros([3])  # Bästa bedömningen
    epoc = -1

    for i in range(iterations):

        NN.backwards(DM.training_inputs[i * batch_size % 864:(i * batch_size + batch_size) % 864],
                     DM.training_outputs[i * batch_size % 864:(i * batch_size + batch_size) % 864])

        if iterations % 864 == 0 or i == 0:

            epoc+=1
            NN.epocs.append(epoc)
            NN.train_acc.append(NN.compare(DM.training_inputs, DM.training_outputs, 0))
            NN.val_acc.append(NN.compare(DM.validation_data, DM.validation_result, 0))
            NN.test_acc.append(NN.compare(DM.test_data, DM.test_result, 0))

        if best_accuracy[1] <  NN.val_acc[-1]:

            best_accuracy[0] = NN.train_acc[-1]
            best_accuracy[1] = NN.val_acc[-1]
            best_accuracy[2] = NN.test_acc[-1]
            print("\nNew best validation accuracy: {}\nNew best test accuracy: {}".format(best_accuracy[1],
                                                                                        best_accuracy[2]))

    plt.plot(NN.epocs,NN.train_acc)
    plt.plot(NN.epocs,NN.val_acc)
    plt.plot(NN.epocs,NN.test_acc)
    plt.legend(('Training','Valdiation','Testing'))
    plt.ylabel("Accuracy")
    plt.xlabel("Epocs")
    plt.show()


