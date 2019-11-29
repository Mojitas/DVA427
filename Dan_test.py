from Mathias_test import *

# TODO: Det går att skriva saker som måste göras på detta sätt. Då dyker de up under TODO-fliken

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':  # typ våran main tror jag.

    DM.segmentation()

    size = 18
    amount = 16 * 54
    rate = 0.2

    weight1 = (2 * np.random.random((size, size)) - 1)
    weight2 = (2 * np.random.random(size) - 1)

    inputs = DM.training_inputs

    s1 = (size, 3)
    layer1 = np.zeros(s1)  # 0 = value, 1 = error, 2 = w0i

    output_w0 = (2 * np.random.random_sample() - 1)/20

    for i in range(size):  # Setting W0i
        layer1[i, 2] = (2 * np.random.random_sample() - 1)/20

    for A in range(1000):
        for R in range(amount):

            for i in range(size):  # Calculating output for layer 1 nodes
                sum1 = 0
                sum1 += layer1[i, 2]  # Constant weight

                for j in range(size):  # Summing all inputs TO an node
                    sum1 += inputs[0, 0, j] * weight1[j, i]

                layer1[i, 0] = sigmoid(sum1)  # Value of node

            sum2 = 0
            sum2 += output_w0

            for i in range(size):  # Calculating output for layer 2 nodes (total output in this case)

                sum2 += layer1[i, 0] * weight2[i]

            output = sigmoid(sum2)

            output_error = output * (1 - output) * (DM.training_outputs[(math.floor(R / 54)), R % 54] - output)
            #print({math.floor(R / 54)}, {R % 54})
            #print("Output: %5f target: %5f error: %5f" %(output, DM.training_outputs[math.floor(R / 54), R % 54], output_error))

            # The floor function is used due to the training data being split in groups of 54

            output_w0 += rate * output_error

            for i in range(size):
                layer1[i, 1] = layer1[i, 0] * (1 - layer1[i, 0]) * weight2[i] * output_error  # Error for layer1 i
                # print(layer1[i,1])
                layer1[i, 2] += rate * layer1[i, 1]  # W0i for Layer1 nodes
                weight2[i] += rate * output_error * layer1[i, 0]  # New weights for layer2
                #print("Layer 1: %5f, change in weight: %f, new weight: %5f" %(layer1[i, 0], rate * output_error * layer1[i, 0], weight2[i]))

            for i in range(size):  # Inputs
                for j in range(size):  # Layer1
                    weight1[i, j] = rate * layer1[j, 1] * inputs[R % 16, math.floor(R / 16), i]  # New weight1
            print(weight2[i])


for R in range(115):
    output = 0

    for i in range(size):  # Calculating output for layer 1 nodes

        sum1 = layer1[i, 2]  # Constant weight

        for j in range(size):  # Summing all inputs TO an node
            sum1 += DM.validation_data[R, j] * weight1[j, i]



        layer1[i, 0] = sigmoid(sum1)  # Value of node

    sum2 = 0
    sum2 += output_w0  # Needs to be like this for some reason otherwise output_w0 increases???????

    for i in range(size):  # Calculating output for layer 2 nodes (total output in this case)

        sum2 += layer1[i, 0] * weight2[i]

    output = sigmoid(sum2)

    #print(output)
    #print(DM.validation_result[R])
