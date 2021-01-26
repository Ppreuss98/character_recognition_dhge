import csv
import numpy as np
import tensorflow as tf
import time

tested_data_size = 10
trained_data_size = 600


def load_training_data_x():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train


def load_training_data_y():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return y_train


def load_test_data_x():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_test


def load_test_data_y():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return y_test


def convert_x_data():
    train_3d = load_training_data_x()
    sub_dataset = np.zeros((trained_data_size, 28, 28))
    converted_dataset = np.zeros((trained_data_size, 28, 28))

    # Trim dataset to x entries
    for i in range(trained_data_size):
        for j in range(28):
            sub_dataset[i][j] = train_3d[i][j]

    # Covert 0-255 to 0 and 1
    for i in range(trained_data_size):
        for j in range(28):
            for k in range(28):
                if sub_dataset[i][j][k] > 127:
                    converted_dataset[i][j][k] = 1
                else:
                    converted_dataset[i][j][k] = 0

    return converted_dataset


def convert_x_test_data():
    train_3d = load_test_data_x()
    sub_dataset = np.zeros((tested_data_size, 28, 28))
    converted_dataset = np.zeros((tested_data_size, 28, 28))

    # Trim dataset to x entries
    for i in range(tested_data_size):
        for j in range(28):
            sub_dataset[i][j] = train_3d[i][j]

    # Covert 0-255 to 0 and 1
    for i in range(tested_data_size):
        for j in range(28):
            for k in range(28):
                if sub_dataset[i][j][k] > 127:
                    converted_dataset[i][j][k] = 1
                else:
                    converted_dataset[i][j][k] = 0

    return converted_dataset


def convert_y_data():
    train_3d = load_training_data_y()
    converted_dataset = np.zeros(6000)
    for i in range(6000):
        converted_dataset[i] = train_3d[i]
    return converted_dataset


def convert_y_test_data():
    train_3d = load_test_data_y()
    converted_dataset = np.zeros(6000)
    for i in range(6000):
        converted_dataset[i] = train_3d[i]
    return converted_dataset


def start():
    time_start = time.time()
    trained_x = convert_x_data()
    trained_y = convert_y_data()
    test_x = convert_x_test_data()
    test_y = convert_y_test_data()
    check_array = np.zeros((tested_data_size, 28))

    # Check trained data vs tested data
    # False, array checks only corresponding entries
    # Array 2 has to iterate through whole trained data (array 2)
    index_list = []
    probability_list = []
    is_correct_number_list = []
    amount_correct = 0
    # 1st loop through tested data
    for i in range(tested_data_size):
        # reset of probability on each tested object
        # index - index of best trained element for the tested element
        probability_best = 0
        index = 0
        current_tested_x = test_x[i]
        # 2nd loop through trained data for each tested data
        for j in range(trained_data_size):
            # list_ones, zeroes for probability check
            current_trained_x = trained_x[j]
            list_ones = []
            list_zeroes = []
            # loops for each array in the tested date (28x28 image)
            for k in range(28):
                for m in range(28):
                    if current_tested_x[k][m] == current_trained_x[k][m]:
                        list_ones.append(1)
                    else:
                        list_zeroes.append(0)
            probability = len(list_zeroes) / len(list_ones)
            # check if current trained data has better probability, set index if yes
            if probability > probability_best:
                probability_best = probability
                index = j
            # at the end of 2nd loop the index_list and probability_lists are updated
            if j == trained_data_size - 1:
                index_list.append(index)
                probability_list.append(probability_best)
    # Check if descriptor of trained data corresponds to the test data descriptor
    for i in index_list:
        if trained_y[i] == test_y[i]:
            print('Trained number: ' + str(trained_y[i]) + ', Tested number:' + str(test_y[i]))
            is_correct_number_list.append(True)
            amount_correct = amount_correct + 1
        else:
            print('Trained number: ' + str(trained_y[i]) + ', Tested number:' + str(test_y[i]))
            is_correct_number_list.append(False)

    print("Indexes of best elements: ")
    print(index_list)
    print("Probabilities of best elements: ")
    print(probability_list)
    print("List of correctly recognized numbers: ")
    print(is_correct_number_list)
    print('Amount of correct matches: ' + str(amount_correct))
    time_end = time.time()
    print(time_end - time_start)


start()
