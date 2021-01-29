import numpy as np
import tensorflow as tf
import time

tested_x_data_size = 1
trained_x_data_size = 1
trained_y_data = 10000


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def convert_x_data(data, size):
    print("Converting x-data to binary...")
    threshold = 127
    width, height = 28, 28
    sub_dataset = data[:size, :width, :height]
    converted_dataset = sub_dataset > threshold
    return converted_dataset


def convert_y_data(data, size):
    print("Converting y-data...")
    converted_dataset = data[:size]
    return converted_dataset


def start():
    print("Starting...")
    time_start = time.time()
    x_train, y_train, x_test, y_test = load_data()
    trained_x = convert_x_data(x_train, trained_x_data_size)
    trained_y = convert_y_data(y_train, trained_x_data_size)
    test_x = convert_x_data(x_test, tested_x_data_size)
    test_y = convert_y_data(y_test, 10000)

    # Check trained data vs tested data
    # False, array checks only corresponding entries
    # Array 2 has to iterate through whole trained data (array 2)
    index_list = []
    probability_list = []
    is_correct_number_list = []
    amount_correct = 0
    # 1st loop through tested data
    print("Testing dataset...")
    for i in range(tested_x_data_size):
        # reset of probability on each tested object
        # index - index of best trained element for the tested element
        probability_best = 0
        index = 0
        current_tested_x = test_x[i]
        # 2nd loop through trained data for each tested data
        for j in range(trained_x_data_size):
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
            if j == trained_x_data_size - 1:
                index_list.append(index)
                probability_list.append(str(round(probability_best * 100, 2)) + '%')

    # create final list of 'recognized' numbers
    final_list = []
    for i in index_list:
        final_list.append(trained_y[i])

    for i in range(len(probability_list)):
        print('Probability: ' + probability_list[i] + ' | Number: ' + str(final_list[i]))

    time_end = time.time()
    print('Runtime: ' + str(round(time_end - time_start, 2)) + ' seconds with ' + str(trained_x_data_size)
          + ' training data entries and ' + str(tested_x_data_size) + ' test data entries')


start()
