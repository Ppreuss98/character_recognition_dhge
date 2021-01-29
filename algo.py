import numpy as np
import tensorflow as tf
import time

tested_x_data_size = 10
trained_x_data_size = 60000


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


def initialize_datasets():
    x_train, y_train, x_test, y_test = load_data()
    trained_x = convert_x_data(x_train, trained_x_data_size)
    trained_y = convert_y_data(y_train, trained_x_data_size)
    test_x = convert_x_data(x_test, tested_x_data_size)
    test_y = convert_y_data(y_test, tested_x_data_size)
    return trained_x, trained_y, test_x, test_y


def print_results(pr_list, t_numbers, final_list, correct, time_input):
    for i in range(len(pr_list)):
        print('Probability: ' + str(pr_list[i]) + ' | Number: ' + str(
            final_list[i]) + ' | Correct number: ' + str(t_numbers[i]))

    time_end = time.time()
    print('Amount of correct numbers: ' + str(correct))
    print('Runtime: ' + str(round(time_end - time_input, 2)) + ' seconds with ' + str(trained_x_data_size)
          + ' training data entries and ' + str(tested_x_data_size) + ' test data entries')


def check_matches(trained_x, trained_y, test_x, test_y):
    index_list = []
    probability_list = []

    # Iteration of tested data entries
    for i in range(tested_x_data_size):
        probability_best = 0
        index = 0
        current_tested_x = test_x[i]

        # Iteration of all trained data entries for one test data entry
        for j in range(trained_x_data_size):
            current_trained_x = trained_x[j]
            matches = current_tested_x[:28, :28] == current_trained_x[:28, :28]
            probability = np.count_nonzero(matches == 0) / np.count_nonzero(matches)

            # If probability of an entry is higher, then probability and index for the corresponding y-value get updated
            if probability > probability_best:
                probability_best = probability
                index = j

        # Creation of final probability list which contains likeness of found number with the tested number
        index_list.append(index)
        probability_list.append(round(probability_best * 100, 2))

    # Counter for correct amount of recognized numbers
    # (y-trained entries at index of index_list compared with y-tested entries)
    final_list = np.take(trained_y, index_list)
    test_numbers = test_y[:tested_x_data_size]
    amount_correct = np.count_nonzero(test_numbers == final_list)

    return probability_list, test_numbers, final_list, amount_correct


def start():
    print("Starting...")
    time_start = time.time()
    print("Testing dataset...")

    trained_x, trained_y, test_x, test_y = initialize_datasets()
    probability_list, test_numbers, final_list, amount_correct = check_matches(trained_x, trained_y, test_x, test_y)

    print_results(pr_list=probability_list, t_numbers=test_numbers, final_list=final_list, correct=amount_correct,
                  time_input=time_start)


start()
