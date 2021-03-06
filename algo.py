import numpy as np
import tensorflow as tf
import time

tested_x_data_size = 100
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


def print_results(sm_list, t_numbers, final_list, correct, time_input, m_sm):
    for i in range(len(sm_list)):
        print('Similarity: ' + str(sm_list[i]) + ' | Number: ' + str(
            final_list[i]) + ' | Correct number: ' + str(t_numbers[i]))

    time_end = time.time()
    print('Amount of correct numbers: ' + str(correct))
    print('Mean similarity: ' + str(round(m_sm, 2)))
    print('Runtime: ' + str(round(time_end - time_input, 2)) + ' seconds with ' + str(trained_x_data_size)
          + ' training data entries and ' + str(tested_x_data_size) + ' test data entries')


def check_matches(trained_x, trained_y, test_x, test_y):
    print("Checking matches...")
    print("\n")
    index_list = []
    probability_list = []

    # Iteration of tested data entries
    for i in range(tested_x_data_size):
        similarity_best = 0
        index = 0
        current_tested_x = test_x[i]

        # Iteration of all trained data entries for one test data entry
        for j in range(trained_x_data_size):
            current_trained_x = trained_x[j]
            matches = current_tested_x[:28, :28] == current_trained_x[:28, :28]
            similarity = np.count_nonzero(matches) / 784

            # If probability of an entry is higher, then probability and index for the corresponding y-value get updated
            if similarity > similarity_best:
                similarity_best = similarity
                index = j

        # Creation of final probability list which contains likeness of found number with the tested number
        index_list.append(index)
        probability_list.append(round(similarity_best * 100, 2))

    # Counter for correct amount of recognized numbers
    # (y-trained entries at index of index_list compared with y-tested entries)
    final_list = np.take(trained_y, index_list)
    test_numbers = test_y[:tested_x_data_size]
    amount_correct = np.count_nonzero(test_numbers == final_list)
    mean_similarity = np.sum(probability_list) / np.size(probability_list)

    return probability_list, test_numbers, final_list, amount_correct, mean_similarity


def start():
    print("Starting...")
    time_start = time.time()

    trained_x, trained_y, test_x, test_y = initialize_datasets()
    similarity_list, test_numbers, final_list, amount_correct, mean_similarity = \
        check_matches(trained_x, trained_y, test_x, test_y)

    print_results(sm_list=similarity_list, t_numbers=test_numbers, final_list=final_list, correct=amount_correct,
                  time_input=time_start, m_sm=mean_similarity)

    print("\n")
    print("Exiting...")


start()
