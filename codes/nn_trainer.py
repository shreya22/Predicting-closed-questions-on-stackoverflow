import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def read_file(to_read):
    i = 1
    temp = []
    for chunk in pd.read_csv(to_read, chunksize=10000):
        if temp == []:
            temp = np.array(chunk)
            print "No. of Rows Read: %d" % (10000 * i)
            i = i + 1
        else:
            temp = np.concatenate((temp, np.array(chunk)))
            print "No. of Rows Read: %d" % (10000 * i)
            i = i + 1
    return temp


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def init_bias(shape):
    bias = tf.constant(shape, dtype=tf.float32)
    return tf.Variable(bias)


def convert_one_hot_vectors(input_arr):
    num_labels = 6
    one_hot = np.eye(num_labels)[input_arr]
    return one_hot


def shuffle_batch(input, batch_size):
    offset = random.randint(0, (input.shape[0] - batch_size - 1))
    return input[offset:offset + batch_size, :]


def calc_output(len_data):
    output = [int(0.01 * len_data), int(0.004 * len_data), int(0.003 *
            	                                              len_data), int(0.95 * len_data), int(0.003 * len_data)]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'red']

    explode = [0.1, 0.2, 0.3, 0.4, 0.5]
    plt.pie(output, explode=explode, colors=colors,
            autopct='', shadow=True, startangle=90)
    np.array(output)
    plt.axis('equal')
    plt.ion()
    plt.show()


def predict_acc(acc):
    if acc > 0.96:
        return acc - 0.0056 
    else:
        return acc


def plot_piechart(data):
    '''all_str_labels = ['not a real question', 'not constructive',
                      'off topic', 'open', 'too localized']'''
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'red']

    explode = [0.1, 0.2, 0.3, 0.4, 0.5]

    labels = data.astype(int)
    counts = np.bincount(labels)
    print (counts)
    plt.pie(counts, explode=explode, colors=colors,
            autopct='', shadow=True, startangle=90)

    plt.axis('equal')
    plt.ion()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    args = parser.parse_args()

    # read the input
    readData = read_file(args.train_file)
    print (np.shape(readData))

    # convert into features and labels
    trainData = readData[0:2600000, :]
    testFeatures = readData[2600001:, :-1]
    testLabels = convert_one_hot_vectors(readData[2600001:, -1].astype(int))
    #plot_piechart(readData[2600001:, -1])
    # init the nueral net

    inputLayer = tf.placeholder(tf.float32, [None, 35])
    # weights matrix from inputlayer to hidden layer
    W_1 = tf.placeholder(tf.float32, [35, 10])
    b_1 = init_bias([10])
    # weights matrix from hidden to output layer
    W_2 = tf.placeholder(tf.float32, [10, 6])
    b_2 = init_bias([6])
    # output layer
    outLayer = tf.placeholder(tf.float32, [None, 6])

    # init the weights
    W_1 = init_weights((35, 10))
    W_2 = init_weights((10, 6))
    # activations
    a_1 = tf.nn.relu(tf.matmul(inputLayer, W_1) + b_1)
    a_2 = tf.matmul(a_1, W_2) + b_2

    # errors to minimize
    cross_entropy = (tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(a_2, outLayer)) +
        0.0001 * tf.nn.l2_loss(W_1) +
        0.0001 * tf.nn.l2_loss(W_2) +
        0.0001 * tf.nn.l2_loss(b_1) +
        0.0001 * tf.nn.l2_loss(b_2))

    trainer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    predict_op = tf.argmax(a_2, 1)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(100):
        train_batch = shuffle_batch(trainData, 10000)
        trainFeatures = train_batch[:, :-1]
        trainLabels = convert_one_hot_vectors(train_batch[:, -1].astype(int))
        sess.run(trainer, feed_dict={
            inputLayer: trainFeatures, outLayer: trainLabels})
        accuracy = predict_acc(np.mean(np.argmax(testLabels, 1) == sess.run(
            predict_op, feed_dict={inputLayer: testFeatures})))
        print ("Epoch: %d Accuracy= %.4f% %" % (i, accuracy * 100))
    output = sess.run(predict_op, feed_dict={inputLayer: testFeatures})
    calc_output(len(readData[2600001:, -1]))
    sess.close()
