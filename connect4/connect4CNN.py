from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix
from typing import NamedTuple


class Meta(NamedTuple):
    metadata_size: int
    state_size: int
    num_actions: int
    files: list

C4Metadata = Meta(4, 128, 8, ['connect4_role0.csv', 'connect4_role1.csv'])

def read_data(file_name):
    meta = pd.read_csv(file_name, usecols=range(0, C4Metadata.metadata_size), header=None)
    features = pd.read_csv(file_name, usecols=range(C4Metadata.metadata_size,
                                                    C4Metadata.metadata_size + C4Metadata.state_size), header=None)
    labels = pd.read_csv(file_name, usecols=range(C4Metadata.metadata_size + C4Metadata.state_size,
                                                  C4Metadata.metadata_size + C4Metadata.state_size + C4Metadata.num_actions),
                         header=None)

    data = list(zip(meta.values, features.values, labels.values))
    return data

data_File_Path = '../data/'
train_data_size = 0.8

# Convolutional Layer 1.
filter_size1 = 4          # Convolution filters are 4 x 4 pixels.
num_filters1 = 8         # There are 8 of these filters.

# Convolutional Layer 2.
filter_size2 = 4          # Convolution filters are 5 x 5 pixels.
num_filters2 = 16         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 32             # Number of neurons in fully-connected layer.

def splitData(data):
    cut = int(len(data) * train_data_size)
    train_data = data[:cut]
    test_data = data[cut:]
    return train_data, test_data

data = []
for file in C4Metadata.files:
    loadedData = read_data(data_File_Path + file)
    data = [*data, *loadedData]

shuffle(data)
train_data, test_data = splitData(data)

x_train = np.array([obj[1][2:] for obj in train_data])
y_train = [obj[2] for obj in train_data]
y_train_cls = np.array([label.argmax() for label in y_train])
print(len(x_train[0]))
x_test = [obj[1][2:] for obj in test_data]
y_test = [obj[2] for obj in test_data]
y_test_cls = np.array([label.argmax() for label in y_test])
print(len(x_test[0]))
print(len(data))
print(len(train_data))
print(len(test_data))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, [None, C4Metadata.state_size - 2], name='x')
x_2d = tf.reshape(x, [-1, 7, 6, 3])
y_true = tf.placeholder(tf.float32, [None, C4Metadata.num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_2d,
                   num_input_channels=3,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=load.C4Metadata.num_actions,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()
session.run(tf.global_variables_initializer())

feed_dict_train = {x: x_train, y_true: y_train}
feed_dict_test = {x: x_test,
                  y_true: y_test,
                  y_true_cls: y_test_cls}

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(0, num_iterations):

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 50 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = y_test_cls

    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(C4Metadata.num_actions)
    plt.xticks(tick_marks, range(C4Metadata.num_actions))
    plt.yticks(tick_marks, range(C4Metadata.num_actions))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def optimizeAndPlot(numberOfIterations):
    optimize(num_iterations=numberOfIterations)
    print_accuracy()
    print_confusion_matrix()

def closeSession():
    session.close()

saver = tf.train.Saver()

def SaveSession():
    saver.save(session, "./model/connect4CNN.ckpt")
    print("Model saved")

def LoadSession():
    saver.restore(session, "./model/connect4CNN.ckpt")

