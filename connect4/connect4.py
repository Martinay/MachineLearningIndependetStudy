from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import load

data_File_Path = '../data/'
train_data_size = 0.8

def splitData(data):
    cut = int(len(data) * train_data_size)
    train_data = data[:cut]
    test_data = data[cut:]
    return train_data, test_data

data = []
for file in load.C4Metadata.files:
    loadedData = load.read_data(data_File_Path + file)
    data = [*data, *loadedData]

shuffle(data)
train_data, test_data = splitData(data)

print(len(data))
print(len(train_data))
print(len(test_data))


#tensorflow
x = tf.placeholder(tf.float32, [None, load.C4Metadata.state_size])
y_true = tf.placeholder(tf.float32, [None, load.C4Metadata.num_actions])
y_true_cls = tf.placeholder(tf.int64, [None])

#variablen, die tensorflow anpassen soll
weights = tf.Variable(tf.zeros([load.C4Metadata.state_size, load.C4Metadata.num_actions]))
biases = tf.Variable(tf.zeros([load.C4Metadata.num_actions]))

#berechnung durch einfache multiplikation
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

#error function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100

x_train = [obj[1] for obj in train_data]
y_train = [obj[2] for obj in train_data]
y_train_cls = np.array([label.argmax() for label in y_train])

x_test = [obj[1] for obj in test_data]
y_test = [obj[2] for obj in test_data]
y_test_cls = np.array([label.argmax() for label in y_test])

def optimize(num_iterations):
    feed_dict_train = {x: x_train,
                       y_true: y_train}

    session.run(optimizer, feed_dict=feed_dict_train)
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

feed_dict_test = {x: x_test,
                  y_true: y_test,
                  y_true_cls: y_test_cls}


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
    tick_marks = np.arange(load.C4Metadata.num_actions)
    plt.xticks(tick_marks, range(load.C4Metadata.num_actions))
    plt.yticks(tick_marks, range(load.C4Metadata.num_actions))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

print_confusion_matrix()
print_accuracy()
for a in range(0,1000):
    optimize(0)
print_accuracy()
print_confusion_matrix()

session.close()