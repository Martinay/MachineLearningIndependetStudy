# MachineLearningIndependetStudy
Independent study in the topic area machine learning  
School of Computer Science  
Reykjavik University, Iceland  


Follow up independent study, previous independent study: [thorgeirk11](https://github.com/thorgeirk11/IndepententStudy)

## Introduction and problem description
With machine learning algorithms it’s possible to predict the action of a player in each state of a
game. The machine learning algorithm which is used in this project is a subset of neural nets. It’s
called convolutional neural network and is mostly applied to image recognition. The goal of this
independent study is to examine, if it’s possible to train policies to a convolutional neural network
in a topic which differs from image recognition.  
In this project player actions which are executed at a given state should be predicted success-
fully. In the first step data of the game connect 4 and in a second step data of an unknown game
are used. Based on that it should be examined how the accuracy of the predictions differ between
structured and unstructured data. The result should then be evaluated with unknown data. After-
wards should the different approaches be compared based on the speed of training and the accuracy.
The program should be developed in the programming language python and should use the tensor-
flow and keras library. These libraries include already an implementation of a convolutional neural
network. The book "Deep Learning" will be used as a first reference
