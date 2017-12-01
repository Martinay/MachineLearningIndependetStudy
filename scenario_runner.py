import os
from random import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from keras import callbacks
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import shutil

class ScenarioRunner:
    test_data_size = 0.2
    validation_data_size = 0.2

    batch_size = 32
    trained_epochs = 0
    def __init__(self, experiment):
        self.experiment = experiment
        self.init_parameter()
        self.init_folder()
        self.load_data()
        self.build_model()
        print('finished loading data and building model')

    ######################################################
    ####Parameter####
    ######################################################

    def init_parameter(self):
        if not 'get_params' in dir(self.experiment):
            return

        params = self.experiment.get_params()
        for key, value in params.items():
            setattr(self, key, value)


    ######################################################
    ####Folder####
    ######################################################

    def init_folder(self):
        folder, experiment_name = self.experiment.file_path()
        inner_folder = os.path.join(folder, 'output')
        inner_folder = os.path.join(inner_folder, experiment_name)
        self.output_dir = os.path.join(os.getcwd(), inner_folder)
        self.model_path = os.path.join(self.output_dir, 'trained_model.h5')
        self.csv_path = os.path.join(self.output_dir, 'trained_history.csv')
        self.evaluation_log_path = os.path.join(self.output_dir, 'summary.txt')
        self.model_plot_path = os.path.join(self.output_dir, 'model.png')
        self.confusion_matrix_path = os.path.join(self.output_dir, 'confusion_matrix')
        self.learning_curve_path = os.path.join(self.output_dir, 'learning_curve')
        tensorboard_log_dir = os.path.join(self.output_dir, 'tensorboard_log')

        self.callback = [callbacks.TerminateOnNaN(),
                    callbacks.CSVLogger(self.csv_path, separator=',', append=True),
                    callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=5, batch_size=self.batch_size,
                                          write_graph=True,
                                          write_grads=True, write_images=True, embeddings_freq=0,
                                          embeddings_layer_names=None, embeddings_metadata=None),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
                                                epsilon=0.0001, cooldown=0, min_lr=0),
                    callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')]


    ######################################################
    ####load data and split into train and test data####
    ######################################################

    def load_data(self):
        print("load data and split into train and test data")

        roleIndex, featuresWithoutRole, labels = self.experiment.load_data()

        data = list(zip(roleIndex, featuresWithoutRole, labels))
        shuffle(data)
        cut = int(len(data) * (1 - self.test_data_size))
        train_data = data[:cut]
        test_data = data[cut:]

        cut = int(len(train_data) * (1 - self.validation_data_size))
        validation_data = train_data[cut:]
        train_data = train_data[:cut]

        self.x_train_role = np.array([obj[0] for obj in train_data])
        self.x_train_features = np.array([obj[1] for obj in train_data])
        self.y_train = np.array([obj[2] for obj in train_data])

        self.x_validation_role = np.array([obj[0] for obj in validation_data])
        self.x_validation_features = np.array([obj[1] for obj in validation_data])
        self.y_validation = np.array([obj[2] for obj in validation_data])

        self.x_test_role = np.array([obj[0] for obj in test_data])
        self.x_test_features = np.array([obj[1] for obj in test_data])
        self.y_test = np.array([obj[2] for obj in test_data])
        self.y_test_cls = [label.argmax() for label in self.y_test]

        self.x_fit_train = {'x_features': self.x_train_features, 'x_role': self.x_train_role}
        self.x_fit_validation = {'x_features': self.x_validation_features, 'x_role': self.x_validation_role}
        self.x_fit_test = {'x_features': self.x_test_features, 'x_role': self.x_test_role}


    ######################################################
    ####Build Model####
    ######################################################

    def build_model(self):
        self.model = self.experiment.build_model(self.x_train_features, self.x_train_role, self.y_train)


    ######################################################
    ####Train Model####
    ######################################################


    def train(self, epochs=1):
        history = self.model.fit(self.x_fit_train, y=self.y_train,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  initial_epoch=self.trained_epochs,
                  shuffle=True,
                  validation_data=(self.x_fit_validation, self.y_validation),
                  callbacks=self.callback)

        self.trained_epochs +=len(history.epoch)

    ######################################################
    ####Evaluate Model####
    ######################################################


    def summarize(self):
        scores = self.model.evaluate(self.x_fit_test, self.y_test, verbose=1)

        self.create_output_dir()
        file = open(self.evaluation_log_path, 'w')
        file.write('run for ' + str(self.trained_epochs) + ' epochs')

        for i, metric in enumerate(self.model.metrics_names):
            text = 'Test ' + metric + " : " + str(scores[i]) + os.linesep
            print(text)
            file.write(text)

        file.write(os.linesep)
        file.write(os.linesep)
        file.write('#######Params#######')
        file.write(os.linesep)
        file.write(str(self.experiment.get_params()))
        file.write(os.linesep)
        file.write(os.linesep)
        file.write('#######Layer#######')
        file.write(os.linesep)
        file.write(str(self.model.layers))
        file.close()

    ######################################################
    ####Predict####
    ######################################################


    def predict(self):
        prediction = self.model.predict(
            x={'x_features': np.array([self.x_train_features[0]]), 'x_role': np.array([self.x_train_role[0]])})

        print(prediction)
        print(self.y_train[0])

    ######################################################
    ####Plot####
    ######################################################

    def plot(self):
        self.plot_confusion_matrix()
        self.plot_learn_curve()
        self.plot_layers()

    def plot_confusion_matrix(self):
        self.create_output_dir()
        filename = self.confusion_matrix_path + str(self.trained_epochs)
        cls_pred = self.model.predict(x=self.x_fit_test)
        cls_pred = [label.argmax() for label in cls_pred]
        cm = confusion_matrix(y_true=self.y_test_cls,
                              y_pred=cls_pred)

        file = open(filename + '.txt', 'w')
        file.write(str(cm))
        file.close()

        #normalize Data
        summedValues = np.sum(cm, axis=1)
        summedValues = np.reshape(summedValues, [-1, 1])
        print(cm)
        cm_normalized = np.divide(cm, summedValues)

        sn.heatmap(cm_normalized, annot=True, linewidths=.2, cmap="YlGnBu")
        num_actions = len(self.y_test[0])
        tick_marks = np.arange(num_actions)
        plt.xticks(tick_marks, range(num_actions))
        plt.yticks(tick_marks, range(num_actions))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(filename + ".png")
        plt.close()

    def plot_learn_curve(self):
        self.create_output_dir()

        loaded_meta = pd.read_csv(self.csv_path)
        accuracies = loaded_meta.val_acc
        episodes = loaded_meta.epoch

        plt.plot(episodes, accuracies, 'o-', linewidth=2, label='Validation data')
        plt.legend(loc='best')
        plt.xlabel("Episode")
        plt.ylabel("Accuracy score")
        plt.savefig(self.learning_curve_path + '_validation_' +'.png')

        plt.plot(episodes, loaded_meta.acc, 'o-', linewidth=2, label='Training data')
        plt.legend(loc='best')
        plt.savefig(self.learning_curve_path + '_both_' +'.png')
        plt.close()

    def plot_layers(self):
        plot_model(model=self.model, to_file=self.model_plot_path, show_shapes=True, show_layer_names=True)

    ######################################################
    ####Load/Save Model####
    ######################################################

    def delete_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def create_output_dir(self):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def save(self):
        self.create_output_dir()
        self.model.save(self.model_path)
        print('Saved trained model at %s ' % self.model_path)

    def load(self):
        self.model.load_weights(self.model_path)
        print('Loaded trained model')

######################################################
####AutoMode####
######################################################
    def run(self, epoches=200):
        self.delete_output_dir()

        self.plot_confusion_matrix()
        self.train(epoches)
        self.save()
        self.summarize()
        self.plot()
