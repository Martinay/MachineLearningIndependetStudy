import numpy as np
import pandas as pd
import keras
import os

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, ZeroPadding2D
from keras.models import Model
from keras import losses


class c4Scenario1:
    folder = 'connect4'
    file_name = os.path.basename(__file__)

    files = ['connect4_role0.csv', 'connect4_role1.csv']
    metadata_size = 4
    state_size = 128
    num_actions = 8

    def get_params(self):
        return {'experiment_description':'first experiment'}

    def file_path(self):
        return self.folder, self.file_name

    def load_data(self):
        role_index = []
        features_without_role = []
        labels = []

        for file_name in self.files:
            data_folder = os.path.join(self.folder, 'data')
            file_path = os.path.join(data_folder, file_name)
            loaded_meta = pd.read_csv(file_path, usecols=range(2, self.metadata_size - 1), header=None)
            loaded_features = pd.read_csv(file_path, usecols=range(self.metadata_size + 2, self.metadata_size + self.state_size),
                                          header=None)
            loaded_labels = pd.read_csv(file_path,
                                        usecols=range(self.metadata_size + self.state_size + 1, self.metadata_size + self.state_size +
                                                      self.num_actions), header=None)

            role_index = [*role_index, * loaded_meta.values]
            features_without_role = [*features_without_role, *loaded_features.values]
            labels = [*labels, *loaded_labels.values]

        features_without_role = np.array(features_without_role).reshape(([-1, 7, 6, 3]))
        return role_index, features_without_role, labels

    def build_model(self, x_train_features, x_train_roles, y_train):
        print("Build CNN Model")

        inputFeatures = Input(shape=x_train_features[0].shape, name="x_features")
        inputRole = Input(shape=(1,), dtype='float32', name="x_role")

        #((top_pad, bottom_pad), (left_pad, right_pad))
        layers = ZeroPadding2D(padding=((1, 0), (0, 0)), data_format="channels_last")(inputFeatures)
        layers = Conv2D(16, (4, 4), padding='same', data_format="channels_last", activation='relu')(layers)
        layers = Flatten()(layers)

        layers = keras.layers.concatenate([inputRole, layers])

        layers = Dense(len(y_train[0]), activation='relu')(layers)

        model = Model(inputs=[inputFeatures, inputRole], outputs=layers)

        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss=losses.mean_squared_logarithmic_error,
                      optimizer=opt,
                      metrics=['accuracy'])

        return model