import numpy as np
import pandas as pd
import keras
import os

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, ZeroPadding2D, UpSampling2D
from keras.models import Model
from keras import losses


class c4Scenario9:
    folder = 'connect4'
    file_name = os.path.basename(__file__)

    files = ['connect4_role0.csv', 'connect4_role1.csv']
    metadata_size = 4
    state_size = 128
    num_actions = 8

    def get_params(self):
        return {'batch_size':128,
                'experiment_description': 'CNN with sqaured field with size 42 * 42 and randomly shuffled, batchsize increased'}

    def file_path(self):
        return self.folder, self.file_name

    def load_data(self):
        role_index = []
        features_without_role = []
        labels = []

        for file_name in self.files:
            data_folder = os.path.join(self.folder, 'data')
            file_path = os.path.join(data_folder, file_name)
            loaded_meta = pd.read_csv(file_path, usecols=range(0, self.metadata_size), header=None)
            loaded_features = pd.read_csv(file_path,
                                          usecols=range(self.metadata_size, self.metadata_size + self.state_size),
                                          header=None)
            loaded_labels = pd.read_csv(file_path,
                                        usecols=range(self.metadata_size + self.state_size,
                                                      self.metadata_size + self.state_size +
                                                      self.num_actions), header=None)

            role_index = [*role_index, *[obj[2] for obj in loaded_meta.values]]
            features_without_role = [*features_without_role, *[obj[2:] for obj in loaded_features.values]]
            labels = [*labels, *loaded_labels.values]

        features_without_role = np.array(features_without_role).reshape(([-1, 42, 1, 3]))
        features_without_role = np.repeat(features_without_role, len(features_without_role[0]), axis=2)
        transposed = np.swapaxes(features_without_role, 0, 2)
        np.random.seed(42)
        for row in transposed:
            row = np.random.shuffle(row)
        features_without_role = np.swapaxes(transposed, 0, 2)
        return role_index, features_without_role, labels

    def build_model(self, x_train_features, x_train_roles, y_train):
        print("Build CNN Model")

        inputFeatures = Input(shape=x_train_features[0].shape, name="x_features")
        inputRole = Input(shape=(1,), dtype='float32', name="x_role")

        # ((top_pad, bottom_pad), (left_pad, right_pad))
        layers = Conv2D(16, (4, 4), padding='same', data_format="channels_last", activation='relu')(inputFeatures)
        layers = MaxPooling2D(pool_size=(2, 2))(layers)

        layers = ZeroPadding2D(padding=((1, 0), (1, 0)), data_format="channels_last")(layers)
        layers = Conv2D(8, (2, 2), padding='same', data_format="channels_last", activation='relu')(layers)
        layers = MaxPooling2D(pool_size=(2, 2))(layers)
        layers = Flatten()(layers)

        layers = keras.layers.concatenate([inputRole, layers])

        layers = Dense(128, activation='relu')(layers)
        layers = Dense(len(y_train[0]), activation='relu')(layers)

        model = Model(inputs=[inputFeatures, inputRole], outputs=layers)

        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss=losses.mean_squared_logarithmic_error,
                      optimizer=opt,
                      metrics=['accuracy'])

        return model
