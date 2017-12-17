import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

files = ['breakthrough_role1.csv']#, 'breakthrough_role0.csv']
metadata_size = 4
state_size = 130
num_actions = 155

role_index = []
features_without_role = []
labels = []

for file_name in files:
    file_path = os.path.join('data', file_name)
    loaded_meta = pd.read_csv(file_path, usecols=range(2, metadata_size - 1), header=None)
    loaded_features = pd.read_csv(file_path,
                                  usecols=range(metadata_size + 2, metadata_size + state_size),
                                  header=None)
    loaded_labels = pd.read_csv(file_path,
                                usecols=range(metadata_size + state_size + 1,
                                              metadata_size + state_size +
                                              num_actions), header=None)

    role_index = [*role_index, *loaded_meta.values]
    features_without_role = [*features_without_role, *loaded_features.values]
    labels = [*labels, *loaded_labels.values]

features_without_role = np.array(features_without_role).reshape(([-1, 8, 8, 2]))

print('Number of entries : {}'.format(len(features_without_role)))

count_action = []
for i, element in enumerate(labels[0]):
    count_action.append(sum(x[i] for x in labels))

figure = plt.figure()

plt.bar(range(len(count_action)), count_action, align='center')
plt.xticks(range(len(count_action)), range(len(count_action)))

plt.title('Action distribution')
plt.xlabel('Action')
plt.ylabel('Number of Actions')
plt.savefig('data/histogramm_role1.png')