import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

files = ['connect4_role0.csv', 'connect4_role1.csv']
metadata_size = 4
state_size = 128
num_actions = 8

role_index = []
features_without_role = []
labels = []

for file_name in files:
    data_folder = os.path.join('connect4', 'data')
    file_path = os.path.join(data_folder, file_name)
    loaded_meta = pd.read_csv(file_path, usecols=range(0, metadata_size), header=None)
    loaded_features = pd.read_csv(file_path, usecols=range(metadata_size, metadata_size + state_size),
                                          header=None)
    loaded_labels = pd.read_csv(file_path,
                                        usecols=range(metadata_size + state_size, metadata_size + state_size +
                                                      num_actions), header=None)

    role_index = [*role_index, *[obj[2] for obj in loaded_meta.values]]
    features_without_role = [*features_without_role, *[obj[2:] for obj in loaded_features.values]]
    labels = [*labels, *[obj[1:] for obj in loaded_labels.values]]

features_without_role = np.array(features_without_role).reshape(([-1, 7, 6, 3]))

print('Number of entries : {}'.format(len(features_without_role)))

count_action0 = sum(x[0] for x in labels)
count_action1 = sum(x[1] for x in labels)
count_action2 = sum(x[2] for x in labels)
count_action3 = sum(x[3] for x in labels)
count_action4 = sum(x[4] for x in labels)
count_action5 = sum(x[5] for x in labels)
count_action6 = sum(x[6] for x in labels)

figure = plt.figure()

D = {u'0':count_action0, u'1': count_action1, u'2':count_action2, u'3':count_action3, u'4':count_action4, u'5':count_action5, u'6':count_action6}

plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())

plt.title('Action distribution')
plt.xlabel('Action')
plt.ylabel('Number of Actions')
plt.savefig('connect4/data/histogramm.png')