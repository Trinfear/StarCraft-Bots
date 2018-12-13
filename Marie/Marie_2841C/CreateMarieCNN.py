#!python3
# generate a CNN for marie and save it
# run one time then let the model be handled elsewhere

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
# generate a CNN model
# train model on Base_Data
# save model

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(160, 144, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(11, activation='softmax'))       # change the number of outputs depending on the number of actions

# just cut off and save here? save to two files and write all the training stuff elsewhere?

learning_rate = 0.001
train_rounds = 25

opt = keras.optimizers.adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

def unbias_data(data, size):
    choices = []
    for i in range(size):
        choices.append([])
    for d in data:  # make an equal number of examples of each case to avoid bias
        choice = np.argmax(d[0])
        choices[int(choice)].append([d[0], d[1]])
    choices = [x for x in choices if x != []]
    lengths = []
    for i in choices:
        lengths.append(len(i))
    print(lengths)
    lowest = min(lengths)
    for i in range(len(choices)):
        random.shuffle(choices[i])
        data[i] = data[i][:lowest]

    return data

# tensorboard = TensorBoard(log_dir'logs/stage1')


eco_data_dir = 'eco_train_data/base_data/'
mil_data_dir = 'mil_train_data/base_data/'
eco_data = []
mil_data = []


##for file in os.listdir(eco_data_dir):
##    data = np.load(file)
##    eco_data.append(Data)

for file in os.listdir(mil_data_dir):
    full_path = os.path.join(mil_data_dir, file)
    data = np.load(full_path)       # this is loading data wrong
    # print(data)
    mil_data.append(data)

for i in range(train_rounds):
    random.shuffle(eco_data)
    random.shuffle(mil_data)        # just train on mil data for now?
    print(len(mil_data))

    n = random.randint(0, len(mil_data)-1)

    data = mil_data[n]
    data = list(data)


    data = unbias_data(data, 11)

    train_data = []
    for j in range(len(data)):
        train_data.append(data[j])

    random.shuffle(train_data)

    test_size = 10
    batch_size = 128

    x_train = np.array([k[1] for k in train_data]).reshape(-1, 160, 144, 3)
    y_train = np.array([k[0] for k in train_data])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              shuffle=True,
              verbose=1)


model.save('MarieCNN2841A')
        


model.save('Marie2841C_Initial_CNN')
