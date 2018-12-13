#!python
# generate training data for Obedient Salvation and clean it
# import OS and run a bunch of times to collect data, saving the data each time
# clean the data to unbias for the NN
# generate a model and train it using cleaned data
# save the model

import numpy as np
from ObedientSalvation import ObedientSalvation
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
import time
import keras
import random
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

games = 100

def generate_data(size):
    for i in range(size):
        OSal = ObedientSalvation()
        result = run_game(maps.get('BelShirVestigeLE'), [
                Bot(Race.Protoss, OSal),
                Computer(Difficulty.Medium)],
                realtime=False)
        if result == Result.Victory:
            np.save('train_data/base_data/{}.npy'.format(str(int(time.time()))), np.array(OSal.train_data))

# load in data and drop all levels towards norm so model isn't biased toward any action?

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
        data[i] = data[i][:lowest]      # is this working right?  instead of removing, multiply smaller ones up to larger?

    return data


def get_data():
    data_dir = 'train_data/base_data/'
    full_data = []
    for file in os.listdir(data_dir):
        full_path = os.path.join(data_dir, file)
        data = np.load(full_path)
        full_data.append(data)

    return full_data


def generate_forest_model():
    clf = RandomForestClassifier(100)
    all_data = get_data()
    random.shuffle(all_data)

    train_rounds = 55

    for i in range(train_rounds):

        n = random.randint(0, len(all_data)-1)
        data = all_data[n]

        data = unbias_data(data, 17)

        train_data = []

        for j in range(len(data)):
            train_data.append(data[j]) # is this necessary at all?

        random.shuffle(train_data)

        batch_size = 128

        x_train = np.array([k[1] for k in train_data]).reshape(-1, 160, 144, 2)
        y_train = np.array([k[0] for k in train_data])
        print(x_train)

        clf.fit(x_train, y_train)
    clf.save('ObedientMind')


def generate_neural_model():
    learning_rate = 0.001

    
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(160, 144, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16, activation='softmax'))

    opt = keras.optimizers.adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    all_data = get_data()
    random.shuffle(all_data)

    train_rounds = 65 

    for i in range(train_rounds):
        print(i)

        random.shuffle(all_data)
        n = random.randint(0, len(all_data)-1)
        data = all_data[n]

        data = unbias_data(data, 16)

        train_data = []

        for j in range(len(data)):
            train_data.append(data[j]) # is this necessary at all?

        random.shuffle(train_data)
        train_data = unbias_data(train_data, 16)

        batch_size = 128

        x_train = np.array([k[1] for k in train_data]).reshape(-1, 160, 144, 3)
        y_train = np.array([k[0] for k in train_data])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  shuffle=True,
                  verbose=1)
    model.save('CuriousMind')


def train_model(model):
    # add in the training stuff above to save space
    # find a way to switch up naming conventions?
    pass
    

# generate_data(5)
generate_neural_model()



