#!python3
# have Obedient salvation play game and then train itself

import numpy as np
import pandas as pd
from ObedientSalvation import ObedientSalvation
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
import time
import keras
import random
import os

difdict = {'Easy': Difficulty.Easy,
           'Medium': Difficulty.Medium,
           'Hard': Difficulty.Hard}

def run_training(rounds, model, difficulty='Easy'):
    wins = 0

    for i in range(rounds):
        OSal = ObedientSalvation(use_model=True, mind=model)
        print('I serve and Obey')
        print(difficulty)

        result = run_game(maps.get('BelShirVestigeLE'), [
                Bot(Race.Protoss, OSal),
                Computer(difdict[difficulty])],
                realtime=False)
        print(result)

        if result == Result.Victory:
            train_model(OSal.train_data, model)
            win += 1
        else:
            data = OSal.train_data
            data = inverse_data(data)
            print(data[0][0])
            train_model(data, model, learning_rate=0.01, rounds=5)

    winrate = wins/rounds
    return winrate


def train_model(data, model, learning_rate=0.001, rounds=1):
    trainee = keras.models.load_model(model)

    for i in range(rounds):
        batch_size = 128
        data = unbias_data(data, 16)
        print(data[0][0])

        x_train = np.array([k[1] for k in data]).reshape(-1, 160, 144, 3)
        y_train = np.array([k[0] for k in data])


        trainee.fit(x_train, y_train,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1)

        os.remove(model)  # delete the model so it can be resaved, fix this later
        trainee.save(model)
        new_model = keras.models.load_model(model)
        if not trainee == new_model:
            print('\n\n\n\n\n\n\n\n\nFUCK THE SAVE ISNT WORKING ITS BEING A BITCH\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')


def master_model(model, step_min = 0.8):
    winrate = 0
    while winrate < step_min:
        winrate = run_training(10, model, 'Easy')
        print(winrate, 'accross 10 games')
        
    winrate = 0
    while winrate < step_min:
        winrate = run_training(10, model, 'Medium')
        print(winrate, 'accross 10 games')
        
    winrate = 0
    while winrate < step_min:
        winrate = run_training(10, model, 'Hard')
        print(winrate, 'accross 10 games')

    print('I have mastered these enemies')
    

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


def phantom_data():
    # generate phantom training data to add in randomness
    # randomly pick a why value from dataset and assign it a random x value
    # add this to dataset
    pass


def inverse_data(data):
    # intake data from a losing section
    # inverse the data such that every choice value is equal to 1-choice
    # this will train away from previous choices
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = 1 - data[i][0][j]
        #print(data[i][0])
    return data

master_model('CuriousMind')
