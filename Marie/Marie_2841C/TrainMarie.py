#!python3
# train Marie2841C
# run through a game using a model
# if result=win collect data and train model for each datapoint
# run again
# train until winrate > 80%?
# randomly reset weights until first win?
# have bots play themself with small random changes each time
# train both bots off of the winner, then make random changes to both

import numpy as np
import pandas as pd
from Marie import Marie2841C
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
import time
import keras
import random

#MarieA = Marie2841C(military_model=True) # add in named of saved models for 
#MarieB = Marie2841C(military_model=True)

# maries will each load in the model saved to file based on given input name
# both models will train off of the winner, then make a few random changes and save
# repeat
# if game too long auto exit and make large changes to both sides
# if one commander wins a large number of games in a row, erase the other commander and replace it with a copy of its superior

'''
results = []
for i in range(5):
    result = run_game(maps.get('ProximaStationLE'), [
            Bot(Race.Terran, MarieA),
            Bot(Race.Terran, MarieB)],
            realtime=False)
    results.append(result)
    if result == Result.Victory:
        if Marie.Military_model:
            np.save('mil_train_data/{}.npy'.format(str(int(time.time()))), np.array(Marie.military_data))
        if Marie.economic_model:
            np.save('eco_train_data/{}.npy'.format(str(int(time.time()))), np.array(Marie.economy_data))
print(results)
'''
# create initial training data using decsion tree, by having decision tree append a number and screen pic on its decisions?
'''
for i in range(5):
    MarieC = Marie2841C()  # this has to get declared here so there is not an issue with stacking up data from previous games
    start_time = time.time()
    result = run_game(maps.get('BelShirVestigeLE'), [
        Bot(Race.Terran, MarieC),
        Computer(Difficulty.Easy)],
        realtime=False)
    total_time = time.time() - start_time
    print('\n\n', result, total_time)
    if result == Result.Victory:
        print('A+')
        np.save('mil_train_data/{}.npy'.format(str(int(time.time()))), np.array(MarieC.military_data))
        print(np.array(MarieC.military_data))
        np.save('eco_train_data/{}.npy'.format(str(int(time.time()))), np.array(MarieC.economy_data))
    # train NN model
    # make random changes to CNN Model
    # save model and to be reloaded
'''
difdict = {'Easy': Difficulty.Easy,
           'Medium': Difficulty.Medium,
           'Hard': Difficulty.Hard}

def run_training(rounds, econ_model=None, mil_model=None, difficulty='Easy'):
    wins = 0
    use_econ = False
    use_mil = False
    if econ_model:
        use_econ = True
    if mil_model:
        use_mil = True

    for i in range(rounds):
        MarieC = Marie2841C(economy_model=use_econ, military_model=use_mil,
                            economist_name=econ_model, commander_name=mil_model)
        print('our girls ready to go')
        print(difficulty)
        result = run_game(maps.get('BelShirVestigeLE'), [
            Bot(Race.Terran, MarieC),
            Computer(difdict[difficulty])],
            realtime=False)
        print(result)
        if result == Result.Victory:
            #if econ_model:
            #    train_model(MarieC.economy_data, econ_model)
            if mil_model:
                train_model(MarieC.military_data, mil_model)
            wins+=1
        else:
            if mil_model:
                data = MarieC.military_data
                data = inverse_data(data)       # this is converting all data to nonetype now?
                train_model(data, mil_model, learning_rate=0.01, rounds=5)
    winrate = wins/rounds
    return winrate


def train_model(data, model, learning_rate=0.001, rounds=1):
    # load model from file
    # train model for rounds on data
    # add some random change to model
    # save model
    trainee = keras.models.load_model(model)    # is this a different map than original?
    for i in range(rounds):
        
        batch_size = 128

        data = unbias_data(data, 11)

        x_train = np.array([k[1] for k in data]).reshape(-1, 160, 144, 3)
        y_train = np.array([k[0] for k in data])

        trainee.fit(x_train, y_train,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1)
    if trainee == keras.models.load_model(model):
        print('\n\n\n\n\n\n\n\nthe model didnt change AT ALL\n\n\n\n\n\n\nNO CHANGE NO CHANGE\n\n\n\n\n\n\n\n\n\n\n\n')

    trainee.save(model) # is model getting saved correctly?
    if not trainee == keras.models.load_model(model):
        print('\n\n\n\n\n\n\nTHE SAVE DIDNT WORK\n\n\n\n\n\n\nNEW NODEL IS NOT A REAL NEW FRIEND\n\n\n\n\n\n\n\n\n\n\n\n\n')


def master_model(eco_model=None, mil_model=None, step_minimum=0.8):
    winrate = 0
    while winrate < step_minimum:
        print('this far okay?')
        winrate = run_training(10, eco_model, mil_model, 'Easy')
        print(winrate, ' across 10 games')

    winrate = 0
    while winrate < step_minimum:
        winrate = run_training(10, eco_model, mil_model, 'Medium')

    winrate = 0
    while winrate < step_minimum:
        winrate = run_training(10, eco_model, mil_model, 'Hard')

    print('Marie is now a master starcraft player')


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
    pass


def inverse_data(data):
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = 1 - data[i][0][j]
    return data

master_model(mil_model='MarieCNN2841A')

