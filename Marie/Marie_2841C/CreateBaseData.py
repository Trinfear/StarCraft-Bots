#!python3
# generate training data for Marie2841C and clean it
# better to do through here isntead of extending Marie into spaghetti

import numpy as np
import pandas as pd
from Marie import Marie2841C
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
import time
import keras
import random



