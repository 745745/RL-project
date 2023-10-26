from gameInput import KeyboardInput
from random import choice
import numpy as np
import readData

def randomTest():
    action_key = ['UP', 'LEFT', 'DOWN', 'RIGHT', 'up+d', 'left+d', 'down+d', 'right+d']
    while True:
        KeyboardInput(choice(action_key))
        if dataReader.getAllData()[1]==0.0:
            return dataReader.getAllData()[0]

dataReader = readData.gameData()
randomTime = [ ]
for i in range(10):
    randomTime.append(randomTest())
    while True:
        KeyboardInput("space")
        if dataReader.getAllData()[1] == 1500.0:
            break

print(np.mean(np.array(randomTime)))
# avg time:17.89s