import torch
def Build_ADJ():
    ADJ = torch.zeros([3508, 3508])

    for i in range(468 * 3):
        for j in range(468*3):
            ADJ[i][j] =1
    for i in range(21 * 3):
        for j in range(21 * 3):
            x = i+468*3
            y = j+468*3
            ADJ[x][y] =1
    for i in range(33 * 3):
        for j in range(33 * 3):
            x = i + 468 * 3+21 * 3
            y = j + 468 * 3+21 * 3
            ADJ[x][y] = 1
    for i in range(21 * 3):
        for j in range(21 * 3):
            x = i + 468 * 3+21 * 3+33 * 3
            y = j + 468 * 3+21 * 3+33 * 3
            ADJ[x][y] = 1
    for i in range(250):
        for j in range(250):
            x = i + 468 * 3+21 * 3+33 * 3+21 * 3
            y = j + 468 * 3+21 * 3+33 * 3+21 * 3
            ADJ[x][y] = 1
    return ADJ

ADJ = Build_ADJ()
ADJ_numpy = ADJ.numpy()
import numpy as np

np.save('./relation.npy', ADJ_numpy)