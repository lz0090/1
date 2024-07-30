import random
import numpy as np

def split_number(num, parts):
    numbers = [random.randint(1, num // parts) for i in range(parts - 1)]
    numbers.append(num - sum(numbers))
    return numbers

# test1 = np.load('E:/project/PPO_static/loc_delay/loc_delay_static.npy', allow_pickle = True)
# print(test1)
#
# test2 = np.load('E:/project/PPO_static/loc_delay/loc_dynamic_all.npy', allow_pickle = True)
# print(test2)