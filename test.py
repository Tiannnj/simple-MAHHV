import random
import math

import numpy
import numpy as np
import gym
import csv

from itertools import chain

print(random.uniform(0.1,0.6))
i = [1, 2, 3, 4, 5]
print(sum(i))


dis_o2v = 70
height_o = 50
# the angle used to calculate the los probability
ars = dis_o2v / height_o
# path loss
PL = (20 * math.log10(83.78 * dis_o2v + 0.00001)) + 1  # 计算path loss
# sum of vehicles to serve
num_v = sum([1, 1, 1, 1, 1, 0])
band_even = 5 / 6
SNR = (0.2 * 1000 * 10 ** (- PL / 10)) / ((10 ** -17.4) * (band_even * 10 ** 6))
link_tran = band_even * math.log2(1 + SNR)
print(link_tran)
num_o_receive = [1, 2, 3]
total_num_o_r = 0
total_num_o_r = sum(num_o_r ** 2 for num_o_r in num_o_receive)
print(total_num_o_r)

remain_ddl = [[0] * 6, [] * 6]
print(remain_ddl)


self = [1,2,3,4,5,6]
pre_state = self[1*2 : 1*4]
print(pre_state)

action = [[1,2,3], [1,2], [3.6]]

ou_action = action[-2:]
for temp in enumerate(ou_action):
    print(temp)
print(ou_action)


t = [[0] * 6] * 2
print(t)

for v in range(0,6):
    i=1
print(v)


r_pre_state = [1,2,3]
print(r_pre_state[0:3])
r_pre_state[0:3]= np.multiply(r_pre_state[0:3], 0)
print(r_pre_state)

rewards_ou = [1,2,3]
rewards_ru = [3,4]
rewards = rewards_ou + rewards_ru
print(rewards)

new_state = [1,2,3,4,5,6,7]
num_o_receive = [14, 15]
new_state[-4: -2] = num_o_receive[:]
print(new_state)


o_receive = np.array([[0], [0]])
print(o_receive[:])
print(o_receive.flatten())

obs = [[[1], [2] ,[3], [4]]]
print('obs', type(obs))
share_obs = np.array(list(obs[0][2]))
print(share_obs)


x = np.array([[1, 2, 3],
     [4, 5, 6]])
y = x[0 : 2, 0 : 1]
print(y)


print(gym.__version__)



from scipy.stats import entropy

entropy_a = entropy([0.5004, 0.4996])

print(entropy_a)

agents_action = [[[1],       [1],       [1],       [0],       [1],       [1],       [1],       [0]],
                  [[1],       [0],       [1],       [0],       [0],       [0],       [0],       [1]]]
print(agents_action[0])
fixed = [[1],       [1],       [1],       [1]]
for w in agents_action:
      w = fixed + w
      print(w)


print(1//2)

a = [5, 2, 3]
b = numpy.prod(a)
print(b)

a = list([1 ,2, 3])
b = list([[2, 4], [2, 4], [2, 5]])
c = list([0, 0, 0])
d = list([1, 0, 0])
x = a + b + c
y = a + b + d
x = x + y
print(x)
for i in range(0, 2):
    print(x[i*9: (i+1)*9])
with open('my_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(0, 2):
        writer.writerow(x[i * 9: (i + 1) * 9])

result_record_name = 'result_record' + str(23) + '.csv'
print(result_record_name)