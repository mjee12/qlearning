from environment import MountainCar
import sys
import random
import numpy as np

"""learnR = 0.01
gamma = 0.99
epsilon = 0.05
maxIter = 200
episodes = 4
mode = 'raw'
car = MountainCar(mode)"""

def vectorize(car, state, mode):
  if mode == "tile":
    x = np.zeros(car.state_space)
    x[list(state.keys())] = 1
  if mode == 'raw':
    x = np.array([state[0], state[1]])
  return x

#weights = np.zeros((car.state_space, car.action_space))
#bias = 0

def q_learning(car, weights, bias, episodes, maxIter, learnR, gamma, epsilon, mode):
  rewardList = []
  for i in range(episodes):
    score = 0
    state = car.reset()
    state = vectorize(car, state, mode)
    for j in range(maxIter):
      Q = np.dot(state, weights) + bias
      if random.randint(0, 1) < epsilon:
        action = random.randint(0,2)
      else:
        action = np.argmax(Q)
      Q = Q[action]
      state_, reward, done = car.step(action) 
      state_ = vectorize(car, state_, mode)
      score += reward
      Q_ = max(np.dot(state_, weights) + bias)
      weights[:,action] -= learnR * (Q - (reward + gamma * Q_)) * state
      bias -= learnR * (Q - (reward + gamma * Q_))
      state = state_
      if done:
        break
    rewardList.append(score)
  return bias, weights, rewardList

#bias, weights, rewardList = q_learning(weights, bias, episodes, maxIter, learnR, gamma, epsilon, mode)

def main(args):
  mode = str(args[1])
  weights_out = args[2]
  returns_out = args[3]
  episodes = int(args[4])
  maxIter = int(args[5])
  epsilon = float(args[6])
  gamma = float(args[7])
  learnR = float(args[8])

  car = MountainCar(mode)

  weights = np.zeros((car.state_space, car.action_space))
  bias = 0

  bias, weights, rewardList = q_learning(car, weights, bias, episodes, maxIter, learnR, gamma, epsilon, mode)

  with open(weights_out, "w") as file:
    file.write("%f\n" % bias)
    for i in range(len(weights)):
      for j in range(len(weights[i])):
        file.write("%f\n" % weights[i][j])
    file.close()

  with open(returns_out, "w") as file:
    for i in rewardList:
      file.write("%f\n" % i)
    file.close()

if __name__ == "__main__":
    main(sys.argv)