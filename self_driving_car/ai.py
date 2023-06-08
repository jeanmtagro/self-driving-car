# AI for Self Driving Car

# Importing the libraries
import numpy
import os
import random

import torch
import torch.nn as nn  # neural netk
import torch.nn.functional as F  # pytorch utils
import torch.optim as optim
from torch.autograd import Variable  # tensor -> tensor+gradient


# Creating the architecture of the Neural Network
class Network(nn.Module):
    """input_size : entries : num of sensors : 5 -> voir map.py - vecteur last_signal
    fc : fully connected
    """

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action  # action2rotation (map.py)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    # propagation -> predict q-values
    def forward(self, state):  # state: etat de l'env, à l'entrée
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Implementing Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity  # taille N max de la memoire
        self.memory = []  # transitions qu'on va retenir de taille capacity

    def push(self, event):  # event: transition
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]  # retirer l'evenement le plus ancien

    def sample(self, batch_size):  # choix aleatoire de `batch_size` transitions
        # transition=(s,a,s+1,reward)
        # ex 2 trans : [(2,4,5,7), (4,5,7,3)]
        samples_ = random.sample(self.memory, batch_size)
        # -> isoler ensemble a instant t les action, les etats, rewards...
        samples = zip(*samples_)  # -> [(2,4), (4,5), (5,7), (7,3)]

        # -> see https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
        # torch.cat(x, 0) ->
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning
class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # lr: taux apprentissage = alpha : tempor diff?
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # unsqueeze(0): rajouter une dimension
        self.last_action = 0  # indice 0 des actions action2rotation=[0,-20,20]
        self.last_reward = 0

    def select_action(self, state):
        """dim =1 pour recup les actions, et pas par ex les états..
        T: Temperature pour exacerber les probas : +haut->++haut ; bas->++bas
        T trop fort (ex: 100) -> aucune incertitude -> que de l'exploitation
        T trop faible (ex: 0) -> que de l'exploration"""
        T = 100
        probs = F.softmax(self.model(state) * T, dim=1)  # T=100? 0? # add dim=1  ###Variable(state, volatile=True)
        action = probs.multinomial(num_samples=1)  # choisi 1 index avec une propa = la valeur correspondante
        return action.data[0, 0]  # inverse de unsqueeze(0)

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[
            0]  # le max sur les actions - puis on recup les etats
        target = batch_reward + self.gamma * next_outputs
        td_loss = F.smooth_l1_loss(outputs, target)  # fc de coût
        self.optimizer.zero_grad()  # clean les calculs de gradient effectués avant
        td_loss.backward()  # back-propagation
        self.optimizer.step()  # enregitre les results (poids ajustés) ds l'optimizer

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
