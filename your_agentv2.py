from collections import defaultdict
import random
import numpy as np


class ReinforcementLearningAgent:
  def __init__(self):
    self.q_table = defaultdict(lambda: [0.0, 0.0])
    self.alpha = 0.6
    self.gamma = 0.95
    self.epsilon = 0.3  # começa alto
    self.epsilon_decay = 0.99
    self.epsilon_min = 0.05
    self.last_state = None
    self.last_action = None

  def get_name(self):
    return "MyRLAgent"

  def extract_state(self, your_karma, his_karma):
    return (np.sign(your_karma), np.sign(his_karma))

  def decision(self, amount, rounds_left, your_karma, his_karma):
    state = self.extract_state(your_karma, his_karma)

    # Atualiza Q-table se tiver histórico
    if self.last_state is not None and self.last_action is not None:
      reward = self.last_reward  # vindo do último result()
      action_index = ["split", "steal"].index(self.last_action)
      old_value = self.q_table[self.last_state][action_index]
      next_max = max(self.q_table[state])
      self.q_table[self.last_state][action_index] = \
          (1 - self.alpha) * old_value + self.alpha * \
          (reward + self.gamma * next_max)

    # Política epsilon-greedy
    if random.random() < self.epsilon:
      action = random.choice(["split", "steal"])
    else:
      action = ["split", "steal"][np.argmax(self.q_table[state])]

    # Decaimento de epsilon
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    self.last_state = state
    self.last_action = action
    return action

  def result(self, your_action, his_action, total_possible, reward):
    # Reward shaping mais indulgente
    if your_action == "split" and his_action == "split":
      self.last_reward = 1
    elif your_action == "steal" and his_action == "split":
      self.last_reward = 2
    elif your_action == "split" and his_action == "steal":
      self.last_reward = 0
    elif your_action == "steal" and his_action == "steal":
      self.last_reward = -1
