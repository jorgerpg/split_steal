from collections import defaultdict
import random
import numpy as np
import os
import pickle


class RLAgent:
  def __init__(self):
    self.q_table = defaultdict(lambda: [0.0, 0.0])
    self.alpha = 0.6
    self.gamma = 0.95
    self.epsilon = 0.3
    self.epsilon_decay = 0.99
    self.epsilon_min = 0.05
    self.last_state = None
    self.last_action = None
    self.last_reward = 0

    self.q_table_filename = "mem/qtable_MyRLAgentV2.pkl"
    self.load_q_table()

  def get_name(self):
    return "RLAgentV2"

  def karma_bucket(self, karma):
    if karma < -10:
      return -2
    elif karma < 0:
      return -1
    elif karma < 10:
      return 0
    else:
      return 1

  def extract_state(self, your_karma, his_karma):
    return (self.karma_bucket(your_karma), self.karma_bucket(his_karma))

  def decision(self, amount, rounds_left, your_karma, his_karma):
    state = self.extract_state(your_karma, his_karma)

    # Atualiza Q-table com Bellman se tiver histórico
    if self.last_state is not None and self.last_action is not None:
      action_index = ["split", "steal"].index(self.last_action)
      old_value = self.q_table[self.last_state][action_index]
      next_max = max(self.q_table[state])
      new_value = (1 - self.alpha) * old_value + self.alpha * \
          (self.last_reward + self.gamma * next_max)
      self.q_table[self.last_state][action_index] = new_value

    if random.random() < self.epsilon:
      action = random.choice(["split", "steal"])
    else:
      action = ["split", "steal"][np.argmax(self.q_table[state])]

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    self.last_state = state
    self.last_action = action
    return action

  def result(self, your_action, his_action, total_possible, reward):
    # Reward shaping mais cooperativo
    if your_action == "split" and his_action == "split":
      self.last_reward = 3
    elif your_action == "steal" and his_action == "split":
      self.last_reward = 1
    elif your_action == "split" and his_action == "steal":
      self.last_reward = -2
    elif your_action == "steal" and his_action == "steal":
      self.last_reward = -3

  def save_q_table(self):
    with open(self.q_table_filename, "wb") as f:
      pickle.dump(dict(self.q_table), f)

  def load_q_table(self):
    if os.path.exists(self.q_table_filename):
      with open(self.q_table_filename, "rb") as f:
        self.q_table.update(pickle.load(f))
