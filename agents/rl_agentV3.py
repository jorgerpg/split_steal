import random
from collections import defaultdict
import numpy as np
import pickle
import os


class RLAgent:
  def __init__(self):
    self.q_table = defaultdict(lambda: [0.0, 0.0])  # [split, steal]
    self.alpha = 0.5
    self.gamma = 0.9
    self.epsilon = 0.1  # valor fixo ou use decay leve
    self.last_state = None
    self.last_action = None
    self.last_reward = 0

    self.action_list = ["split", "steal"]
    self.q_table_filename = "mem/qtable_MyRLAgentV3.pkl"
    self.load_q_table()

  def get_name(self):
    return "RLAgentV3"

  def extract_state(self, rounds_left, your_karma, his_karma):
    round_bucket = 0 if rounds_left <= 2 else 1
    return (round_bucket, np.sign(your_karma), np.sign(his_karma))

  def decision(self, amount, rounds_left, your_karma, his_karma):
    state = self.extract_state(rounds_left, your_karma, his_karma)

    # Atualização Q-table com Bellman
    if self.last_state is not None and self.last_action is not None:
      action_index = self.action_list.index(self.last_action)
      old_value = self.q_table[self.last_state][action_index]
      next_max = max(self.q_table[state])
      new_value = (1 - self.alpha) * old_value + self.alpha * (self.last_reward + self.gamma * next_max)
      self.q_table[self.last_state][action_index] = new_value

    # Escolha de ação com exploração
    if random.random() < self.epsilon:
      action = random.choice(self.action_list)
    else:
      action = self.action_list[np.argmax(self.q_table[state])]

    self.last_state = state
    self.last_action = action
    return action

  def result(self, your_action, his_action, total_possible, reward):
    # Política de recompensa adaptativa
    if your_action == "split" and his_action == "split":
      self.last_reward = 1
    elif your_action == "steal" and his_action == "split":
      self.last_reward = 2
    elif your_action == "split" and his_action == "steal":
      self.last_reward = -1
    elif your_action == "steal" and his_action == "steal":
      self.last_reward = 0

  def save_q_table(self):
    os.makedirs(os.path.dirname(self.q_table_filename), exist_ok=True)
    with open(self.q_table_filename, "wb") as f:
      pickle.dump(dict(self.q_table), f)

  def load_q_table(self):
    if os.path.exists(self.q_table_filename):
      with open(self.q_table_filename, "rb") as f:
        self.q_table.update(pickle.load(f))
