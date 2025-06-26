from collections import defaultdict
import random
import numpy as np
import pickle
import os


class RLAgent:
  def __init__(self):
    self.q_table = defaultdict(lambda: [0.0, 0.0])
    self.alpha = 0.5
    self.gamma = 0.9
    self.epsilon = 0.3
    self.epsilon_decay = 0.98
    self.epsilon_min = 0.01
    self.last_state = None
    self.last_action = None

    self.q_table_filename = "mem/qtable_MyRLAgentV2.pkl"
    self.load_q_table()

  def get_name(self):
    return "RLAgentV2"

  def discretize_rounds(self, rounds_left):
    if rounds_left <= 2:
      return 0  # fim de jogo
    elif rounds_left <= 5:
      return 1
    else:
      return 2  # início

  def karma_bucket(self, karma):
    if karma < -10:
      return -2
    elif karma < 0:
      return -1
    elif karma < 10:
      return 0
    else:
      return 1

  def extract_state(self, rounds_left, your_karma, his_karma):
    rounds_bucket = self.discretize_rounds(rounds_left)
    your_k = self.karma_bucket(your_karma)
    his_k = self.karma_bucket(his_karma)
    return (rounds_bucket, your_k, his_k)

  def decision(self, amount, rounds_left, your_karma, his_karma):
    state = self.extract_state(rounds_left, your_karma, his_karma)

    if random.random() < self.epsilon:
      action = random.choice(["split", "steal"])
    else:
      action = ["split", "steal"][np.argmax(self.q_table[state])]

    self.last_state = state
    self.last_action = action

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    return action

  def result(self, your_action, his_action, total_possible, reward):
    action_index = ["split", "steal"].index(self.last_action)
    next_state = self.last_state

    # Reward shaping mais cooperativo
    if your_action == "split" and his_action == "split":
      reward = 3
    elif your_action == "steal" and his_action == "split":
      reward = 1
    elif your_action == "split" and his_action == "steal":
      reward = -2
    elif your_action == "steal" and his_action == "steal":
      reward = -3

    # Bellman update
    old_value = self.q_table[self.last_state][action_index]
    next_max = max(self.q_table[next_state])
    new_value = (1 - self.alpha) * old_value + self.alpha * \
        (reward + self.gamma * next_max)
    self.q_table[self.last_state][action_index] = new_value

  def save_q_table(self):
    with open(self.q_table_filename, "wb") as f:
      pickle.dump(dict(self.q_table), f)

  def load_q_table(self):
    if os.path.exists(self.q_table_filename):
      with open(self.q_table_filename, "rb") as f:
        self.q_table.update(pickle.load(f))
