from collections import defaultdict
import random
import numpy as np
import pickle
import os


class RLAgent:
  def __init__(self):
    self.q_table = defaultdict(lambda: [0.0, 0.0])
    self.alpha = 0.5       # Taxa de aprendizado
    self.gamma = 0.9       # Fator de desconto
    self.epsilon = 0.1     # Exploração inicial
    self.last_state = None
    self.last_action = None

    self.q_table_filename = "mem/qtable_MyRLAgentV2.pkl"
    self.load_q_table()

  def get_name(self):
    return "RLAgentV2"

  def extract_state(self, rounds_left, your_karma, his_karma):
    return (rounds_left, np.sign(your_karma), np.sign(his_karma))

  def decision(self, amount, rounds_left, your_karma, his_karma):
    state = self.extract_state(rounds_left, your_karma, his_karma)

    # Política ε-greedy
    if random.random() < self.epsilon:
      action = random.choice(["split", "steal"])
    else:
      action_index = np.argmax(self.q_table[state])
      action = ["split", "steal"][action_index]

    self.last_state = state
    self.last_action = action
    return action

  def result(self, your_action, his_action, total_possible, reward):
    # Define próxima situação (estado não muda muito entre rodadas)
    next_state = self.last_state
    action_index = ["split", "steal"].index(self.last_action)

    # Novo reward shaping mais competitivo
    if your_action == "split" and his_action == "split":
      reward = 2
    elif your_action == "steal" and his_action == "split":
      reward = 3
    elif your_action == "split" and his_action == "steal":
      reward = -2
    elif your_action == "steal" and his_action == "steal":
      reward = -1

    # Atualiza Q-table com regra de Bellman
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
        loaded = pickle.load(f)
        self.q_table.update(loaded)
