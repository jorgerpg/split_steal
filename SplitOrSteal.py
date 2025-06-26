from collections import defaultdict
import random
import numpy as np
from itertools import combinations

from agents import simple_opponents, rl_agentV2, rl_agent

mean = 100
variance = 10000


class Game:
  def __init__(self, total_rounds):
    self.rounds_played = 0
    self.total_rounds = total_rounds
    self.current_amount = 0

  def isOver(self):
    return self.rounds_played >= self.total_rounds

  def prepare_round(self):
    self.current_amount = max(mean, np.random.normal(mean, np.sqrt(variance)))

  def play_round(self, left_agent, right_agent, remaining):
    self.rounds_played += 1

    left_decision = left_agent.decision(
        self.current_amount, remaining, left_agent.karma, right_agent.karma)
    right_decision = right_agent.decision(
        self.current_amount, remaining, right_agent.karma, left_agent.karma)

    if left_decision == right_decision == "steal":
      left_reward, right_reward = 0, 0
    elif left_decision == right_decision == "split":
      left_reward = right_reward = self.current_amount / 2
    elif left_decision == 'steal':
      left_reward, right_reward = self.current_amount, 0
    else:
      left_reward, right_reward = 0, self.current_amount

    left_agent.total_amount += left_reward
    right_agent.total_amount += right_reward

    left_agent.result(left_decision, right_decision,
                      self.current_amount, left_reward)
    right_agent.result(right_decision, left_decision,
                       self.current_amount, right_reward)

    left_agent.add_karma(-1 if left_decision == "steal" else 1)
    right_agent.add_karma(-1 if right_decision == "steal" else 1)


class Player:
  def __init__(self, agent):
    self.name = agent.get_name()
    self.agent = agent
    self.total_amount = 0
    self.karma = 0

  def reset_karma(self):
    self.karma = 0

  def add_karma(self, value):
    self.karma = min(max(self.karma + value, -5), 5)

  def decision(self, total_amount, rounds_played, your_karma, his_karma):
    return self.agent.decision(total_amount, rounds_played, your_karma, his_karma)

  def result(self, your_action, his_action, total_possible, reward):
    self.agent.result(your_action, his_action, total_possible, reward)


# Definir agentes
agents = [
    Player(simple_opponents.Karmine()),
    Player(simple_opponents.Karmine()),
    Player(rl_agent.RLAgent()),
    Player(rl_agentV2.RLAgent())
]

nrematches = 10
nfullrounds = 100

total_rounds = len(agents) * (len(agents) - 1) * nfullrounds * nrematches // 2
game = Game(total_rounds)

# Estatísticas de partidas
matches_played = defaultdict(lambda: 0)

# Jogar partidas
while not game.isOver():
  random.shuffle(agents)
  for a in agents:
    a.reset_karma()

  for player1, player2 in combinations(agents, 2):
    matches_played[player1.name] += 1
    matches_played[player2.name] += 1

    for remaining in reversed(range(nrematches)):
      game.prepare_round()
      game.play_round(player1, player2, remaining)

# Exibir resultados finais
print(matches_played)
max_score = -1
best = None
for a in agents:
  print(f"O agente '{a.name}' obteve {a.total_amount:.2f}")
  if a.total_amount > max_score:
    best = a
    max_score = a.total_amount

print(f"Vencedor: {best.name}")
print(f"Score: {max_score:.2f}")
