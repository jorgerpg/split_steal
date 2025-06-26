from collections import defaultdict, Counter
import random
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import os
import pandas as pd


from agents import simple_opponents, rl_agentV2, rl_agentV3, rl_agent

# Parâmetros da recompensa
mean = 100
variance = 10000

# Criação da pasta para gráficos
os.makedirs("NSH/graphs", exist_ok=True)


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
    self.history = []
    self.action_counter = Counter()

  def reset_karma(self):
    self.karma = 0

  def add_karma(self, value):
    self.karma = min(max(self.karma + value, -5), 5)

  def decision(self, total_amount, rounds_played, your_karma, his_karma):
    action = self.agent.decision(
        total_amount, rounds_played, your_karma, his_karma)
    self.action_counter[action] += 1
    return action

  def result(self, your_action, his_action, total_possible, reward):
    self.agent.result(your_action, his_action, total_possible, reward)
    self.history.append(self.total_amount)


# Lista de agentes
AGENT_TYPES = [
    simple_opponents.Karmine,
    simple_opponents.Opportunist,
    simple_opponents.Pretender,
    simple_opponents.Randy,
    rl_agent.RLAgent,
    rl_agentV2.RLAgent,
    rl_agentV3.RLAgent,
    simple_opponents.Copycat
]

agents = [Player(agent()) for agent in AGENT_TYPES]

# Parâmetros do torneio
nrematches = 10
nfullrounds = 500
total_rounds = len(agents) * (len(agents) - 1) * nfullrounds * nrematches // 2
game = Game(total_rounds)

matches_played = defaultdict(lambda: 0)

# Simulação
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

# Exibição dos resultados
print("\n--- RESULTADOS FINAIS ---")
for agent in agents:
  print(f"{agent.name} - Score final: {agent.total_amount:.2f}")

winner = max(agents, key=lambda a: a.total_amount)
print(f"\n🏆 Vencedor: {winner.name} com {winner.total_amount:.2f} pontos.")

# Gráfico geral de desempenho acumulado
plt.figure(figsize=(12, 6))
for agent in agents:
  plt.plot(agent.history, label=agent.name)
plt.title("Desempenho acumulado dos agentes")
plt.xlabel("Rodadas")
plt.ylabel("Recompensa acumulada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("NSH/graphs/desempenho_geral.png")
plt.close()


decision_data = {
    'Agente': [a.name for a in agents],
    'split': [a.action_counter.get('split', 0) for a in agents],
    'steal': [a.action_counter.get('steal', 0) for a in agents]
}
decision_df = pd.DataFrame(decision_data).set_index('Agente')

plt.figure(figsize=(10, 6))
decision_df.plot(kind='bar', stacked=True, colormap='Paired')
plt.title('Decisões (split/steal) por Agente')
plt.ylabel('Frequência')
plt.xlabel('Agentes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('NSH/graphs/decisions_by_agent.png')
plt.close()
