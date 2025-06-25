from collections import defaultdict, Counter
import random
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import os
import pandas as pd

import simple_opponents
import your_agent
import rl_agent

# Parâmetros da recompensa
mean = 100
variance = 10000

# Criação da pasta para gráficos
os.makedirs("graphs", exist_ok=True)


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


# Configurações do torneio por eliminação
AGENTS_PER_TYPE = 10
ELIMINATE_BOTTOM_N = 5
N_REMATCHES = 25

AGENT_TYPES = [
    simple_opponents.Karmine,
    simple_opponents.Opportunist,
    simple_opponents.Pretender,
    simple_opponents.Randy,
    rl_agent.RLAgent,
    your_agent.ReinforcementLearningAgent,
    simple_opponents.Copycat
]

# Criar agentes iniciais
agents = []
for agent_class in AGENT_TYPES:
  for _ in range(AGENTS_PER_TYPE):
    agents.append(Player(agent_class()))

# Armazenar o melhor score total por tipo de agente
best_score_by_type = defaultdict(float)
type_totals = defaultdict(list)
rl_performance = defaultdict(list)
qtable_evolution = defaultdict(list)

round_index = 1

while len(agents) > 1:
  print(f"\n=== RODADA DE ELIMINAÇÃO {round_index} ===")
  for agent in agents:
    agent.total_amount = 0
    agent.history = []
    agent.karma = 0
    agent.action_counter = Counter()

  game = Game(total_rounds=1e9)

  for player1, player2 in combinations(agents, 2):
    for remaining in reversed(range(N_REMATCHES)):
      game.prepare_round()
      game.play_round(player1, player2, remaining)

  print("\nPontuação atual:")
  for agent in agents:
    print(f"{agent.name} - Score: {agent.total_amount:.2f}")
    if agent.total_amount > best_score_by_type[agent.name]:
      best_score_by_type[agent.name] = agent.total_amount
    type_totals[agent.name].append(agent.total_amount)

    if agent.name in ["SimpleRL", "MyRLAgent"]:
      rl_performance[agent.name].append(agent.total_amount)
      qtable = getattr(agent.agent, "q_table", getattr(agent.agent, "Q", {}))
      qtable_evolution[agent.name].append(len(qtable))

  # Gráfico de RL por rodada
  rl_agents = [agent for agent in agents if agent.name in [
      "SimpleRL", "MyRLAgent"]]
  if rl_agents:
    plt.figure(figsize=(12, 6))
    for agent in rl_agents:
      plt.plot(agent.history, label=agent.name)
    plt.title(f"Desempenho dos Agentes de RL - Rodada {round_index}")
    plt.xlabel("Partidas")
    plt.ylabel("Recompensa acumulada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graphs/rl_rodada_{round_index}.png")
    plt.close()

  # Eliminar os piores
  agents.sort(key=lambda x: x.total_amount)
  eliminated = agents[:min(ELIMINATE_BOTTOM_N, len(agents)-1)]
  agents = agents[min(ELIMINATE_BOTTOM_N, len(agents)-1):]

  print(f"\n❌ Eliminados ({len(eliminated)}): {[a.name for a in eliminated]}")
  print(f"✅ Restantes: {len(agents)} agentes\n")
  round_index += 1

# Final
print("\n=== AGENTE VENCEDOR ===")
print(f"🏆 {agents[0].name} com score final: {agents[0].total_amount:.2f}")

# Gráfico final:
# Gráfico de linha com preenchimento
plt.figure(figsize=(10, 5))
for name, scores in type_totals.items():
  plt.plot(scores, label=name)
  plt.fill_between(range(len(scores)), scores, alpha=0.2)
plt.title("Linha Preenchida - Média por Tipo")
plt.xlabel("Rodadas")
plt.ylabel("Recompensa Total")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/melhor_media_linha_preenchida.png")
plt.close()

# Gráfico final dos agentes de RL, se ainda existirem
rl_agents = [agent for agent in agents if agent.name in [
    "SimpleRL", "MyRLAgent"]]
if rl_agents:
  plt.figure(figsize=(12, 6))
  for agent in rl_agents:
    plt.plot(agent.history, label=agent.name)
  plt.title("Histórico dos Agentes de RL na Rodada Final")
  plt.xlabel("Partidas")
  plt.ylabel("Recompensa acumulada")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig("graphs/rl_final.png")
  plt.close()

# Gráfico: Comparação de desempenho por rodada dos agentes de RL
plt.figure(figsize=(10, 6))
for name, scores in rl_performance.items():
  plt.plot(range(1, len(scores)+1), scores, marker='o', label=name)
  plt.fill_between(range(1, len(scores)+1), scores, alpha=0.1)
plt.title("Comparação de Desempenho Médio por Rodada (RL Agents)")
plt.xlabel("Rodadas")
plt.ylabel("Recompensa Média")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/comparacao_performance_rl_agents.png")
plt.close()

# Gráfico: Evolução do tamanho da Q-table dos agentes de RL
plt.figure(figsize=(10, 6))
for name, sizes in qtable_evolution.items():
  plt.plot(range(1, len(sizes)+1), sizes, marker='s',
           label=f"{name} - Q-table Size")
plt.title("Evolução do Tamanho da Q-table por Rodada")
plt.xlabel("Rodadas")
plt.ylabel("Entradas não nulas na Q-table")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/evolucao_qtable_rl_agents.png")
plt.close()
