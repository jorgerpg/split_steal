import random


def always_split_callback(total_amount, rounds_left, your_karma, his_karma):
  return 'split'


def always_steal_callback(total_amount, rounds_left, your_karma, his_karma):
  return 'steal'


def always_random_callback(total_amount, rounds_left, your_karma, his_karma):
  return random.choice(['steal', 'split'])


def always_his_karma_callback(total_amount, rounds_left, your_karma, his_karma):
  return "split" if his_karma >= 0 else "steal"


def always_steal_on_last_round_callback(total_amount, rounds_left, your_karma, his_karma):
  return "steal" if rounds_left <= 0 else "split"


def always_karma_positive_callback(total_amount, rounds_left, your_karma, his_karma):
  return "steal" if your_karma >= 1 else "split"


class StaticAgent:

  def __init__(self, name, decision_callback):
    self.decision_callback = decision_callback
    self.name = name

  def get_name(self):
    return self.name

  def decision(self, total_amount, rounds_left, your_karma, his_karma):
    return self.decision_callback(total_amount, rounds_left, your_karma, his_karma)

  def result(self, your_action, his_action, total_possible, reward):
    pass


class Splitter(StaticAgent):
  def __init__(self):
    StaticAgent.__init__(self, "Splitter", always_split_callback)


class Randy(StaticAgent):
  def __init__(self):
    StaticAgent.__init__(self, "Randy", always_random_callback)


class Stealer(StaticAgent):
  def __init__(self):
    StaticAgent.__init__(self, "Stealer", always_steal_callback)


class Karmine(StaticAgent):
  def __init__(self):
    StaticAgent.__init__(self, "Karmine", always_his_karma_callback)


class Opportunist(StaticAgent):
  def __init__(self):
    StaticAgent.__init__(self, "Opportunist",
                         always_steal_on_last_round_callback)


class Pretender(StaticAgent):
  def __init__(self):
    StaticAgent.__init__(self, "Pretender", always_karma_positive_callback)

# Implementação do agente Copycat


class Copycat(StaticAgent):
  def __init__(self):
    # Armazena a última ação do oponente
    self.last_opponent_action = None
    # Indica se é a última rodada
    self.last_round = False
    # Inicializa a classe base com o nome e o callback de decisão
    super().__init__("Copycat", self.copycat_callback)

  def copycat_callback(self, total_amount, rounds_left, your_karma, his_karma):
    # Atualiza o status se é a última rodada
    self.last_round = rounds_left == 0
    # Na primeira rodada, sempre "split"
    if self.last_opponent_action is None:
      return "split"
    # Nas demais, copia a última ação do oponente
    return self.last_opponent_action

  def result(self, your_action, his_action, total_possible, reward):
    # Se for a última rodada, reseta a memória
    if self.last_round:
      self.last_opponent_action = None
    else:
      # Armazena a última ação do oponente para a próxima rodada
      self.last_opponent_action = his_action
