# Reimportar pacotes após reset
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

# Simulação de dados fictícios para demonstração
type_totals = {
    "Karmine": [1200, 1300, 1100],
    "Opportunist": [1000, 950],
    "Pretender": [1150, 1250, 1400, 1300],
    "Randy": [900, 850],
    "SimpleRL": [1050, 1100],
    "MyRLAgent": [1080, 1120],
    "Copycat": [1350, 1300, 1320]
}

os.makedirs("graphs", exist_ok=True)

# Geração dos dados médios por tipo
avg_by_type = {k: np.mean(v) for k, v in type_totals.items()}
types = list(avg_by_type.keys())
values = list(avg_by_type.values())

# Gráfico de linha simples
plt.figure(figsize=(10, 5))
for name, scores in type_totals.items():
  plt.plot(scores, marker='o', label=name)
plt.title("Linha - Média por Tipo (Histórico por Rodada)")
plt.xlabel("Rodadas")
plt.ylabel("Recompensa Total")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/melhor_media_linha.png")
plt.close()

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

# Gráfico de barras
plt.figure(figsize=(10, 5))
plt.bar(types, values, color='skyblue')
plt.title("Barras - Melhor Média por Tipo de Agente")
plt.ylabel("Média Máxima de Recompensa")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graphs/melhor_media_barras.png")
plt.close()

# Gráfico de barras horizontais
plt.figure(figsize=(10, 5))
plt.barh(types, values, color='lightcoral')
plt.title("Barras Horizontais - Melhor Média por Tipo de Agente")
plt.xlabel("Média Máxima de Recompensa")
plt.tight_layout()
plt.savefig("graphs/melhor_media_barras_horizontais.png")
plt.close()

# Gráfico de radar
categories = types
N = len(categories)
values_circle = values + [values[0]]
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += [angles[0]]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values_circle, linewidth=2, linestyle='solid')
ax.fill(angles, values_circle, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title("Radar - Melhor Média por Tipo")
plt.tight_layout()
plt.savefig("graphs/melhor_media_radar.png")
plt.close()

os.listdir("graphs")
