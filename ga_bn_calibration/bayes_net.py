import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import json
import ast
import time


from sklearn.metrics import mean_squared_error
import itertools

import pickle

# ================================
# Class Bayesian Network
# ================================
class BNetwork:
    def __init__(self):
        self.model = BayesianNetwork()
        self.nodes = {}
        self.evidence = {}

    def createNode(self, node_id, name, outcomes):
        self.nodes[node_id] = {"name": name, "outcomes": outcomes}
        self.model.add_node(node_id)

    def addEdge(self, parent_id, child_id):
        self.model.add_edge(parent_id, child_id)

    def setNodeCPD(self, node_id, cpt_values):
        parent_ids = list(self.model.get_parents(node_id))
        parent_cards = [len(self.nodes[p]["outcomes"]) for p in parent_ids]
        var_card = len(self.nodes[node_id]["outcomes"])
        cpt_values = np.array(cpt_values)
        if cpt_values.shape != (var_card, np.prod(parent_cards)):
            raise ValueError(f"Incorrect format for {node_id}. Expected {(var_card, np.prod(parent_cards))}, but received {cpt_values.s
    cpd = TabularCPD(
            variable=node_id,
            variable_card=var_card,
            values=cpt_values.tolist(),
            evidence=parent_ids if parent_ids else None,
            evidence_card=parent_cards if parent_cards else None,
            state_names={node_id: self.nodes[node_id]["outcomes"], **{pid: self.nodes[pid]["outcomes"] for pid in parent_ids}}
        )
        self.model.add_cpds(cpd)

    def updateBeliefs(self):
        self.model.check_model()
        infer = VariableElimination(self.model)
        evidence_dict = self.evidence if self.evidence else {}
        return {
            node_id: infer.query([node_id], evidence=evidence_dict).values.tolist()
            for node_id in self.nodes if node_id not in evidence_dict
        }

    def setEvidence(self, node_id, state_name):
        self.evidence[node_id] = state_name

    def calculateTPN(self, node_id):
        infer = VariableElimination(self.model)
        return infer.query([node_id], evidence=self.evidence).values.tolist()



# ================================
# Beginning of the Bayesian Network creation
# ================================
bn = BNetwork()

states = {
    'VL': {'lower': 0.0, 'upper': 0.2},
    'L':  {'lower': 0.2, 'upper': 0.4},
    'M':  {'lower': 0.4, 'upper': 0.6},
    'H':  {'lower': 0.6, 'upper': 0.8},
    'VH': {'lower': 0.8, 'upper': 1.0}
}

bn.createNode("AT", "Technical Aptitude", list(states.keys()))
bn.createNode("AC", "Collaborative Aptitude", list(states.keys()))
bn.createNode("AE", "Team Aptitude", list(states.keys()))


bn.addEdge("AT", "AE")
bn.addEdge("AC", "AE")

cpd_at = [[0.2], [0.2], [0.2], [0.2], [0.2]]
cpd_ac = [[0.2], [0.2], [0.2], [0.2], [0.2]]


#wmean:
def wmean(*args):
    if len(args) % 2 != 0:
        raise ValueError("An even number of arguments (weight-value pairs) is required.")

    partial_sum_value = 0.0
    partial_sum_weight = 0.0

    for i in range(0, len(args), 2):
        w = args[i]
        x = args[i + 1]
        partial_sum_value += w * x
        partial_sum_weight += w

    if partial_sum_weight == 0:
        return None

    return partial_sum_value / partial_sum_weight


#wmin:
def wmin(*args):
    if len(args) % 2 != 0:
        raise ValueError("An even number of arguments (weight-value pairs) is required.")

    n = len(args) // 2
    if n < 2:
        return None

    weights = [args[2 * i] for i in range(n)]
    values = [args[2 * i + 1] for i in range(n)]

    if sum(weights) == 0:
        return None

    S = sum(values)
    current_min = float('inf')
    for i in range(n):
        w_i = weights[i]
        denom = w_i + (n - 1)
        numerator = w_i * values[i] + (S - values[i])
        e_i = numerator / denom
        current_min = np.minimum(current_min, e_i)  # Comparação elemento a elemento (alteracao feita no codigo)

    return current_min

#wmax:
def wmax(*args):
    if len(args) % 2 != 0:
        return None

    n = len(args) // 2
    if n < 2:
        return None

    weights = []
    values = []
    for i in range(n):
        w_i = args[2 * i]
        x_i = args[2 * i + 1]
        if w_i < 0 or not np.all((0 <= x_i) & (x_i <= 1)): #adaptando essa linha para trabalhar com vetores de amostras, e não com valores escalares.
            return None
        weights.append(w_i)
        values.append(x_i)

    all_zero_denominators = True
    for i in range(n):
        denom = weights[i] + (n - 1)
        if denom != 0:
            all_zero_denominators = False
            break
    if all_zero_denominators:
        return None

    max_e = None
    sum_all = sum(values)
    for i in range(n):
        w_i = weights[i]
        x_i = values[i]
        denom = w_i + (n - 1)
        numerator = w_i * x_i + (sum_all - x_i)
        e_i = numerator / denom
        if max_e is None: #adaptando aqui para comparar arrays posição a posição.
            max_e = e_i
        else:
            max_e = np.maximum(max_e, e_i)
    return max_e

#mixminmax:
def mixminmax(*args):
    """
    Calculates:
      (wmin * min(values) + wmax * max(values)) / (wmin + wmax)
    Subject to:
      - All weights must be non-negative
      - Return None if sum of weights is zero
      - Raise ValueError if invalid arguments are passed
    Expected Input:
      mixminmax(w1, x1, w2, x2, ..., wn, xn)
    """
    if len(args) % 2 != 0:
        raise ValueError("An even number of arguments (weight-value pairs) is required.")

    n = len(args) // 2
    weights = [args[2 * i] for i in range(n)]
    values = [args[2 * i + 1] for i in range(n)]

    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")
    if sum(weights) == 0:
        return None

    values_array = np.array(values)  # matriz N x amostras
    mins = np.min(values_array, axis=0)
    maxs = np.max(values_array, axis=0)

    return (weights[0] * mins + weights[1] * maxs) / sum(weights)

# ============================================


# ================================
# Step 2 – Mixing with the chosen function and conversion using TNormal
# ================================

def mix_and_transform_with_tnormal(estados_pais, pesos, repository, variance, func_comb):
    #Validation
    if not estados_pais:
        raise KeyError("estados_pais is empty")

    pesos = np.array(pesos, dtype=float)
    if len(pesos) != len(estados_pais):
        raise ValueError("length mismatch between pesos and estados_pais")
    if np.any(pesos < 0):
        raise ValueError("invalid weights (negative values)")
    if np.sum(pesos) <= 0:
        raise ValueError("invalid weights (sum ≤ 0)")

    amostras_por_pai = []
    for estado in estados_pais:
        if estado not in repository:
            raise KeyError(f"Estado {estado} not in repository")
        samples = repository[estado]['amostras']
        if len(samples) < 10000:
            raise ValueError("less than 10 000 samples")
        #amostras_por_pai.append(np.random.choice(samples, size=10000, replace=False))
        amostras_por_pai.append(np.array(samples[:10000]))

    intercalado = [item for pair in zip(pesos, amostras_por_pai) for item in pair] #adaptado para empacotar pesos e amostras
    valores_continuos = func_comb(*intercalado)

    
    if (not isinstance(valores_continuos, np.ndarray) or
        valores_continuos.ndim != 1 or
        len(valores_continuos) != 10000 or
        np.any(valores_continuos < 0) or
        np.any(valores_continuos > 1)):
        raise ValueError("incompatible shape or values returned by func_comb")

    mean = np.mean(valores_continuos)
    if variance <= 0:
        variance = 0.0001 #ATRIBUINDO VALOR MINIMO PRA EVITAR ERERRO COM VARIANCIA NULA OU NEGATICA
    max_variance = mean * (1 - mean)
    if variance > max_variance:
        variance = max(max_variance, 0.0001)  # garante limite mínimo pra variancia
        #raise ValueError("variance exceeds μ(1-μ)")
    std = np.sqrt(variance)
    
    try:
        dist = truncnorm((0 - mean) / std, (1 - mean) / std, loc=mean, scale=std)
    except (FloatingPointError, ZeroDivisionError):
        raise ValueError("invalid distribution parameters")

    bins = np.linspace(0, 1, 6)
    probs = np.array([dist.cdf(bins[i+1]) - dist.cdf(bins[i]) for i in range(5)])
    probs = np.round(probs, 3)

    if np.abs(np.sum(probs) - 1) > 1e-6:
        probs /= np.sum(probs)  # Renormaliza

    return np.round(probs, 3)



functions = {
    "WMEAN": wmean,
    "WMIN": wmin,
    "WMAX": wmax,
    "MIXMINMAX": mixminmax
}
def carregar_amostras_json(caminho_arquivo='repository.json'):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        repository = json.load(f)
        # Converte as listas de volta para arrays do NumPy
        for estado in repository:
            repository[estado]['amostras'] = np.array(repository[estado]['amostras'])
        return repository

# Exemplo de uso
repository = carregar_amostras_json()
