import time
import numpy as np
from sklearn.metrics import mean_squared_error
from bn_as_fitness_v2 import mix_and_transform_with_tnormal,  functions
import itertools


# ============================
# Tabela com os dados do especialista
# ============================
expert_data = [
                               
    {"AT": "VL", "AC": "VH", "AE_expert": [0.274, 0.323, 0.274, 0.081, 0.048]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0.172, 0.259, 0.345, 0.172, 0.052]},
    {"AT": "VL",  "AC": "VL",  "AE_expert": [0.333, 0.333, 0.283, 0.050, 0.0]},
    {"AT": "VH",  "AC": "VH", "AE_expert": [0.0, 0.055, 0.273, 0.309, 0.364]},
    {"AT": "VL",  "AC": "M", "AE_expert": [0.2, 0.3, 0.34, 0.1, 0.06]},
    {"AT": "M",   "AC": "VL", "AE_expert": [0.357, 0.357, 0.179, 0.107, 0.0]},
]


# Geração do repositório de amostras
#repositorio = gerar_amostras_base_por_estado()
import pickle
with open('repositorio.pkl', 'rb') as f:
    repositorio = pickle.load(f)

pesos_possiveis = list(itertools.product(range(1, 6), repeat=2))  # gerar pesos possiveis para 2 pais
#pesos_possiveis = list(itertools.product(range(1, 6), repeat=3)) #pra 3 pais

variancias = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]



inicio = time.time()

melhor_erro = float("inf")
melhor_config = {}
print("Iniciando calibração por força bruta...\n")

for nome_func, func in functions.items():
  
    for pesos in pesos_possiveis:
        variancias = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
        for variance in variancias:
            erros = []
            for c in expert_data:
                estados_pais = [c["AT"], c["AC"]]  #  testando com 2 pais
                    
                probs_model = mix_and_transform_with_tnormal(
                    estados_pais, 
                    pesos, 
                    repositorio,
                    variance=round(variance, 2),
                    func_comb=func
                )
                erro = mean_squared_error(c["AE_expert"], probs_model)
                erros.append(erro)
            media_brier = np.mean(erros)
            # LOG de cada combinação testada
            #print(f"Função: {nome_func:<10} | Estados_pais: {estados_pais} | Pesos: {pesos} | Variância: {variance:.2f} | Brier Médio: {media_brier:.5f}")


            if media_brier < melhor_erro:
                melhor_erro = media_brier
                melhor_config = {
                    "funcao": nome_func,
                    "pesos": pesos,
                    "variance": round(variance, 2),
                    "brier_medio": round(media_brier, 8)
                }

print("Melhor configuração encontrada:")
print(melhor_config)
# Fim da medição
fim = time.time()

# ⏱ Resultado

print(f"Tempo de execução força bruta: {fim - inicio:.4f} segundos")

minutos = int((fim - inicio) // 60)
segundos = (fim - inicio) % 60
print(f"Tempo de execução: {minutos} minutos e {segundos:.2f} segundos")
