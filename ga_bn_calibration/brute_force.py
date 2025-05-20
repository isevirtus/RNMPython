
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from bn_as_fitness_v2_translated import mix_and_transform_with_tnormal,  functions
import itertools

# ============================
# expert data
# ============================
expert_data = [
    {"AT": "VL", "AC": "VH", "AE_expert": [0.274, 0.323, 0.274, 0.081, 0.048]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0.172, 0.259, 0.345, 0.172, 0.052]},
    {"AT": "VL",  "AC": "VL",  "AE_expert": [0.333, 0.333, 0.283, 0.050, 0.0]},
    {"AT": "VH",  "AC": "VH", "AE_expert": [0.0, 0.055, 0.273, 0.309, 0.364]},
    {"AT": "VL",  "AC": "M", "AE_expert": [0.2, 0.3, 0.34, 0.1, 0.06]},
    {"AT": "M",   "AC": "VL", "AE_expert": [0.357, 0.357, 0.179, 0.107, 0.0]},
]



import pickle
with open('repository.pkl', 'rb') as f:
    repository = pickle.load(f)

possible_weights = list(itertools.product(range(1, 6), repeat=2))  

variances = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]

start = time.time()

best_error = float("inf")
best_config = {}
print("begin brute force ...\n")

for nome_func, func in functions.items():
    for weights in possible_weights:
        variances = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
        for variance in variances:
            errors = []
            for c in expert_data:
                parent_states = [c["AT"], c["AC"]]
                probs_model = mix_and_transform_with_tnormal(
                    parent_states,
                    weights,
                    repository,
                    variance=round(variance, 2),
                    func_comb=func
                )
                error = mean_squared_error(c["AE_expert"], probs_model)
                errors.append(error)
            mean_brier = np.mean(errors)

            if mean_brier < best_error:
                best_error = mean_brier
                best_config = {
                    "function": nome_func,
                    "weights": weights,
                    "variance": round(variance, 2),
                    "mean_brier": round(mean_brier, 8)
                }

print("Best config found:")
print(best_config)

end = time.time()

print(f"execution time: {end - start:.4f} sec")


