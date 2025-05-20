from bn_as_fitness_v2_translated import mix_and_transform_with_tnormal, repository, functions

import random       
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import csv


TPN1 = [
    {"AT": "VL", "AC": "VH", "AE": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VL", "AC": "VL", "AE": "VH", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VL", "AC": "VH", "AE": "VH", "AE_expert": [0, 0, 1, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE": "VH", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VH", "AE": "VL", "AE_expert": [0, 0, 1, 0, 0]}
]
TPN2 = [
    {"AT": "VL", "AC": "VH", "AE": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VL", "AC": "VL", "AE": "VH", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VL", "AC": "VH", "AE": "VH", "AE_expert": [0, 0, 1, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE": "VH", "AE_expert": [0, 0, 1, 0, 0]},
    {"AT": "VH", "AC": "VH", "AE": "VL", "AE_expert": [0, 0, 1, 0, 0]}
]
TPN3 = [
    {"AT": "VL", "AC": "VH", "AE": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE": "VL", "AE_expert": [1, 0, 0, 0, 0]},
    {"AT": "VL", "AC": "VL", "AE": "VH", "AE_expert": [1, 0, 0, 0, 0]},
    {"AT": "VL", "AC": "VH", "AE": "VH", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE": "VH", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VH", "AE": "VL", "AE_expert": [0, 1, 0, 0, 0]}
]
TPN4 = [
    {"AT": "VL", "AC": "VH", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VL", "AC": "M",  "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "M",  "AC": "VL", "AE_expert": [0, 1, 0, 0, 0]}
]
TPN5 = [
    {"AT": "VL", "AC": "VH", "AE_expert": [0, 0, 0, 1, 0]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "VL", "AC": "M",  "AE_expert": [0, 1, 0, 0, 0]},
    {"AT": "M",  "AC": "VL", "AE_expert": [0, 1, 0, 0, 0]}
]
AT_AC_AE = [
    {"AT": "VL", "AC": "VH", "AE_expert": [0.274, 0.323, 0.274, 0.081, 0.048]},
    {"AT": "VH", "AC": "VL", "AE_expert": [0.172, 0.259, 0.345, 0.172, 0.052]},
    {"AT": "VL",  "AC": "VL",  "AE_expert": [0.333, 0.333, 0.283, 0.050, 0.0]},
    {"AT": "VH",  "AC": "VH", "AE_expert": [0.0, 0.055, 0.273, 0.309, 0.364]},
    {"AT": "VL",  "AC": "M", "AE_expert": [0.2, 0.3, 0.34, 0.1, 0.06]},
    {"AT": "M",   "AC": "VL", "AE_expert": [0.357, 0.357, 0.179, 0.107, 0.0]},
]
expert_data = TPN3  # or TPN1, TPN2...

def generate_weights(n_parents):
    return np.random.randint(1, 6, size=n_parents)  # Values between 1 and 5 

variances = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]

# Individual representation
class Individual:
    def __init__(self, n_parents):
        self.function = random.choice(list(functions.keys()))
        self.weights = generate_weights(n_parents)
        self.variance = random.choice(variances)
        self.fitness = None

    def evaluate(self, repository, verbose=False):
        errors = []
        self.probs_per_scenario = []  # List to save results

        for c in expert_data:
            if "AE" in c:
                parent_states = [c["AT"], c["AC"], c["AE"]]
            else:
                parent_states = [c["AT"], c["AC"]]

            probs_model = mix_and_transform_with_tnormal(
                parent_states, self.weights, repository,
                variance=self.variance,
                func_comb=functions[self.function]
            )

            error = mean_squared_error(c["AE_expert"], probs_model)
            errors.append(error)

            self.probs_per_scenario.append({
                "scenario": parent_states,
                "expected": c["AE_expert"],
                "calculated": probs_model.tolist(),
                "brier": error
            })

            if verbose:
                print(f"Scenario: {parent_states}")
                print(f"Expected: {c['AE_expert']}")
                print(f"Calculated: {np.round(probs_model, 4).tolist()}")
                print(f"Brier Score: {error:.5f}\n")

        self.fitness = np.mean(errors)
        return self.fitness

def export_results_excel_like(individual):
    print("\nVarA\tVarB\tVL\tL\tM\tH\tVH\tVL\tL\tM\tH\tVH\tBrier")
    for r in individual.probs_per_scenario:
        a, b = r["scenario"][:2]
        expected = "\t".join(str(e) for e in r["expected"])
        calculated = "\t".join(f"{p:.3f}" for p in r["calculated"])
        print(f"{a}\t{b}\t{expected}\t{calculated}\t{r['brier']:.5f}")

# Population Initialization
def initialize_population(pop_size, n_parents):
    return [Individual(n_parents) for _ in range(pop_size)]

# Tournament Selection
def tournament_selection(population, k=3):
    return min(random.sample(population, k), key=lambda ind: ind.fitness)

# One-Point Crossover
def crossover(parent1, parent2):
    child = Individual(len(parent1.weights))
    child.function = parent1.function if random.random() < 0.5 else parent2.function
    point = random.randint(1, len(parent1.weights) - 1)
    child.weights = np.concatenate((parent1.weights[:point], parent2.weights[point:]))
    child.variance = random.choice([parent1.variance, parent2.variance])
    return child

# Mutation
def mutation(ind, mutation_rate):
    if random.random() < mutation_rate:
        ind.function = random.choice(list(functions.keys()))
    if random.random() < mutation_rate:
        ind.weights = generate_weights(len(ind.weights))
    if random.random() < mutation_rate:
        ind.variance = random.choice(variances)
    return ind

# Main Genetic Algorithm
def genetic_algorithm(pop_size, n_parents, max_gen, mutation_rate, repository, functions):
    population = initialize_population(pop_size, n_parents)
    for ind in population:
        ind.evaluate(repository)

    for generation in range(max_gen):
        new_population = []
        elite = min(population, key=lambda ind: ind.fitness)
        new_population.append(elite)  # Elitism

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < 0.8:
                child = crossover(parent1, parent2)
            else:
                child = tournament_selection(population)

            child = mutation(child, mutation_rate)
            child.evaluate(repository)
            new_population.append(child)

        population = new_population
        
        best = sorted(population, key=lambda ind: ind.fitness)
        #print(f"[GEN {generation}] Best Brier: {best[0].fitness:.5f} | Function: {best[0].function} | Weights: {best[0].weights} | Variance: {best[0].variance}")

    best = min(population, key=lambda ind: ind.fitness)
    print("\nBest configuration found:")
    print(f"Function: {best.function}, Weights: {best.weights}, Variance: {best.variance}, Brier Score: {best.fitness}")
    print("\n📋 Probabilities and Brier Score per scenario:")
    best.evaluate(repository, verbose=True)
    return best

# Execution Example
start = time.time()
n_parents = len([k for k in expert_data[0] if k.startswith("A") and k != "AE_expert"])
best_ind = genetic_algorithm(pop_size=50, n_parents=n_parents, max_gen=10, mutation_rate=0.1, repository=repository, functions=functions)

print(best_ind)
end = time.time()

print(f"Execution time GA:  {end - start:.4f} seconds")

minutes = int((end - start) // 60)
seconds = (end - start) % 60
print(f"Execution time GA: {minutes} minutes and {seconds:.2f} seconds")

export_results_excel_like(best_ind)

def save_results_csv(individual, filename="ga_results.csv"):
    import csv

    header = ["VarA", "VarB", "VL_exp", "L_exp", "M_exp", "H_exp", "VH_exp",
              "VL_calc", "L_calc", "M_calc", "H_calc", "VH_calc", "Brier"]

    with open(filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(header)

        for r in individual.probs_per_scenario:
            states = r["scenario"]
            expected = r["expected"]
            calculated = [f"{p:.5f}".replace('.', ',') for p in r["calculated"]]
            brier = f"{r['brier']:.5f}".replace('.', ',')

            if len(states) == 2:
                row = [states[0], states[1]] + expected + calculated + [brier]
            elif len(states) == 3:
                row = [states[0], states[1], states[2]] + expected + calculated + [brier]
                if len(header) == 13:
                    header.insert(2, "VarC")
                    writer.writerow(header)

            writer.writerow(row)

    print(f"\n CSV saved to: {filename}")

save_results_csv(best_ind)
