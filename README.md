# RNM-GA Calibration

**Automatic Calibration of Ranked Nodes in Bayesian Networks using Genetic Algorithms**

This repository contains the complete implementation of the method proposed in the paper _“Automatic Calibration of Ranked Nodes in Bayesian Networks Using Genetic Algorithms”_. It provides the source code, the Bayesian inference model using the Ranked Nodes Method (RNM), and a unified XLS file with expert-elicited scenarios and validation analyses.

---

## 📁 Repository Structure

All files are located inside the `ga_bn_calibration` folder:

```
📦 ga_bn_calibration/
├── ga_calibration.py             # Genetic Algorithm implementation
├── bn_ranked_nodes.py            # RNM-based Bayesian Network implementation
├── brute_force.py                # Brute-force calibration (for comparison purposes)
├── repository.json               # Expert-elicited scenarios (JSON format)
├── repository.pkl                # Same scenarios (binary format for fast loading)
├── data_validation_softcom.xlsx # Unified spreadsheet with scenarios and validation analyses
```

---

## 🧠 Description

This project automates the calibration of the **Ranked Nodes Method (RNM)** for Bayesian Networks using a **Genetic Algorithm (GA)** to optimize:
- Aggregation function (e.g., WMEAN, WMIN, WMAX, MIXMINMAX)
- Parent weights
- Variance

The optimization goal is to **minimize the Brier Score**, which quantifies the accuracy of probabilistic predictions in comparison to expert-defined expectations.

---

## 📊 Data Validation File

The file `data_validation_softcom.xlsx` contains both the input data and all validation analyses. It includes:

### 📌 Node Tabs (`TPN#1` to `TPN#5`)
Each sheet corresponds to a node:
- `TPN#1 - PBO`: Product Backlog Ordering  
- `TPN#2 - SR`: Sprint Review Quality  
- `TPN#3 - SE`: Software Engineering Techniques Quality  
- `TPN#4 - PBQ`: Product Backlog Quality  
- `TPN#5 - WVQ`: Work Validation Quality  

Each tab includes:
- Scenario configurations (parent states)
- Expected distributions (expert-elicited)
- Calculated distributions using:
  - GA-calibrated RNM
  - Production Rule (PR) configuration
  - Original configuration
- Brier Scores for each method
- Selected function type, weights, variance, and execution time

### `Data Analysis` Tab
- Boxplots comparing GA vs PR
- Q-Q plots for normality
- Shapiro-Wilk and Wilcoxon test results
- Summary of statistically significant results

---

## 🚀 How to Use

To execute the calibration process for Ranked Nodes using both the Genetic Algorithm and the Brute-Force method, follow the steps below. **All scripts and resources are located inside the `ga_bn_calibration` folder.**

### 1. Prerequisites
Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pgmpy
```

---

### 2. Input Files
Ensure the following input files are present:

- `repository.json` or `repository.pkl`: Contains the expert-elicited samples for each state.
- The Python modules: `bn_ranked_nodes.py`, `ga.py`, and `brute_force.py`

---

### 3. Run the Genetic Algorithm (GA)
This will search the best combination of aggregation function, weights, and variance to minimize the Brier Score across expert scenarios.

```bash
cd ga_bn_calibration
python ga_calibration.py
```

Results:
- Best configuration (function, weights, variance)
- Brier Score per scenario
- CSV file: `resultados_ag.csv`

---

### 4. Run the Brute-Force Calibration (Optional)
This will exhaustively evaluate all combinations of parameters to find the optimal configuration.

```bash
cd ga_bn_calibration
python brute_force.py
```

Results:
- Best configuration found
- Comparison with expert distributions
- CSV file: `resultados_forca_bruta.csv`

---

## 📜 License and Citation

This project is licensed under the GNU General Public License v3.0. You can find the full text of the license in the LICENSE file in this repository.
