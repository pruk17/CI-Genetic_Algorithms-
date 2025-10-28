"""
GeneticAlgo_HW3_660610817.py  (Terminal-only output)

Genetic Algorithm (GA) to search Multilayer Perceptron (MLP) architectures
and hyperparameters, evaluated via 10-fold stratified cross-validation (i.e., 10% per fold).

- Expects 'wdbc.csv' in the same folder with headers:
  id_number, diagnosis, feature_1 ... feature_30  (diagnosis ∈ {M,B})
- Genome encodes:
  * n_layers ∈ {1,2,3}
  * n1,n2,n3 ∈ [4,128] nodes (n2 used if n_layers≥2, n3 if n_layers=3)
  * activation ∈ {relu,tanh,logistic}
  * alpha ∈ [1e-6, 1e-1] (log-uniform)
  * lr_init ∈ [1e-4, 1e-1] (log-uniform)
  * seed ∈ ℕ (random_state for MLP)
- Fitness = mean accuracy (10-fold stratified CV).

This version adds:
  • Clear legend at top (no abbreviations in meaning)
  • Friendlier numeric formatting for alpha/lr (e.g., 0.0983 instead of 9.83e-02)
  • Stronger architecture/seed mutation to avoid same hls/seed every generation
  • Lower elitism (1) + slightly higher crossover to encourage exploration
  • Per-generation diversity stats (unique HLS patterns, unique activations)

Run:
  python GeneticAlgo_HW3_660610817.py --generations 10 --population 24
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Utility
# -----------------------------

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_wdbc(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    expected_first_cols = ["id_number", "diagnosis"]
    if list(df.columns[:2]) != expected_first_cols:
        print(
            f"[WARN] First two columns are {list(df.columns[:2])}, expected {expected_first_cols}. Proceeding anyway.",
            file=sys.stderr
        )
    y = df["diagnosis"].map({"M": 1, "B": 0}).values
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values.astype(np.float64)
    return X, y

# -----------------------------
# GA Encoding
# -----------------------------

ACTIVATIONS = ["relu", "tanh", "logistic"]

@dataclass
class Genome:
    n_layers: int
    n1: int
    n2: int
    n3: int
    activation: str
    alpha: float
    lr_init: float
    seed: int

    def hidden_layer_sizes(self) -> Tuple[int, ...]:
        if self.n_layers == 1:
            return (self.n1,)
        elif self.n_layers == 2:
            return (self.n1, self.n2)
        else:
            return (self.n1, self.n2, self.n3)

def random_genome() -> Genome:
    return Genome(
        n_layers=random.randint(1, 3),
        n1=random.randint(4, 128),
        n2=random.randint(4, 128),
        n3=random.randint(4, 128),
        activation=random.choice(ACTIVATIONS),
        alpha=10 ** random.uniform(-6, -1),
        lr_init=10 ** random.uniform(-4, -1),
        seed=random.randint(0, 10_000)
    )

def mutate(g: Genome) -> Genome:
    """เพิ่มโอกาสกลายพันธุ์ของสถาปัตยกรรมและ seed เพื่อให้หลากหลายมากขึ้น"""
    g = Genome(**asdict(g))
    pm_struct = 0.5
    pm_hyper = 0.35
    pm_act = 0.3

    if random.random() < pm_struct:
        g.n_layers = random.randint(1, 3)
    if random.random() < pm_struct:
        g.n1 = int(np.clip(int(round(np.random.normal(g.n1, 10))), 4, 128))
    if random.random() < pm_struct:
        g.n2 = int(np.clip(int(round(np.random.normal(g.n2, 10))), 4, 128))
    if random.random() < pm_struct:
        g.n3 = int(np.clip(int(round(np.random.normal(g.n3, 10))), 4, 128))
    if random.random() < pm_act:
        g.activation = random.choice(ACTIVATIONS)
    if random.random() < pm_hyper:
        g.alpha = float(np.clip(g.alpha * (10 ** np.random.uniform(-0.7, 0.7)), 1e-7, 1e-0))
    if random.random() < pm_hyper:
        g.lr_init = float(np.clip(g.lr_init * (10 ** np.random.uniform(-0.7, 0.7)), 1e-5, 1e-0))
    if random.random() < pm_struct:
        g.seed = random.randint(0, 10_000)
    return g

def crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    fa = list(asdict(a).values())
    fb = list(asdict(b).values())
    point = random.randint(1, len(fa) - 1)
    ca = fa[:point] + fb[point:]
    cb = fb[:point] + fa[point:]
    keys = list(asdict(a).keys())
    ga = Genome(**{k: v for k, v in zip(keys, ca)})
    gb = Genome(**{k: v for k, v in zip(keys, cb)})
    return ga, gb

# -----------------------------
# Evaluation (10-fold CV)
# -----------------------------

def evaluate_genome(g: Genome, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> float:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    scores = []
    for train_idx, test_idx in kfold.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=g.hidden_layer_sizes(),
                activation=g.activation,
                solver="adam",
                alpha=g.alpha,
                learning_rate_init=g.lr_init,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=g.seed,
            )
        )
        clf.fit(X_tr, y_tr)
        y_pr = clf.predict(X_te)
        scores.append(accuracy_score(y_te, y_pr))
    return float(np.mean(scores))

# -----------------------------
# Helper display
# -----------------------------

def print_legend_once():
    print(
        "\nLegend for per-generation output:\n"
        "  GEN   = Generation index\n"
        "  best  = Best mean accuracy from 10-fold cross validation\n"
        "  avg   = Average mean accuracy of all individuals in the generation\n"
        "  hls   = Hidden layer sizes (tuple of nodes per hidden layer)\n"
        "  act   = Activation function (relu/tanh/logistic)\n"
        "  alpha = L2 regularization strength\n"
        "  lr    = Initial learning rate for Adam optimizer\n"
        "  seed  = Random seed for the model (controls reproducibility)\n"
    )

def fmt_float(x: float) -> str:
    return f"{x:.6f}".rstrip('0').rstrip('.') if x < 1 else f"{x:.4f}"

# -----------------------------
# GA main loop
# -----------------------------

def tournament_select(pop: List[Tuple[Genome, float]], k: int = 3) -> Genome:
    contenders = random.sample(pop, k)
    return max(contenders, key=lambda t: t[1])[0]

def run_ga(X, y, generations=10, population_size=24, crossover_rate=0.9, elitism=1, seed=42, verbose=True):
    set_global_seed(seed)
    population = [random_genome() for _ in range(population_size)]
    best_tuple = None
    fitness_cache = {}

    def fitness(g):
        key = json.dumps(asdict(g), sort_keys=True)
        if key not in fitness_cache:
            fitness_cache[key] = evaluate_genome(g, X, y)
        return fitness_cache[key]

    if verbose:
        print_legend_once()

    for gen in range(1, generations + 1):
        scored = [(g, fitness(g)) for g in population]
        scored.sort(key=lambda t: t[1], reverse=True)
        best_g, best_f = scored[0]
        avg_f = np.mean([f for _, f in scored])
        unique_hls = len(set(tuple(g.hidden_layer_sizes()) for g, _ in scored))
        unique_act = len(set(g.activation for g, _ in scored))

        print(
            f"[GEN {gen:02d}] best={best_f:.4f} avg={avg_f:.4f} "
            f"hls={best_g.hidden_layer_sizes()} act={best_g.activation} "
            f"alpha={fmt_float(best_g.alpha)} lr={fmt_float(best_g.lr_init)} seed={best_g.seed} "
            f"| diversity: unique_hls={unique_hls}, unique_act={unique_act}"
        )

        if best_tuple is None or best_f > best_tuple[1]:
            best_tuple = (best_g, best_f)

        new_pop = [g for g, _ in scored[:elitism]]
        while len(new_pop) < population_size:
            p1, p2 = tournament_select(scored), tournament_select(scored)
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            new_pop.extend([mutate(c1), mutate(c2)])
        population = new_pop[:population_size]

    return best_tuple

# -----------------------------
# Entry point
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm MLP Search with 10-fold CV")
    parser.add_argument("--csv", type=str, default="wdbc.csv")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    X, y = load_wdbc(args.csv)
    best_g, best_f = run_ga(X, y, generations=args.generations, population_size=args.population, seed=args.seed)

    print("\n=== BEST CONFIGURATION (10-fold CV) ===")
    print(f"hidden_layer_sizes: {best_g.hidden_layer_sizes()}")
    print(f"activation        : {best_g.activation}")
    print(f"alpha (L2)        : {fmt_float(best_g.alpha)}")
    print(f"learning_rate_init: {fmt_float(best_g.lr_init)}")
    print(f"seed              : {best_g.seed}")
    print(f"mean_accuracy     : {best_f:.4f}")
    print("[DONE]")

if __name__ == "__main__":
    main()
