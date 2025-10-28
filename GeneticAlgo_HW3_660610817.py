#!/usr/bin/env python3
"""
GeneticAlgo_HW3_660610817.py

Genetic Algorithm (GA) to search Multilayer Perceptron (MLP) architectures
and hyperparameters, evaluated via 10-fold stratified cross-validation (i.e., 10% per fold).

- Expects 'wdbc.csv' in the same folder with headers:
  id_number, diagnosis, feature_1 ... feature_30
  where diagnosis is 'M' or 'B'.
- GA genome encodes: number of hidden layers (1..3), nodes per layer,
  activation ('relu' or 'tanh' or 'logistic'), alpha (L2), learning_rate_init.
- Fitness = mean cross-validated accuracy (10 folds).
- Best configuration is reported and saved.

Run:
  python GeneticAlgo_HW3_660610817.py --generations 10 --population 24

Notes:
- This script uses scikit-learn, numpy, pandas. Install if missing:
    pip install numpy pandas scikit-learn
- Comments are in English as requested.
"""

import argparse
import json
import math
import os
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
    """
    Load WDBC dataset from csv_path.
    - Returns X (n_samples, 30), y (n_samples,) with y in {0,1} where 1=malignant (M), 0=benign (B).
    """
    df = pd.read_csv(csv_path)
    expected_first_cols = ["id_number", "diagnosis"]
    if list(df.columns[:2]) != expected_first_cols:
        print(f"[WARN] First two columns are {list(df.columns[:2])}, expected {expected_first_cols}. Proceeding anyway.", file=sys.stderr)
    # Map labels
    y = df["diagnosis"].map({"M": 1, "B": 0}).values
    # Use all feature_1..feature_30 columns
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values.astype(np.float64)
    return X, y

# -----------------------------
# GA Encoding
# -----------------------------

ACTIVATIONS = ["relu", "tanh", "logistic"]

@dataclass
class Genome:
    # Architecture
    n_layers: int              # 1..3
    n1: int                    # 4..64
    n2: int                    # 4..64 (used if n_layers>=2 else ignored)
    n3: int                    # 4..64 (used if n_layers==3 else ignored)
    activation: str            # 'relu' | 'tanh' | 'logistic'
    # Optimization hyperparams
    alpha: float               # L2 regularization (1e-6..1e-1)
    lr_init: float             # learning_rate_init (1e-4..1e-1)
    # Random state (for reproducibility across folds)
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
        n1=random.randint(4, 64),
        n2=random.randint(4, 64),
        n3=random.randint(4, 64),
        activation=random.choice(ACTIVATIONS),
        alpha=10 ** random.uniform(-6, -1),       # log-uniform
        lr_init=10 ** random.uniform(-4, -1),     # log-uniform
        seed=random.randint(0, 10_000)
    )

def mutate(g: Genome, pm: float = 0.25) -> Genome:
    """ Simple mutation operator. Each field mutates with probability pm. """
    g = Genome(**asdict(g))  # copy
    if random.random() < pm:
        g.n_layers = random.randint(1, 3)
    if random.random() < pm:
        g.n1 = int(np.clip(int(round(np.random.normal(g.n1, 8))), 4, 128))
    if random.random() < pm:
        g.n2 = int(np.clip(int(round(np.random.normal(g.n2, 8))), 4, 128))
    if random.random() < pm:
        g.n3 = int(np.clip(int(round(np.random.normal(g.n3, 8))), 4, 128))
    if random.random() < pm:
        g.activation = random.choice(ACTIVATIONS)
    if random.random() < pm:
        g.alpha = float(np.clip(g.alpha * (10 ** np.random.uniform(-0.5, 0.5)), 1e-7, 1e-0))
    if random.random() < pm:
        g.lr_init = float(np.clip(g.lr_init * (10 ** np.random.uniform(-0.5, 0.5)), 1e-5, 1e-0))
    if random.random() < pm:
        g.seed = random.randint(0, 10_000)
    return g

def crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    """ One-point crossover across the tuple of fields. """
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
    """
    Build an MLP with this genome and return mean accuracy over 10-fold stratified CV.
    Uses a StandardScaler inside the pipeline to avoid data leakage.
    Uses early_stopping to reduce overfitting; max_iter is moderate to keep runtime reasonable.
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)  # 10% per fold
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
        acc = accuracy_score(y_te, y_pr)
        scores.append(acc)

    mean_acc = float(np.mean(scores))
    if verbose:
        print(f"[EVAL] {g.hidden_layer_sizes()} act={g.activation} alpha={g.alpha:.2e} lr={g.lr_init:.2e} -> {mean_acc:.4f}")
    return mean_acc

# -----------------------------
# GA Loop
# -----------------------------

def tournament_select(pop: List[Tuple[Genome, float]], k: int = 3) -> Genome:
    """ Tournament selection: pick k random individuals and return the best. """
    contenders = random.sample(pop, k)
    winner = max(contenders, key=lambda t: t[1])
    return winner[0]

def run_ga(
    X: np.ndarray,
    y: np.ndarray,
    generations: int = 10,
    population_size: int = 24,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.25,
    elitism: int = 2,
    seed: int = 42,
    verbose: bool = True,
):
    set_global_seed(seed)
    # Initialize population
    population = [random_genome() for _ in range(population_size)]
    fitness_cache = {}

    def fitness(g: Genome) -> float:
        key = json.dumps(asdict(g), sort_keys=True)
        if key not in fitness_cache:
            fitness_cache[key] = evaluate_genome(g, X, y, verbose=False)
        return fitness_cache[key]

    history = []
    best_tuple = None

    for gen in range(1, generations + 1):
        # Evaluate population
        scored = [(g, fitness(g)) for g in population]
        scored.sort(key=lambda t: t[1], reverse=True)

        best_g, best_f = scored[0]
        avg_f = float(np.mean([f for _, f in scored]))
        history.append({"generation": gen, "best_fitness": best_f, "avg_fitness": avg_f, "best_genome": asdict(best_g)})
        if verbose:
            hls = best_g.hidden_layer_sizes()
            print(f"[GEN {gen:02d}] best={best_f:.4f} avg={avg_f:.4f} "
                  f"hls={hls} act={best_g.activation} alpha={best_g.alpha:.2e} lr={best_g.lr_init:.2e} seed={best_g.seed}")

        if best_tuple is None or best_f > best_tuple[1]:
            best_tuple = (best_g, best_f)

        # Elitism: carry top 'elitism' genomes
        new_pop = [g for g, _ in scored[:elitism]]

        # Generate rest via crossover + mutation using tournament selection
        while len(new_pop) < population_size:
            parent1 = tournament_select(scored, k=3)
            parent2 = tournament_select(scored, k=3)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            child1 = mutate(child1, pm=mutation_rate)
            child2 = mutate(child2, pm=mutation_rate)
            new_pop.extend([child1, child2])

        population = new_pop[:population_size]

    return best_tuple, history

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="GA to search MLP architectures with 10-fold CV on WDBC.")
    parser.add_argument("--csv", type=str, default="wdbc.csv", help="Path to wdbc.csv (same folder by default).")
    parser.add_argument("--generations", type=int, default=10, help="Number of GA generations.")
    parser.add_argument("--population", type=int, default=24, help="Population size.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--no-verbose", action="store_true", help="Reduce console output.")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"[ERROR] Cannot find '{csv_path}'. Place it in the same folder or pass --csv PATH.", file=sys.stderr)
        sys.exit(1)

    set_global_seed(args.seed)
    X, y = load_wdbc(csv_path)

    # Run GA
    best, history = run_ga(
        X, y,
        generations=args.generations,
        population_size=args.population,
        crossover_rate=0.8,
        mutation_rate=0.25,
        elitism=2,
        seed=args.seed,
        verbose=(not args.no_verbose),
    )

    best_g, best_f = best
    print("\n=== BEST CONFIGURATION (10-fold CV) ===")
    print(f"hidden_layer_sizes: {best_g.hidden_layer_sizes()}")
    print(f"activation        : {best_g.activation}")
    print(f"alpha (L2)        : {best_g.alpha:.3e}")
    print(f"learning_rate_init: {best_g.lr_init:.3e}")
    print(f"seed              : {best_g.seed}")
    print(f"mean_accuracy     : {best_f:.4f}")

    # Save artifacts
    out_dir = "ga_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_config.json"), "w") as f:
        json.dump({"best_fitness": best_f, "genome": asdict(best_g)}, f, indent=2)

    hist_path = os.path.join(out_dir, "history.jsonl")
    with open(hist_path, "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    # Optional: train best config on full dataset and save the scaler+model for reuse
    # (Not used for scoring; this is just a convenience artifact.)
    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=best_g.hidden_layer_sizes(),
            activation=best_g.activation,
            solver="adam",
            alpha=best_g.alpha,
            learning_rate_init=best_g.lr_init,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=best_g.seed,
        )
    )
    pipeline.fit(X, y)
    # Persist via joblib
    try:
        import joblib
        joblib.dump(pipeline, os.path.join(out_dir, "final_model.joblib"))
        print(f"[INFO] Saved trained pipeline to {os.path.join(out_dir, 'final_model.joblib')}")
    except Exception as e:
        print(f"[WARN] joblib not available or failed to save model: {e}")

    # Also save a short CSV summary for quick reading
    pd.DataFrame([
        {
            "hidden_layer_sizes": best_g.hidden_layer_sizes(),
            "activation": best_g.activation,
            "alpha": best_g.alpha,
            "learning_rate_init": best_g.lr_init,
            "seed": best_g.seed,
            "cv_mean_accuracy": best_f
        }
    ]).to_csv(os.path.join(out_dir, "best_summary.csv"), index=False)

    print(f"[INFO] GA history written to {hist_path}")
    print("[DONE]")

if __name__ == "__main__":
    main()
