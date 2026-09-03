# Multi-User Dueling Bandits

This repository contains the experimental code for [*Multi-User Dueling Bandits: A Fair Approach using Nash Social Welfare*](https://arxiv.org/abs/2605.01961).

The paper studies dueling bandits with multiple users whose preferences may disagree.  Rather than optimizing average preference alone, the proposed methods optimize **Nash social welfare**, which balances utility across users.  This code compares:

- Fair Explore-Then-Commit (Fair-ETC)
- Fair epsilon-greedy
- Utilitarian versions of both methods
- A uniform-over-users/winners baseline

## Setup

Use Python 3.9 or newer, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy scikit-learn matplotlib
```

## Run an experiment

Run the default synthetic-preference experiment:

```bash
python main.py
```

For a smaller custom experiment, pass the number of users (`U`), arms (`K`), horizon (`T`), and number of independent runs:

```bash
python main.py --U 5 --K 5 --T 50000 --run_counts 10 --minimum_gap 0.1
```

Results are saved as NumPy arrays and a `params.txt` file under `results/`.  By default, the output directory is named from the experiment parameters, for example `results/5_5_50000_0.1/`.

Useful options:

```bash
# Generate correlated/clustered synthetic users
python main.py --clustered_users --fraction_clustered 0.5

# Save somewhere else
python main.py --base_directory ./my_results
```

Use `python main.py --help` to see all simulation and algorithm hyperparameters.

## Plot results

Create plots for one experiment folder:

```bash
python plot_results.py results/5_5_50000_0.1
```

This writes regret, minimum-welfare, Nash-social-welfare, utilitarian-welfare, and Gini-coefficient figures (`.pdf`, `.png`, and `.svg`) plus a metrics table.  The plotting script uses LaTeX for labels, so a working LaTeX installation is required.

To plot every completed experiment in a results directory:

```bash
python plot_all.py --base_directory ./results --out_directory ./Figures
```

## Sushi experiment (optional)

The Sushi experiment expects the Sushi ranking file at:

```
data/sushi3-2016/sushi3a.5000.10.order
```

After placing the dataset there, run:

```bash
python main.py --sushi --sushi_clusters 5
```

## Main files

- `main.py` runs the simulations and saves results.
- `algorithms.py` implements the fair and baseline algorithms.
- `utils.py` generates synthetic preferences, identifies/uses Condorcet winners, and loads Sushi data.
- `plot_results.py` plots one experiment.
- `plot_all.py` plots all experiment folders.
