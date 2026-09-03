import numpy as np
from numpy.random import Generator
import argparse
from algorithms import  explore_then_commit, eps_greedy, uniform_over_winners, rucb
from utils import generate_user_prefs, generate_clustered_user_prefs, load_sushi_data
import os
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dueling Bandits Simulation")
    
    # Simulation Parameters
    parser.add_argument("--U", type=int, default=8, help="Number of users")
    parser.add_argument("--K", type=int, default=10, help="Number of arms")
    parser.add_argument("--T", type=int, default=10000, help="Total time horizon")
    parser.add_argument("--run_counts", type=int, default=30, help="Number of independent seeds/trials to run")
    parser.add_argument("--minimum_gap", type=float, default=0.25, help="Minimum gap for preference probabilities")
    
    # Algorithm Hyperparameters
    parser.add_argument("--L_scaling", type=float, default=0.25, help="Scaling factor for ETC exploration length L")
    parser.add_argument("--eps_scaling", type=float, default=0.1, help="Scaling factor for Epsilon-Greedy decay")
    parser.add_argument("--delta_scaling", type=float, default=0.0025, help="Scaling factor for DKWT confidence delta")
    parser.add_argument("--iters_fw", type=int, default=300, help="Frank-Wolfe iterations")
    parser.add_argument("--update_freq", type=int, default=50, help="Frequency of policy updates for Eps-Greedy")
    parser.add_argument("--alpha", type=float, default=0.51, help="Exploration parameter for RUCB")
    parser.add_argument(
        "--clustered_users", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Generate clustered user preferences"
    )
    parser.add_argument(
        "--sushi", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Use the sushi dataset"
    )
    parser.add_argument("--sushi_clusters", type=int, default=5, help="Number of clusters for the sushi dataset")
    parser.add_argument("--fraction_clustered", type=float, default=0.5, help="Fraction of users in the cluster")
    
    parser.add_argument(
        "--force_global_winner", 
        action=argparse.BooleanOptionalAction, 
        default=False,
        help="Ensure a Condorcet winner exists in the average preference matrix"
    )

    parser.add_argument("--base_directory", type=str, default="./results/", help="Base directory for saving results")

    args = parser.parse_args()

    # Assign to variables
    U = args.U
    K = args.K
    minimum_gap = args.minimum_gap
    T = args.T
    force_global_winner = args.force_global_winner
    base_directory = args.base_directory
    run_counts = args.run_counts
    L_scaling = args.L_scaling
    eps_scaling = args.eps_scaling
    delta_scaling = args.delta_scaling
    iters_fw = args.iters_fw
    update_freq = args.update_freq
    rng_seeder = np.random.default_rng(seed=42)
    clustered_users = args.clustered_users
    fraction_clustered = args.fraction_clustered
    sushi = args.sushi
    sushi_clusters = args.sushi_clusters

    alpha = args.alpha
    seeds = rng_seeder.integers(0, 100000, size=run_counts)
    # Storage for ETC
    if sushi:
        U = sushi_clusters
        K = 10
        print("Loading sushi data once...", flush=True)
        pref_matrices, condorcet_winners, sushi_rankings = load_sushi_data(n_clusters=sushi_clusters)
        sushi_scores = K - sushi_rankings
        print("Sushi data loaded.", flush=True)
    else:
        sushi_scores = None
    if sushi:
        U = sushi_clusters
        K = 10
        etc_sushi_utils = np.zeros((run_counts, 5000))
        eps_sushi_utils = np.zeros((run_counts, 5000))
        uniform_sushi_utils = np.zeros((run_counts, 5000))
        etc_util_sushi_utils = np.zeros((run_counts, 5000))
        eps_util_sushi_utils = np.zeros((run_counts, 5000))
    etc_regret = np.zeros((run_counts, T))
    etc_id_steps = np.zeros(run_counts)
    etc_ex_steps = np.zeros(run_counts)
    etc_scores_1 = np.zeros((run_counts, T, U))
    etc_scores_2 = np.zeros((run_counts, T, U))

    # Storage for Eps-Greedy
    eps_regret = np.zeros((run_counts, T))
    eps_id_steps = np.zeros(run_counts)
    eps_ex_steps = np.zeros(run_counts)
    eps_scores_1 = np.zeros((run_counts, T, U))
    eps_scores_2 = np.zeros((run_counts, T, U))

    # Storage for Utilitarian
    etc_util_regret = np.zeros((run_counts, T))
    etc_util_scores_1 = np.zeros((run_counts, T, U))
    etc_util_scores_2 = np.zeros((run_counts, T, U))
    eps_util_regret = np.zeros((run_counts, T))
    eps_util_scores_1 = np.zeros((run_counts, T, U))
    eps_util_scores_2 = np.zeros((run_counts, T, U))

    # Storage for Uniform over Winners
    uniform_regret = np.zeros((run_counts, T))
    uniform_id_steps = np.zeros(run_counts)
    uniform_scores_1 = np.zeros((run_counts, T, U))
    uniform_scores_2 = np.zeros((run_counts, T, U))

    # Storage for RUCB
    rucb_regret = np.zeros((run_counts, T)) 
    rucb_scores_1 = np.zeros((run_counts, T, U))
    rucb_scores_2 = np.zeros((run_counts, T, U))
    for i in range(run_counts):
        s = seeds[i]
        print(f"Running seed {s} ({i+1}/{run_counts})...", flush=True)
        if not sushi:
            rng = np.random.default_rng(seed=s)
            if clustered_users:
                pref_matrices, condorcet_winners = generate_clustered_user_prefs(
                    U=U, 
                    K=K, 
                    rng=rng, 
                    minimum_gap=minimum_gap, 
                    fraction_clustered=fraction_clustered, 
                    force_global_winner=force_global_winner
                )
            else:
                pref_matrices, condorcet_winners = generate_user_prefs(U=U, K=K, rng=rng, minimum_gap=minimum_gap, force_global_winner=force_global_winner)
        
        rng = np.random.default_rng(seed=s)
        # Run ETC
        print("Running Fair-ETC...", flush=True)
        etc_results = explore_then_commit(
            P_users=pref_matrices,
            condorcet_winners=condorcet_winners,
            T=T,
            rng=rng,
            iters_fw=iters_fw,
            L_scaling=L_scaling,
            delta_scaling=delta_scaling,
            sushi_scores=sushi_scores
        )
        if sushi:
            etc_regret[i], etc_id_steps[i], etc_ex_steps[i], etc_scores_1[i], etc_scores_2[i], etc_sushi_utils[i] = etc_results
        else:
            etc_regret[i], etc_id_steps[i], etc_ex_steps[i], etc_scores_1[i], etc_scores_2[i] = etc_results

        
        rng = np.random.default_rng(seed=s)
        # Run ETC Utilitarian
        print("Running Utilitarian-ETC...", flush=True)
        etc_util_results = explore_then_commit(
            P_users=pref_matrices,
            condorcet_winners=condorcet_winners,
            T=T,
            rng=rng,
            iters_fw=iters_fw,
            L_scaling=L_scaling,
            delta_scaling=delta_scaling,
            utilitarian=True,
            sushi_scores=sushi_scores
        )
        if sushi:
            etc_util_regret[i], _, _, etc_util_scores_1[i], etc_util_scores_2[i], etc_util_sushi_utils[i] = etc_util_results
        else:
            etc_util_regret[i], _, _, etc_util_scores_1[i], etc_util_scores_2[i] = etc_util_results
        

        rng = np.random.default_rng(seed=s)
        print("Running Utilitarian-Epsilon-Greedy...", flush=True)
        eps_util_results = eps_greedy(
            P_users=pref_matrices,
            condorcet_winners=condorcet_winners,
            T=T,
            rng=rng,
            iters_fw=iters_fw,
            eps_scaling=eps_scaling,
            delta_scaling=delta_scaling,
            utilitarian=True,
            update_freq=update_freq,
            sushi_scores=sushi_scores
        )
        if sushi:
            eps_util_regret[i], _, eps_util_scores_1[i], eps_util_scores_2[i], eps_util_sushi_utils[i] = eps_util_results
        else:
            eps_util_regret[i], _, eps_util_scores_1[i], eps_util_scores_2[i] = eps_util_results

        rng = np.random.default_rng(seed=s)
        print("Running Fair-Epsilon-Greedy...", flush=True)
        eps_results = eps_greedy(
            P_users=pref_matrices,
            condorcet_winners=condorcet_winners,
            T=T,
            rng=rng,
            iters_fw=iters_fw,
            eps_scaling=eps_scaling,
            delta_scaling=delta_scaling,
            update_freq=update_freq,
            sushi_scores=sushi_scores
        )
        if sushi:
            eps_regret[i], eps_id_steps[i], eps_scores_1[i], eps_scores_2[i], eps_sushi_utils[i] = eps_results
        else:
            eps_regret[i], eps_id_steps[i], eps_scores_1[i], eps_scores_2[i] = eps_results

        rng = np.random.default_rng(seed=s)
        print("Running Uniform-Over-Winners...", flush=True)
        uniform_results = uniform_over_winners(
            P_users=pref_matrices,
            condorcet_winners=condorcet_winners,
            T=T,
            rng=rng,
            iters_fw=iters_fw,
            delta_scaling=delta_scaling,
            sushi_scores=sushi_scores
        )
        if sushi:
            uniform_regret[i], uniform_id_steps[i], uniform_scores_1[i], uniform_scores_2[i], uniform_sushi_utils[i] = uniform_results
        else:
            uniform_regret[i], uniform_id_steps[i], uniform_scores_1[i], uniform_scores_2[i] = uniform_results

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_directory}/{U}_{K}_{T}_{minimum_gap}_{fraction_clustered}" if clustered_users else f"{base_directory}/{U}_{K}_{T}_{minimum_gap}"
    output_dir += f"_sushi_{sushi_clusters}" if sushi else ""

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}...")

    np.save(os.path.join(output_dir, "etc_regret.npy"), etc_regret)
    np.save(os.path.join(output_dir, "etc_scores_1.npy"), etc_scores_1)
    np.save(os.path.join(output_dir, "etc_scores_2.npy"), etc_scores_2)

    np.save(os.path.join(output_dir, "eps_regret.npy"), eps_regret)
    np.save(os.path.join(output_dir, "eps_scores_1.npy"), eps_scores_1)
    np.save(os.path.join(output_dir, "eps_scores_2.npy"), eps_scores_2)

    np.save(os.path.join(output_dir, "etc_util_regret.npy"), etc_util_regret)
    np.save(os.path.join(output_dir, "etc_util_scores_1.npy"), etc_util_scores_1)
    np.save(os.path.join(output_dir, "etc_util_scores_2.npy"), etc_util_scores_2)

    np.save(os.path.join(output_dir, "eps_util_regret.npy"), eps_util_regret)
    np.save(os.path.join(output_dir, "eps_util_scores_1.npy"), eps_util_scores_1)
    np.save(os.path.join(output_dir, "eps_util_scores_2.npy"), eps_util_scores_2)

    np.save(os.path.join(output_dir, "uniform_regret.npy"), uniform_regret)
    np.save(os.path.join(output_dir, "uniform_scores_1.npy"), uniform_scores_1)
    np.save(os.path.join(output_dir, "uniform_scores_2.npy"), uniform_scores_2)

    if sushi:
        np.save(os.path.join(output_dir, "etc_sushi_utils.npy"), etc_sushi_utils)
        np.save(os.path.join(output_dir, "eps_sushi_utils.npy"), eps_sushi_utils)
        np.save(os.path.join(output_dir, "uniform_sushi_utils.npy"), uniform_sushi_utils)
        np.save(os.path.join(output_dir, "etc_util_sushi_utils.npy"), etc_util_sushi_utils)
        np.save(os.path.join(output_dir, "eps_util_sushi_utils.npy"), eps_util_sushi_utils)

    # Save parameters for reference
    with open(os.path.join(output_dir, "params.txt"), "w") as f:
        f.write(f"U={U}\nK={K}\nT={T}\nrun_counts={run_counts}\n")
        f.write(f"minimum_gap={minimum_gap}\n")
        f.write(f"L_scaling={L_scaling}\n")
        f.write(f"eps_scaling={eps_scaling}\n")
        f.write(f"delta_scaling={delta_scaling}\n")
        f.write(f"iters_fw={iters_fw}\n")
        f.write(f"update_freq={update_freq}\n")
        f.write(f"alpha={alpha}\n")
        f.write(f"clustered_users={clustered_users}\n")
        f.write(f"fraction_clustered={fraction_clustered}\n")
        f.write(f"force_global_winner={force_global_winner}\n")
