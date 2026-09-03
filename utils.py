import numpy as np
from numpy.random import Generator
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
import os

def ensure_global_condorcet(
    pref_matrices: np.ndarray, 
    condorcet_winners: np.ndarray,
    minimum_gap: float = 0.05
) -> Tuple[np.ndarray, int]:
    """
    Post-process preference matrices to ensure that the Average Preference Matrix
    has a valid Condorcet Winner.
    
    """
    U, K, _ = pref_matrices.shape
    P_avg = np.mean(pref_matrices, axis=0)

    copeland_scores = np.sum(P_avg > 0.5, axis=1)
    global_winner = np.argmax(copeland_scores)

    for j in range(K):
        if j == global_winner:
            continue
            
        current_avg = P_avg[global_winner, j]
        target_avg = 0.5 + minimum_gap
        count = 0
        while current_avg < target_avg:
            winners_count = np.sum(np.logical_and(condorcet_winners == j, np.abs(pref_matrices[:, j, global_winner] - 0.5 - minimum_gap) <= 1e-8))
            cap_count = np.sum(pref_matrices[:, global_winner, j] >= 1.0 - 1e-8)
            min_gap_count = np.sum(np.abs(np.abs(pref_matrices[:, global_winner, j] - 0.5) - minimum_gap) <= 1e-8)

            deficit = target_avg - current_avg
            deficit = deficit * U / (U -  winners_count + cap_count + min_gap_count)
            assert (deficit >= 0)
            for u in range(U):
                new_val = pref_matrices[u, global_winner, j] + deficit
                new_val = min(new_val, 1.0)
                if j == condorcet_winners[u] and new_val > 0.5 - 1e-8:
                    new_val = 0.5 - minimum_gap
                if new_val >= 0.5 and new_val - 0.5 < minimum_gap:
                    new_val = 0.5 + minimum_gap
                elif new_val < 0.5 and 0.5 - new_val < minimum_gap:
                    if j == condorcet_winners[u]:
                        new_val = 0.5 - minimum_gap
                    else:
                        new_val = 0.5 + minimum_gap
                
                pref_matrices[u, global_winner, j] = new_val
                pref_matrices[u, j, global_winner] = 1.0 - new_val
            current_avg = np.mean(pref_matrices[:, global_winner, j])
            count += 1
            if count >= 1000:
                raise ValueError("Could not enforce global condorcet winner.")
    P_avg_new = np.mean(pref_matrices, axis=0)
    assert(np.all(P_avg_new[global_winner, np.arange(K) != global_winner] > 0.5))
    assert(np.all(P_avg_new[global_winner, np.arange(K) != global_winner] >= 0.5+minimum_gap-1e-8))
    
    return pref_matrices, global_winner

def generate_user_prefs(U: int, K: int, rng: Generator, minimum_gap=0.1, force_global_winner=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate U user-specific preference matrices over K arms.
    Sample a condorcet winner for each user. The condorcet winner is always preferred to any other arm. Generate the rest of the probabilities such that P(i,j) = 1-P(j,i)
    """
    condorcet_winners = np.zeros(U, dtype=int)
    pref_matrices = np.zeros((U,K, K))
    for u in range(U):
        condorcet_winner = rng.integers(0, K)
        condorcet_winners[u] = condorcet_winner
        for i in range(K):
            for j in range(i,K):
                if i == j:
                    pref_matrices[u, i, j] = 0.5
                elif i == condorcet_winner:
                    pref_matrices[u, i, j] = 0.5 + minimum_gap + (0.5-minimum_gap) * rng.random()
                elif j == condorcet_winner:
                    pref_matrices[u, i, j] = (0.5-minimum_gap) * rng.random()
                else:
                    pref_matrices[u, i, j] = rng.random()
                    if pref_matrices[u, i, j] > 0.5 and pref_matrices[u, i, j] - 0.5 < minimum_gap:
                        pref_matrices[u, i, j] = 0.5 + minimum_gap
                    elif pref_matrices[u, i, j] <= 0.5 and 0.5 - pref_matrices[u, i, j] < minimum_gap:
                        pref_matrices[u, i, j] = 0.5 - minimum_gap
                pref_matrices[u, j, i] = 1 - pref_matrices[u, i, j]

    if force_global_winner:
        pref_matrices, _ = ensure_global_condorcet(pref_matrices, condorcet_winners, minimum_gap=minimum_gap)
    for u in range(U):
        assert(np.allclose(pref_matrices[u,:,:] + pref_matrices[u,:,:].T, np.ones((K,K))))
        assert(np.allclose(np.diag(pref_matrices[u,:,:]), 0.5 * np.ones(K)))
        condorcet_winner = condorcet_winners[u]
        assert(np.all(pref_matrices[u, condorcet_winner, np.array([i for i in range(condorcet_winner)]+[i for i in range(condorcet_winner+1,K)])] > 0.5))
        assert(np.all(np.abs(pref_matrices[u] + 0.5 * np.eye(K) -0.5) >= minimum_gap-1e-8))

    return pref_matrices, condorcet_winners

def generate_clustered_user_prefs(
    U: int, 
    K: int, 
    rng: Generator, 
    minimum_gap: float = 0.1, 
    fraction_clustered: float = 0.5,
    force_global_winner: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate U user-specific preference matrices with a clustered structure.
    A fraction of users share the same Condorcet winner with close probabilities.
    The rest have winners sampled uniformly from remaining arms.
    """
    condorcet_winners = np.zeros(U, dtype=int)
    pref_matrices = np.zeros((U, K, K))
    
    n_clustered = int(np.ceil(U * fraction_clustered))
    common_winner = rng.integers(0, K)
    remaining_arms = [i for i in range(K) if i != common_winner]
    if not remaining_arms: remaining_arms = [common_winner]

    for u in range(U):
        if u <= n_clustered:
            condorcet_winner = common_winner
        else:
            condorcet_winner = rng.choice(remaining_arms)
        
        condorcet_winners[u] = condorcet_winner
        
        for i in range(K):
            for j in range(i, K):
                if i == j:
                    pref_matrices[u, i, j] = 0.5
                elif i == condorcet_winner:
                        if u <= n_clustered and u > 0:
                            pref_matrices[u, i, j] = np.clip(pref_matrices[0, i, j] + rng.normal(0, 0.1), 0.5+minimum_gap, 1)
                        elif u > n_clustered and j == common_winner:
                            pref_matrices[u, i, j] = 1 - (0.1) * rng.random()
                        else:
                            pref_matrices[u, i, j] = 0.5 + minimum_gap + (0.5 - minimum_gap) * rng.random()
                elif j == condorcet_winner:
                        if u <= n_clustered and u > 0:
                            pref_matrices[u, i, j] = np.clip(pref_matrices[0, i, j] + rng.normal(0, 0.1), 0, 0.5-minimum_gap)
                        elif u > n_clustered and i == common_winner:
                            pref_matrices[u, i, j] = (0.1) * rng.random()
                        else:
                            pref_matrices[u, i, j] = (0.5 - minimum_gap) * rng.random()
                else:
                    pref_matrices[u, i, j] = rng.random()
                    if pref_matrices[u, i, j] > 0.5 and pref_matrices[u, i, j] - 0.5 < minimum_gap:
                        pref_matrices[u, i, j] = 0.5 + minimum_gap
                    elif pref_matrices[u, i, j] <= 0.5 and 0.5 - pref_matrices[u, i, j] < minimum_gap:
                        pref_matrices[u, i, j] = 0.5 - minimum_gap
                
                pref_matrices[u, j, i] = 1.0 - pref_matrices[u, i, j]

    if force_global_winner:
        pref_matrices, _ = ensure_global_condorcet(pref_matrices, condorcet_winners, minimum_gap=minimum_gap)

    for u in range(U):
        assert(np.allclose(pref_matrices[u,:,:] + pref_matrices[u,:,:].T, np.ones((K,K))))
        assert(np.allclose(np.diag(pref_matrices[u,:,:]), 0.5 * np.ones(K)))
        condorcet_winner = condorcet_winners[u]
        others = np.array([i for i in range(condorcet_winner)]+[i for i in range(condorcet_winner+1,K)])
        if len(others) > 0:
            assert(np.all(pref_matrices[u, condorcet_winner, others] > 0.5))
        assert(np.all(np.abs(pref_matrices[u] + 0.5 * np.eye(K) - 0.5) >= minimum_gap - 1e-8))

    return pref_matrices, condorcet_winners

def score_from_preferences(P: np.ndarray, condorcet_winners: np.ndarray) -> np.ndarray:
    """
    Given a preference matrix P, compute the score for each arm.
    Score for arm i is the sum of 2 * P(i,condorcet_winner).
    """
    scores = np.zeros((P.shape[0], P.shape[1]))
    for u in range(P.shape[0]):
        condorcet_winner = condorcet_winners[u]
        for i in range(P.shape[1]):
            scores[u, i] = 2 * P[u, i, condorcet_winner]
    return scores

def objective_and_grad(x: np.ndarray, scores: np.ndarray, utilitarian: bool = False) -> Tuple[float, np.ndarray]:
    """
    Compute f(x) = sum_u log(s_u^T x)  (concave), and gradient of g(x) = -f(x).
    We return f(x) and grad_g(x), where grad_g(x) = -∑_u b_u / (b_u^T x).
    If utilitarian is True, compute f(x) = sum_u s_u^T x and grad_g(x) = -sum_u s_u.
    """
    if utilitarian:
        vals = scores.dot(x)
        f = np.sum(vals)
        grad_g = -np.sum(scores, axis=0)
        return f, grad_g

    # Numeric stability: ensure positivity
    eps = 1e-12
    vals = scores.dot(x)
    vals = np.clip(vals, eps, None)
    f = np.sum(np.log(vals))
    grad_g = np.zeros_like(x)
    for u, v in enumerate(vals):
        grad_g -= scores[u] / v
    return f, grad_g


def frank_wolfe_simplex(scores: np.ndarray, K: int, iters: int = 500, tol: float = 2e-4, utilitarian: bool = False) -> Tuple[np.ndarray, List[float]]:
    """
    Frank-Wolfe to minimize g(x) = -∑_u log(b_u^T x) over the probability simplex Δ_K.
    (Equivalently, maximize f(x) = ∑_u log(b_u^T x).)

    Returns:
        x: optimized stochastic policy in the simplex
        f_hist: history of the concave objective f(x) per iteration
    """
    x = np.ones(K) / K  # start uniform
    f_hist = []
    f_val, grad_g = objective_and_grad(x, scores, utilitarian=utilitarian)
    for t in range(iters):
        f_hist.append(f_val)

        # Linear minimization oracle over simplex: pick vertex minimizing <s, grad_g>
        k = np.abs(grad_g - np.min(grad_g)) <= 1e-12
        s = np.zeros(K)
        s[k] = 1.0/np.sum(k)
        # print("Grad", grad_g)
        # Step size (classic diminishing step sizes)
        gamma = 2.0 / (t + 2.0)

        x_new = (1 - gamma) * x + gamma * s
        # print("policy", x_new)

        # Convergence check (duality gap proxy: <x - s, grad_g> )
        # gap = float((x - s) @ grad_g)
        x = x_new
        f_val_new, grad_g = objective_and_grad(x, scores, utilitarian=utilitarian) 
        if abs(f_val_new - f_val) < tol:
            break
        f_val = f_val_new
    return x, f_hist

def sample_duel(P: np.ndarray, i: int, j: int, rng: Generator) -> np.ndarray:
    """
    Sample the outcome of a duel between i and j from preference matrix P.
    Returns 1 if i wins, 0 if j wins.
    """
    outcome = rng.random((len(P),)) < P[:, i, j]
    return outcome


def load_sushi_data(n_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the Sushi dataset, clusters users based on rankings, and creates a preference matrix for each cluster.

    The preference P(i, j) for a cluster is the fraction of users in that cluster who prefer item i over j.
    The number of "users" returned for the bandit algorithms is `n_clusters`.
    Clustering is performed using Agglomerative Clustering with Spearman distance.

    Args:
        n_clusters: The number of user clusters to create. This will be the number of "users" (U) for the algorithms.


    Returns:
        pref_matrices: A (n_clusters, K, K) numpy array of preference matrices.
        condorcet_winners: A (n_clusters,) numpy array of Condorcet winner indices for each cluster.
        rank_vectors_k: A (num_valid_users, K) numpy array of the original user rankings.
    """

    base_path = "data/sushi3-2016"
    filepath = os.path.join(base_path, "sushi3a.5000.10.order")

    K = 10
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sushi dataset file '{filepath}' not found. Please download it to 'data/sushi3-2016/'.")


    # Read rankings
    all_rankings = []
    with open(filepath, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 3:
                continue
            all_rankings.append(parts[2:])

    num_users = len(all_rankings)
    rank_vectors = np.full((num_users, 10), -1, dtype=int)
    for i, ranking in enumerate(all_rankings):
        for rank, item_id in enumerate(ranking):
            rank_vectors[i, item_id] = rank

    valid_users_mask = np.all(rank_vectors[:, :K] != -1, axis=1)
    rank_vectors_k = rank_vectors[valid_users_mask, :K]

    rho_matrix, _ = spearmanr(rank_vectors_k, axis=1, nan_policy='raise')
    dist_matrix = (1.0 - rho_matrix) / 2.0
    np.fill_diagonal(dist_matrix, 0)
    # Use Agglomerative Clustering with the precomputed distance matrix
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    cluster_labels = agg_clustering.fit_predict(dist_matrix)

    pref_matrices = np.zeros((n_clusters, K, K))
    condorcet_winners = np.full(n_clusters, -1, dtype=int)
    for c in range(n_clusters):
        user_indices_in_cluster = np.where(cluster_labels == c)[0]
        if len(user_indices_in_cluster) == 0:
            raise ValueError(f"Cluster {c} has no users.")

        num_users_in_cluster = len(user_indices_in_cluster)
        cluster_rank_vectors = rank_vectors_k[user_indices_in_cluster]

        pref_matrix_c = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1, K):
                i_is_preferred = np.sum(cluster_rank_vectors[:, i] < cluster_rank_vectors[:, j])
                i_score = np.sum(K - cluster_rank_vectors[:, i])/num_users_in_cluster
                j_score = np.sum(K - cluster_rank_vectors[:, j])/num_users_in_cluster
                #pref_matrix_c[i, j] = 0.5+0.5*(i_score - j_score)/K
                pref_matrix_c[i, j] = i_is_preferred/num_users_in_cluster
                pref_matrix_c[j, i] = 1 - pref_matrix_c[i, j]
        gap = 0.1
        for i in range(K):
            for j in range(i + 1, K):
                if pref_matrix_c[i,j] == 0.5:
                    pref_matrix_c[i,j] = 0.5 + gap
                    pref_matrix_c[j,i] = 0.5 - gap
        pref_matrix_c[np.logical_and(pref_matrix_c < 0.5, pref_matrix_c > 0.5-gap)] = 0.5-gap
        pref_matrix_c[np.logical_and(pref_matrix_c > 0.5, pref_matrix_c < 0.5+gap)] = 0.5+gap
        #assert(not np.any(np.abs(pref_matrix_c - 0.5) < gap))
        np.fill_diagonal(pref_matrix_c, 0.5)
        assert(np.allclose(pref_matrix_c[:,:] + pref_matrix_c[:,:].T, np.ones((K,K))))
        pref_matrices[c, :, :] = pref_matrix_c

        copeland_scores = np.sum(pref_matrix_c > 0.5, axis=1)
        potential_winners = np.where(copeland_scores == K - 1)[0]

        assert len(potential_winners) > 0, f"Cluster {c} has no Condorcet winner."
        assert len(potential_winners) == 1, f"Cluster {c} has multiple Condorcet winners ({potential_winners}), which is impossible."

        condorcet_winners[c] = potential_winners[0]

    return pref_matrices, condorcet_winners, rank_vectors_k