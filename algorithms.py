import numpy as np
import math
from numpy.random import Generator
from typing import List, Tuple
from utils import sample_duel, objective_and_grad, frank_wolfe_simplex, score_from_preferences
def find_positive_real_roots(coeffs: List[float]) -> List[float]:
    """
    Finds the single positive real root of a polynomial equation given its coefficients.
    Asserts that exactly one positive real root exists.
    Args:
        coeffs: A list of coefficients [a, b, c, d, ...] for ax^n + bx^(n-1) + ... = 0.
    Returns:
        The positive real root as a float.
    """
    if not coeffs or coeffs[0] == 0:
        raise ValueError("Coefficients list is empty or first coefficient is zero.")
    roots = np.roots(coeffs)
    # Filter for real roots (imaginary part is close to zero)
    real_roots = roots[np.isclose(roots.imag, 0)].real
    # Filter for positive real roots
    positive_real_roots = real_roots[real_roots > 0]
    if len(positive_real_roots) != 1:
         raise ValueError(f"Expected 1 positive real root, but found {len(positive_real_roots)}: {positive_real_roots} from roots {roots}")
    return positive_real_roots[0]
    
def dkw_mode_identification(P: np.ndarray, delta: float,  u: int, i: int, j: int, r: int, F: np.ndarray, rng: Generator) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    DKW-based mode identification algorithm to find the condorcet winner.
    Args:
        P: U x K x K preference matrices for U users and K arms.
        delta: confidence parameter.
        i, j: arms to duel.
        F: reamining arms for users.
        rng: random number generator.
    Returns:
        estimated_winners: estimated duel winners for each user.
        estimated_losers: estimated duel losers for each user.
        total_samples: total number of samples used.
        total_outcomes_i: total outcomes where i won.
        total_outcomes_j: total outcomes where j won.
    """
    U, K, _ = P.shape
    estimated_winners = -1 * np.ones(U, dtype=int)
    estimated_losers = -1 * np.ones(U, dtype=int)

    # Identify users for whom both i and j are still candidates
    active_users_mask = np.logical_and(F[:, i] , F[:, j])

    # If no users are active for this pair, return immediately.
    if not np.any(active_users_mask):
        return estimated_winners, estimated_losers, 0, np.zeros(U), np.zeros(U)
    
    total_samples = 0
    total_outcomes_i = np.zeros(U)
    total_outcomes_j = np.zeros(U)
    assert i != j

    # Loop until all active users have a winner
    #while np.any(np.logical_and((estimated_winners == -1), active_users_mask)):
    while estimated_winners[u] == -1:
        delta_bar = 6 * delta / (math.pi**2 * r**2)
        h = 2 ** (-r-1)
        T_samples = math.ceil(8*math.log(4 / delta_bar) / (h**2))

        round_outcomes_i = np.zeros(U)
        for _ in range(T_samples):
            outcomes = sample_duel(P, i, j, rng)
            round_outcomes_i += outcomes

        total_samples += T_samples
        total_outcomes_i += round_outcomes_i
        total_outcomes_j += T_samples - round_outcomes_i

        p_hat_i = round_outcomes_i / T_samples
        p_hat_j = 1 - p_hat_i

        
        estimated_winners[p_hat_i - p_hat_j > h] = i
        estimated_losers[p_hat_i - p_hat_j > h] = j
        estimated_winners[p_hat_j - p_hat_i > h] = j
        estimated_losers[p_hat_j - p_hat_i > h] = i

        # undecided_active_mask = (estimated_winners == -1) & active_users_mask

        # i_wins_mask = (p_hat_i - p_hat_j > h) & undecided_active_mask
        # estimated_winners[i_wins_mask] = i
        # estimated_losers[i_wins_mask] = j

        # j_wins_mask = (p_hat_j - p_hat_i > h) & undecided_active_mask
        # estimated_winners[j_wins_mask] = j
        # estimated_losers[j_wins_mask] = i

        r += 1
    return estimated_winners, estimated_losers, total_samples, total_outcomes_i, total_outcomes_j, r

def dkwt(P: np.ndarray, s_true: np.ndarray, delta: float, rng: Generator, sushi_scores: np.ndarray = None) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, List, List, List]:
    """
    DKW-based tournament algorithm to find the condorcet winner.
    Args:
        P: U x K x K preference matrices for U users and K arms.
        s_true: true score vectors for each user.
        delta: confidence parameter.
        rng: random number generator.
        sushi_scores: (optional) scores from original sushi data to compute true utility.
    Returns:
        estimated_winners: estimated condorcet winners for each user.
        total_samples: total number of samples used.
        wins: number of wins for each arm pair and user.
        counts: number of duels for each arm pair.
        objectives: list of objective values observed during the duels.
        scores_1: list of score vectors for the first arm in each duel.
        scores_2: list of score vectors for the second arm in each duel.
        sushi_utils: list of utilitarian sushi utilities if sushi_scores is provided.
    """
    U, K, _ = P.shape
    estimated_winners = np.zeros(U, dtype=int)
    delta_bar = delta / (K)
    F = np.ones((U, K), dtype=bool)
    wins = np.zeros((K, K, U,), dtype=float)
    counts = np.zeros((K,K,), dtype=float)
    total_samples = 0
    objectives = list()
    scores_1 = list()
    scores_2 = list()
    sushi_utils = 0
    r = np.ones((K,K),)
    for u in range(U):
        remaining_arms = np.where(F[u,:])[0]
        remaining_arms_count = remaining_arms.shape[0]
        i = rng.integers(0, remaining_arms_count)
        i = remaining_arms[i]
        while remaining_arms_count > 1:
            j = rng.integers(0, remaining_arms_count-1)
            if remaining_arms[j] >= i:
                j += 1
            j = remaining_arms[j]
            round_winners, round_losers, samples, total_outcomes_i, total_outcomes_j, r[i][j] = dkw_mode_identification(P, delta_bar, u, i, j, r[i][j], F, rng)
            r[j][i] = r[i][j]
            x_hat_1 = np.zeros(K)
            x_hat_1[i] = 1.0
            x_hat_2 = np.zeros(K)
            x_hat_2[j] = 1.0
            fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
            fx_val_2, _ = objective_and_grad(x_hat_2, s_true)
            fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
            objectives+= [fx for _ in range(samples)]
            scores_1.extend([s_true[:, i]] * samples)
            scores_2.extend([s_true[:, j]] * samples)            
            if sushi_scores is not None:
                sushi_utils += 0.5 * (sushi_scores[:, i] + sushi_scores[:, j])
            counts[i, j] += samples
            counts[j, i] += samples
            wins[i, j, :] += total_outcomes_i
            wins[j, i, :] += total_outcomes_j
            for u_prime in range(U):
                if round_winners[u_prime] != -1:
                    F[u_prime, round_losers[u_prime]] = False
            i = round_winners[u]
            remaining_arms = np.where(F[u,:])[0]
            remaining_arms_count = remaining_arms.shape[0]
            total_samples += samples
        estimated_winners[u] = remaining_arms[0]
    objectives = np.array(objectives).flatten()
    print("DKWT finished.", flush=True)
    return estimated_winners, total_samples, wins, counts, objectives, scores_1, scores_2, sushi_utils

def explore_then_commit(
    P_users: np.ndarray,
    condorcet_winners: np.ndarray,
    T: int,
    rng: Generator,
    iters_fw: int = 300,
    L_scaling: float = 0.25,
    delta_scaling: float = 0.0025,
    utilitarian: bool = False,
    sushi_scores: np.ndarray = None,
) -> Tuple:
    """
    Explore-then-commit algorithm for learning a stochastic policy over K arms
    to maximize the expected log-welfare across U users, given user-specific
    preference matrices P_users.
    Args:
        P_users: List of U user-specific preference matrices (K x K numpy arrays).
        condorcet_winners: True condorcet winners for each user (numpy array of length U).
        T: Total number of rounds.
        rng: Random number generator.
        iters_fw: Number of Frank-Wolfe iterations for optimization.
        L_scaling: Scaling factor for L (number of exploration duels per arm pair).
        delta_scaling: Scaling factor for delta in DKWT.
        sushi_scores: (optional) scores from original sushi data to compute true utility.
    Returns:
        cumulative_regret: Cumulative regret at each round t (numpy array of length T).
        identification_steps: Number of steps taken for condorcet winner identification.
        exploration_steps: Number of steps taken for exploration after identification.
        scores_1: array of score vectors for the first arm in each duel.
        scores_2: array of score vectors for the second arm in each duel.
        sushi_utils: (if sushi_scores is not None) array of utilitarian sushi utilities.
    """
    U, K, _ = P_users.shape

    s_true = score_from_preferences(P_users, condorcet_winners)
    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true) # f(x*)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)

    delta = (K * np.log(K/2))/(delta_scaling * T)
    assert(delta < 1 and delta > 0)
    estimated_winners, t, wins, counts, objectives, scores_1, scores_2, sushi_utils = dkwt(P_users, s_true, delta=delta, rng=rng, sushi_scores=sushi_scores)
    identification_steps = t
    if(np.any(estimated_winners != condorcet_winners)):
        print("Warning: Estimated winners do not match true condorcet winners.")
    assert (t< T), f"Identificaiton steps {t} exceeded horizon {T}"
    cumulative_regret[:len(objectives)] = np.cumsum(f_star - objectives)
    scores_est = np.zeros((U, K))

    #analytical L
    a = 1
    b = -U*math.sqrt(math.log(U*K*(T-t)))/(len(estimated_winners))
    c = 0
    d = -U*(T-t)*math.sqrt(math.log(U*K*(T-t)))/(K*len(estimated_winners))

    # Find the positive real roots of the cubic equation for L.
    L_analytical = math.ceil(find_positive_real_roots([a, b, c, d])** 2) 

    L = L_scaling*math.ceil((K ** (-2/3) * T ** (2/3) * U ** (2/3) * (math.log(U*K*T)) ** (1/3)) / len(estimated_winners) ** (2/3))
    print(L_analytical, L)
    if (abs(L_scaling - 1.0) < 1e-5):
        L = L_analytical
    wins = np.zeros((K, K, U,), dtype=float)
    counts = np.zeros((K,K,), dtype=float)
    for u in range(U):
        for i in range(K):
            if i == estimated_winners[u]:
                continue
            while counts[i, estimated_winners[u]] < L:
                outcome = sample_duel(P_users, i, estimated_winners[u], rng)
                counts[i, estimated_winners[u]] += 1
                counts[estimated_winners[u], i] += 1
                wins[i, estimated_winners[u], :] += outcome
                wins[estimated_winners[u], i, :] += 1 - outcome
                x_hat_1 = np.zeros(K)
                x_hat_1[i] = 1.0
                x_hat_2 = np.zeros(K)
                x_hat_2[estimated_winners[u]] = 1.0
                scores_1.append(s_true[:, i])
                scores_2.append(s_true[:, estimated_winners[u]])
                if sushi_scores is not None:
                    sushi_utils += 0.5 * (sushi_scores[:, i] + sushi_scores[:, estimated_winners[u]])

                fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
                fx_val_2, _ = objective_and_grad(x_hat_2, s_true)
                fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
                cumulative_regret[t] = (f_star - fx) + cumulative_regret[t-1]
                t += 1
                if t >= T:
                    break
            if t >= T: break
        if t >= T: 
            print("Warning, exploration steps exceeded time")
            break
    exploration_steps = t
    assert np.all(counts == counts.T)
    assert np.all(wins + wins.transpose((1,0,2)) == counts[:,:,np.newaxis])
    for i in range(K):
        wins[i,i,:] = 0.5
        counts[i,i] = 1.0
    scores_est = 2*np.array([wins[:, estimated_winners[u],u]/counts[:, estimated_winners[u]] for u in range(U)])
    scores_est = np.clip(scores_est, 0.0, 1.0)
    x_hat, _ = frank_wolfe_simplex(scores_est, K, iters=iters_fw, utilitarian=utilitarian)
    fx_val, _ = objective_and_grad(x_hat, s_true)
    fx = np.exp(fx_val)
    if t > 0:
        cumulative_regret[t:] = cumulative_regret[t - 1] + np.cumsum(np.full(T-t, f_star - fx))
    else:
        cumulative_regret[t:] = np.cumsum(np.full(T-t, f_star - fx))

    for _ in range(t, T):
        i,j = rng.choice(K, size=2, p=x_hat)
        scores_1.append(s_true[:, i]) 
        scores_2.append(s_true[:, j])
        if sushi_scores is not None:
            sushi_utils += 0.5 * (sushi_scores[:, i] + sushi_scores[:, j])
        outcome = sample_duel(P_users, i, j, rng)
    if sushi_scores is not None:
        return cumulative_regret, identification_steps, exploration_steps, np.array(scores_1), np.array(scores_2), sushi_utils
    return cumulative_regret, identification_steps, exploration_steps, np.array(scores_1), np.array(scores_2)

def eps_greedy(
    P_users: np.ndarray,
    condorcet_winners: np.ndarray,
    T: int,
    rng: Generator,
    iters_fw: int = 300,
    eps_scaling: float = 0.125,
    delta_scaling: float = 0.0025,
    utilitarian: bool = False,
    update_freq: int = 50,
    sushi_scores: np.ndarray = None,
) -> Tuple:
    """
    Epsilon-Greedy algorithm.
    
    1. Identification: Uses DKWT to find Condorcet winners (necessary to define scores).
    2. Eps-Greedy Loop: 
       - With prob epsilon_t: Explore (Duel random arm vs winner to refine score estimates).
       - With prob 1-epsilon_t: Exploit (Play the best policy found so far).
    Args:
        P_users: List of U user-specific preference matrices (K x K numpy arrays).
        condorcet_winners: True condorcet winners for each user (numpy array of length U).
        T: Total number of rounds.
        rng: Random number generator.
        iters_fw: Number of Frank-Wolfe iterations for optimization.
        eps_scaling: Scaling factor for epsilon_t.
        delta_scaling: Scaling factor for delta in DKWT.
        update_freq: Number of exploration steps to wait before updating the exploitation policy.
        sushi_scores: (optional) scores from original sushi data to compute true utility.
    Returns:
        cumulative_regret: Cumulative regret at each round t (numpy array of length T).
        identification_steps: Number of steps taken for condorcet winner identification.
        scores_1: array of score vectors for the first arm in each duel.
        scores_2: array of score vectors for the second arm in each duel.
        sushi_utils: (if sushi_scores is not None) array of utilitarian sushi utilities.
    """
    U, K, _ = P_users.shape

    s_true = score_from_preferences(P_users, condorcet_winners)
    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)

    delta = (K * np.log(K/2))/(delta_scaling * T)
    estimated_winners, t, wins, counts, objectives, scores_1, scores_2, sushi_utils = dkwt(P_users, s_true, delta=delta, rng=rng, sushi_scores=sushi_scores)
    
    assert(delta < 1 and delta > 0)
    identification_steps = t

    
    assert (t< T), f"Identificaiton steps {t} exceeded horizon {T}"

    cumulative_regret[:t] = np.cumsum(f_star - objectives)

    for i in range(K):
        wins[i,i,:] = 0.5
        counts[i,i] = 1.0
    pending_updates = update_freq

    pairs = [(i, estimated_winners[u]) for u in range(U) for i in range(K) if i != estimated_winners[u]]
    pairs = list(set(pairs))
    count_exploration = [counts[i, w] for i, w in pairs]
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}

    for step in range(t, T):
        epsilon = min(1.0, eps_scaling* U**(2/3) * (K) ** (1/3) * len(estimated_winners) ** (1/3) * (np.log(U*K*(step - identification_steps+1))) ** (1/3) / ((step - identification_steps+1) ** (1/3)))
        is_exploration = rng.random() < epsilon
        if is_exploration:
            
            u = rng.integers(0, U)
            winner = estimated_winners[u]

            i, winner = pairs[np.argmin(count_exploration)]

            outcome = sample_duel(P_users, i, winner, rng)
            scores_1.append(s_true[:, i])
            scores_2.append(s_true[:, winner])
            if sushi_scores is not None:
                sushi_utils += 0.5 * (sushi_scores[:, i] + sushi_scores[:, winner])
            # Update Stats
            counts[i, winner] += 1
            counts[winner, i] += 1
            wins[i, winner, :] += outcome
            wins[winner, i, :] += 1 - outcome
            count_exploration[pair_index[(i, winner)]] += 1
            if (winner, i) in pair_index:
                count_exploration[pair_index[(winner, i)]] += 1

            x_hat_1 = np.zeros(K); x_hat_1[i] = 1.0
            x_hat_2 = np.zeros(K); x_hat_2[winner] = 1.0
            
            fx_val_1, _ = objective_and_grad(x_hat_1, s_true)
            fx_val_2, _ = objective_and_grad(x_hat_2, s_true)
            fx = 0.5 * np.exp(fx_val_1) + 0.5 * np.exp(fx_val_2)
            
            cumulative_regret[step] = cumulative_regret[step-1] + (f_star - fx)
            pending_updates += 1

        else:
            if pending_updates >= update_freq:
                current_counts = counts.copy()
                current_counts[current_counts == 0] = 1.0
                scores_est = np.zeros((U, K))
                for u_idx in range(U):
                    w_idx = estimated_winners[u_idx]
                    scores_est[u_idx, :] = 2 * wins[:, w_idx, u_idx] / current_counts[:, w_idx]
                
                scores_est = np.clip(scores_est, 0.0, 1.0)
                x_current, _ = frank_wolfe_simplex(scores_est, K, iters=iters_fw, utilitarian=utilitarian)
                pending_updates = 0
            i,j = rng.choice(K, size=2, p=x_current)
            scores_1.append(s_true[:, i])
            scores_2.append(s_true[:, j])
            if sushi_scores is not None:
                sushi_utils += 0.5 * (sushi_scores[:, i] + sushi_scores[:, j])
            # Calculate Regret for playing the current best policy
            fx_val, _ = objective_and_grad(x_current, s_true)
            fx = np.exp(fx_val)
            cumulative_regret[step] = cumulative_regret[step-1] + (f_star - fx)

    if sushi_scores is not None:
        return cumulative_regret, identification_steps, np.array(scores_1), np.array(scores_2), sushi_utils
    return cumulative_regret, identification_steps, np.array(scores_1), np.array(scores_2)


def uniform_over_winners(
    P_users: np.ndarray,
    condorcet_winners: np.ndarray,
    T: int,
    rng: Generator,
    iters_fw: int = 300,
    delta_scaling: float = 0.0025,
    sushi_scores: np.ndarray = None,
) -> Tuple:
    """
    Identifies the Condorcet winner for each user using DKWT, then plays a 
    uniformly random policy across the set of identified winners.
    """
    U, K, _ = P_users.shape

    s_true = score_from_preferences(P_users, condorcet_winners)
    x_star, _ = frank_wolfe_simplex(s_true, K, iters=iters_fw)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)

    delta = (K * np.log(K/2))/(delta_scaling * T)
    estimated_winners, t, wins, counts, objectives, scores_1, scores_2, sushi_utils = dkwt(P_users, s_true, delta=delta, rng=rng, sushi_scores=sushi_scores)
    
    identification_steps = t
    
    
    assert (t< T), f"Identificaiton steps {t} exceeded horizon {T}"
        
    cumulative_regret[:t] = np.cumsum(f_star - objectives)
    
    # Construct uniform policy over winners
    x_uniform = np.zeros(K)
    for u in range(U):
        x_uniform[estimated_winners[u]] += 1.0 / U

    # Calculate welfare of this policy
    fx_val, _ = objective_and_grad(x_uniform, s_true)
    fx = np.exp(fx_val)
    # Fill remaining regret
    cumulative_regret[t:] = cumulative_regret[t-1] + np.cumsum(np.full(T - t, f_star - fx))
    
    for _ in range(t, T):
        i, j = rng.choice(K, size=2, p=x_uniform)
        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])
        if sushi_scores is not None:
            sushi_utils += 0.5 * (sushi_scores[:, i] + sushi_scores[:, j])

    if sushi_scores is not None:
        return cumulative_regret, identification_steps, np.array(scores_1), np.array(scores_2), sushi_utils
    return cumulative_regret, identification_steps, np.array(scores_1), np.array(scores_2)

def rucb(
    P_users: np.ndarray,
    condorcet_winners: np.ndarray,
    T: int,
    rng: Generator,
    alpha: float = 0.51
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RUCB algorithm (Zoghi et al. 2014) applied to the average preference matrix.
    
    Args:
        alpha: Exploration parameter (usually > 0.5).
    """
    U, K, _ = P_users.shape

    s_true = score_from_preferences(P_users, condorcet_winners)
    x_star, _ = frank_wolfe_simplex(s_true, K, iters=300)
    f_star_val, _ = objective_and_grad(x_star, s_true)
    f_star = np.exp(f_star_val)

    cumulative_regret = np.zeros(T, dtype=float)
    scores_1 = list()
    scores_2 = list()

    wins_matrix = np.zeros((K, K)) 
    n_matrix = np.zeros((K, K))

    t = 0
    
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((i, j))
    
    # --- Initialization Loop ---
    for i, j in pairs:
        if t >= T: break
        
        outcome = np.mean(sample_duel(P_users, i, j, rng))
        
        wins_matrix[i, j] += outcome
        wins_matrix[j, i] += 1 - outcome
            
        n_matrix[i, j] += 1
        n_matrix[j, i] += 1

        # Record scores for output
        scores_1.append(s_true[:, i])
        scores_2.append(s_true[:, j])

        x_hat_1 = np.zeros(K)
        x_hat_1[i] = 1.0
        x_hat_2 = np.zeros(K)
        x_hat_2[j] = 1.0
        fx = 0.5 * np.exp(objective_and_grad(x_hat_1, s_true)[0]) + 0.5 * np.exp(objective_and_grad(x_hat_2, s_true)[0])
        
        if t == 0:
            cumulative_regret[t] = f_star - fx
        else:
            cumulative_regret[t] = cumulative_regret[t-1] + (f_star - fx)
        t += 1

    # --- Main RUCB Loop ---
    while t < T:
        means = np.divide(wins_matrix, n_matrix, out=np.zeros_like(wins_matrix), where=n_matrix!=0)
        
        exploration_term = np.sqrt(np.divide(alpha * np.log(t), n_matrix, out=np.full_like(n_matrix, 1), where=n_matrix!=0))
        
        ucb_matrix = means + exploration_term
        np.fill_diagonal(ucb_matrix, 0.5)
        candidates = np.arange(K)[np.all(ucb_matrix >= 0.5, axis=1)]
        
        if len(candidates) == 0:
            i_arm = rng.integers(0, K)
        else:
            i_arm = rng.choice(candidates)
            
        ucb_against_i = ucb_matrix[:, i_arm].copy()
        #ucb_against_i[i_arm] = -1.0 # exclude self
        j_arm = np.argmax(ucb_against_i)
        

        outcome = np.mean(sample_duel(P_users, i_arm, j_arm, rng))

        wins_matrix[i_arm, j_arm] += outcome
        wins_matrix[j_arm, i_arm] += 1 - outcome
            
        n_matrix[i_arm, j_arm] += 1
        n_matrix[j_arm, i_arm] += 1
        
        scores_1.append(s_true[:, i_arm])
        scores_2.append(s_true[:, j_arm])

        x_hat_1 = np.zeros(K)
        x_hat_1[i_arm] = 1.0
        x_hat_2 = np.zeros(K)
        x_hat_2[j_arm] = 1.0
        fx = 0.5 * np.exp(objective_and_grad(x_hat_1, s_true)[0]) + 0.5 * np.exp(objective_and_grad(x_hat_2, s_true)[0])
        
        cumulative_regret[t] = cumulative_regret[t-1] + (f_star - fx)
        t += 1

    return cumulative_regret, np.array(scores_1), np.array(scores_2)