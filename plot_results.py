import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def compute_user_utilities(scores_1: np.ndarray, scores_2: np.ndarray) -> np.ndarray:
    total_scores = 0.5*scores_1 + 0.5*scores_2
    return np.cumsum(total_scores, axis=1)

def min_welfare(user_utilities: np.ndarray) -> np.ndarray:
    return np.min(user_utilities, axis=2)


def utlitarian_welfare(user_utilities: np.ndarray) -> np.ndarray:
    return np.mean(user_utilities, axis=2)

def nsw_welfare(user_utilities: np.ndarray) -> np.ndarray:
    U = user_utilities.shape[2]
    return np.prod(user_utilities, axis=2) ** (1.0 / U)

def gini_welfare(user_utilities: np.ndarray) -> np.ndarray:
    runs, T, U = user_utilities.shape
    sum_abs_diff = np.zeros((runs, T))
    for i in range(U):
        for j in range(U):
            sum_abs_diff += np.abs(user_utilities[:, :, i] - user_utilities[:, :, j])
    sum_xi = np.sum(user_utilities, axis=2)
    gini = sum_abs_diff / (2 * U * sum_xi)
    return np.nan_to_num(gini, nan=0.0)

def plot_with_ci(ax, data, label, color, linestyle="-"):
    run_counts, T = data.shape
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(run_counts)
    ax.plot(mean, label=label, color=color, linestyle=linestyle, rasterized=True)
    ax.fill_between(np.arange(T), mean - ci, mean + ci, color=color, alpha=0.2, rasterized=True)

def plot_results_from_folder(folder, output_dir=None, suffix=None):
    if output_dir is None:
        output_dir = folder
    
    # Load data
    try:
        etc_regret = np.load(os.path.join(folder, "etc_regret.npy"))
        etc_scores_1 = np.load(os.path.join(folder, "etc_scores_1.npy"))
        etc_scores_2 = np.load(os.path.join(folder, "etc_scores_2.npy"))
        
        eps_regret = np.load(os.path.join(folder, "eps_regret.npy"))
        eps_scores_1 = np.load(os.path.join(folder, "eps_scores_1.npy"))
        eps_scores_2 = np.load(os.path.join(folder, "eps_scores_2.npy"))

        etc_util_regret = np.load(os.path.join(folder, "etc_util_regret.npy"))
        etc_util_scores_1 = np.load(os.path.join(folder, "etc_util_scores_1.npy"))
        etc_util_scores_2 = np.load(os.path.join(folder, "etc_util_scores_2.npy"))

        eps_util_regret = np.load(os.path.join(folder, "eps_util_regret.npy"))
        eps_util_scores_1 = np.load(os.path.join(folder, "eps_util_scores_1.npy"))
        eps_util_scores_2 = np.load(os.path.join(folder, "eps_util_scores_2.npy"))
        
        uniform_regret = np.load(os.path.join(folder, "uniform_regret.npy"))
        uniform_scores_1 = np.load(os.path.join(folder, "uniform_scores_1.npy"))
        uniform_scores_2 = np.load(os.path.join(folder, "uniform_scores_2.npy"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Infer parameters
    run_counts, T = etc_regret.shape
    _, _, U = etc_scores_1.shape
    
    # Compute utilities
    etc_utilities = compute_user_utilities(etc_scores_1, etc_scores_2)
    eps_utilities = compute_user_utilities(eps_scores_1, eps_scores_2)
    etc_util_utilities = compute_user_utilities(etc_util_scores_1, etc_util_scores_2)
    eps_util_utilities = compute_user_utilities(eps_util_scores_1, eps_util_scores_2)
    uniform_utilities = compute_user_utilities(uniform_scores_1, uniform_scores_2)
    

    # Update matplotlib parameters to use LaTeX and Times font
    plt.rcParams.update({
        "text.usetex": True,                  # Use LaTeX to render text
        "font.family": "serif",               # Use serif font family
        "font.serif": ["Times"],              # Specify Times as the serif font
        # The preamble helps ensure the exact same packages are used (optional but recommended)
        "text.latex.preamble": r"\usepackage{times} \usepackage{amsmath}", 
        "axes.labelsize": 24,                 # Adjust to match paper (usually 10-12pt)
        "font.size": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })


    # Define accessible colors (Okabe-Ito palette)
    color_etc = "#0072B2"      # Blue
    color_eps = "#D55E00"      # Vermilion
    color_uniform = "#009E73"  # Bluish Green

    # Plot Regret
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_ci(ax, etc_regret, "Fair-ETC (Ours)", color_etc)
    plot_with_ci(ax, etc_util_regret, "Utilitarian-ETC", color_etc, linestyle="--")
    plot_with_ci(ax, eps_regret, "Fair-$\epsilon$-Greedy (Ours)", color_eps)
    plot_with_ci(ax, eps_util_regret, "Utilitarian-$\epsilon$-Greedy", color_eps, linestyle="--")
    plot_with_ci(ax, uniform_regret, "Uniform-Over-Users", color_uniform)
    
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Cumulative Regret")
    #ax.set_title(f"Cumulative Regret (U={U})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    reg_filename = f"regret_comparison_{suffix}.pdf" if suffix else "regret_comparison.pdf"
    reg_path = os.path.join(output_dir, reg_filename)
    
    fig.savefig(reg_path)
    fig.savefig(reg_path.replace(".pdf", ".svg"))
    fig.savefig(reg_path.replace(".pdf", ".png"))
    plt.close(fig)
    
    # Calculate Min Welfare
    etc_min_welfare = min_welfare(etc_utilities)
    eps_min_welfare = min_welfare(eps_utilities)
    etc_util_min_welfare = min_welfare(etc_util_utilities)
    eps_util_min_welfare = min_welfare(eps_util_utilities)
    uniform_min_welfare = min_welfare(uniform_utilities)
    
    # Plot Welfare
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_with_ci(ax2, etc_min_welfare, "Fair-ETC (Ours)", color_etc)
    plot_with_ci(ax2, etc_util_min_welfare, "Utilitarian-ETC", color_etc, linestyle="--")
    plot_with_ci(ax2, eps_min_welfare, "Fair-$\epsilon$-Greedy (Ours)", color_eps)
    plot_with_ci(ax2, eps_util_min_welfare, "Utilitarian-$\epsilon$-Greedy", color_eps, linestyle="--")
    plot_with_ci(ax2, uniform_min_welfare, "Uniform-Over-Users", color_uniform)
    
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Minimum User Welfare")
    #ax2.set_title(f"Max-Min Fairness (U={U})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    wel_filename = f"min_welfare_comparison_{suffix}.pdf" if suffix else "min_welfare_comparison.pdf"
    wel_path = os.path.join(output_dir, wel_filename)
    
    fig2.savefig(wel_path)
    fig2.savefig(wel_path.replace(".pdf", ".svg"))
    fig2.savefig(wel_path.replace(".pdf", ".png"))
    plt.close(fig2)

    # Calculate NSW Welfare
    etc_nsw_welfare = nsw_welfare(etc_utilities)
    eps_nsw_welfare = nsw_welfare(eps_utilities)
    etc_util_nsw_welfare = nsw_welfare(etc_util_utilities)
    eps_util_nsw_welfare = nsw_welfare(eps_util_utilities)
    uniform_nsw_welfare = nsw_welfare(uniform_utilities)

    # Plot NSW Welfare
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_with_ci(ax3, etc_nsw_welfare, "Fair-ETC (Ours)", color_etc)
    plot_with_ci(ax3, etc_util_nsw_welfare, "Utilitarian-ETC", color_etc, linestyle="--")
    plot_with_ci(ax3, eps_nsw_welfare, "Fair-$\epsilon$-Greedy (Ours)", color_eps)
    plot_with_ci(ax3, eps_util_nsw_welfare, "Utilitarian-$\epsilon$-Greedy", color_eps, linestyle="--")
    plot_with_ci(ax3, uniform_nsw_welfare, "Uniform-Over-Users", color_uniform)

    ax3.set_xlabel("Rounds")
    ax3.set_ylabel("NSW User Welfare")
    #ax3.set_title(f"NSW Welfare (U={U})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    nsw_filename = f"nsw_welfare_comparison_{suffix}.pdf" if suffix else "nsw_welfare_comparison.pdf"
    nsw_path = os.path.join(output_dir, nsw_filename)

    fig3.savefig(nsw_path)
    fig3.savefig(nsw_path.replace(".pdf", ".svg"))
    fig3.savefig(nsw_path.replace(".pdf", ".png"))
    plt.close(fig3)

    # Calculate Utilitarian Welfare
    etc_util_welfare = utlitarian_welfare(etc_utilities)
    eps_util_welfare = utlitarian_welfare(eps_utilities)
    etc_util_util_welfare = utlitarian_welfare(etc_util_utilities)
    eps_util_util_welfare = utlitarian_welfare(eps_util_utilities)
    uniform_util_welfare = utlitarian_welfare(uniform_utilities)

    # Plot Utilitarian Welfare
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    plot_with_ci(ax4, etc_util_welfare, "Fair-ETC (Ours)", color_etc)
    plot_with_ci(ax4, etc_util_util_welfare, "Utilitarian-ETC", color_etc, linestyle="--")
    plot_with_ci(ax4, eps_util_welfare, "Fair-$\epsilon$-Greedy (Ours)", color_eps)
    plot_with_ci(ax4, eps_util_util_welfare, "Utilitarian-$\epsilon$-Greedy", color_eps, linestyle="--")
    plot_with_ci(ax4, uniform_util_welfare, "Uniform-Over-Users", color_uniform)
    ax4.set_xlabel("Rounds")
    ax4.set_ylabel("Utilitarian User Welfare")
    #ax4.set_title(f"Utilitarian Welfare (U={U})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    util_filename = f"utilitarian_welfare_comparison_{suffix}.pdf" if suffix else "utilitarian_welfare_comparison.pdf"
    util_path = os.path.join(output_dir, util_filename)
    fig4.savefig(util_path)
    fig4.savefig(util_path.replace(".pdf", ".svg"))
    fig4.savefig(util_path.replace(".pdf", ".png"))
    plt.close(fig4)


    # Calculate Gini Coefficient
    etc_gini = gini_welfare(etc_utilities)
    eps_gini = gini_welfare(eps_utilities)
    etc_util_gini = gini_welfare(etc_util_utilities)
    eps_util_gini = gini_welfare(eps_util_utilities)
    uniform_gini = gini_welfare(uniform_utilities)

    # Plot Gini Coefficient
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    plot_with_ci(ax5, etc_gini, "Fair-ETC (Ours)", color_etc)
    plot_with_ci(ax5, etc_util_gini, "Utilitarian-ETC", color_etc, linestyle="--")
    plot_with_ci(ax5, eps_gini, "Fair-$\epsilon$-Greedy (Ours)", color_eps)
    plot_with_ci(ax5, eps_util_gini, "Utilitarian-$\epsilon$-Greedy", color_eps, linestyle="--")
    plot_with_ci(ax5, uniform_gini, "Uniform-Over-Users", color_uniform)
    
    ax5.set_xlabel("Rounds")
    ax5.set_ylabel("Gini Coefficient")
    #ax6.set_title(f"Gini Coefficient (Inequality) (U={U})")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    gini_filename = f"gini_coefficient_{suffix}.pdf" if suffix else "gini_coefficient.pdf"
    gini_path = os.path.join(output_dir, gini_filename)
    fig5.savefig(gini_path)
    fig5.savefig(gini_path.replace(".pdf", ".svg"))
    fig5.savefig(gini_path.replace(".pdf", ".png"))
    plt.close(fig5)

    # Save tables of mean and CI at the last time step
    metrics_summary = {
        "Cumulative Regret": {
            "Fair-ETC (Ours)": etc_regret[:, -1],
            "Utilitarian-ETC": etc_util_regret[:, -1],
            "Fair-$\epsilon$-Greedy (Ours)": eps_regret[:, -1],
            "Utilitarian-$\epsilon$-Greedy": eps_util_regret[:, -1],
            "Uniform-Over-Users": uniform_regret[:, -1]
        },
        "Min Welfare": {
            "Fair-ETC (Ours)": etc_min_welfare[:, -1],
            "Utilitarian-ETC": etc_util_min_welfare[:, -1],
            "Fair-$\epsilon$-Greedy (Ours)": eps_min_welfare[:, -1],
            "Utilitarian-$\epsilon$-Greedy": eps_util_min_welfare[:, -1],
            "Uniform-Over-Users": uniform_min_welfare[:, -1]
        },
        "NSW Welfare": {
            "Fair-ETC (Ours)": etc_nsw_welfare[:, -1],
            "Utilitarian-ETC": etc_util_nsw_welfare[:, -1],
            "Fair-$\epsilon$-Greedy (Ours)": eps_nsw_welfare[:, -1],
            "Utilitarian-$\epsilon$-Greedy": eps_util_nsw_welfare[:, -1],
            "Uniform-Over-Users": uniform_nsw_welfare[:, -1]
        },
        "Utilitarian Welfare": {
            "Fair-ETC (Ours)": etc_util_welfare[:, -1],
            "Utilitarian-ETC": etc_util_util_welfare[:, -1],
            "Fair-$\epsilon$-Greedy (Ours)": eps_util_welfare[:, -1],
            "Utilitarian-$\epsilon$-Greedy": eps_util_util_welfare[:, -1],
            "Uniform-Over-Users": uniform_util_welfare[:, -1]
        },
        "Gini Coefficient": {
            "Fair-ETC (Ours)": etc_gini[:, -1],
            "Utilitarian-ETC": etc_util_gini[:, -1],
            "Fair-$\epsilon$-Greedy (Ours)": eps_gini[:, -1],
            "Utilitarian-$\epsilon$-Greedy": eps_util_gini[:, -1],
            "Uniform-Over-Users": uniform_gini[:, -1]
        }
    }

    table_filename = f"metrics_table_{suffix}.txt" if suffix else "metrics_table.txt"
    table_path = os.path.join(output_dir, table_filename)
    
    with open(table_path, "w") as f:
        for metric_name, alg_data in metrics_summary.items():
            f.write(f"Metric: {metric_name}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'Algorithm':<25} | {'Mean':<12} | {'CI':<12}\n")
            f.write("-" * 55 + "\n")
            for alg, data in alg_data.items():
                # data shape: (runs, T)
                mean_val = np.mean(data)
                std_val = np.std(data)
                n_runs = len(data)
                ci_val = 1.96 * std_val / np.sqrt(n_runs)
                
                f.write(f"{alg:<25} | {mean_val:<12.4f} | {ci_val:<12.4f}\n")
            f.write("\n")
    
    print(f"Saved metrics table to: {table_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from folder")
    parser.add_argument("folder", type=str, help="Folder containing .npy files")
    parser.add_argument("--output_dir", type=str, help="Output directory for figures", default=None)
    parser.add_argument("--suffix", type=str, help="Suffix for filenames", default=None)
    args = parser.parse_args()
    
    plot_results_from_folder(args.folder, args.output_dir, args.suffix)
    
    plt.show()