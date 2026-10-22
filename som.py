import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import rand_score
from sklearn.metrics.cluster import contingency_matrix as CM

Data_Fol = "DATA"
Res_Fol = "RESULT"
Embedding_Dimension = 300
Epochs = 70
Initial_LearningRate = 0.8
Random_Seed = 50

random.seed(Random_Seed)
np.random.seed(Random_Seed)

os.makedirs(Res_Fol, exist_ok=True)
os.makedirs(os.path.join(Res_Fol, "CM"), exist_ok=True)


def load_data(Data_Fol):
    categories = ["animals","countries","fruits","veggies"]
    words, labels, vectors = [] ,[] ,[]
    

    for category in categories:
        file_path = os.path.join(Data_Fol, f"{category}.txt") 

        if not os.path.exists(file_path):
            print(f" File is not found -> {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip() 

                if not line:
                    continue

                values = line.split()
                word = values[0] 
                vector = np.asarray(values[1:], dtype=np.float32)

                if vector.shape[0] != Embedding_Dimension:
                    print(
                        f"WARNING: Skipping '{word}' - expected "
                        f"{Embedding_Dimension}d, got {vector.shape[0]}d" )
                    continue

                words.append(word)
                labels.append(category)
                vectors.append(vector)

    print(f"Total words loaded: {len(words)}")
    print(f"Categories: {dict(Counter(labels))}")

    return np.array(vectors), labels, words


def initialize_som_weights(k, input_dim, data):
    """
    Initialize the SOM weights randomly.

    Parameters:
    k : int
        Number of neurons/clusters.
    input_dim : int
        Number of features in each input vector.

    Returns:
    weights : numpy array
        Matrix of shape (k, input_dim).
    """
    indices = np.random.choice(len(data), size=k, replace=False)

    weights = data[indices].copy().astype(np.float64)
    weights += np.random.randn(*weights.shape) * 0.01
    return weights


def calculate_euclidean_distances(input_vector, weights):
    """
    Calculate the Euclidean distance between one input vector
    and all SOM neurons.
    """

    distances = np.sqrt(np.sum((weights - input_vector) ** 2, axis=1))
    return distances


def find_bmu(input_vector, weights):
    """
    Find the Best Matching Unit (BMU).

    The BMU is the neuron with the smallest Euclidean distance
    from the input vector.
    """

    distances = calculate_euclidean_distances(input_vector, weights)
    bmu_index = np.argmin(distances)

    return bmu_index


def train_som(X, num_neurons, Epochs=70, learning_rate =0.8):
    np.random.seed(num_neurons)
    weights = initialize_som_weights (num_neurons, X.shape[1],X)

    for i in range(Epochs):
        current_lr = learning_rate * (1 - i / Epochs)
        shuffled = X[np.random.permutation(len(X))]

        for input_vector in shuffled:
            bmu_index = find_bmu (input_vector, weights)
            weights[bmu_index] = weights[bmu_index] + current_lr * (input_vector - weights[bmu_index])
        print(f"Epoch {i+1}/{Epochs} - LR: {current_lr:.4f}")  
    return weights


def build_contingency(labels, clusters):
    cats = sorted(set(labels))
    cls = sorted(set(clusters))

    tab = np.zeros((len(cats), len(cls)), dtype=int)
    for t, c in zip(labels, clusters):
        tab[cats.index(t)][cls.index(c)] += 1

    return tab, cats, cls

def compute_purity(tab):
    return tab.max(axis=0).sum() / tab.sum()

def pairs_count(x):
    return x * (x - 1) // 2

def compute_rand_index(labels, clusters):
    tab, _, _ = build_contingency(labels, clusters)
    n = len(labels)

    same_cluster = sum(pairs_count(c) for c in tab.sum(axis=0))
    same_class = sum(pairs_count(r) for r in tab.sum(axis=1))
    same_both = sum(pairs_count(v) for v in tab.flatten())
    total = pairs_count(n)

    TP = same_both
    FP = same_cluster - TP
    FN = same_class - TP
    TN = total - TP - FP - FN

    ri = (TP + TN) / total
    return ri, TP, FP, FN, TN

def print_table(tab, cats, cls):
    cols = ["C" + str(c) for c in cls]
    w = max(max(len(r) for r in cats), 10) + 2
    print(" " * w + "".join(f"{c:>7}" for c in cols))
    for i, r in enumerate(cats):
        line = f"{r:<{w}}" + "".join(f"{tab[i][j]:>7}" for j in range(len(cols)))
        print(line)

def save_table(f, tab, cats, cls):
    cols = ["C" + str(c) for c in cls]
    w = max(max(len(r) for r in cats), 10) + 2
    f.write(" " * w + "".join(f"{c:>7}" for c in cols) + "\n")
    for i, r in enumerate(cats):
        line = f"{r:<{w}}" + "".join(f"{tab[i][j]:>7}" for j in range(len(cols)))
        f.write(line + "\n")

def assign_clusters(X, weights):
    n = X.shape[0]
    clusters = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.linalg.norm(weights - X[i], axis=1)
        clusters[i] = np.argmin(d)
    return clusters

def evaluate_for_k(X, labels, k, train_fn, out_dir):
    weights = train_fn(X, num_neurons=k)
    clusters = assign_clusters(X, weights)

    tab, cats, cls = build_contingency(labels, clusters)
    p = compute_purity(tab)
    ri, TP, FP, FN, TN = compute_rand_index(labels, clusters)

    print(f"\nk = {k}")
    print(f"Purity = {p:.4f}")
    print(f"Rand Index = {ri:.4f}")
    print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print_table(tab, cats, cls)

    cm_path = os.path.join(out_dir, "CM", f"k_{k}.txt")
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(f"k = {k}\n")
        f.write(f"Purity = {p:.4f}\n")
        f.write(f"Rand Index = {ri:.4f}\n")
        f.write(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}\n\n")
        save_table(f, tab, cats, cls)

    return p, ri, tab, cats, cls, (TP, FP, FN, TN)

def run_full_evaluation(X, labels, train_fn, k_values=range(2, 11), out_dir="RESULT"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "CM"), exist_ok=True)

    ks, purities, rand_scores = [], [], []

    all_path = os.path.join(out_dir, "contingency_tables.txt")
    csv_path = os.path.join(out_dir, "results.csv")

    f_all = open(all_path, "w", encoding="utf-8")
    f_csv = open(csv_path, "w", encoding="utf-8")
    f_csv.write("k,purity,rand_index\n")

    for k in k_values:
        p, ri, tab, cats, cls, pairs = evaluate_for_k(X, labels, k, train_fn, out_dir)

        f_all.write(f"\nk = {k}\n")
        f_all.write(f"Purity = {p:.4f}\n")
        f_all.write(f"Rand Index = {ri:.4f}\n")
        f_all.write(f"TP={pairs[0]}  FP={pairs[1]}  FN={pairs[2]}  TN={pairs[3]}\n")
        save_table(f_all, tab, cats, cls)

        f_csv.write(f"{k},{p:.4f},{ri:.4f}\n")

        ks.append(k)
        purities.append(p)
        rand_scores.append(ri)

    f_all.close()
    f_csv.close()

    plot_metrics(ks, purities, rand_scores)
    
    print("\n--- check with sklearn ---")
    for k in k_values:
        weights = train_fn(X, num_neurons=k)
        clusters = assign_clusters(X, weights)
        mine, *_ = compute_rand_index(labels, clusters)
        sk = rand_score(labels, clusters)
        print(f"k={k}: mine={mine:.4f}  sklearn={sk:.4f}")

    return ks, purities, rand_scores

MIN_K = 2
MAX_K = 10

METRICS_FIG_SIZE = (7, 4.5)
CM_FIG_HEIGHT = 4.8
CM_DPI = 140


def plot_contingency_matrix_from_table(tab, cats, cls, k):
    """
    Plot and save the contingency matrix as a heatmap.
    """

    fig_width = max(5.5, len(cls) * 0.7 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, CM_FIG_HEIGHT))

    im = ax.imshow(tab, cmap="Purples", aspect="auto")

    colorbar = plt.colorbar(im, ax=ax)
    colorbar.set_label("Count")

    cluster_labels = ["C" + str(c) for c in cls]

    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels(cluster_labels, rotation=35, ha="right")

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)

    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            value = tab[i, j]
            text_color = "white" if value > tab.max() / 2 else "black"

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold"
            )

    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Category")
    ax.set_title(f"Contingency Matrix - k = {k}")

    plt.tight_layout()

    output_path = os.path.join(Res_Fol, "CM", f"contingency_k_{k}.png")
    plt.savefig(output_path, dpi=CM_DPI)
    plt.close()

    print(f"Saved contingency matrix image -> {output_path}")


def plot_metrics(k_values, purity_scores, rand_scores):
    """
    Plot Purity and Rand Index for all k values.
    """

    plt.figure(figsize=METRICS_FIG_SIZE)

    plt.plot(
    k_values,
    purity_scores,
    marker="o",
    color="#C084FC",
    linewidth=2.5,
    label="Purity"
)

    plt.plot(
    k_values,
    rand_scores,
    marker="s",
    color="#6B21A8",
    linewidth=2.5,
    label="Rand Index"
)

    plt.xlabel("Number of Neurons (k)")
    plt.ylabel("Score")
    plt.title("SOM Clustering Evaluation")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(Res_Fol, "metrics_plot.png")
    plt.savefig(output_path, dpi=CM_DPI)
    plt.close()

    print(f"Saved metrics plot -> {output_path}")


def main():
    print("=" * 55)
    print("SOM WORD CLUSTERING")
    print("=" * 55)

    X, labels, words = load_data(Data_Fol)

    print(f"Dataset shape: {X.shape}")

    k_values = []
    purity_scores = []
    rand_scores = []

    results_path = os.path.join(Res_Fol, "results.csv")
    summary_path = os.path.join(Res_Fol, "summary_results.txt")

    with open(results_path, "w", encoding="utf-8") as csv_file, \
         open(summary_path, "w", encoding="utf-8") as summary_file:

        csv_file.write("k,purity,rand_index\n")

        for k in range(MIN_K, MAX_K + 1):
            print("\n" + "=" * 55)
            print(f"Training SOM with k = {k}")
            print("=" * 55)

            purity, ri, tab, cats, cls, pairs = evaluate_for_k(
                X,
                labels,
                k,
                train_som,
                Res_Fol
            )

            k_values.append(k)
            purity_scores.append(purity)
            rand_scores.append(ri)

            csv_file.write(f"{k},{purity:.4f},{ri:.4f}\n")

            summary_file.write(f"k = {k}\n")
            summary_file.write(f"Purity = {purity:.4f}\n")
            summary_file.write(f"Rand Index = {ri:.4f}\n")
            summary_file.write(
                f"TP={pairs[0]}  FP={pairs[1]}  FN={pairs[2]}  TN={pairs[3]}\n\n"
            )

            plot_contingency_matrix_from_table(tab, cats, cls, k)

    plot_metrics(k_values, purity_scores, rand_scores)

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)

    for i in range(len(k_values)):
        print(
            f"k = {k_values[i]:2d} | "
            f"Purity = {purity_scores[i]:.4f} | "
            f"Rand Index = {rand_scores[i]:.4f}"
        )

    print("=" * 55)
    print("All results saved in:", Res_Fol)
    print("=" * 55)


if __name__ == "__main__":
    main()
