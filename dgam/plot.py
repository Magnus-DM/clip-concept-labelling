"""Plot DGAM results: accuracy versus number of concepts, and one fitted histogram.

Example:
    python -m dgam.plot --dataset SYNTHETIC --concept 40
"""

import argparse
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dgam.evaluate import pickle_path  # noqa: E402
from dgam.method import (  # noqa: E402
    double_gaussian_approximation,
    double_gaussian_correction,
    get_threshold,
)


def plot_curve(result, out_file):
    curve = result["balanced_accuracy_curve"]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(curve) + 1), curve, label="DGAM")
    plt.axvline(
        result["cutoff"], color="grey", linestyle="--", label=f"cutoff = {result['cutoff']}"
    )
    plt.xlabel("Number of selected concepts (DGAM ranking)")
    plt.ylabel("Balanced accuracy")
    plt.title(result["dataset"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_histogram(cosine_similarity_list, concept, out_file):
    params, r_squared = double_gaussian_approximation(cosine_similarity_list)
    params, _ = double_gaussian_correction(cosine_similarity_list, params, r_squared)
    threshold = get_threshold(params)
    amp1, mu1, sigma1, amp2, mu2, sigma2 = params

    data = np.array([x[0] for x in cosine_similarity_list])
    labels = np.array([x[1] for x in cosine_similarity_list])
    bins = np.histogram_bin_edges(data, bins=50)
    x = np.linspace(data.min(), data.max(), 500)

    plt.figure(figsize=(8, 5))
    plt.hist(data[labels == 0], bins=bins, alpha=0.5, label="absent (true label)")
    plt.hist(data[labels == 1], bins=bins, alpha=0.5, label="present (true label)")
    plt.plot(x, amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2)), "--", label="Gaussian 1")
    plt.plot(x, amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2)), "--", label="Gaussian 2")
    plt.axvline(threshold, color="black", label="DGAM threshold")
    plt.xlabel("Cosine similarity (image, concept)")
    plt.ylabel("Count")
    plt.title(f"Concept {concept}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--concept", type=int, help="concept id for the histogram plot")
    parser.add_argument("--similarity-dir", default="data/similarities")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(os.path.join(args.results_dir, f"{args.dataset}_dgam.json")) as f:
        result = json.load(f)
    curve_file = os.path.join(args.results_dir, f"{args.dataset}_accuracy_vs_concepts.png")
    plot_curve(result, curve_file)
    print(f"Saved {curve_file}")

    concept = args.concept or result["ranking"][0]["concept"]
    with open(pickle_path(args.similarity_dir, args.dataset, concept), "rb") as f:
        cosine_similarity_list = pickle.load(f)
    hist_file = os.path.join(args.results_dir, f"{args.dataset}_concept_{concept}_fit.png")
    plot_histogram(cosine_similarity_list, concept, hist_file)
    print(f"Saved {hist_file}")


if __name__ == "__main__":
    main()
