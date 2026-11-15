"""Run DGAM on precomputed cosine similarities and report balanced accuracy.

Expects one pickle per concept in ``<similarity_dir>/<DATASET>/``, named
``attribute_<n>_im_percentage_100.pkl``, each holding a list of
``(cosine_similarity, label)`` tuples (see ``dgam.similarities``).

Example:
    python -m dgam.evaluate --dataset SYNTHETIC --similarity-dir data/similarities
"""

import argparse
import json
import os
import pickle

from dgam.method import fit_concept, get_balanced_accuracy

# Concept ids used in the report. CelebA concept 25 ("No Beard") is a negated
# concept and was removed because CLIP does not handle negations on their own.
CONCEPTS = {
    "CUB": list(range(1, 313)),
    "VOC": list(range(1, 21)),
    "CELEBA": [i for i in range(1, 41) if i != 25],
}

CUTOFF = 20


def pickle_path(similarity_dir, dataset, attribute):
    return os.path.join(similarity_dir, dataset, f"attribute_{attribute}_im_percentage_100.pkl")


def concepts_for(dataset, similarity_dir):
    if dataset in CONCEPTS:
        return CONCEPTS[dataset]
    names = os.listdir(os.path.join(similarity_dir, dataset))
    return sorted(int(n.split("_")[1]) for n in names if n.startswith("attribute_"))


def evaluate(dataset, similarity_dir, cutoff=CUTOFF):
    attributes = concepts_for(dataset, similarity_dir)

    cosine_similarity_list_dict = {}
    score_dict = {}
    threshold_dict = {}
    for attribute in attributes:
        with open(pickle_path(similarity_dir, dataset, attribute), "rb") as f:
            cosine_similarity_list = sorted(pickle.load(f), key=lambda x: x[0])
        cosine_similarity_list_dict[attribute] = cosine_similarity_list
        score, threshold = fit_concept(cosine_similarity_list)
        score_dict[attribute] = [attribute, score]
        threshold_dict[attribute] = threshold

    curve = [
        get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, k)[0]
        for k in range(1, len(attributes) + 1)
    ]
    ranking = sorted(score_dict.values(), key=lambda x: x[1], reverse=True)
    return {
        "dataset": dataset,
        "n_concepts": len(attributes),
        "cutoff": cutoff,
        "balanced_accuracy_at_cutoff": curve[min(cutoff, len(curve)) - 1],
        "balanced_accuracy_all_concepts": curve[-1],
        "balanced_accuracy_curve": curve,
        "ranking": [{"concept": a, "score": s, "threshold": threshold_dict[a]} for a, s in ranking],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="CUB, VOC, CELEBA or SYNTHETIC")
    parser.add_argument("--similarity-dir", default="data/similarities")
    parser.add_argument("--cutoff", type=int, default=CUTOFF)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    result = evaluate(args.dataset, args.similarity_dir, args.cutoff)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.dataset}_dgam.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Dataset: {result['dataset']} ({result['n_concepts']} concepts)")
    print(
        f"Balanced accuracy, top {result['cutoff']} concepts: "
        f"{100 * result['balanced_accuracy_at_cutoff']:.2f}%"
    )
    print(f"Balanced accuracy, all concepts: {100 * result['balanced_accuracy_all_concepts']:.2f}%")
    print("Top 5 concepts by DGAM score:", [r["concept"] for r in result["ranking"][:5]])
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
