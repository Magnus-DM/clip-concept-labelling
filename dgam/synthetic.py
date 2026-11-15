"""Generate a synthetic dataset of CLIP-like cosine similarities.

Each concept gets ``n_images`` (similarity, label) pairs. Absent images (label 0)
and present images (label 1) are drawn from two normal distributions whose gap
varies per concept, from well separated (easy for CLIP) to fully overlapping
(CLIP cannot see the concept). This mimics what real CLIP similarities look like
and lets the whole pipeline run in seconds without downloading any dataset.

Example:
    python -m dgam.synthetic --out-dir data/similarities --n-concepts 40
"""

import argparse
import os
import pickle

import numpy as np


def make_concept(rng, n_images, prevalence, separation):
    labels = (rng.random(n_images) < prevalence).astype(int)
    sigma = rng.uniform(0.010, 0.016)
    mu_absent = rng.uniform(0.19, 0.23)
    mu_present = mu_absent + separation * sigma
    similarities = np.where(
        labels == 1,
        rng.normal(mu_present, sigma * rng.uniform(0.8, 1.2), n_images),
        rng.normal(mu_absent, sigma, n_images),
    )
    return sorted(zip(similarities.tolist(), labels.tolist(), strict=True), key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="data/similarities")
    parser.add_argument("--n-concepts", type=int, default=40)
    parser.add_argument("--n-images", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(args.out_dir, "SYNTHETIC")
    os.makedirs(out_dir, exist_ok=True)

    truth = []
    for concept in range(1, args.n_concepts + 1):
        separation = rng.uniform(0.0, 5.0)
        prevalence = rng.uniform(0.1, 0.5)
        data = make_concept(rng, args.n_images, prevalence, separation)
        path = os.path.join(out_dir, f"attribute_{concept}_im_percentage_100.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        truth.append((concept, separation, prevalence))

    print(f"Wrote {args.n_concepts} concepts x {args.n_images} images to {out_dir}")
    easiest = sorted(truth, key=lambda t: t[1], reverse=True)[:5]
    print("Best separated concepts (ground truth):", [c for c, _, _ in easiest])


if __name__ == "__main__":
    main()
