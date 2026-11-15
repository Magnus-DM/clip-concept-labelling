"""Step 1 of DGAM: CLIP cosine similarities between every image and every concept.

Writes one pickle per concept to ``<out-dir>/<DATASET>/`` with a sorted list of
``(cosine_similarity, label)`` tuples. The labels are the dataset's ground truth
and are only used for evaluation.

Expected dataset layouts (see the README for download links):
    data/CUB_200_2011/   images.txt, images/, attributes/image_attribute_labels.txt
    data/VOC2012/        JPEGImages/, ImageSets/Main/<class>_{train,val,trainval}.txt
    data/celebA/         list_attr_celeba.csv, Celebrity_Faces_Dataset/<image files>

Example:
    python -m dgam.similarities --dataset VOC --data-root data/VOC2012
"""

import argparse
import csv
import os
import pickle
import re

import torch
from PIL import Image

CELEBA_MAX_IMAGES = 50_000  # subset used in the report (hardware limits)
CUB_PHRASES = os.path.join(os.path.dirname(__file__), "..", "concepts", "cub_concept_phrases.txt")


def load_cub(root):
    """CUB-200-2011: 11,788 images, 312 concepts phrased as sentences."""
    image_ids, image_paths = [], []
    with open(os.path.join(root, "images.txt")) as f:
        for line in f:
            image_id, rel_path = line.split()
            image_ids.append(int(image_id))
            image_paths.append(os.path.join(root, "images", rel_path))

    with open(CUB_PHRASES) as f:
        concepts = [re.sub(r"^\d+\.\s*", "", line.strip()) for line in f if line.strip()]

    labels = {}  # (image_id, concept_id) -> 0/1
    with open(os.path.join(root, "attributes", "image_attribute_labels.txt")) as f:
        for line in f:
            parts = line.split()
            labels[(int(parts[0]), int(parts[1]))] = int(parts[2])

    label_rows = [[labels[(i, c)] for c in range(1, len(concepts) + 1)] for i in image_ids]
    return image_paths, concepts, label_rows


def load_voc(root):
    """Pascal VOC 2012 classification: 11,540 images, the 20 class names as concepts."""
    main_dir = os.path.join(root, "ImageSets", "Main")
    splits = ("train", "val", "trainval")

    ids = set()
    for split in splits:
        with open(os.path.join(main_dir, f"{split}.txt")) as f:
            ids.update(f.read().splitlines())
    image_ids = sorted(ids)
    image_paths = [os.path.join(root, "JPEGImages", f"{i}.jpg") for i in image_ids]

    concepts = sorted(
        {
            name.split("_")[0]
            for name in os.listdir(main_dir)
            if name not in {f"{s}.txt" for s in splits}
        }
    )

    labels = {}
    for c, name in enumerate(concepts):
        for split in splits:
            path = os.path.join(main_dir, f"{name}_{split}.txt")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    image_id, value = line.split()
                    # -1 = absent; 1 = present; 0 = "difficult" (counted as present)
                    labels[(image_id, c)] = 0 if int(value) == -1 else 1

    label_rows = [[labels[(i, c)] for c in range(len(concepts))] for i in image_ids]
    return image_paths, concepts, label_rows


def load_celeba(root):
    """CelebA: the first 50,000 images and the 40 attribute names as concepts."""
    image_paths, label_rows = [], []
    with open(os.path.join(root, "list_attr_celeba.csv"), newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        concepts = [name.replace("_", " ") for name in header[1:]]
        for row in reader:
            if len(image_paths) >= CELEBA_MAX_IMAGES:
                break
            image_paths.append(os.path.join(root, "Celebrity_Faces_Dataset", row[0]))
            label_rows.append([1 if int(v) == 1 else 0 for v in row[1:]])
    return image_paths, concepts, label_rows


LOADERS = {"CUB": load_cub, "VOC": load_voc, "CELEBA": load_celeba}


@torch.no_grad()
def compute_similarities(image_paths, concepts, model, preprocess, device, batch_size=64):
    import clip

    text = clip.tokenize(concepts).to(device)
    text_features = model.encode_text(text).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    rows = []
    for start in range(0, len(image_paths), batch_size):
        batch = [preprocess(Image.open(p)) for p in image_paths[start : start + batch_size]]
        image_features = model.encode_image(torch.stack(batch).to(device)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        rows.append(image_features @ text_features.T)
        print(f"  encoded {min(start + batch_size, len(image_paths))}/{len(image_paths)} images")
    return torch.cat(rows).cpu()  # (n_images, n_concepts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="data/similarities")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import clip

    model, preprocess = clip.load("ViT-B/32", device=args.device)
    image_paths, concepts, label_rows = LOADERS[args.dataset](args.data_root)
    print(f"{args.dataset}: {len(image_paths)} images, {len(concepts)} concepts")

    similarities = compute_similarities(image_paths, concepts, model, preprocess, args.device)

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for c in range(len(concepts)):
        pairs = sorted(
            ((similarities[i, c].item(), label_rows[i][c]) for i in range(len(image_paths))),
            key=lambda x: x[0],
        )
        with open(os.path.join(out_dir, f"attribute_{c + 1}_im_percentage_100.pkl"), "wb") as f:
            pickle.dump(pairs, f)
    print(f"Wrote {len(concepts)} concept files to {out_dir}")


if __name__ == "__main__":
    main()
